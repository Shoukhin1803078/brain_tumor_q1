"""
Step 1+2: Pretrained-backbone + mask-guided ROI cropping for brain tumor classification.

Improvements over the paper's from-scratch 4-conv CNN + RF:
  - EfficientNetB0 backbone, ImageNet-pretrained, two-stage fine-tuning
  - BatchNorm in the head (missing in the original)
  - Patient-level train/val/test split (using cjdata.PID) -> no leakage
  - Class-weighted loss (meningioma is underrepresented and hardest class)
  - Mask-guided ROI cropping: crop around the tumor mask (with margin) before
    resizing, instead of feeding the whole 512x512 slice with skull/background.
    This directly targets the meningioma<->pituitary/glioma confusion seen in
    the whole-slice baseline, since meningioma sits near the skull and gets
    confused with surrounding structure when the model sees the full image.
  - Test-time augmentation (TTA) at inference for a free accuracy bump.

Run on Kaggle with GPU enabled. DATA_DIR points at the figshare .mat files.
"""

import os
import glob
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/ashkhagan/figshare-brain-tumor-dataset/dataset/data"
IMG_SIZE = 224                                     # EfficientNetB0 native size
BATCH_SIZE = 16
HEAD_EPOCHS = 10          # stage 1: train head only, backbone frozen
FINE_TUNE_EPOCHS = 60     # stage 2: unfreeze top of backbone, fine-tune end-to-end
FINE_TUNE_AT = 0.8        # unfreeze the top 80% of backbone layers (was 0.5)
CROP_MARGIN = 0.25        # fraction of tumor bbox size added as margin on each side
TTA_ROUNDS = 8            # number of augmented views averaged at test time
SEED = 42
NUM_CLASSES = 3
LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary"}  # cjdata.label convention


# ---------------------------------------------------------------------------
# 1. Load .mat files -> images, masks, labels, patient IDs
# ---------------------------------------------------------------------------
def load_dataset(data_dir):
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")

    images, masks, labels, pids = [], [], [], []
    for f in mat_files:
        with h5py.File(f, "r") as h:
            cjdata = h["cjdata"]
            img = np.array(cjdata["image"]).astype(np.float32)
            mask = np.array(cjdata["tumorMask"]).astype(np.float32)
            label = int(np.array(cjdata["label"]).item())
            pid = np.array(cjdata["PID"])
            pid = "".join(chr(c) for c in pid.flatten()) if pid.dtype.kind in "uS" else str(int(pid.item()))

            images.append(img)
            masks.append(mask)
            labels.append(label - 1)  # -> 0,1,2
            pids.append(pid)

    return images, masks, np.array(labels), np.array(pids)


def crop_to_mask(img, mask, margin=CROP_MARGIN):
    """Crop image to the tumor mask's bounding box, with a margin, so the
    model sees tumor + local context instead of the whole brain + skull."""
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return img  # no mask -> fall back to full image
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    pad_y, pad_x = int(h * margin) + 1, int(w * margin) + 1
    y0 = max(0, y0 - pad_y)
    y1 = min(img.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(img.shape[1], x1 + pad_x)
    return img[y0:y1, x0:x1]


def preprocess(img, mask):
    img = crop_to_mask(img, mask)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    img = tf.image.resize(img[..., None], [IMG_SIZE, IMG_SIZE]).numpy()
    img = np.repeat(img, 3, axis=-1)  # grayscale -> 3-channel for ImageNet backbone
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Patient-level stratified split
# ---------------------------------------------------------------------------
def patient_level_split(images, labels, pids, seed=SEED):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(images, labels, groups=pids))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)  # 0.125*0.8=0.1 -> 70/10/20
    train_idx, val_idx = next(gss2.split(
        [images[i] for i in trainval_idx],
        labels[trainval_idx],
        groups=pids[trainval_idx],
    ))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    assert set(pids[train_idx]).isdisjoint(pids[val_idx])
    assert set(pids[train_idx]).isdisjoint(pids[test_idx])
    assert set(pids[val_idx]).isdisjoint(pids[test_idx])

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# 3. Model: EfficientNetB0 backbone + BatchNorm head
# ---------------------------------------------------------------------------
def build_model():
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, backbone


def make_augmenter():
    return tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
    ])


def tta_predict(model, augmenter, X, rounds=TTA_ROUNDS, batch_size=BATCH_SIZE):
    """Average predictions over several augmented views of each test image."""
    probs = np.zeros((X.shape[0], NUM_CLASSES), dtype=np.float32)
    # one clean (non-augmented) pass
    probs += model.predict(X, batch_size=batch_size, verbose=0)
    for _ in range(rounds):
        X_aug = augmenter(X, training=True).numpy()
        probs += model.predict(X_aug, batch_size=batch_size, verbose=0)
    probs /= (rounds + 1)
    return probs


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset...")
    images_raw, masks_raw, labels, pids = load_dataset(DATA_DIR)
    print(f"Loaded {len(images_raw)} slices from {len(set(pids))} patients")

    train_idx, val_idx, test_idx = patient_level_split(images_raw, labels, pids)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    print("Preprocessing images (mask-guided crop + resize/normalize)...")
    X = np.stack([preprocess(images_raw[i], masks_raw[i]) for i in range(len(images_raw))])
    y = labels

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    augmenter = make_augmenter()

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED)
                .batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model, backbone = build_model()

    class_weight = dict(enumerate(
        compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    ))
    print(f"Class weights {{meningioma, glioma, pituitary}}: {class_weight}")

    # --- Stage 1: train head only, backbone frozen ---
    model.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n=== Stage 1: training head (backbone frozen) ===")
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, class_weight=class_weight)

    # --- Stage 2: unfreeze top of backbone, fine-tune end-to-end ---
    backbone.trainable = True
    n_layers = len(backbone.layers)
    freeze_until = int(n_layers * (1 - FINE_TUNE_AT))
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(1e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        callbacks.ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True),
    ]
    print("\n=== Stage 2: fine-tuning top of backbone ===")
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              callbacks=cbs, class_weight=class_weight)

    # --- Evaluate: plain single-pass prediction ---
    print("\n=== Test set evaluation (single pass) ===")
    y_pred_plain = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE), axis=1)
    print(classification_report(y_test, y_pred_plain, target_names=list(LABEL_MAP.values())))
    print(confusion_matrix(y_test, y_pred_plain))

    # --- Evaluate: test-time augmentation ---
    print(f"\n=== Test set evaluation (TTA, {TTA_ROUNDS} rounds) ===")
    probs_tta = tta_predict(model, augmenter, X_test)
    y_pred_tta = np.argmax(probs_tta, axis=1)
    print(classification_report(y_test, y_pred_tta, target_names=list(LABEL_MAP.values())))
    print(confusion_matrix(y_test, y_pred_tta))


if __name__ == "__main__":
    main()
