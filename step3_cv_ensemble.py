"""
Step 3: Patient-level 5-fold cross-validation + CNN-embedding ensemble classifier.

Builds on step1_baseline.py (EfficientNetB0 backbone, mask-guided ROI cropping,
class weighting, TTA). This script adds:

  - 5-fold patient-level GroupKFold CV, so every patient is held out exactly
    once -> gives a mean +/- std accuracy instead of one noisy single-split
    number. This is the statistically defensible way to report results for
    a journal, especially with only 233 patients.
  - Per-fold: train the CNN, then extract its penultimate-layer embedding and
    fit a soft-voting ensemble (Random Forest + XGBoost + SVM) on top of it,
    compared against the CNN's own softmax head (with and without TTA).

Run on Kaggle with GPU enabled. This trains 5 models sequentially -- budget
several hours. If Kaggle's 9-12h session limit is a concern, reduce N_FOLDS
or FINE_TUNE_EPOCHS below.
"""

import os
import gc
import glob
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/ashkhagan/figshare-brain-tumor-dataset/dataset/data"
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
CROP_MARGIN = 0.25
TTA_ROUNDS = 5
N_FOLDS = 5
SEED = 42
NUM_CLASSES = 3
LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def crop_to_mask(img, mask, margin=CROP_MARGIN):
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return img
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
    img = np.repeat(img, 3, axis=-1)
    return (img * 255).astype(np.uint8)


def load_and_preprocess_dataset(data_dir):
    """Load + crop/resize each .mat file one at a time and write straight into
    the final uint8 array, instead of holding all 3000+ raw float32 images and
    masks in memory at once (~6GB peak) before preprocessing them. Same
    preprocessing per file, same file order -> identical output to the
    two-pass version, just without the memory spike that was causing the
    Kaggle notebook to OOM and restart."""
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")

    n = len(mat_files)
    X = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    pids = []

    for i, f in enumerate(mat_files):
        with h5py.File(f, "r") as h:
            cjdata = h["cjdata"]
            img = np.array(cjdata["image"]).astype(np.float32)
            mask = np.array(cjdata["tumorMask"]).astype(np.float32)
            label = int(np.array(cjdata["label"]).item())
            pid = np.array(cjdata["PID"])
            pid = "".join(chr(c) for c in pid.flatten()) if pid.dtype.kind in "uS" else str(int(pid.item()))

        X[i] = preprocess(img, mask)
        labels[i] = label - 1
        pids.append(pid)
        del img, mask

        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  processed {i + 1}/{n}")

    return X, labels, np.array(pids)


# ---------------------------------------------------------------------------
# 2. Model: EfficientNetB0 backbone, with an embedding-extraction hook
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
    embedding = layers.Dense(128, activation="relu", name="embedding")(x)
    x = layers.BatchNormalization()(embedding)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    embedding_model = models.Model(inputs, embedding)
    return model, backbone, embedding_model


def make_augmenter():
    return tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
    ])


def tta_predict(model, augmenter, X, rounds=TTA_ROUNDS, batch_size=BATCH_SIZE):
    probs = np.zeros((X.shape[0], NUM_CLASSES), dtype=np.float32)
    probs += model.predict(X, batch_size=batch_size, verbose=0)
    for _ in range(rounds):
        X_aug = augmenter(X, training=True).numpy()
        probs += model.predict(X_aug, batch_size=batch_size, verbose=0)
    probs /= (rounds + 1)
    return probs


def train_fold_model(X_train, y_train, X_val, y_val, class_weight, augmenter):
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED)
                .batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model, backbone, embedding_model = build_model()

    model.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              class_weight=class_weight, verbose=2)

    backbone.trainable = True
    n_layers = len(backbone.layers)
    freeze_until = int(n_layers * (1 - FINE_TUNE_AT))
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(1e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              callbacks=cbs, class_weight=class_weight, verbose=2)

    return model, embedding_model


def fit_ensemble(embeddings_train, y_train):
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced")
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         random_state=SEED, eval_metric="mlogloss")
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("svm", svm)],
        voting="soft",
    )
    ensemble.fit(embeddings_train, y_train)
    return ensemble


# ---------------------------------------------------------------------------
# 3. Cross-validation driver
# ---------------------------------------------------------------------------
def main():
    print("Loading + preprocessing dataset (mask-guided crop + resize/normalize)...")
    X, y, pids = load_and_preprocess_dataset(DATA_DIR)
    print(f"Loaded {len(X)} slices from {len(set(pids))} patients")
    gc.collect()

    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_results = {"cnn": [], "cnn_tta": [], "ensemble": []}
    all_val_frac = 0.15

    for fold_idx, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups=pids)):
        print(f"\n{'='*70}\nFOLD {fold_idx + 1}/{N_FOLDS}\n{'='*70}")

        gss = GroupShuffleSplit(n_splits=1, test_size=all_val_frac, random_state=SEED)
        tr_sub, val_sub = next(gss.split(
            trainval_idx, y[trainval_idx], groups=pids[trainval_idx]
        ))
        train_idx = trainval_idx[tr_sub]
        val_idx = trainval_idx[val_sub]

        assert set(pids[train_idx]).isdisjoint(pids[test_idx])
        assert set(pids[val_idx]).isdisjoint(pids[test_idx])
        assert set(pids[train_idx]).isdisjoint(pids[val_idx])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

        class_weight = dict(enumerate(
            compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
        ))
        augmenter = make_augmenter()

        model, embedding_model = train_fold_model(
            X_train, y_train, X_val, y_val, class_weight, augmenter
        )

        # --- CNN softmax, single pass ---
        y_pred_cnn = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
        acc_cnn = accuracy_score(y_test, y_pred_cnn)
        fold_results["cnn"].append(acc_cnn)

        # --- CNN softmax + TTA ---
        probs_tta = tta_predict(model, augmenter, X_test)
        y_pred_tta = np.argmax(probs_tta, axis=1)
        acc_tta = accuracy_score(y_test, y_pred_tta)
        fold_results["cnn_tta"].append(acc_tta)

        # --- Embedding ensemble (RF + XGBoost + SVM) ---
        emb_train = embedding_model.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
        emb_test = embedding_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        ensemble = fit_ensemble(emb_train, y_train)
        y_pred_ens = ensemble.predict(emb_test)
        acc_ens = accuracy_score(y_test, y_pred_ens)
        fold_results["ensemble"].append(acc_ens)

        print(f"\nFold {fold_idx + 1} results: CNN={acc_cnn:.4f}  CNN+TTA={acc_tta:.4f}  Ensemble={acc_ens:.4f}")
        print("\n-- CNN+TTA classification report --")
        print(classification_report(y_test, y_pred_tta, target_names=list(LABEL_MAP.values())))
        print(confusion_matrix(y_test, y_pred_tta))
        print("\n-- Embedding ensemble classification report --")
        print(classification_report(y_test, y_pred_ens, target_names=list(LABEL_MAP.values())))
        print(confusion_matrix(y_test, y_pred_ens))

        del model, embedding_model, ensemble, emb_train, emb_test
        tf.keras.backend.clear_session()
        gc.collect()

    print(f"\n{'='*70}\nFINAL 5-FOLD CROSS-VALIDATION SUMMARY\n{'='*70}")
    for key, label in [("cnn", "CNN (softmax)"), ("cnn_tta", "CNN + TTA"), ("ensemble", "CNN-embedding ensemble (RF+XGB+SVM)")]:
        accs = np.array(fold_results[key])
        print(f"{label:40s}: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}%   (folds: {[f'{a*100:.2f}' for a in accs]})")


if __name__ == "__main__":
    main()
