"""
Step 5: Grad-CAM visualization for the trained EfficientNetB0 brain tumor
classifier, for interpretability figures in the paper.

Trains one model (same recipe as step4_kaggle4class.py, single fold -- no
need to redo the full 5-fold CV for this) on the 4-class dataset, then
generates Grad-CAM heatmaps overlaid on sample test images from each class,
showing which regions of the MRI the model attended to when making its
prediction.

Run on Kaggle (GPU enabled). Outputs PNGs to /kaggle/working/gradcam/ --
download these and place them in images/ for the paper.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
SEED = 42
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
OUT_DIR = "/kaggle/working/gradcam"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading (same as step4)
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
    paths, labels = [], []
    for split_dir in ["Training", "Testing"]:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, split_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*")))
            paths.extend(files)
            labels.extend([class_idx] * len(files))
    return paths, np.array(labels)


def load_and_preprocess(paths):
    n = len(paths)
    X = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        X[i] = img.numpy().astype(np.uint8)
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"  processed {i + 1}/{n}")
    return X


# ---------------------------------------------------------------------------
# 2. Model (same architecture as step4, but we need direct access to the
#    backbone submodel to locate the last conv layer for Grad-CAM)
# ---------------------------------------------------------------------------
def build_and_train_model(X_train, y_train, X_val, y_val, class_weight):
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    conv_features = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(conv_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    augmenter = tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
    ])

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED)
                .batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

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

    return model, backbone


# ---------------------------------------------------------------------------
# 3. Grad-CAM
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, backbone, pred_index=None):
    """Standard Grad-CAM: gradient of the predicted class score w.r.t. the
    backbone's last conv feature map, global-average-pooled into per-channel
    weights, then a weighted sum of the feature maps -> a coarse localization
    heatmap showing which spatial regions drove the prediction.

    NOTE: building a second Model that reaches into
    `backbone.get_layer(...).output` fails in Keras 3 when the backbone was
    called as a single nested layer inside the outer model (it raises a
    KeyError -- the backbone's internal tensors aren't part of the outer
    model's graph). Instead, we replay the head layers manually inside the
    GradientTape, on top of the backbone's own output tensor -- for
    EfficientNetB0 with include_top=False, that output IS already the final
    conv feature map (post top_conv/top_bn/top_activation), so no need to
    dig into a named internal layer at all.
    """
    backbone_idx = model.layers.index(backbone)
    pre_layers = model.layers[1:backbone_idx]   # skip InputLayer, keep preprocessing op(s)
    post_layers = model.layers[backbone_idx + 1:]  # GAP, BN, Dropout, Dense, ... softmax

    x = img_array
    for layer in pre_layers:
        x = layer(x)

    with tf.GradientTape() as tape:
        conv_output = backbone(x, training=False)
        tape.watch(conv_output)
        h = conv_output
        for layer in post_layers:
            h = layer(h, training=False)
        predictions = h
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def overlay_and_save(orig_img, heatmap, save_path, alpha=0.4):
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    superimposed = jet_heatmap * alpha + orig_img * (1 - alpha)
    superimposed = np.uint8(superimposed)

    _, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(orig_img.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------
def main():
    print("Listing image paths...")
    paths, y = list_image_paths(DATA_DIR)
    print(f"Found {len(paths)} images across {NUM_CLASSES} classes: {CLASS_NAMES}")

    print("Loading + preprocessing images...")
    X = load_and_preprocess(paths)

    trainval_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=y
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.15, random_state=SEED, stratify=y[trainval_idx]
    )

    class_weight = dict(enumerate(
        compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y[train_idx])
    ))

    print("Training model for Grad-CAM visualization (single split, not full CV)...")
    model, backbone = build_and_train_model(
        X[train_idx], y[train_idx], X[val_idx], y[val_idx], class_weight
    )

    print(f"\nGenerating Grad-CAM visualizations, saving to {OUT_DIR} ...")
    rng = np.random.RandomState(SEED)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_test_idx = test_idx[y[test_idx] == class_idx]
        chosen = rng.choice(class_test_idx, size=min(3, len(class_test_idx)), replace=False)
        for i, idx in enumerate(chosen):
            img = X[idx]
            img_array = np.expand_dims(img, axis=0).astype(np.float32)
            heatmap, pred_class, pred_probs = make_gradcam_heatmap(
                img_array, model, backbone
            )
            correct = "correct" if pred_class == class_idx else "MISCLASSIFIED"
            save_path = os.path.join(
                OUT_DIR, f"gradcam_{class_name}_{i}_{correct}.png"
            )
            overlay_and_save(img, heatmap, save_path)
            print(f"  {class_name} sample {i}: predicted={CLASS_NAMES[pred_class]} "
                  f"(confidence {pred_probs[pred_class]:.2f}) -- {correct}")

    print(f"\nDone. Download the images in {OUT_DIR} and place the best "
          f"representative ones (e.g. one per class) into images/ for the paper.")


if __name__ == "__main__":
    main()
