"""
Step 4: 4-class Kaggle "Brain Tumor MRI Dataset" (masoudnickparvar) benchmark.

Dataset: /kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma, meningioma, notumor, pituitary
Structure (standard Kaggle release):
    brain-tumor-mri-dataset/
        Training/
            glioma/*.jpg
            meningioma/*.jpg
            notumor/*.jpg
            pituitary/*.jpg
        Testing/
            glioma/*.jpg
            meningioma/*.jpg
            notumor/*.jpg
            pituitary/*.jpg

IMPORTANT LIMITATION vs. the figshare pipeline (step3_cv_ensemble.py):
  - This dataset provides no tumor segmentation masks -> no mask-guided ROI
    cropping here. The whole slice is used as-is (center-cropped/resized).
  - This dataset provides no patient ID field -> cross-validation here is
    standard stratified k-fold over images, NOT patient-level. If you find
    documentation confirming no patient overlap between Training/Testing (or
    within-class duplicates), note that in the paper; otherwise disclose this
    as a limitation, since some sources reused for this merged dataset are
    known to contain near-duplicate slices from the same scan.

Otherwise reuses the same recipe as step3: EfficientNetB0 backbone, two-stage
fine-tuning, class weighting, TTA, 5-fold CV, CNN-embedding ensemble
(RF + XGBoost + SVM).
"""

import os
import gc
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
TTA_ROUNDS = 5
N_FOLDS = 5
SEED = 42
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# 1. Load + preprocess: stream each file straight into a uint8 array
#    (same memory-safety approach as step3, to avoid Kaggle OOM restarts)
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
    """Combine Training/ and Testing/ folders -- we re-split ourselves via
    k-fold rather than relying on the dataset's original train/test split,
    since we want cross-validated, not single-split, accuracy."""
    paths, labels = [], []
    for split_dir in ["Training", "Testing"]:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, split_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*")))
            paths.extend(files)
            labels.extend([class_idx] * len(files))

    if not paths:
        raise FileNotFoundError(
            f"No images found under {data_dir}/Training or /Testing. "
            f"Check DATA_DIR and folder names match {CLASS_NAMES}."
        )
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
    print("Listing image paths...")
    paths, y = list_image_paths(DATA_DIR)
    print(f"Found {len(paths)} images across {NUM_CLASSES} classes: {CLASS_NAMES}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == i).sum()}")

    print("\nLoading + preprocessing images...")
    X = load_and_preprocess(paths)
    gc.collect()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = {"cnn": [], "cnn_tta": [], "ensemble": []}

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*70}\nFOLD {fold_idx + 1}/{N_FOLDS}\n{'='*70}")

        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=0.15, random_state=SEED, stratify=y[trainval_idx]
        )

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

        y_pred_cnn = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
        acc_cnn = accuracy_score(y_test, y_pred_cnn)
        fold_results["cnn"].append(acc_cnn)

        probs_tta = tta_predict(model, augmenter, X_test)
        y_pred_tta = np.argmax(probs_tta, axis=1)
        acc_tta = accuracy_score(y_test, y_pred_tta)
        fold_results["cnn_tta"].append(acc_tta)

        emb_train = embedding_model.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
        emb_test = embedding_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        ensemble = fit_ensemble(emb_train, y_train)
        y_pred_ens = ensemble.predict(emb_test)
        acc_ens = accuracy_score(y_test, y_pred_ens)
        fold_results["ensemble"].append(acc_ens)

        print(f"\nFold {fold_idx + 1} results: CNN={acc_cnn:.4f}  CNN+TTA={acc_tta:.4f}  Ensemble={acc_ens:.4f}")
        print("\n-- CNN+TTA classification report --")
        print(classification_report(y_test, y_pred_tta, target_names=CLASS_NAMES))
        print(confusion_matrix(y_test, y_pred_tta))
        print("\n-- Embedding ensemble classification report --")
        print(classification_report(y_test, y_pred_ens, target_names=CLASS_NAMES))
        print(confusion_matrix(y_test, y_pred_ens))

        del model, embedding_model, ensemble, emb_train, emb_test
        tf.keras.backend.clear_session()
        gc.collect()

    print(f"\n{'='*70}\nFINAL 5-FOLD CROSS-VALIDATION SUMMARY (4-class Kaggle dataset)\n{'='*70}")
    for key, label in [("cnn", "CNN (softmax)"), ("cnn_tta", "CNN + TTA"), ("ensemble", "CNN-embedding ensemble (RF+XGB+SVM)")]:
        accs = np.array(fold_results[key])
        print(f"{label:40s}: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}%   (folds: {[f'{a*100:.2f}' for a in accs]})")


if __name__ == "__main__":
    main()
