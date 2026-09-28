
"""
Step 6: Multi-backbone comparison on the 4-class Kaggle "Brain Tumor MRI
Dataset" (masoudnickparvar), under the exact same protocol as
step4_kaggle4class.py.

Why this script exists
-----------------------
The paper currently reports results for one backbone (EfficientNetB0) only.
A journal-grade "Results" section needs more than one classifier/backbone
compared under an identical protocol -- this is the CNN-image-pipeline
analogue of comparing multiple classical ML models, the way tabular-data
papers do. This script generalizes step4's model-building code to run the
SAME 5-fold stratified CV + two-stage fine-tuning + TTA + CNN-embedding
ensemble recipe across FOUR backbones:

    - efficientnetb0  (already have results for this from step4 -- rerun it
                        here too if you want *timing* numbers measured with
                        the same instrumentation as the other backbones;
                        otherwise you can skip it and reuse step4's accuracy
                        numbers, noting the timing wasn't measured for it.)
    - resnet50
    - densenet121
    - mobilenetv2

How to run
-----------
Set BACKBONE below to ONE of the four names and run the whole script as one
Kaggle session. Do NOT try to loop over all four backbones in a single
session -- each backbone's 5-fold CV takes on the order of 1.5-2.5 hours on
a single Tesla T4 (EfficientNetB0's run in step4 took about that long), so
four backbones back-to-back would blow past Kaggle's ~12h session limit and
risks losing all progress if the kernel gets killed partway through. Run one
backbone per session (four sessions total), across as many days as your
weekly GPU quota allows.

Each run appends its fold-by-fold results (accuracy AND timing, per fold,
per method) as one JSON line per fold to /kaggle/working/backbone_sweep.jsonl.
The file is opened in append mode and flushed after every fold, so a killed
kernel only loses the fold in progress, not prior folds -- restart the
script (same BACKBONE) and previously-completed folds' rows are still in the
file; you do not need to deduplicate anything downstream, just concatenate
the folds across your four sessions and average by (backbone, method).

Download backbone_sweep.jsonl after each session and paste/attach its
contents back for the paper write-up -- do not hand-summarize it, the raw
per-fold rows are what let us compute an honest mean +/- std and rerun the
Wilcoxon test.
"""

import os
import gc
import glob
import json
import time
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
# Config -- EDIT THIS before each Kaggle session
# ---------------------------------------------------------------------------
BACKBONE = "resnet50"  # one of: efficientnetb0, resnet50, densenet121, mobilenetv2

DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
RESULTS_PATH = "/kaggle/working/backbone_sweep.jsonl"
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
# Backbone registry -- each entry supplies the Keras application class and
# its matching preprocess_input function. Using the wrong preprocess_input
# for a given backbone (e.g. feeding raw 0-255 pixels to a backbone that
# expects the caffe-style BGR-mean-subtracted input ResNet50 uses) silently
# tanks accuracy without erroring, so this table is the important part of
# generalizing step4's code -- don't just swap the model class and keep
# EfficientNet's preprocess_input.
# ---------------------------------------------------------------------------
BACKBONE_REGISTRY = {
    "efficientnetb0": dict(
        app=tf.keras.applications.EfficientNetB0,
        preprocess=tf.keras.applications.efficientnet.preprocess_input,
    ),
    "resnet50": dict(
        app=tf.keras.applications.ResNet50,
        preprocess=tf.keras.applications.resnet50.preprocess_input,
    ),
    "densenet121": dict(
        app=tf.keras.applications.DenseNet121,
        preprocess=tf.keras.applications.densenet.preprocess_input,
    ),
    "mobilenetv2": dict(
        app=tf.keras.applications.MobileNetV2,
        preprocess=tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
}


# ---------------------------------------------------------------------------
# 1. Load + preprocess (identical to step4)
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
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
# 2. Model: generalized over BACKBONE, with an embedding-extraction hook
# ---------------------------------------------------------------------------
def build_model(backbone_name):
    cfg = BACKBONE_REGISTRY[backbone_name]
    backbone = cfg["app"](
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = cfg["preprocess"](inputs)
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


def train_fold_model(backbone_name, X_train, y_train, X_val, y_val, class_weight, augmenter):
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED)
                .batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model, backbone, embedding_model = build_model(backbone_name)

    model.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              class_weight=class_weight, verbose=2)
    t_stage1 = time.perf_counter() - t0

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
    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              callbacks=cbs, class_weight=class_weight, verbose=2)
    t_stage2 = time.perf_counter() - t0

    return model, embedding_model, t_stage1, t_stage2


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


def append_result(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# 3. Cross-validation driver
# ---------------------------------------------------------------------------
def main():
    assert BACKBONE in BACKBONE_REGISTRY, f"BACKBONE must be one of {list(BACKBONE_REGISTRY)}"
    print(f"Backbone: {BACKBONE}")

    print("Listing image paths...")
    paths, y = list_image_paths(DATA_DIR)
    print(f"Found {len(paths)} images across {NUM_CLASSES} classes: {CLASS_NAMES}")

    print("\nLoading + preprocessing images...")
    X = load_and_preprocess(paths)
    gc.collect()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = {"cnn": [], "cnn_tta": [], "ensemble": []}

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*70}\n{BACKBONE} -- FOLD {fold_idx + 1}/{N_FOLDS}\n{'='*70}")

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

        model, embedding_model, t_stage1, t_stage2 = train_fold_model(
            BACKBONE, X_train, y_train, X_val, y_val, class_weight, augmenter
        )
        train_time_s = t_stage1 + t_stage2

        # -- CNN (single pass), timed --
        t0 = time.perf_counter()
        probs_cnn = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        t_cnn_infer = time.perf_counter() - t0
        y_pred_cnn = np.argmax(probs_cnn, axis=1)
        acc_cnn = accuracy_score(y_test, y_pred_cnn)
        fold_results["cnn"].append(acc_cnn)
        ms_per_image_cnn = 1000.0 * t_cnn_infer / len(X_test)

        # -- CNN + TTA, timed --
        t0 = time.perf_counter()
        probs_tta = tta_predict(model, augmenter, X_test)
        t_tta_infer = time.perf_counter() - t0
        y_pred_tta = np.argmax(probs_tta, axis=1)
        acc_tta = accuracy_score(y_test, y_pred_tta)
        fold_results["cnn_tta"].append(acc_tta)
        ms_per_image_tta = 1000.0 * t_tta_infer / len(X_test)

        # -- Embedding ensemble, timed (embedding extraction + ensemble predict) --
        emb_train = embedding_model.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
        t0 = time.perf_counter()
        emb_test = embedding_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        ensemble = fit_ensemble(emb_train, y_train)
        y_pred_ens = ensemble.predict(emb_test)
        t_ens_infer = time.perf_counter() - t0
        acc_ens = accuracy_score(y_test, y_pred_ens)
        fold_results["ensemble"].append(acc_ens)
        ms_per_image_ens = 1000.0 * t_ens_infer / len(X_test)

        print(f"\nFold {fold_idx + 1} [{BACKBONE}] results: "
              f"CNN={acc_cnn:.4f} ({ms_per_image_cnn:.2f} ms/img)  "
              f"CNN+TTA={acc_tta:.4f} ({ms_per_image_tta:.2f} ms/img)  "
              f"Ensemble={acc_ens:.4f} ({ms_per_image_ens:.2f} ms/img)  "
              f"train_time={train_time_s:.1f}s "
              f"(stage1={t_stage1:.1f}s, stage2={t_stage2:.1f}s)")
        print("\n-- CNN+TTA classification report --")
        print(classification_report(y_test, y_pred_tta, target_names=CLASS_NAMES))
        print(confusion_matrix(y_test, y_pred_tta))
        print("\n-- Embedding ensemble classification report --")
        print(classification_report(y_test, y_pred_ens, target_names=CLASS_NAMES))
        print(confusion_matrix(y_test, y_pred_ens))

        append_result({
            "backbone": BACKBONE,
            "fold": fold_idx + 1,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "train_time_stage1_s": round(t_stage1, 2),
            "train_time_stage2_s": round(t_stage2, 2),
            "train_time_total_s": round(train_time_s, 2),
            "cnn_acc": round(acc_cnn, 4), "cnn_ms_per_image": round(ms_per_image_cnn, 3),
            "tta_acc": round(acc_tta, 4), "tta_ms_per_image": round(ms_per_image_tta, 3),
            "ensemble_acc": round(acc_ens, 4), "ensemble_ms_per_image": round(ms_per_image_ens, 3),
        })

        del model, embedding_model, ensemble, emb_train, emb_test
        tf.keras.backend.clear_session()
        gc.collect()

    print(f"\n{'='*70}\nFINAL 5-FOLD SUMMARY -- {BACKBONE}\n{'='*70}")
    for key, label in [("cnn", "CNN (softmax)"), ("cnn_tta", "CNN + TTA"),
                        ("ensemble", "CNN-embedding ensemble (RF+XGB+SVM)")]:
        accs = np.array(fold_results[key])
        print(f"{label:40s}: {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}%   "
              f"(folds: {[f'{a*100:.2f}' for a in accs]})")
    print(f"\nPer-fold rows appended to {RESULTS_PATH} -- download this file "
          f"and paste/attach its contents for the paper write-up.")


if __name__ == "__main__":
    main()
