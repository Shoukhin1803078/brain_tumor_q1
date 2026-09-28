"""
Standalone plotting script: generates accuracy_vs_epoch.png, loss_vs_epoch.png,
and confusion_matrix.png for the paper, using the actual Fold 1 training log
and confusion matrix already produced by step4_kaggle4class.py (pasted into
the conversation), without re-running training.

Run locally: python3 plot_curves.py
Outputs go to images/ next to this script.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is undefined when running inside a Jupyter/Kaggle notebook cell
    SCRIPT_DIR = os.getcwd()

OUT_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------------------------------------------------------------------------
# Fold 1 training log (Stage 1: head-only, 8 epochs; Stage 2: fine-tune, 35 epochs)
# Values transcribed directly from the Kaggle run output.
# ---------------------------------------------------------------------------
stage1_acc = [0.7382, 0.8078, 0.8270, 0.8356, 0.8409, 0.8487, 0.8603, 0.8552]
stage1_val_acc = [0.7280, 0.7894, 0.8171, 0.8137, 0.8322, 0.8252, 0.8310, 0.8368]
stage1_loss = [0.7545, 0.5124, 0.4550, 0.4422, 0.4214, 0.3996, 0.3728, 0.3777]
stage1_val_loss = [0.7105, 0.5433, 0.5090, 0.4565, 0.4356, 0.4694, 0.4635, 0.4276]

stage2_acc = [
    0.6750, 0.7592, 0.8033, 0.8145, 0.8419, 0.8433, 0.8632, 0.8703, 0.8787, 0.8826,
    0.8909, 0.8930, 0.9044, 0.9148, 0.9101, 0.9124, 0.9238, 0.9167, 0.9279, 0.9322,
    0.9371, 0.9350, 0.9287, 0.9442, 0.9498, 0.9483, 0.9489, 0.9520, 0.9522, 0.9610,
    0.9565, 0.9608, 0.9545, 0.9577, 0.9618,
]
stage2_val_acc = [
    0.7500, 0.7894, 0.8090, 0.8194, 0.8299, 0.8391, 0.8449, 0.8634, 0.8692, 0.8796,
    0.8854, 0.8900, 0.9016, 0.9016, 0.9086, 0.9120, 0.9236, 0.9190, 0.9271, 0.9306,
    0.9352, 0.9433, 0.9444, 0.9491, 0.9537, 0.9560, 0.9572, 0.9583, 0.9606, 0.9572,
    0.9630, 0.9664, 0.9676, 0.9641, 0.9653,
]
stage2_loss = [
    0.9135, 0.6627, 0.5401, 0.4934, 0.4536, 0.4281, 0.3875, 0.3553, 0.3293, 0.3174,
    0.2963, 0.2864, 0.2610, 0.2355, 0.2508, 0.2285, 0.2133, 0.2288, 0.2039, 0.1922,
    0.1834, 0.1843, 0.1952, 0.1694, 0.1491, 0.1458, 0.1453, 0.1381, 0.1334, 0.1276,
    0.1315, 0.1194, 0.1317, 0.1196, 0.1182,
]
stage2_val_loss = [
    0.6670, 0.5733, 0.5250, 0.4875, 0.4633, 0.4219, 0.4150, 0.3805, 0.3657, 0.3426,
    0.3180, 0.3085, 0.2818, 0.2794, 0.2660, 0.2494, 0.2286, 0.2224, 0.2016, 0.2036,
    0.1858, 0.1721, 0.1729, 0.1639, 0.1568, 0.1492, 0.1470, 0.1401, 0.1366, 0.1372,
    0.1317, 0.1214, 0.1171, 0.1205, 0.1238,
]

train_acc = stage1_acc + stage2_acc
val_acc = stage1_val_acc + stage2_val_acc
train_loss = stage1_loss + stage2_loss
val_loss = stage1_val_loss + stage2_val_loss
epochs = range(1, len(train_acc) + 1)
stage_boundary = len(stage1_acc)  # epoch index where fine-tuning begins

# Fold 1 CNN+TTA confusion matrix (rows = true, cols = predicted),
# class order: glioma, meningioma, notumor, pituitary
confusion = np.array([
    [349,   7,   3,   1],
    [  2, 349,   3,   6],
    [  1,   0, 357,   2],
    [  0,   6,   0, 354],
])


def plot_accuracy():
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.axvline(stage_boundary, color="gray", linestyle="--", linewidth=1,
                label="Fine-tuning starts")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy (Representative Fold)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "accuracy_vs_epoch.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def plot_loss():
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.axvline(stage_boundary, color="gray", linestyle="--", linewidth=1,
                label="Fine-tuning starts")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (Representative Fold)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "loss_vs_epoch.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def plot_confusion_matrix():
    plt.figure(figsize=(5, 4.5))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Confusion Matrix (CNN+TTA, Representative Fold)")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = confusion.max() / 2.0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, format(confusion[i, j], "d"), ha="center", va="center",
                      color="white" if confusion[i, j] > thresh else "black")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_accuracy()
    plot_loss()
    plot_confusion_matrix()
