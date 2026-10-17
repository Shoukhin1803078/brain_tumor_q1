"""
Step 10: Paper figures in the style of the two comparison papers --
Mahmud et al., Informatics and Health 3
(2026) 290-311 -- "Uncertainty-aware hybrid CNN-transformer ensemble".

Single self-contained file. Every value is hardcoded in the DATA block below,
so this runs anywhere with numpy + matplotlib: no logs, no dataset, no GPU, no
network.

    python3 step10_paper_figures.py              # all available figures
    python3 step10_paper_figures.py --list
    python3 step10_paper_figures.py --only roc

What this mirrors
-----------------
Mahmud et al.'s plot figures, and the equivalent for our RRF-Net:

    their Fig. 10/11   confusion matrix                 -> fig01
    their Fig. 12      training/validation curves       -> fig02
    their Fig. 13      "Histogram of Model Accuracies"  -> fig03
    their Table 11     ablation study                   -> fig04
    their Table 6      per-class performance            -> fig05
    their Table 10     efficiency analysis              -> fig06
    their Fig. 8/9     ROC curves, per-class AUC        -> fig07  (needs probs)
    their Fig. 18      reliability diagrams             -> fig08  (needs probs)
    their Fig. 19      uncertainty distributions        -> fig09  (needs probs)
    their Fig. 20      coverage-risk / selective pred.  -> fig10  (needs probs)

and the plot figures from Jayaraman et al., Sci. Reports 15:45769 (CAFNet)
that Mahmud et al. do not have an equivalent for:

    their Fig. 17      train vs test accuracy + trendline   -> fig11
    their Fig. 16      train/val curves + test reference    -> fig12
    their Fig. 11      5-fold cross-validation accuracy     -> fig13
    their Fig. 14/15   per-model train/val accuracy & loss  -> fig14
    their Fig. 2/3     dataset class distribution           -> fig15

CAFNet's Figs. 5-8 (image width and height histograms, the width-height
scatter, and the pixel-intensity distribution) have no counterpart: the EDA
recorded only the min/max ranges, not the underlying distributions, so there
is nothing to plot without re-reading all 7,200 images.

CAFNet's Figs. 9/10 (decision tree, naive Bayes and LDA baselines) are also
absent -- this project never trained those classifiers.

Their Figs. 1-7 and 14-17 are architecture diagrams, augmentation samples,
Grad-CAM and LIME overlays. Those are images of the model or of MRI slices, not
plots of results, so they are out of scope here.

The four "(needs probs)" figures
--------------------------------
ROC curves, reliability diagrams, uncertainty distributions and coverage-risk
analysis cannot be drawn from summary statistics. Every one of them needs the
per-sample predicted probability for each test image, and our logs only ever
recorded scalars (accuracy, ECE, NLL). They are NOT hardcoded here, because
inventing per-sample values would mean inventing the curves.

step9_rrf_fusion.py now writes what they need (SAVE_PROBS = True ->
/kaggle/working/probs/*.npz). Download that folder into ./probs/ after the next
run and these four figures generate automatically, from real data, with no
change to this file.

Provenance of the hardcoded numbers
-----------------------------------
    RRF-Net v3          rrf_fusion_v3.log   (variant rrf_full@v3_vit4, 5 folds)
    RRF-Net v1 / v2     rrf_fusion.log, rrf_fusion_v2.log  (ablation arms)
    CNN baselines       step_6_backbone_sweep_logs.log     (5 folds each)
    Mahmud / CAFNet     the two PDFs in new_vit_cnn_methodology/
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "images/paper"
PROBS_DIR = "probs"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ===========================================================================
# DATA -- every number below is measured. Edit here, not in the figure code.
# ===========================================================================

# --- RRF-Net v3 (EfficientNetV2-B0 + ViT-B/16, last 4 ViT blocks fine-tuned),
#     5-fold image-level CV, 1440 test images per fold. rrf_fusion_v3.log ------
FOLDS = [
    dict(fold=1, cnn=92.15, vit=96.67, fused=97.08, tta=98.26, ece=0.0287,
         nll=0.1017, entropy=0.1382, beta=0.413, disagree=101, route_acc=0.7327,
         ms=20.77, tta_ms=100.35, train_s=2796.3),
    dict(fold=2, cnn=92.85, vit=95.14, fused=97.01, tta=97.99, ece=0.0365,
         nll=0.1148, entropy=0.1607, beta=0.360, disagree=113, route_acc=0.6814,
         ms=20.34, tta_ms=99.23, train_s=2665.8),
    dict(fold=3, cnn=91.87, vit=96.18, fused=96.53, tta=97.85, ece=0.0264,
         nll=0.1057, entropy=0.1299, beta=0.398, disagree=112, route_acc=0.8036,
         ms=20.62, tta_ms=99.18, train_s=2791.5),
    dict(fold=4, cnn=92.15, vit=96.18, fused=97.22, tta=98.12, ece=0.0325,
         nll=0.1094, entropy=0.1414, beta=0.345, disagree=106, route_acc=0.8019,
         ms=20.44, tta_ms=99.29, train_s=2791.6),
    dict(fold=5, cnn=94.31, vit=96.53, fused=97.36, tta=98.33, ece=0.0317,
         nll=0.0974, entropy=0.1292, beta=0.356, disagree=78, route_acc=0.6538,
         ms=20.32, tta_ms=98.62, train_s=2786.8),
]

# --- Confusion matrix, TTA predictions, all 5 folds pooled. Rows = true class.
#     Sums to 7,200: every image is the test item of exactly one fold. --------
CONFUSION = np.array([[1739,   47,   12,    2],
                      [  17, 1762,    4,   17],
                      [  11,    9, 1776,    4],
                      [   2,   10,    1, 1787]])

# --- Per-class precision / recall / F1 (%), mean of the 5 per-fold reports ---
PER_CLASS = {
    "glioma":     dict(precision=98.4, recall=96.6, f1=97.4),
    "meningioma": dict(precision=96.4, recall=98.0, f1=97.2),
    "notumor":    dict(precision=99.2, recall=98.6, f1=99.0),
    "pituitary":  dict(precision=98.8, recall=99.2, f1=99.0),
}

# --- Mean learning curves over the 5 folds. Stage 1 = both backbones frozen
#     (5 epochs), stage 2 = fine-tuning (18 epochs). step_6/rrf logs. ---------
CURVE_S1 = dict(
    acc=[0.7823, 0.8598, 0.8799, 0.8961, 0.9070],
    val_acc=[0.8396, 0.8426, 0.8181, 0.8657, 0.8764],
    loss=[0.6778, 0.4300, 0.3785, 0.3367, 0.3136],
    val_loss=[0.5144, 0.4919, 0.5222, 0.4211, 0.4109],
    route=[0.7311, 0.6565, 0.6360, 0.6273, 0.6101],
    val_route=[0.6844, 0.6903, 0.6492, 0.5961, 0.6263])
CURVE_S2 = dict(
    acc=[0.9318, 0.9446, 0.9524, 0.9598, 0.9643, 0.9689, 0.9716, 0.9748,
         0.9765, 0.9803, 0.9816, 0.9823, 0.9853, 0.9857, 0.9876, 0.9896,
         0.9898, 0.9900],
    val_acc=[0.9194, 0.9282, 0.9299, 0.9421, 0.9414, 0.9445, 0.9509, 0.9528,
             0.9546, 0.9593, 0.9632, 0.9611, 0.9625, 0.9630, 0.9660, 0.9657,
             0.9620, 0.9667],
    loss=[0.2523, 0.2174, 0.1923, 0.1768, 0.1630, 0.1475, 0.1355, 0.1255,
          0.1150, 0.1107, 0.1027, 0.0970, 0.0857, 0.0838, 0.0797, 0.0701,
          0.0691, 0.0628],
    val_loss=[0.3075, 0.2845, 0.2816, 0.2538, 0.2503, 0.2465, 0.2265, 0.2238,
              0.2076, 0.2079, 0.2026, 0.1920, 0.2012, 0.1923, 0.1903, 0.1837,
              0.2016, 0.1952],
    route=[0.5475, 0.5126, 0.4663, 0.4544, 0.4326, 0.3999, 0.3804, 0.3470,
           0.3279, 0.3243, 0.3056, 0.2924, 0.2558, 0.2520, 0.2486, 0.2177,
           0.2141, 0.1878],
    val_route=[0.5602, 0.5316, 0.5265, 0.4869, 0.5274, 0.5107, 0.4887, 0.4678,
               0.4517, 0.4455, 0.4649, 0.4380, 0.4598, 0.4178, 0.4229, 0.4147,
               0.4532, 0.4474])

# --- Independently trained CNN baselines, per-fold TTA accuracy (%).
#     Same folds, same seed, same TTA. step_6_backbone_sweep_logs.log ---------
BASELINES = {
    "ResNet50":       dict(folds=[98.75, 98.12, 98.54, 98.26, 98.19], ms=35.36),
    "DenseNet121":    dict(folds=[98.54, 98.26, 97.92, 98.40, 98.47], ms=37.67),
    "MobileNetV2":    dict(folds=[97.78, 98.47, 97.50, 98.06, 97.99], ms=23.49),
    "EfficientNetB0": dict(folds=[97.64, 97.50, 97.57, 98.19, 97.85], ms=28.06),
}

# --- Ablation over ViT fine-tuning capacity. v2 is a SINGLE fold -- it is
#     marked as such everywhere it appears and must not be read as a 5-fold
#     result. rrf_fusion.log / rrf_fusion_v2.log / rrf_fusion_v3.log ----------
ABLATION = [
    dict(name="ViT frozen\n(0 params)",      n_folds=5, vit=90.32, fused=95.76,
         tta=97.08, beta=0.713, ece=0.0470, gain=1.97),
    dict(name="Last 4 blocks\n(28.4 M)",     n_folds=5, vit=96.14, fused=97.04,
         tta=98.11, beta=0.374, ece=0.0312, gain=0.90),
    dict(name="All 12 blocks\n(85.1 M)",     n_folds=1, vit=98.40, fused=98.40,
         tta=98.68, beta=0.107, ece=0.0205, gain=0.00),
]

# --- Efficiency, Mahmud's Table 10 layout. Parameter counts are the sum of the
#     two backbones plus the fusion head; latency is measured on a Kaggle T4. --
EFFICIENCY = [
    ("Trainable ViT parameters (M)",   28.35),
    ("Total ViT parameters (M)",       85.80),
    ("CNN backbone parameters (M)",     5.92),
    ("Inference time - GPU (ms/image)", 20.50),
    ("Inference time + TTA (ms/image)", 99.33),
    ("Training time per fold (min)",    46.10),
]

# --- Stage-2 learning curves for the independently trained CNN baselines,
#     mean over the 5 folds (35 fine-tuning epochs each), plus final-epoch
#     train / validation accuracy. step_6_backbone_sweep_logs.log ------------
BACKBONE_CURVES = {
    "EfficientNetB0": dict(
        acc=[0.6674, 0.7588, 0.7943, 0.8158, 0.8310, 0.8452, 0.8537, 0.8677,
             0.8802, 0.8811, 0.8861, 0.8984, 0.9051, 0.9113, 0.9106, 0.9187,
             0.9242, 0.9248, 0.9276, 0.9339, 0.9371, 0.9376, 0.9364, 0.9422,
             0.9431, 0.9481, 0.9480, 0.9527, 0.9543, 0.9554, 0.9568, 0.9566,
             0.9572, 0.9613, 0.9565],
        val_acc=[0.7683, 0.8021, 0.8197, 0.8347, 0.8444, 0.8528, 0.8600, 0.8699,
                 0.8794, 0.8835, 0.8898, 0.8956, 0.9021, 0.9056, 0.9107, 0.9155,
                 0.9181, 0.9259, 0.9298, 0.9312, 0.9336, 0.9363, 0.9405, 0.9419,
                 0.9433, 0.9456, 0.9447, 0.9477, 0.9505, 0.9505, 0.9530, 0.9551,
                 0.9534, 0.9560, 0.9577],
        loss=[0.9085, 0.6612, 0.5612, 0.4981, 0.4584, 0.4329, 0.3933, 0.3623,
              0.3279, 0.3280, 0.3051, 0.2779, 0.2658, 0.2452, 0.2462, 0.2226,
              0.2105, 0.2123, 0.1995, 0.1882, 0.1753, 0.1762, 0.1787, 0.1647,
              0.1596, 0.1469, 0.1464, 0.1355, 0.1305, 0.1321, 0.1244, 0.1229,
              0.1258, 0.1162, 0.1259],
        val_loss=[0.6465, 0.5647, 0.5217, 0.4837, 0.4531, 0.4278, 0.4026, 0.3801,
                  0.3568, 0.3402, 0.3184, 0.2982, 0.2811, 0.2672, 0.2561, 0.2417,
                  0.2333, 0.2181, 0.2072, 0.2035, 0.1989, 0.1906, 0.1811, 0.1732,
                  0.1685, 0.1653, 0.1661, 0.1588, 0.1559, 0.1544, 0.1442, 0.1414,
                  0.1454, 0.1441, 0.1386],
        final_train=95.65, final_val=95.77),
    "ResNet50": dict(
        acc=[0.8570, 0.8971, 0.9159, 0.9337, 0.9401, 0.9470, 0.9564, 0.9624,
             0.9687, 0.9693, 0.9748, 0.9766, 0.9805, 0.9824, 0.9817, 0.9843,
             0.9852, 0.9854, 0.9864, 0.9878, 0.9890, 0.9898, 0.9891, 0.9917,
             0.9919, 0.9931, 0.9926, 0.9922, 0.9936, 0.9942, 0.9942, 0.9949,
             0.9938, 0.9942, 0.9935],
        val_acc=[0.8275, 0.8706, 0.8838, 0.9081, 0.9072, 0.9204, 0.9222, 0.9285,
                 0.9333, 0.9368, 0.9394, 0.9449, 0.9437, 0.9479, 0.9493, 0.9497,
                 0.9542, 0.9532, 0.9574, 0.9576, 0.9627, 0.9618, 0.9613, 0.9636,
                 0.9685, 0.9664, 0.9690, 0.9693, 0.9702, 0.9734, 0.9743, 0.9722,
                 0.9728, 0.9740, 0.9725],
        loss=[0.3930, 0.2830, 0.2290, 0.1833, 0.1657, 0.1524, 0.1270, 0.1100,
              0.0917, 0.0895, 0.0786, 0.0726, 0.0640, 0.0574, 0.0578, 0.0507,
              0.0471, 0.0462, 0.0444, 0.0385, 0.0377, 0.0352, 0.0371, 0.0294,
              0.0283, 0.0248, 0.0262, 0.0257, 0.0227, 0.0223, 0.0224, 0.0206,
              0.0225, 0.0205, 0.0239],
        val_loss=[0.5785, 0.4076, 0.3609, 0.2945, 0.2801, 0.2445, 0.2329, 0.2252,
                  0.2219, 0.2033, 0.1944, 0.1701, 0.1812, 0.1647, 0.1526, 0.1602,
                  0.1504, 0.1490, 0.1452, 0.1402, 0.1295, 0.1273, 0.1285, 0.1288,
                  0.1166, 0.1216, 0.1140, 0.1089, 0.1108, 0.1041, 0.0970, 0.1015,
                  0.1008, 0.1044, 0.0997],
        final_train=99.35, final_val=97.25),
    "DenseNet121": dict(
        acc=[0.8320, 0.8729, 0.8933, 0.9087, 0.9196, 0.9234, 0.9319, 0.9421,
             0.9481, 0.9501, 0.9549, 0.9598, 0.9623, 0.9697, 0.9665, 0.9711,
             0.9758, 0.9747, 0.9766, 0.9784, 0.9789, 0.9806, 0.9805, 0.9833,
             0.9845, 0.9862, 0.9853, 0.9879, 0.9878, 0.9887, 0.9894, 0.9906,
             0.9881, 0.9906, 0.9890],
        val_acc=[0.8632, 0.8894, 0.9039, 0.9130, 0.9190, 0.9245, 0.9345, 0.9361,
                 0.9403, 0.9465, 0.9498, 0.9528, 0.9542, 0.9599, 0.9620, 0.9648,
                 0.9664, 0.9660, 0.9692, 0.9676, 0.9674, 0.9722, 0.9725, 0.9722,
                 0.9739, 0.9727, 0.9729, 0.9741, 0.9755, 0.9750, 0.9764, 0.9771,
                 0.9757, 0.9773, 0.9748],
        loss=[0.4495, 0.3473, 0.2926, 0.2530, 0.2291, 0.2146, 0.1949, 0.1678,
              0.1501, 0.1445, 0.1338, 0.1222, 0.1135, 0.0966, 0.1028, 0.0897,
              0.0807, 0.0851, 0.0762, 0.0697, 0.0678, 0.0652, 0.0641, 0.0561,
              0.0524, 0.0489, 0.0492, 0.0428, 0.0423, 0.0419, 0.0398, 0.0367,
              0.0392, 0.0359, 0.0392],
        val_loss=[0.3652, 0.3025, 0.2715, 0.2491, 0.2291, 0.2152, 0.1953, 0.1847,
                  0.1714, 0.1574, 0.1490, 0.1431, 0.1379, 0.1308, 0.1224, 0.1121,
                  0.1104, 0.1086, 0.1018, 0.1017, 0.1027, 0.0981, 0.0948, 0.0968,
                  0.0942, 0.0923, 0.0909, 0.0898, 0.0897, 0.0900, 0.0877, 0.0849,
                  0.0836, 0.0843, 0.0856],
        final_train=98.90, final_val=97.48),
    "MobileNetV2": dict(
        acc=[0.7575, 0.8211, 0.8475, 0.8662, 0.8765, 0.8851, 0.8932, 0.9054,
             0.9159, 0.9174, 0.9216, 0.9273, 0.9322, 0.9414, 0.9390, 0.9457,
             0.9508, 0.9491, 0.9492, 0.9563, 0.9570, 0.9583, 0.9581, 0.9623,
             0.9635, 0.9676, 0.9669, 0.9689, 0.9712, 0.9724, 0.9724, 0.9734,
             0.9713, 0.9760, 0.9725],
        val_acc=[0.8576, 0.8651, 0.8690, 0.8741, 0.8826, 0.8842, 0.8868, 0.8900,
                 0.8975, 0.9016, 0.9028, 0.9081, 0.9111, 0.9123, 0.9174, 0.9174,
                 0.9220, 0.9278, 0.9301, 0.9317, 0.9315, 0.9303, 0.9366, 0.9366,
                 0.9398, 0.9438, 0.9449, 0.9472, 0.9431, 0.9451, 0.9421, 0.9454,
                 0.9479, 0.9514, 0.9511],
        loss=[0.6653, 0.4932, 0.4253, 0.3647, 0.3432, 0.3171, 0.2873, 0.2616,
              0.2348, 0.2284, 0.2194, 0.2025, 0.1920, 0.1685, 0.1748, 0.1568,
              0.1441, 0.1474, 0.1458, 0.1297, 0.1254, 0.1236, 0.1229, 0.1107,
              0.1059, 0.0971, 0.0971, 0.0926, 0.0855, 0.0862, 0.0844, 0.0804,
              0.0861, 0.0764, 0.0833],
        val_loss=[0.4687, 0.4449, 0.4350, 0.4045, 0.3596, 0.3391, 0.3186, 0.3054,
                  0.2842, 0.2698, 0.2563, 0.2430, 0.2334, 0.2223, 0.2155, 0.2146,
                  0.2098, 0.1988, 0.1957, 0.1886, 0.1885, 0.1852, 0.1779, 0.1739,
                  0.1725, 0.1671, 0.1642, 0.1603, 0.1624, 0.1605, 0.1609, 0.1552,
                  0.1493, 0.1482, 0.1444],
        final_train=97.25, final_val=95.11),
}

# --- Dataset composition and the per-fold split. The 4-class Kaggle release is
#     exactly balanced; every fold uses the same split of the 7,200 images.
#     eda/results/eda_a_summary.json + the rrf_fusion_v3.log fold headers.
#     NOTE: only the width/height RANGES are recorded, not the distributions,
#     which is why CAFNet's Figs. 5-8 have no counterpart here. ---------------
DATASET = dict(n_images=7200, per_class={c: 1800 for c in CLASS_NAMES},
               n_train=4896, n_val=864, n_test=1440,
               width_range=(150, 1375), height_range=(167, 1446))

# --- Published results, transcribed from the two PDFs. Different split
#     protocols, so NOT directly comparable -- shown for context only. --------
PUBLISHED = {
    "Mahmud et al. 2026": dict(acc=98.10, protocol="10-fold similarity-group CV"),
    "CAFNet (Jayaraman 2025)": dict(acc=96.41, protocol="single 80/10/10 split"),
}
MAHMUD_ECE_WITH_MCDROPOUT = 0.017

OURS = "RRF-Net (proposed)"

# ===========================================================================
# Style -- close to the comparison paper: default matplotlib, light grid,
# centred bold titles, skyblue bars with the proposed model highlighted red.
# ===========================================================================
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "-", "legend.fontsize": 9, "legend.frameon": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
BAR_COLOR, HILITE = "skyblue", "red"
TRAIN_COLOR, VAL_COLOR = "tab:blue", "red"
MANIFEST = []


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(path)
    plt.close(fig)
    MANIFEST.append(name)
    print(f"  wrote {path}")


def _foot(fig, text):
    """Caption under a figure, wrapped to the figure's own width. An unwrapped
    line sits inside the tight bounding box and silently widens the canvas."""
    import textwrap
    width = max(60, int(fig.get_figwidth() * 13.5))
    fig.text(0.5, -0.02, "\n".join(textwrap.wrap(text, width)), ha="center",
             va="top", fontsize=8.5, style="italic", color="dimgray")


def _mean(key):
    return float(np.mean([f[key] for f in FOLDS]))


def _std(key):
    return float(np.std([f[key] for f in FOLDS]))


# ===========================================================================
# fig01 -- Confusion matrix            (their Fig. 10 / 11)
# ===========================================================================
def fig01_confusion():
    cm = CONFUSION
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    thresh = cm.max() * 0.55
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")
    acc = 100 * np.trace(cm) / cm.sum()
    ax.set_title(f"Confusion Matrix - RRF-Net (5-fold pooled)\n"
                 f"Accuracy = {acc:.2f}%   n = {cm.sum():,}")
    ax.set_xticks(range(4), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Count", fraction=0.046, pad=0.04)
    _save(fig, "fig01_confusion_matrix")


# ===========================================================================
# fig02 -- Training and validation curves          (their Fig. 12)
# ===========================================================================
def fig02_training_curves():
    n1 = len(CURVE_S1["acc"])
    ep = np.arange(1, n1 + len(CURVE_S2["acc"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    panels = [("Training and Validation Loss", "loss", "val_loss", "Loss"),
              ("Training and Validation Accuracy", "acc", "val_acc", "Accuracy"),
              ("Routing Loss (supervised gate)", "route", "val_route", "Route BCE")]
    for ax, (title, tk, vk, ylab) in zip(axes, panels):
        tr = CURVE_S1[tk] + CURVE_S2[tk]
        va = CURVE_S1[vk] + CURVE_S2[vk]
        ax.plot(ep, tr, color=TRAIN_COLOR, lw=1.6, label=f"Training {ylab}")
        ax.plot(ep, va, color=VAL_COLOR, lw=1.6, label=f"Validation {ylab}")
        ax.axvline(n1 + 0.5, color="gray", lw=1.0, ls="--")
        ax.text(n1 + 0.9, 0.93, "backbones unfrozen", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform(), va="top")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.legend(loc="best")
    fig.suptitle("RRF-Net: training and validation metrics (mean of 5 folds)",
                 fontsize=13, fontweight="bold", y=1.03)
    _save(fig, "fig02_training_curves")


# ===========================================================================
# fig03 -- Histogram of model accuracies            (their Fig. 13)
# ===========================================================================
def fig03_model_comparison():
    names = ["CNN branch\n(EfficientNetV2-B0)", "ViT branch\n(ViT-B/16)"]
    vals = [_mean("cnn"), _mean("vit")]
    for b, d in BASELINES.items():
        names.append(b)
        vals.append(float(np.mean(d["folds"])))
    names.append(OURS)
    vals.append(_mean("tta"))

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    colors = [HILITE if n == OURS else BAR_COLOR for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black",
                  linewidth=0.7, zorder=3)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.12, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=9.5)
    ax.set_title("Histogram of Model Accuracies (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(names)), names, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(90, 100.6)
    ax.text(0.5, 0.965,
            "Branch rows are auxiliary heads inside the joint model, not "
            "independently trained networks",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            style="italic", color="dimgray")
    _save(fig, "fig03_model_comparison")


# ===========================================================================
# fig04 -- Ablation study                 (their Table 11 / Fig. 28 analogue)
# ===========================================================================
def fig04_ablation():
    names = [a["name"] for a in ABLATION]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))
    specs = [("vit", "ViT branch accuracy (%)", "ViT Branch"),
             ("tta", "Fused + TTA accuracy (%)", "Full Model + TTA"),
             ("beta", r"$\bar{\beta}$ (weight on CNN)", "Adaptive Gate Position"),
             ("gain", "Fused - better branch (pp)", "Fusion Contribution")]
    for ax, (k, ylab, title) in zip(axes, specs):
        vals = [a[k] for a in ABLATION]
        colors = [HILITE if a["name"].startswith("Last 4") else BAR_COLOR
                  for a in ABLATION]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7,
                      zorder=3)
        for b_, v, a in zip(bars, vals, ABLATION):
            ax.text(b_.get_x() + b_.get_width() / 2, v,
                    f"{v:.2f}" + ("\n(1 fold)" if a["n_folds"] == 1 else ""),
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x, names, fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.margins(y=0.22)
        if k in ("vit", "tta"):
            ax.set_ylim(min(vals) - 3, 100.5)
    fig.suptitle("Ablation study: ViT fine-tuning capacity vs. fusion contribution",
                 fontsize=13, fontweight="bold", y=1.03)
    _foot(fig,
          "As ViT capacity rises the gate collapses onto it (beta 0.71 -> 0.11) and "
          "the fusion stops contributing: the highest accuracy (98.68%, one fold) "
          "comes from the configuration where the fusion is worth +0.00 pp.")
    _save(fig, "fig04_ablation")


# ===========================================================================
# fig05 -- Per-class performance                  (their Table 6 analogue)
# ===========================================================================
def fig05_per_class():
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1-score"]
    x = np.arange(len(CLASS_NAMES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [PER_CLASS[c][m] for c in CLASS_NAMES]
        bars = ax.bar(x + (i - 1) * w, vals, width=w * 0.92, label=lab,
                      edgecolor="black", linewidth=0.6, zorder=3)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.08, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("RRF-Net per-class performance (TTA, mean of 5 folds)")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(94, 100.6)
    ax.legend(loc="lower right")
    _save(fig, "fig05_per_class")


# ===========================================================================
# fig06 -- Efficiency analysis + fold stability   (their Table 10 analogue)
# ===========================================================================
def fig06_efficiency():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    labels = [k for k, _ in EFFICIENCY][::-1]
    vals = [v for _, v in EFFICIENCY][::-1]
    bars = ax.barh(range(len(vals)), vals, color=BAR_COLOR, edgecolor="black",
                   linewidth=0.7, zorder=3)
    for b_, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.012, b_.get_y() + b_.get_height() / 2,
                f"{v:,.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Value")
    ax.set_title("Efficiency analysis of RRF-Net")
    ax.margins(x=0.18)

    folds = [f["fold"] for f in FOLDS]
    ax2.plot(folds, [f["tta"] for f in FOLDS], "-o", color=HILITE, lw=2,
             ms=8, label=OURS, zorder=4)
    for b, d in BASELINES.items():
        ax2.plot(folds, d["folds"], "-o", lw=1.3, ms=5, alpha=0.75, label=b)
    ax2.set_title("Per-fold accuracy (TTA), same folds for every model")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(folds)
    ax2.legend(loc="lower left", fontsize=8.5, ncol=2)
    m, s = _mean("tta"), _std("tta")
    ax2.text(0.98, 0.04, f"{OURS}: {m:.2f}% $\\pm$ {s:.2f}",
             transform=ax2.transAxes, ha="right", fontsize=9, color=HILITE)
    _save(fig, "fig06_efficiency")


# ===========================================================================
# fig11 -- Train vs Test accuracy with trendline     (CAFNet Fig. 17)
# ===========================================================================
def fig11_train_test_trendline():
    names = list(BACKBONE_CURVES.keys()) + [OURS]
    train = [BACKBONE_CURVES[n]["final_train"] for n in names[:-1]] + \
            [CURVE_S2["acc"][-1] * 100]
    test = [float(np.mean(BASELINES[n]["folds"])) for n in names[:-1]] + \
           [_mean("tta")]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    b1 = ax.bar(x - w / 2, train, w, label="Train Accuracy", color="mediumseagreen",
                edgecolor="black", linewidth=0.7, zorder=3)
    b2 = ax.bar(x + w / 2, test, w, label="Test Accuracy", color="coral",
                edgecolor="black", linewidth=0.7, zorder=3)
    for bars, vals in ((b1, train), (b2, test)):
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.12, f"{v:.2f}%",
                    ha="center", va="bottom", fontsize=8.5)
    z = np.polyfit(x, test, 1)
    ax.plot(x, np.poly1d(z)(x), "--", color="tab:blue", lw=1.8,
            label="Test Trendline", zorder=4)
    gaps = [t - s_ for t, s_ in zip(train, test)]
    ax.set_title("Train & Test Accuracy with Trendline")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, names, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(90, 101.5)
    ax.legend(loc="lower left")
    _foot(fig,
          "Train - test gap:  " +
          "   ".join(f"{n.split()[0]} {g:+.2f}" for n, g in zip(names, gaps)) +
          ".  Train accuracy is the final fine-tuning epoch without TTA while test "
          "accuracy includes it, so a negative gap means TTA is worth more than the "
          "train/test difference -- the same caveat applies to CAFNet's Fig. 17.")
    _save(fig, "fig11_train_test_trendline")


# ===========================================================================
# fig12 -- Training, validation and test accuracy    (CAFNet Fig. 16)
# ===========================================================================
def fig12_train_val_test():
    n1 = len(CURVE_S1["acc"])
    tr = np.array(CURVE_S1["acc"] + CURVE_S2["acc"]) * 100
    va = np.array(CURVE_S1["val_acc"] + CURVE_S2["val_acc"]) * 100
    ep = np.arange(1, len(tr) + 1)
    test = _mean("tta")
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.plot(ep, tr, "-o", color=TRAIN_COLOR, ms=4, lw=1.6, label="Training Accuracy")
    ax.plot(ep, va, "-o", color=VAL_COLOR, ms=4, lw=1.6, label="Validation Accuracy")
    ax.axhline(test, color="green", ls="--", lw=1.6,
               label=f"Test Accuracy: {test:.2f}%")
    ax.annotate(f"{test:.2f}%", (ep[-1], test), textcoords="offset points",
                xytext=(-6, 6), ha="right", color="green", fontsize=9)
    ax.annotate(f"{tr[-1]:.2f}%", (ep[-1], tr[-1]), textcoords="offset points",
                xytext=(-4, 6), ha="right", color=TRAIN_COLOR, fontsize=9)
    ax.annotate(f"{va[-1]:.2f}%", (ep[-1], va[-1]), textcoords="offset points",
                xytext=(-4, -14), ha="right", color=VAL_COLOR, fontsize=9)
    ax.axvline(n1 + 0.5, color="gray", lw=1.0, ls="--")
    ax.text(n1 + 0.9, 0.04, "backbones unfrozen", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())
    ax.set_title("Training, Validation, and Test Accuracy - RRF-Net")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")
    _save(fig, "fig12_train_val_test_accuracy")


# ===========================================================================
# fig13 -- 5-fold cross-validation accuracy          (CAFNet Fig. 11)
# ===========================================================================
def fig13_cv_accuracy():
    folds = [f["fold"] for f in FOLDS]
    acc = np.array([f["tta"] for f in FOLDS]) / 100
    mean = acc.mean()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for a, zoom in ((ax, False), (ax2, True)):
        a.plot(folds, acc, "-o", color=TRAIN_COLOR, lw=1.8, ms=7,
               label="Fold Accuracy")
        a.axhline(mean, color=VAL_COLOR, ls="--", lw=1.5,
                  label=f"Mean Accuracy: {mean:.4f}")
        a.set_xlabel("Fold")
        a.set_ylabel("Accuracy")
        a.set_xticks(folds)
        if zoom:
            pad = (acc.max() - acc.min()) * 0.9 + 0.001
            a.set_ylim(acc.min() - pad, acc.max() + pad)
            for f_, v in zip(folds, acc):
                a.annotate(f"{v:.4f}", (f_, v), textcoords="offset points",
                           xytext=(0, 9), ha="center", fontsize=8.5)
            a.set_title("Same data, zoomed to the fold range")
        else:
            a.set_ylim(0, 1.03)
            a.set_title("5-Fold Cross-Validation Accuracy")
            a.legend(loc="lower left")
    fig.suptitle(f"RRF-Net 5-fold cross-validation (TTA): "
                 f"{_mean('tta'):.2f}% $\\pm$ {_std('tta'):.2f}",
                 fontsize=13, fontweight="bold", y=1.02)
    _foot(fig,
          "The left panel uses CAFNet's full 0-1 axis; at this accuracy that hides "
          "all the between-fold variation, which is why the zoomed panel is shown "
          "beside it.")
    _save(fig, "fig13_cv_accuracy")


# ===========================================================================
# fig14 -- All models: train/validation accuracy and loss  (CAFNet Fig. 14/15)
# ===========================================================================
def fig14_all_model_curves():
    curves = dict(BACKBONE_CURVES)
    curves[OURS] = dict(acc=CURVE_S1["acc"] + CURVE_S2["acc"],
                        val_acc=CURVE_S1["val_acc"] + CURVE_S2["val_acc"],
                        loss=CURVE_S1["loss"] + CURVE_S2["loss"],
                        val_loss=CURVE_S1["val_loss"] + CURVE_S2["val_loss"])
    # Explicit colours: matplotlib's 4th default is red, which would collide
    # with the highlight colour reserved for the proposed model.
    base_colors = {"EfficientNetB0": "tab:blue", "ResNet50": "tab:orange",
                   "DenseNet121": "tab:green", "MobileNetV2": "tab:purple"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    fig.subplots_adjust(hspace=0.42)
    panels = [("acc", "Training Accuracy", "Accuracy"),
              ("val_acc", "Validation Accuracy", "Accuracy"),
              ("loss", "Training Loss", "Loss"),
              ("val_loss", "Validation Loss", "Loss")]
    for ax, (k, title, ylab) in zip(axes.ravel(), panels):
        for name, c in curves.items():
            y = np.array(c[k])
            ep = np.arange(1, len(y) + 1)
            ax.plot(ep, y, lw=2.2 if name == OURS else 1.4,
                    color=HILITE if name == OURS else base_colors[name],
                    zorder=4 if name == OURS else 3, label=name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
    axes[0, 0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Transfer-learning models and RRF-Net: training and validation "
                 "metrics (mean of 5 folds)", fontsize=13, fontweight="bold",
                 y=1.0)
    _foot(fig,
          "Backbones fine-tune for 35 epochs, RRF-Net for 5 + 18. RRF-Net starts "
          "far higher because its first 5 epochs train the fusion head on frozen "
          "backbones, so the epoch axis is not a like-for-like schedule "
          "comparison.")
    _save(fig, "fig14_all_model_curves")


# ===========================================================================
# fig15 -- Dataset composition and fold split        (CAFNet Figs. 2/3)
# ===========================================================================
def fig15_dataset_distribution():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    counts = [DATASET["per_class"][c] for c in CLASS_NAMES]
    bars = ax.bar(CLASS_NAMES, counts, color=BAR_COLOR, edgecolor="black",
                  linewidth=0.7, zorder=3)
    for b_, v in zip(bars, counts):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 12, f"{v:,}", ha="center",
                va="bottom", fontsize=10)
    ax.set_title(f"Dataset class distribution (n = {DATASET['n_images']:,})")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts) * 1.16)

    split = [("Train", DATASET["n_train"]), ("Validation", DATASET["n_val"]),
             ("Test", DATASET["n_test"])]
    tot = sum(v for _, v in split)
    left = 0
    for (lab, v), col in zip(split, ["mediumseagreen", "gold", "coral"]):
        ax2.barh([0], [v], height=0.42, left=left, color=col, edgecolor="black",
                 linewidth=0.7, zorder=3, label=f"{lab}: {v:,} ({100*v/tot:.0f}%)")
        ax2.text(left + v / 2, 0, f"{v:,}", ha="center", va="center", fontsize=10)
        left += v
    ax2.set_yticks([])
    ax2.set_ylim(-0.45, 1.25)
    ax2.set_xlabel("Images per fold")
    ax2.set_title("Per-fold split (identical for every fold and model)")
    ax2.legend(loc="upper center", ncol=1, fontsize=9)
    _foot(fig,
          "The 4-class Kaggle release is exactly balanced at 1,800 images per "
          "class, so no class weighting is applied anywhere. Image sizes range "
          f"{DATASET['width_range'][0]}-{DATASET['width_range'][1]} px wide and "
          f"{DATASET['height_range'][0]}-{DATASET['height_range'][1]} px high; only "
          "these ranges were recorded, not the full distributions, which is why "
          "CAFNet's Figs. 5-8 (width/height histograms, the width-height scatter "
          "and the pixel-intensity distribution) have no counterpart here.")
    _save(fig, "fig15_dataset_distribution")


# ===========================================================================
# Probability-based figures -- their Figs. 8/9, 18, 19, 20.
# These need the per-sample arrays written by step9's SAVE_PROBS block.
# ===========================================================================
def load_probs(tag="rrf_full_v3_vit4", mode="image"):
    files = sorted(glob.glob(os.path.join(PROBS_DIR, f"{tag}_{mode}_fold*.npz")))
    return [dict(np.load(f)) for f in files]


def fig07_roc(folds):
    """Their Fig. 8/9 -- one-vs-rest ROC with per-class AUC."""
    from sklearn.metrics import roc_curve, auc
    y = np.concatenate([f["y_true"] for f in folds])
    p = np.concatenate([f["p_tta"] for f in folds])
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    aucs = []
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((y == i).astype(int), p[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, lw=1.8, label=f"{c} (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Random Classifier")
    ax.set_title(f"ROC Curves - Multi-class Classification\n"
                 f"Macro-average AUC = {np.mean(aucs):.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, "fig07_roc_curves")


def fig08_reliability(folds, n_bins=10):
    """Their Fig. 18 -- reliability diagram, overall and per class, with the
    sample-density shading that explains why ECE is dominated by the top bin."""
    y = np.concatenate([f["y_true"] for f in folds])
    p = np.concatenate([f["p_tta"] for f in folds])
    panels = [("Overall (max-softmax)", None)] + [(c, i) for i, c in enumerate(CLASS_NAMES)]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    edges = np.linspace(0, 1, n_bins + 1)
    for ax, (title, cls) in zip(axes.ravel(), panels):
        if cls is None:
            conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
        else:
            conf, correct = p[:, cls], (y == cls).astype(float)
        ece, xs, ys, dens = 0.0, [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            dens.append(m.mean())
            if m.sum():
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.bar(centres, dens, width=1 / n_bins, color="lightgray", zorder=1,
               label="Sample density")
        for c_, a_, cf in zip(centres, ys, xs):
            if np.isnan(a_):
                continue
            ax.bar([c_], [a_], width=1 / n_bins, zorder=2, alpha=0.55,
                   color="lightcoral" if a_ < cf else "lightskyblue")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, zorder=3)
        ax.plot(xs, ys, "-o", color="tab:blue", lw=1.6, ms=4, zorder=4)
        ax.text(0.03, 0.95, f"ECE = {ece:.4f}", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(fc="white", ec="gray", lw=0.6))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].legend(
        handles=[plt.Line2D([], [], color="tab:blue", marker="o", label="Model calibration"),
                 plt.Line2D([], [], color="k", ls="--", label="Perfect calibration"),
                 plt.Rectangle((0, 0), 1, 1, fc="lightcoral", alpha=.55, label="Overconfident bins"),
                 plt.Rectangle((0, 0), 1, 1, fc="lightskyblue", alpha=.55, label="Underconfident bins"),
                 plt.Rectangle((0, 0), 1, 1, fc="lightgray", label="Sample density")],
        loc="center", fontsize=9.5)
    fig.suptitle("Reliability Diagrams (Calibration Plots) - 4-class classification",
                 fontsize=13, fontweight="bold", y=1.0)
    _save(fig, "fig08_reliability_diagrams")


def fig09_uncertainty(folds):
    """Their Fig. 19 -- confidence and entropy, correct vs incorrect."""
    from scipy.stats import ks_2samp, gaussian_kde
    y = np.concatenate([f["y_true"] for f in folds])
    p = np.concatenate([f["p_tta"] for f in folds])
    ok = p.argmax(1) == y
    conf = p.max(1)
    ent = -np.sum(p * np.log(p + 1e-12), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    for ax, vals, title, xlab in ((axes[0], conf, "Max-Softmax Confidence Distribution",
                                   "Max-Softmax Confidence"),
                                  (axes[1], ent, "Predictive Entropy Distribution",
                                   "Predictive Entropy")):
        ks = ks_2samp(vals[ok], vals[~ok])
        grid = np.linspace(vals.min(), vals.max(), 400)
        for mask, lbl, col in ((ok, "Correct", "green"), (~ok, "Incorrect", "red")):
            if mask.sum() < 2:
                continue
            ax.fill_between(grid, gaussian_kde(vals[mask])(grid), alpha=0.35,
                            color=col, label=lbl)
            ax.axvline(vals[mask].mean(), color=col, ls="--", lw=1.4,
                       label=f"{lbl} $\\mu$={vals[mask].mean():.3f}")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.text(0.02, 0.97, f"KS stat={ks.statistic:.3f}\np={ks.pvalue:.2e}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="gray", lw=0.6))
    fig.suptitle("Uncertainty distributions for correct vs. incorrect predictions",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "fig09_uncertainty_distributions")


def fig10_coverage_risk(folds):
    """Their Fig. 20 -- selective prediction / coverage-risk."""
    y = np.concatenate([f["y_true"] for f in folds])
    p = np.concatenate([f["p_tta"] for f in folds])
    conf, ok = p.max(1), (p.argmax(1) == y)
    order = np.argsort(-conf)
    ok_sorted = ok[order]
    n = len(ok_sorted)
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(ok_sorted) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else \
        float(np.trapz(risk, cov))
    full_risk = 1 - ok.mean()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.0))
    ax.fill_between(cov, risk, color="lightskyblue", alpha=0.5,
                    label="Abstention benefit")
    ax.plot(cov, risk, color="tab:blue", lw=1.6, label=f"AURC = {aurc:.4f}")
    ax.axhline(full_risk, color="gray", ls="--", lw=1.0,
               label=f"Random baseline ({full_risk:.3f})")
    for c in (0.7, 0.8, 0.9):
        i = int(c * n) - 1
        ax.plot([c], [risk[i]], "o", color=HILITE, ms=5)
        ax.annotate(f"Risk@{int(c*100)}%={risk[i]:.3f}", (c, risk[i]),
                    textcoords="offset points", xytext=(-6, 9), fontsize=8,
                    color=HILITE, ha="right")
    ax.set_title("Coverage-Risk Curve (Selective Prediction)")
    ax.set_xlabel("Coverage (fraction of samples predicted)")
    ax.set_ylabel("Risk (error rate)")
    ax.legend(loc="upper left", fontsize=8.5)

    sel = np.cumsum(ok_sorted) / np.arange(1, n + 1)
    ax2.plot(cov, sel, color="green", lw=1.6, label="Selective accuracy")
    ax2.axhline(ok.mean(), color="k", ls="--", lw=1.0,
                label=f"Full-data accuracy ({ok.mean():.4f})")
    i90 = int(0.9 * n) - 1
    ax2.axvline(0.9, color=HILITE, lw=1.0)
    ax2.annotate(f"Acc@90% = {sel[i90]:.4f}", (0.9, sel[i90]),
                 textcoords="offset points", xytext=(-8, -16), fontsize=8.5,
                 color=HILITE, ha="right")
    ax2.set_title("Selective Accuracy vs Coverage")
    ax2.set_xlabel("Coverage")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Selective Prediction / Coverage-Risk Analysis "
                 "(abstention via max-softmax confidence)",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "fig10_coverage_risk")


# ===========================================================================
# Driver
# ===========================================================================
HARDCODED = [("confusion", fig01_confusion), ("curves", fig02_training_curves),
             ("comparison", fig03_model_comparison), ("ablation", fig04_ablation),
             ("perclass", fig05_per_class), ("efficiency", fig06_efficiency),
             ("trendline", fig11_train_test_trendline),
             ("trainvaltest", fig12_train_val_test),
             ("cv", fig13_cv_accuracy),
             ("allcurves", fig14_all_model_curves),
             ("dataset", fig15_dataset_distribution)]
NEEDS_PROBS = [("roc", fig07_roc), ("reliability", fig08_reliability),
               ("uncertainty", fig09_uncertainty), ("coverage", fig10_coverage_risk)]

MISSING = """
Figures 7-10 (ROC / reliability / uncertainty / coverage-risk) were NOT drawn.

They are the four figures in Mahmud et al. that need a predicted probability
for every individual test image, and our logs only ever recorded scalars
(accuracy, ECE, NLL). They are deliberately not hardcoded: inventing
per-sample values would mean inventing the curves.

step9_rrf_fusion.py now saves exactly what they need:

    SAVE_PROBS = True  ->  /kaggle/working/probs/*.npz

Download that folder into ./{d}/ after the next run and re-run this script --
the four figures then generate from real data, with no edit to this file.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on figure name")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n, _ in HARDCODED:
            print(f"{n:14s} (hardcoded)")
        for n, _ in NEEDS_PROBS:
            print(f"{n:14s} (needs ./{PROBS_DIR}/)")
        return

    print(f"RRF-Net paper figures -> {OUT_DIR}/")
    print(f"  5-fold mean, TTA: {_mean('tta'):.2f}% +/- {_std('tta'):.2f}   "
          f"ECE {_mean('ece'):.4f}   NLL {_mean('nll'):.4f}\n")
    for name, fn in HARDCODED:
        if args.only and args.only not in name:
            continue
        fn()

    probs = load_probs()
    if probs:
        print(f"\nFound {len(probs)} per-fold probability files in {PROBS_DIR}/")
        for name, fn in NEEDS_PROBS:
            if args.only and args.only not in name:
                continue
            fn(probs)
    print(f"\n{len(MANIFEST)} figures written.")
    if not probs:
        print(MISSING.format(d=PROBS_DIR))


if __name__ == "__main__":
    main()
