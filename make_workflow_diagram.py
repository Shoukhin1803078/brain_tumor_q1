"""
Generates the architecture/workflow diagram referenced in brain_tumor.tex as
images/Brain tumor classification.drawio.png -- built programmatically with
matplotlib (no drawio needed) so it exactly matches the pipeline described in
Section III (Methodology): EfficientNetB0 backbone with two-stage fine-tuning,
feeding a dual inference head (softmax+TTA, and an embedding-based
RF/XGBoost/SVM soft-voting ensemble).

Run locally: python3 make_workflow_diagram.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "Brain tumor classification.drawio.png")

fig, ax = plt.subplots(figsize=(14, 5.3))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.4)
ax.axis("off")

BOX_STYLE = dict(boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.4)

COLORS = {
    "input":    "#dbe9f6",
    "prep":     "#d7f0e3",
    "backbone": "#fde3cf",
    "head":     "#e8ddf7",
    "branch":   "#fff3cd",
    "output":   "#f7d7da",
}


def box(x, y, w, h, text, color, fontsize=9.5, weight="normal"):
    rect = FancyBboxPatch((x, y), w, h, facecolor=color, edgecolor="black",
                           **BOX_STYLE, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, zorder=3, linespacing=1.4)
    return (x, y, w, h)


def arrow(b1, b2, from_side="right", to_side="left", **kw):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = {"right": (x1 + w1, y1 + h1 / 2), "top": (x1 + w1 / 2, y1 + h1),
          "bottom": (x1 + w1 / 2, y1), "left": (x1, y1 + h1 / 2)}[from_side]
    p2 = {"right": (x2 + w2, y2 + h2 / 2), "top": (x2 + w2 / 2, y2 + h2),
          "bottom": (x2 + w2 / 2, y2), "left": (x2, y2 + h2 / 2)}[to_side]
    fa = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                          linewidth=1.3, color="black", zorder=1, **kw)
    ax.add_patch(fa)


# ---------------------------------------------------------------------------
# Stage 1: Input -> Preprocessing -> Backbone
# ---------------------------------------------------------------------------
b_input = box(0.2, 2.4, 1.5, 1.2, "Input MRI\n224$\\times$224$\\times$3", COLORS["input"])
b_prep = box(2.1, 2.2, 1.9, 1.6,
             "Preprocessing\n• Resize / normalize\n• Class-weighted\n  sampling\n• Augmentation\n  (rot./zoom/flip)",
             COLORS["prep"], fontsize=8.3)
b_backbone = box(4.4, 1.9, 2.3, 2.2,
                  "EfficientNetB0\nBackbone\n(ImageNet-pretrained)\n\nStage 1: head-only\nStage 2: fine-tune\ntop 80% layers",
                  COLORS["backbone"], fontsize=8.3)
b_head = box(7.1, 1.9, 2.1, 2.2,
             "GAP $\\to$ BatchNorm\n$\\to$ Dropout(0.4)\n$\\to$ Dense(128)\n[Embedding]\n$\\to$ BatchNorm\n$\\to$ Dropout(0.3)",
             COLORS["head"], fontsize=8.0)

arrow(b_input, b_prep)
arrow(b_prep, b_backbone)
arrow(b_backbone, b_head)

# ---------------------------------------------------------------------------
# Stage 2: dual inference branches
# ---------------------------------------------------------------------------
b_branch_top = box(9.6, 3.5, 2.1, 1.6,
                    "Softmax\n(4 classes)\n+ Test-Time\nAugmentation\n(5$\\times$ avg.)",
                    COLORS["branch"], fontsize=8.3)
b_branch_bot = box(9.6, 0.7, 2.1, 2.0,
                    "Embedding\n$\\to$ Random Forest\n$\\to$ XGBoost\n$\\to$ RBF SVM\n$\\to$ Soft Voting",
                    COLORS["branch"], fontsize=8.3)

arrow(b_head, b_branch_top, from_side="right", to_side="left")
arrow(b_head, b_branch_bot, from_side="right", to_side="left")

b_output = box(12.1, 2.15, 1.7, 1.6, "Predicted\nClass\n(glioma /\nmeningioma /\nno tumor /\npituitary)",
               COLORS["output"], fontsize=8.3)

arrow(b_branch_top, b_output, from_side="right", to_side="left")
arrow(b_branch_bot, b_output, from_side="right", to_side="left")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_PATH}")
