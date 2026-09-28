"""
Step 7: Generate comparison figures for the multi-backbone sweep, parsing
directly from step_6_backbone_sweep_logs.log (the raw Kaggle console output
for the resnet50 / densenet121 / mobilenetv2 runs of step6_backbone_sweep.py),
plus the EfficientNetB0 numbers already reported from step4_kaggle4class.py.

Run locally (no GPU needed -- this only parses text logs and plots):
    python3 step7_backbone_figures.py

Produces, under images/:
    backbone_accuracy_bar.png       -- grouped bar, mean+/-std accuracy per
                                        backbone x method (CNN / TTA / Ensemble)
    backbone_fold_variance.png      -- per-fold accuracy scatter/strip per
                                        backbone, TTA method, showing spread
    backbone_latency_tradeoff.png   -- accuracy (TTA) vs single-pass CNN
                                        inference latency, one point per backbone
    backbone_training_time.png      -- stacked bar: stage-1 vs stage-2 training
                                        time per backbone (timed backbones only)

Colors use the Okabe-Ito colorblind-safe qualitative palette, assigned in a
fixed order across all four figures (same backbone -> same color everywhere).
"""

import re
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "step_6_backbone_sweep_logs.log"

# EfficientNetB0 is now also parsed directly from the log (it was re-run
# through step6_backbone_sweep.py's timing-instrumented code, so it no longer
# needs to be hardcoded or excluded from the latency/training-time figures).

# Okabe-Ito colorblind-safe palette, fixed assignment order (same order used
# in every figure below).
COLORS = {
    "EfficientNetB0": "#0072B2",  # blue
    "ResNet50": "#E69F00",        # orange
    "DenseNet121": "#009E73",     # bluish green
    "MobileNetV2": "#D55E00",     # vermillion
}
ORDER = ["EfficientNetB0", "ResNet50", "DenseNet121", "MobileNetV2"]


def parse_log(path):
    """Extract per-fold accuracy + timing rows for each backbone from the raw
    Kaggle console log. Returns {backbone: {"cnn":[...], "tta":[...], "ens":[...],
    "train_time":[...], "cnn_ms":[...], "tta_ms":[...], "ens_ms":[...]}}."""
    text = open(path).read()
    pattern = re.compile(
        r"Fold \d+ \[(?P<backbone>\w+)\] results: "
        r"CNN=(?P<cnn>[\d.]+) \((?P<cnn_ms>[\d.]+) ms/img\)\s+"
        r"CNN\+TTA=(?P<tta>[\d.]+) \((?P<tta_ms>[\d.]+) ms/img\)\s+"
        r"Ensemble=(?P<ens>[\d.]+) \((?P<ens_ms>[\d.]+) ms/img\)\s+"
        r"train_time=(?P<train_time>[\d.]+)s"
    )
    data = {}
    for m in pattern.finditer(text):
        b = m.group("backbone")
        d = data.setdefault(b, {"cnn": [], "tta": [], "ens": [],
                                 "train_time": [], "cnn_ms": [], "tta_ms": [], "ens_ms": []})
        d["cnn"].append(100 * float(m.group("cnn")))
        d["tta"].append(100 * float(m.group("tta")))
        d["ens"].append(100 * float(m.group("ens")))
        d["train_time"].append(float(m.group("train_time")))
        d["cnn_ms"].append(float(m.group("cnn_ms")))
        d["tta_ms"].append(float(m.group("tta_ms")))
        d["ens_ms"].append(float(m.group("ens_ms")))
    return data


def main():
    data = parse_log(LOG_PATH)
    # Rename to display names matching COLORS/ORDER
    name_map = {"resnet50": "ResNet50", "densenet121": "DenseNet121",
                "mobilenetv2": "MobileNetV2", "efficientnetb0": "EfficientNetB0"}
    data = {name_map.get(k, k): v for k, v in data.items()}
    for name in ORDER:
        assert name in data, f"missing {name} in parsed log"
        for key in ("cnn", "tta", "ens"):
            assert len(data[name][key]) == 5, f"{name} {key} does not have 5 folds"

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })

    # -----------------------------------------------------------------
    # Fig 1: grouped bar -- mean +/- std accuracy, backbone x method
    # -----------------------------------------------------------------
    methods = ["cnn", "tta", "ens"]
    method_labels = ["CNN (softmax)", "CNN + TTA", "Embedding ensemble"]
    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, name in enumerate(ORDER):
        means = [np.mean(data[name][m]) for m in methods]
        stds = [np.std(data[name][m]) for m in methods]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                       label=name, color=COLORS[name], edgecolor="white", linewidth=0.5)
        for bar, mean_v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean_v + 0.55,
                     f"{mean_v:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("5-fold accuracy (%)")
    ax.set_ylim(90, 100.5)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    fig.tight_layout()
    fig.savefig("images/backbone_accuracy_bar.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Fig 2: per-fold spread (strip plot) for CNN+TTA, one column per backbone
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, name in enumerate(ORDER):
        vals = data[name]["tta"]
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.12
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=COLORS[name],
                   s=45, zorder=3, edgecolor="white", linewidth=0.6)
        ax.hlines(np.mean(vals), i - 0.22, i + 0.22, color=COLORS[name], linewidth=2.2, zorder=2)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=15, ha="right")
    ax.set_ylabel("Per-fold accuracy, CNN+TTA (%)")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/backbone_fold_variance.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Fig 3: accuracy vs. inference latency trade-off (CNN single-pass ms/img
    # on the x-axis, mean TTA accuracy on the y-axis -- one axis pair, no
    # dual-axis; all four backbones now share identical timing instrumentation)
    # -----------------------------------------------------------------
    timed = list(ORDER)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name in timed:
        lat = np.mean(data[name]["cnn_ms"])
        acc = np.mean(data[name]["tta"])
        ax.scatter(lat, acc, color=COLORS[name], s=140, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(name, (lat, acc), textcoords="offset points", xytext=(8, 6), fontsize=9.5)
    ax.set_xlabel("Mean CNN inference latency (ms/image, single pass)")
    ax.set_ylabel("Mean CNN+TTA accuracy (%)")
    ax.grid(color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/backbone_latency_tradeoff.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Fig 4: stacked bar -- stage-1 vs stage-2 training time per backbone
    # -----------------------------------------------------------------
    # stage-1/stage-2 split isn't stored separately in the regex above; re-parse.
    text = open(LOG_PATH).read()
    stage_pattern = re.compile(
        r"Fold \d+ \[(?P<backbone>\w+)\].*?train_time=[\d.]+s \(stage1=(?P<s1>[\d.]+)s, stage2=(?P<s2>[\d.]+)s\)"
    )
    stage_data = {}
    for m in stage_pattern.finditer(text):
        b = name_map.get(m.group("backbone"), m.group("backbone"))
        d = stage_data.setdefault(b, {"s1": [], "s2": []})
        d["s1"].append(float(m.group("s1")))
        d["s2"].append(float(m.group("s2")))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    s1_means = [np.mean(stage_data[n]["s1"]) for n in timed]
    s2_means = [np.mean(stage_data[n]["s2"]) for n in timed]
    x = np.arange(len(timed))
    ax.bar(x, s1_means, width=0.5, label="Stage 1 (head only)", color="#B0B0B0", edgecolor="white")
    ax.bar(x, s2_means, width=0.5, bottom=s1_means, label="Stage 2 (fine-tuning)",
           color=[COLORS[n] for n in timed], edgecolor="white")
    for i, n in enumerate(timed):
        total = s1_means[i] + s2_means[i]
        ax.text(i, total + 30, f"{total/60:.1f} min", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(timed)
    ax.set_ylabel("Mean training time per fold (s)")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig("images/backbone_training_time.png", dpi=200)
    plt.close(fig)

    print("Saved 4 figures to images/: backbone_accuracy_bar.png, "
          "backbone_fold_variance.png, backbone_latency_tradeoff.png, "
          "backbone_training_time.png")


if __name__ == "__main__":
    main()
