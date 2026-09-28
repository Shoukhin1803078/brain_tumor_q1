"""
Step 8: Exploratory Data Analysis (EDA) for the 4-class Kaggle "Brain Tumor
MRI Dataset" (masoudnickparvar) used throughout this study.

Runs entirely on CPU -- no TensorFlow/GPU needed, only image reading (PIL),
hashing, and plotting (numpy, matplotlib, scikit-learn). Safe to run in a
Kaggle CPU-only session, as long as the dataset is attached as an input
(same path as step4_kaggle4class.py / step6_backbone_sweep.py).

Produces, under images/ and results/:
    eda_a_class_distribution.png
    eda_a_image_dimensions.png
    eda_a_sample_grid.png
    eda_a_intensity_boxplot.png
    eda_a_mean_images.png
    eda_a_near_duplicate_examples.png     -- only if CONFIRMED pairs exist
    results/eda_a_near_duplicates.csv     -- only CONFIRMED pairs (Stage 2)
    results/eda_a_summary.json            -- headline numbers for the paper

NEAR-DUPLICATE / LEAKAGE AUDIT -- history and design rationale
----------------------------------------------------------------
A first version of this script used a single-stage 8x8 (64-bit) average
hash with a loose Hamming-distance threshold. That approach was DISCARDED
after failing a basic sanity check: even at Hamming distance 0 (the
strictest possible match), only ~49% of flagged pairs shared the same class
label -- essentially the 25% base rate you'd expect from 4 balanced classes
if the hash carried *no* signal at all, barely above chance. The cause,
confirmed by manual inspection: at 8x8 resolution, brain MRI slices from
every class share the same coarse layout (dark background, centered bright
oval skull/brain), which dominates a tiny hash and swamps the actual
class-specific content. A hash that can't even distinguish glioma from
pituitary at distance 0 cannot be trusted to detect genuine near-duplicates.

This version fixes that with a two-stage design, which is the standard,
correct way to do near-duplicate detection at scale:

  Stage 1 (cheap, high-recall candidate filter): a finer 16x16 (256-bit)
  average hash, compared pairwise via vectorized Hamming distance, with a
  deliberately loose threshold. This is NOT the final answer -- it only
  narrows ~26 million possible pairs down to a manageable candidate set.

  Stage 2 (expensive, high-precision verification): for every Stage-1
  candidate pair, compute the actual root-mean-square pixel difference
  (RMSE) between 64x64 grayscale thumbnails of the two images. Only pairs
  below a strict RMSE threshold are called CONFIRMED near-duplicates. This
  is what actually decides "near-duplicate" status -- the hash is only
  ever used to avoid an O(n^2) pixel comparison over the full dataset.

The script prints the same-class rate among CONFIRMED pairs as a sanity
check; if that rate is not high, treat the audit as still unreliable for
this dataset rather than trusting the leakage numbers derived from it.

Only CONFIRMED pairs are used for the leakage question (whether a
near-duplicate pair straddles two different folds under the exact
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) split used in
step4_kaggle4class.py / step6_backbone_sweep.py), and only CONFIRMED pairs
are written to the CSV, so the output stays small (the Stage-1-only,
64-bit-hash version of this script produced an 867 MB CSV of mostly
false positives; this version should produce at most a few hundred rows).

Usage: edit DATA_DIR below if your Kaggle input path differs, then run the
whole script. Download the images/eda_a_*.png files and
results/eda_a_summary.json / eda_a_near_duplicates.csv for the paper.
"""

import os
import glob
import json
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
N_FOLDS = 5
SEED = 42

HASH_SIZE = 16                  # 16x16 -> 256-bit average hash (Stage 1 filter)
HASH_WORDS = (HASH_SIZE * HASH_SIZE) // 64   # 256 bits -> 4 uint64 words
CANDIDATE_THRESHOLD = 40        # Stage 1: Hamming distance <= this / 256 bits
                                 # (~15.6%) counts as a candidate worth verifying.
                                 # Deliberately loose -- Stage 2 does the real
                                 # filtering, this just bounds the search space.

VERIFY_SIZE = 64                # thumbnail resolution used for Stage 2 pixel
                                 # comparison (0-255 grayscale)
CONFIRM_RMSE = 8.0              # Stage 2: root-mean-square pixel difference
                                 # (0-255 scale) at or below this is a
                                 # CONFIRMED near-duplicate. 8.0 corresponds to
                                 # images that are visually indistinguishable;
                                 # tighten/loosen and re-examine the same-class
                                 # rate printed at the end if needed.

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)

COLOR_A = "#0072B2"
CLASS_COLORS = {"glioma": "#0072B2", "meningioma": "#E69F00",
                 "notumor": "#009E73", "pituitary": "#D55E00"}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
})


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
    """Same order as step4_kaggle4class.py's list_image_paths -- this exact
    order is what determines which StratifiedKFold fold each image falls in,
    so it must match bit-for-bit for the leakage audit below to be valid."""
    paths, labels = [], []
    for split_dir in ["Training", "Testing"]:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, split_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*")))
            paths.extend(files)
            labels.extend([class_idx] * len(files))
    return paths, np.array(labels)


# ---------------------------------------------------------------------------
# Stage 1: coarse candidate hash (256-bit average hash, packed as 4 uint64 words)
# ---------------------------------------------------------------------------
def average_hash_words(pil_img, hash_size=HASH_SIZE):
    gray = pil_img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()  # length hash_size*hash_size
    n_words = (len(bits) + 63) // 64
    words = np.zeros(n_words, dtype=np.uint64)
    for w in range(n_words):
        chunk = bits[w * 64:(w + 1) * 64]
        val = 0
        for b in chunk:
            val = (val << 1) | int(b)
        words[w] = np.uint64(val)
    return words


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount_words(arr_u64):
    """arr_u64: (..., HASH_WORDS) uint64 -> total popcount across all words."""
    flat = arr_u64.reshape(-1)
    b = flat.reshape(-1, 1).view(np.uint8)
    per_word_bits = _POPCOUNT_TABLE[b].sum(axis=1)
    return per_word_bits.reshape(arr_u64.shape).sum(axis=-1)


def find_candidate_pairs(hashes, threshold=CANDIDATE_THRESHOLD, row_chunk=200):
    """hashes: (n, HASH_WORDS) uint64. Returns list of (i, j) with i < j and
    total Hamming distance <= threshold. This is Stage 1 only -- a cheap,
    intentionally loose filter; Stage 2 (verify_candidates) does the real
    near-duplicate decision."""
    n = len(hashes)
    hashes = hashes.astype(np.uint64)
    pairs = []
    for start in range(0, n, row_chunk):
        end = min(start + row_chunk, n)
        chunk = hashes[start:end]                          # (c, W)
        xor = chunk[:, None, :] ^ hashes[None, :, :]        # (c, n, W)
        dist = popcount_words(xor)                          # (c, n)
        for local_i, global_i in enumerate(range(start, end)):
            row = dist[local_i]
            close = np.where(row <= threshold)[0]
            for j in close:
                if j > global_i:
                    pairs.append((global_i, int(j)))
        if (start // row_chunk) % 5 == 0:
            print(f"  Stage 1 (candidate) scan: {end}/{n} images compared, "
                  f"{len(pairs)} candidates so far")
    return pairs


# ---------------------------------------------------------------------------
# Stage 2: pixel-level verification
# ---------------------------------------------------------------------------
def verify_candidates(candidate_pairs, verify_thumbs, rmse_threshold=CONFIRM_RMSE):
    """verify_thumbs: (n, VERIFY_SIZE, VERIFY_SIZE) float32 grayscale, 0-255.
    Returns list of (i, j, rmse) for pairs at or below rmse_threshold."""
    confirmed = []
    for k, (i, j) in enumerate(candidate_pairs):
        diff = verify_thumbs[i] - verify_thumbs[j]
        rmse = float(np.sqrt(np.mean(diff * diff)))
        if rmse <= rmse_threshold:
            confirmed.append((i, j, rmse))
        if (k + 1) % 5000 == 0:
            print(f"  Stage 2 (verify) scan: {k + 1}/{len(candidate_pairs)} "
                  f"candidates checked, {len(confirmed)} confirmed so far")
    return confirmed


def main():
    print("=" * 70)
    print("Dataset EDA")
    print("=" * 70)
    paths, y = list_image_paths(DATA_DIR)
    n = len(paths)
    print(f"Found {n} images across {len(CLASS_NAMES)} classes")

    # --- class distribution ---
    counts = {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    print("Class counts:", counts)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(counts.keys(), counts.values(), color=[CLASS_COLORS[c] for c in counts])
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 20, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/eda_a_class_distribution.png", dpi=200)
    plt.close(fig)

    # --- image dimensions + intensity stats + hashing (single pass over all images) ---
    widths, heights, mean_intensity = [], [], []
    hashes = np.zeros((n, HASH_WORDS), dtype=np.uint64)
    verify_thumbs = np.zeros((n, VERIFY_SIZE, VERIFY_SIZE), dtype=np.float32)
    imgs_by_class_sample = {c: [] for c in CLASS_NAMES}
    sum_img_by_class = {c: np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64) for c in CLASS_NAMES}
    count_by_class = {c: 0 for c in CLASS_NAMES}

    for idx, p in enumerate(paths):
        with Image.open(p) as im:
            widths.append(im.width)
            heights.append(im.height)
            gray = im.convert("L")
            arr_native = np.asarray(gray, dtype=np.float32)
            mean_intensity.append(float(arr_native.mean()))
            hashes[idx] = average_hash_words(im)
            verify_thumbs[idx] = np.asarray(
                gray.resize((VERIFY_SIZE, VERIFY_SIZE), Image.BILINEAR), dtype=np.float32)

            cname = CLASS_NAMES[y[idx]]
            if len(imgs_by_class_sample[cname]) < 5:
                imgs_by_class_sample[cname].append(np.asarray(im.convert("RGB").resize((160, 160))))
            resized = np.asarray(gray.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float64)
            sum_img_by_class[cname] += resized
            count_by_class[cname] += 1
        if (idx + 1) % 1000 == 0 or (idx + 1) == n:
            print(f"  scanned {idx + 1}/{n}")

    widths, heights = np.array(widths), np.array(heights)
    mean_intensity = np.array(mean_intensity)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(widths, bins=30, alpha=0.7, label="width", color=COLOR_A)
    ax.hist(heights, bins=30, alpha=0.7, label="height", color="#D55E00")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Number of images")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/eda_a_image_dimensions.png", dpi=200)
    plt.close(fig)
    print(f"Width range: [{widths.min()}, {widths.max()}], "
          f"Height range: [{heights.min()}, {heights.max()}]")

    # --- sample grid, 5 per class ---
    fig, axes = plt.subplots(len(CLASS_NAMES), 5, figsize=(10, 8))
    for r, cname in enumerate(CLASS_NAMES):
        for c in range(5):
            ax = axes[r, c]
            if c < len(imgs_by_class_sample[cname]):
                ax.imshow(imgs_by_class_sample[cname][c])
            ax.axis("off")
        axes[r, 0].text(-0.15, 0.5, cname, transform=axes[r, 0].transAxes,
                         rotation=90, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig("images/eda_a_sample_grid.png", dpi=200)
    plt.close(fig)

    # --- pixel intensity boxplot by class ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    data_per_class = [mean_intensity[y == i] for i in range(len(CLASS_NAMES))]
    bp = ax.boxplot(data_per_class, labels=CLASS_NAMES, patch_artist=True)
    for patch, cname in zip(bp["boxes"], CLASS_NAMES):
        patch.set_facecolor(CLASS_COLORS[cname])
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean pixel intensity (0-255)")
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/eda_a_intensity_boxplot.png", dpi=200)
    plt.close(fig)

    # --- mean ("average") image per class ---
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(12, 3.2))
    for ax, cname in zip(axes, CLASS_NAMES):
        mean_img = sum_img_by_class[cname] / max(count_by_class[cname], 1)
        ax.imshow(mean_img, cmap="gray")
        ax.set_title(cname, fontsize=10)
        ax.axis("off")
    fig.suptitle("Mean image per class (pixel-wise average, aligned to slice center)")
    fig.tight_layout()
    fig.savefig("images/eda_a_mean_images.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Near-duplicate / leakage audit -- two-stage (see module docstring)
    # -----------------------------------------------------------------
    print("\nStage 1: generating near-duplicate candidates via 256-bit hash...")
    candidates = find_candidate_pairs(hashes)
    print(f"Stage 1 done: {len(candidates)} candidate pairs "
          f"(Hamming distance <= {CANDIDATE_THRESHOLD}/256 bits)")

    print("\nStage 2: verifying candidates by pixel-level RMSE...")
    confirmed = verify_candidates(candidates, verify_thumbs)
    print(f"Stage 2 done: {len(confirmed)} CONFIRMED near-duplicate pairs "
          f"(RMSE <= {CONFIRM_RMSE} on a 0-255 scale)")

    same_class = sum(1 for i, j, r in confirmed if y[i] == y[j])
    cross_class = len(confirmed) - same_class
    same_class_rate = 100 * same_class / len(confirmed) if confirmed else float("nan")
    print(f"Same-class rate among CONFIRMED pairs: {same_class_rate:.1f}% "
          f"({same_class}/{len(confirmed)}) "
          f"-- SANITY CHECK: this should be well above the 25% chance rate "
          f"for 4 balanced classes; if it is not, do not trust the leakage "
          f"numbers below.")

    # Replicate the EXACT fold assignment used in step4_kaggle4class.py / step6
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of = np.full(n, -1, dtype=int)
    dummy_X = np.zeros((n, 1))
    for fold_idx, (_, test_idx) in enumerate(skf.split(dummy_X, y)):
        fold_of[test_idx] = fold_idx

    leaking_pairs = [(i, j, r) for i, j, r in confirmed if fold_of[i] != fold_of[j]]
    affected_images = set()
    for i, j, r in leaking_pairs:
        affected_images.add(i)
        affected_images.add(j)

    print(f"CONFIRMED pairs split across DIFFERENT folds under the actual CV protocol: "
          f"{len(leaking_pairs)} / {len(confirmed)}")
    print(f"Unique images with >=1 CONFIRMED near-duplicate in a different fold: "
          f"{len(affected_images)} / {n} ({100*len(affected_images)/n:.2f}%)" if confirmed else
          "No confirmed near-duplicate pairs found.")

    # save CSV of only the CONFIRMED pairs (should be small)
    with open("results/eda_a_near_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_i", "path_j", "class_i", "class_j", "rmse",
                    "fold_i", "fold_j", "different_fold"])
        for i, j, r in confirmed:
            w.writerow([paths[i], paths[j], CLASS_NAMES[y[i]], CLASS_NAMES[y[j]],
                        round(r, 3), int(fold_of[i]), int(fold_of[j]), fold_of[i] != fold_of[j]])

    # visualize a handful of example confirmed near-duplicate pairs (closest first)
    confirmed_sorted = sorted(confirmed, key=lambda t: t[2])
    n_examples = min(4, len(confirmed_sorted))
    if n_examples > 0:
        fig, axes = plt.subplots(n_examples, 2, figsize=(5, 2.4 * n_examples))
        if n_examples == 1:
            axes = axes[None, :]
        for row, (i, j, r) in enumerate(confirmed_sorted[:n_examples]):
            for col, idx in enumerate((i, j)):
                with Image.open(paths[idx]) as im:
                    axes[row, col].imshow(im.convert("L"), cmap="gray")
                axes[row, col].axis("off")
                axes[row, col].set_title(
                    f"{CLASS_NAMES[y[idx]]}, fold {fold_of[idx]}", fontsize=8)
            axes[row, 0].set_ylabel(f"RMSE={r:.1f}", fontsize=9)
        fig.suptitle("Closest CONFIRMED near-duplicate pairs (Stage 2 pixel RMSE)")
        fig.tight_layout()
        fig.savefig("images/eda_a_near_duplicate_examples.png", dpi=200)
        plt.close(fig)
    else:
        print("No confirmed pairs to visualize -- skipping eda_a_near_duplicate_examples.png")

    summary = {
        "n_images": n,
        "class_counts": counts,
        "image_width_range": [int(widths.min()), int(widths.max())],
        "image_height_range": [int(heights.min()), int(heights.max())],
        "stage1_hash_bits": HASH_SIZE * HASH_SIZE,
        "stage1_candidate_threshold": CANDIDATE_THRESHOLD,
        "stage1_candidate_pairs": len(candidates),
        "stage2_rmse_threshold": CONFIRM_RMSE,
        "confirmed_near_duplicate_pairs": len(confirmed),
        "confirmed_same_class": same_class,
        "confirmed_cross_class": cross_class,
        "confirmed_same_class_rate_pct": round(same_class_rate, 2) if confirmed else None,
        "confirmed_pairs_split_across_folds": len(leaking_pairs),
        "images_with_cross_fold_confirmed_duplicate": len(affected_images),
        "images_with_cross_fold_confirmed_duplicate_pct":
            round(100 * len(affected_images) / n, 3) if confirmed else 0.0,
    }
    with open("results/eda_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/eda_a_summary.json:", json.dumps(summary, indent=2))
    print("\nDone. Download images/eda_a_*.png, results/eda_a_summary.json, "
          "and results/eda_a_near_duplicates.csv for the paper.")
    return summary


if __name__ == "__main__":
    main()
