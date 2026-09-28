"""
Run this on Kaggle (same notebook/session where the brain-tumor-mri-dataset
is already mounted) to build the sample-slice grid figure for the paper's
Dataset section (images/dataset_image.PNG).

Grabs one representative Training image per class and arranges them in a
1x4 grid with class labels. Download the resulting PNG and place it at
images/dataset_image.PNG in the paper repo.
"""

import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
OUT_PATH = "/kaggle/working/dataset_image.PNG"

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, class_name in zip(axes, CLASS_NAMES):
    class_dir = os.path.join(DATA_DIR, "Training", class_name)
    sample_path = sorted(glob.glob(os.path.join(class_dir, "*")))[0]
    img = mpimg.imread(sample_path)
    ax.imshow(img, cmap="gray")
    ax.set_title(class_name.capitalize(), fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_PATH} -- download this and place it at images/dataset_image.PNG")
