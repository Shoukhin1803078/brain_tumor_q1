# Brain Tumor MRI Classification — Hybrid CNN–Ensemble Framework

Research code and manuscript for multi-class brain tumor classification from MRI
slices. The study's angle is **evaluation reliability**, not chasing a single
accuracy number: it reports cross-validated results with error bars across
multiple backbones and validates the gains with a statistical test, rather than
quoting one favorable train/test split for one network.

**Paper title:** *A Hybrid CNN–Ensemble Framework with Multi-Backbone and
Statistically Validated Evaluation for Multi-Class Brain Tumor Classification*
(Md Al Amin Tokder Shoukhin, Tasmia Jannat, Md. Ali Hossain, Nazmul Haque —
Dept. of CSE, Rajshahi University of Engineering and Technology)

## Headline results

4-class, 7,200-image benchmark (glioma / meningioma / pituitary / no-tumor),
stratified 5-fold CV, identical protocol for every backbone:

| Backbone | CNN (softmax) | CNN + TTA | Embedding ensemble |
|---|---|---|---|
| EfficientNetB0 | 96.01% ± 0.38% | 97.75% ± 0.25% | 97.26% ± 0.27% |
| ResNet50 | 97.18% ± 0.56% | **98.38% ± 0.23%** | 97.64% ± 0.41% |
| DenseNet121 | **97.69% ± 0.53%** | 98.32% ± 0.22% | **98.00% ± 0.34%** |
| MobileNetV2 | 95.04% ± 0.72% | 97.96% ± 0.32% | 96.25% ± 0.23% |

- TTA beats the plain softmax for **every** backbone (exact one-sided Wilcoxon
  signed-rank, `p = 0.031` each).
- ResNet50 and DenseNet121 beat EfficientNetB0 in **all five folds**
  (`p = 0.031` each); MobileNetV2's gain is not significant (`p = 0.156`).
- MobileNetV2 is the cheapest on every axis (lowest inference latency, ~14–31%
  faster to train) while still beating the original EfficientNetB0 on accuracy.

## Pipeline

1. **Preprocessing** — decode, resize to 224×224, normalize with each backbone's
   own ImageNet `preprocess_input`. Class-weighted loss via
   `sklearn.compute_class_weight` (classes are actually balanced here, so this
   is a no-op in practice).
2. **Backbone** — ImageNet-pretrained EfficientNetB0 / ResNet50 / DenseNet121 /
   MobileNetV2, with a shared head: `GAP → BatchNorm → Dropout(0.4) →
   Dense(128) [embedding] → BatchNorm → Dropout(0.3) → softmax`.
3. **Two-stage fine-tuning** — Stage 1 freezes the backbone and trains the head
   (Adam 1e-3, 8 epochs); Stage 2 unfreezes the top 80% and fine-tunes
   (Adam 1e-5, ≤35 epochs, early stopping + reduce-on-plateau).
4. **Dual inference** — (a) **TTA**: average softmax over the clean pass plus 5
   augmented views; (b) **embedding ensemble**: soft-voting Random Forest +
   XGBoost + RBF SVM trained on the 128-d penultimate embedding.
5. **Evaluation** — stratified 5-fold CV (every image tested once), mean ± std,
   exact Wilcoxon signed-rank test on paired fold scores, and Grad-CAM.

## Repository layout

### Experiment scripts

| Script | Purpose |
|---|---|
| `step1_baseline.py` | EfficientNetB0 baseline on the **figshare `.mat`** dataset; patient-level split (`cjdata.PID`), **mask-guided ROI cropping** + TTA |
| `step3_cv_ensemble.py` | Same figshare dataset, **patient-level 5-fold GroupKFold CV** + CNN-embedding ensemble |
| `step4_kaggle4class.py` | Main 4-class Kaggle benchmark, **image-level stratified 5-fold CV** + TTA + ensemble |
| `step5_gradcam.py` | Trains one model and produces per-class Grad-CAM (original / heatmap / overlay) |
| `step6_backbone_sweep.py` | Generalizes step4 across 4 backbones with timing instrumentation; appends per-fold rows to `backbone_sweep.jsonl` |
| `step7_backbone_figures.py` | Parses `step_6_backbone_sweep_logs.log` → 4 comparison figures (accuracy bar, fold variance, latency trade-off, training time) |
| `step8_eda.py` | EDA (class balance, dimensions, intensity, sample grid, mean images) + two-stage **near-duplicate/leakage audit** |

> **Two datasets are used.** `step1`/`step3` use the figshare `.mat` release
> (per-slice masks + patient IDs, enabling leakage-free patient-level splits).
> `step4`–`step8` use the 4-class Kaggle JPG release
> (`masoudnickparvar/brain-tumor-mri-dataset`), which has no masks and no
> patient IDs — so CV there is image-level only. The paper is built on the
> latter; the former motivated the mask-guided approach.

### Figure/diagram generators

| Script | Output |
|---|---|
| `plot_curves.py` | `images/accuracy_vs_epoch.png`, `loss_vs_epoch.png`, `confusion_matrix.png` (from the transcribed Fold-1 log) |
| `make_workflow_diagram.py` | `images/Brain tumor classification.drawio.png` (pipeline diagram, pure matplotlib) |
| `make_dataset_sample_grid.py` | `images/dataset_image.PNG` (one sample per class; run on Kaggle) |

All scripts are written to run on **Kaggle with a GPU** (paths point at
`/kaggle/input/...`), except the plotting/diagram scripts which run locally on
CPU.

### Manuscript

| File | Notes |
|---|---|
| `brain_tumor_preprint_verified.tex` | **Current manuscript** (multi-backbone + statistics); uses `reference_verified.bib` |
| `brain_tumor_preprint.tex` | Earlier preprint draft; uses `reference.bib` |
| `brain_tumor.tex` | Early version (single efficientnet backbone framing); uses `reference.bib` |
| `reference_verified.bib` / `reference.bib` | BibTeX (the `_verified` set has author/accuracy corrections) |
| `feedback.md` | Reviewer feedback that drove the multi-backbone + stats revision |

Build: `pdflatex brain_tumor_preprint_verified && bibtex brain_tumor_preprint_verified && pdflatex ...` (twice). PDFs: `Final_bloodstein.pdf`, `target.pdf` (conference templates).

### Notebook & artifacts

- `updated-model-for-brain-tumor-classsification (2).ipynb` — Kaggle notebook
  with the figshare `step1`–`step3` pipeline inline.
- `step_6_backbone_sweep_logs.log` — raw console log for the ResNet50 /
  DenseNet121 / MobileNetV2 sweeps (input to `step7`).
- `images/` — all paper figures (accuracy/loss curves, confusion matrix,
  backbone comparisons, Grad-CAM panels, EDA plots, workflow diagram).
- `eda/images/`, `eda/results/eda_a_summary.json` — EDA outputs.

## Reproducing

```bash
# 1. Multi-backbone sweep — set BACKBONE in step6, run one backbone per Kaggle
#    session (a full 5-fold run is ~1.5–2.5 GPU-hours on a T4):
python3 step6_backbone_sweep.py        # -> backbone_sweep.jsonl + console log

# 2. Figures from the log (local, CPU):
python3 step7_backbone_figures.py      # -> images/backbone_*.png

# 3. EDA + near-duplicate audit (local or Kaggle CPU):
python3 step8_eda.py                   # -> images/eda_a_*.png, results/eda_a_summary.json

# 4. Interpretability:
python3 step5_gradcam.py               # -> gradcam/*.png
```

Set `DATA_DIR` at the top of each script if your Kaggle input path differs.

## Known limitations (and why they're documented)

- **No patient IDs** in the 4-class dataset → image-level CV means two slices
  from the same scan could straddle folds (subject-level leakage). An early
  64-bit average-hash audit failed its sanity check (near-chance same-class
  rate) and was discarded; `step8_eda.py` implements the corrected two-stage
  (coarse hash filter → pixel-RMSE verification) audit.
- **Dataset snapshot differs** from the 7,023-image release some papers cite
  (this one is 7,200, exactly balanced), so cross-paper comparisons share the
  dataset identifier, not byte-identical images.
- **Mixed acquisition planes / one CT slice** present within classes; no-tumor
  slices are systematically brighter.
- **n = 5 folds** → smallest attainable one-sided Wilcoxon `p = 0.031`.
- **GPU non-determinism** adds run-to-run variance beyond the fold std.
- 2D slice classification only; no external validation; Grad-CAM done for
  EfficientNetB0 only.

See `feedback.md` for the external review that shaped the final manuscript.
