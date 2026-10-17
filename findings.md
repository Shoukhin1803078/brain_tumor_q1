# Findings

Running record of verified experimental results, their provenance, and open
issues. Append to this rather than rewriting it; keep the provenance line on
every claim so anything here can be re-checked later.

Last updated: 2026-10-16 (RRF-Net run #1 complete, 5/5 folds)

---

## 1. Multi-backbone sweep (step6) — the paper's current results

4-class Kaggle benchmark (masoudnickparvar snapshot, 7,200 images, exactly
1,800/class), stratified 5-fold CV at the **image level**, identical protocol
per backbone.

| Backbone | CNN (softmax) | CNN + TTA | Embedding ensemble |
|---|---|---|---|
| EfficientNetB0 | 96.01% ± 0.38% | 97.75% ± 0.25% | 97.26% ± 0.27% |
| ResNet50 | 97.18% ± 0.56% | **98.38% ± 0.23%** | 97.64% ± 0.41% |
| DenseNet121 | **97.69% ± 0.53%** | 98.32% ± 0.22% | **98.00% ± 0.34%** |
| MobileNetV2 | 95.04% ± 0.72% | 97.96% ± 0.32% | 96.25% ± 0.23% |

*Provenance:* every per-fold number re-read from `step_6_backbone_sweep_logs.log`
(all 20 fold rows) and confirmed to match the README table exactly.

### Statistical claims — independently recomputed, all hold

Exact one-sided Wilcoxon signed-rank over the 5 paired folds:

- **TTA > plain softmax** for all four backbones, `p = 0.031` each.
  Mean gains: MobileNetV2 +2.92, EfficientNetB0 +1.74, ResNet50 +1.19,
  DenseNet121 +0.62.
- **vs EfficientNetB0+TTA:** ResNet50 wins 5/5 folds (`p = 0.031`),
  DenseNet121 wins 5/5 (`p = 0.031`), MobileNetV2 wins 3/5 (`p = 0.156`, n.s.).
- **Embedding ensemble > plain CNN:** significant for EfficientNetB0, ResNet50,
  MobileNetV2 (`p = 0.031`); **not** for DenseNet121 (`p = 0.1875`).

*Provenance:* recomputed from the fold values with `scipy.stats.wilcoxon(...,
alternative='greater', method='exact')` on 2026-10-16. n=5 means the smallest
attainable one-sided p is 0.031.

> **Caveat that governs all of the above:** these use image-level
> `StratifiedKFold`. Near-duplicate slices can straddle folds. See §4.

---

## 2. RRF-Net run #1 (`rrf_full`, image split) — INVALID as a result

First GPU run of `step9_rrf_fusion.py`, now complete (5/5 folds).
**Do not cite the headline number as an RRF-Net result.** Retained as the
"no ViT fine-tuning" ablation row.

| Fold | CNN br. | ViT br. | Fused | Fused+TTA | ECE | NLL | route_acc | β̄ | disagree |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 93.89 | 90.35 | 95.28 | 97.57 | 0.0523 | 0.1667 | 0.704 | 0.757 | 125/1440 |
| 2 | 94.17 | 90.69 | 94.58 | 97.78 | 0.0422 | 0.1623 | 0.698 | 0.856 | 126/1440 |
| 3 | 91.46 | 89.10 | 92.43 | 97.01 | 0.0351 | 0.2089 | 0.599 | 0.749 | 162/1440 |
| 4 | 88.47 | 89.79 | 93.61 | 96.94 | 0.0497 | 0.1956 | 0.528 | 0.579 | 161/1440 |
| 5 | 91.46 | 91.67 | 94.93 | 96.11 | 0.0556 | 0.1804 | 0.524 | 0.624 | 145/1440 |
| **mean** | 91.89 ± 2.06 | 90.32 ± 0.86 | 94.17 ± 1.03 | **97.08 ± 0.58** | 0.047 | 0.183 | 0.611 | 0.713 | — |

Cost: ~29–36 min/fold on one T4 (1,535–2,161 s train), 21 ms/img single pass,
~104 ms/img with TTA. Total run ~2.7 h.

*Provenance:* `rrf_fusion.log`, 372 lines, all 5 folds complete; means
cross-checked against the script's own SUMMARY block.

### Why it is invalid: the ViT branch never trained

Every fold logged `[vit] 1/4 layers trainable`. Cause: keras-hub returns the
whole ViT as **one nested sub-model**, so the wrapper has only four top-level
layers — `['vit_raw_input', 'vit_rescale', 'vi_t_backbone', 'vit_drop_cls']`.
The original `set_vit_trainable()` iterated only that top level and never
reached any transformer block. None of those four names contains a digit, so
the fallback branch unfroze `layers[3:]` — the class-token `Lambda`, which has
zero weights.

**Net effect: 0 ViT parameters were trainable in stage 2.** The ViT was a
frozen stock ImageNet extractor for the entire run.

*Provenance:* reproduced against a stub with that exact nesting —
old rule → 0 trainable params; fixed rule → blocks 8–11, 11,817,984 of
36,044,544 params (32.8%).

Corroborating evidence in the data itself:

- ViT branch std across folds is **0.86**, versus **2.06** for the CNN branch
  that did train — the flatness signature of a fixed extractor.
- TTA adds **+2.92 pp** over the base fused prediction (94.17 → 97.08,
  `p = 0.031`), where step6's backbones gained 0.6–2.9 pp. A TTA rescue at the
  top of that range indicates an underfit base model.

### How it compares to step6 (same folds, same protocol)

Exact one-sided Wilcoxon, backbone TTA vs RRF fused+TTA over the 5 paired folds:

| Backbone | mean | vs RRF | folds won | p |
|---|---|---|---|---|
| ResNet50 | 98.37 | +1.29 | 5/5 | **0.031** |
| DenseNet121 | 98.32 | +1.24 | 5/5 | **0.031** |
| MobileNetV2 | 97.96 | +0.88 | 5/5 | **0.031** |
| EfficientNetB0 | 97.75 | +0.67 | 4/5 | 0.094 (n.s.) |

RRF-Net run #1 loses significantly to three of the four CNN baselines. Again:
this is the score of a model with a dead branch, not of RRF-Net.

### What this run nonetheless establishes

**The fusion mechanism adds real value over either branch alone.** The fused
prediction beats the better of the two branches in **5/5 folds**
(`p = 0.031`), by +1.39, +0.41, +0.97, +3.82, +3.26 pp. So the architecture is
combining information rather than defaulting to whichever branch is stronger —
and it does that even when one branch is frozen.

**The routing mechanism does real work.** `route_acc` — how often the gate
picks the correct branch on images where the two branches disagree — averages
**0.611**, significantly above the 0.5 chance floor (`p = 0.031`), *while one
branch was crippled*. β̄ = 0.713 shows the router independently learned to
distrust the handicapped ViT. This is the strongest available evidence that N1
(entropy-conditioned routing) and N3 (supervised routing loss) are not
decorative.

**Watch item — route_acc decays monotonically across folds:**
0.704 → 0.698 → 0.599 → 0.528 → 0.524. Folds 4 and 5 are at chance. Note the
fold order is not a training order (each fold is an independent model), so this
is not drift — it tracks something about the folds themselves. Candidates:
insufficient `W_ROUTE`, or the router latching onto "CNN is usually better"
early and losing discrimination as the branches converge in quality. Fold 5 is
the one fold where the ViT branch beat the CNN (91.67 vs 91.46) and it has the
lowest route_acc and the worst Fused+TTA (96.11) — consistent with a router
mis-calibrated toward the CNN. Re-check after the fixed run; if route_acc stays
flat near 0.5 once the ViT is genuinely trained, raise `W_ROUTE` from 0.2.

### Fix applied (`RUN_TAG = "v2_vitfix"`)

- `_iter_sublayers()` walks nested sub-models recursively.
- Unfreezing counts only layers that own weights, and **raises** if the
  unfreeze would train 0 parameters.
- Nested layer names dumped once per run via `debug_vit_structure()`.
- `RUN_TAG` folded into `variant_tag()`, because `completed_folds()` keys on
  `(split_mode, variant)` — without it a fixed re-run would have found five
  `rrf_full`/`image` rows and skipped every fold as "already done".

Regression-tested: all five ablation variants build and train, and the loss
trajectories are **bit-identical** to the pre-patch run (`no cross-attention`:
1.7130 → 1.2649, route_loss 0.1218, acc 0.875 both times), confirming the patch
touched only freezing and tagging, not the model graph.

---

## 3. Competing work on the *same* dataset

Both papers in `new_vit_cnn_methodology/` use the masoudnickparvar 7,023
release — the same Kaggle identifier this project benchmarks.

### Mahmud et al. 2026, *Informatics and Health* — the real bar

EfficientNetV2-B0 + ViT-B/16, both projected to a shared 512-d space,
bidirectional cross-modal attention, then a global adaptive gate
`α = σ(W_g[C;V] + b_g)`, `F = α⊙C + (1−α)⊙V`. Focal loss (γ=2) + label
smoothing (ε=0.1). MC Dropout, temperature scaling (T fit on validation logits
only), ECE/NLL, coverage-risk/AURC. 10-fold stratified **group** CV over
perceptual-similarity clusters. McNemar + Wilcoxon. BraTS 2021 + LODO.

- **98.10%** test (95% CI 97.37–98.76), macro-AUC 0.9988
- 98.19% ± 0.16% F1 over 10 group-wise folds
- MC Dropout cut ECE 0.031 → 0.017
- 93.65M params, 357 MB, 17.23 GFLOPs, 17.85 ms/img GPU

Ablation: EffNetV2-B0 96.78 / ViT 97.46 / no cross-attn 97.62 / no gate 97.48 /
neither 97.21 / no MC dropout 97.96 / **full 98.10**. The entire fusion stack is
worth **+0.64 over ViT alone**.

**Leakage data point (important for §4):** they removed **2,134
near-duplicates** and found **4,825 similarity groups among 7,023 images**.
Roughly a third of this dataset is near-duplicated.

Defects noted on read: Tables 3 and 4 contradict each other on the same splits
(Table 4's columns sum to 4,983/1,491 while its own totals row says
5,107/1,367); Table 4's columns are headed "Patients" in a paper whose central
caveat is that no patient IDs exist; unresolved referee language survives in
the published text ("superiority … should not be claimed until Table 6 has been
recalculated"); MC-dropout uncertainty values (0.001107 / 0.001432) carry no
named measure or units.

### Jayaraman et al. 2025, *Scientific Reports* (CAFNet) — weak

MobileNetV2 + ViT, one-way cross-attention (`Q` from ViT, `K,V` from CNN),
`H = Concat(Q, Z)·W_o`. 80/10/10 stratified random split, **image-level, no
deduplication at all**. **96.41%** test.

Defects: no training hyperparameters reported anywhere (no epochs, optimizer,
LR, batch size, ViT variant, or pretrained/scratch) — not reproducible from the
text. Transfer-learning baselines badly under-trained (ResNet50 68.73%, VGG19
77.65%) against our own ResNet50 at 97.18% on the same data, which inflates
CAFNet's apparent margin. Abstract and Table 5 say 96.41%; the Conclusion says
96.81%.

### Comparability warning

**Our 98.38% is not comparable to their 98.10%.** They split at the
similarity-cluster level; step6 splits at the image level. Any claim that we
beat them would be wrong and a reviewer who has read Mahmud et al. will catch
it immediately.

---

## 4. The leakage gap — the open contribution

Neither paper reports the **delta** between image-level and group-level
splitting on this benchmark. CAFNet never grouped; Mahmud et al. grouped but
never published the ungrouped counterfactual. We already hold half the
experiment: 4 backbones × 5 folds × 3 inference modes under image-level CV,
fully logged.

Re-running that grid under similarity-grouped CV yields, with error bars, *how
many points of published accuracy on this benchmark are near-duplicate
leakage*. That is the headline.

### Blocker: the committed EDA summary is from the discarded audit

`eda/results/eda_a_summary.json` carries keys the current `step8_eda.py` no
longer writes — `near_duplicate_threshold_bits`, `near_duplicate_pairs_total`,
`images_with_cross_fold_near_duplicate_pct: 98.569`. The current script emits
`stage1_*` / `confirmed_*` keys.

So **the corrected two-stage audit has never been run.** The artifact in the
repo is from the v1 average-hash scan whose same-class rate was ~30% — near the
25% chance base rate for four balanced classes — which is exactly why it was
discarded. `eda/images/eda_a_near_duplicate_examples.png` is from that same
stale run.

This matters: 98.57% cross-fold near-duplication is a catastrophic-sounding
number sitting unlabelled in the repo, derived from an audit the manuscript
itself declares untrustworthy. Mahmud et al.'s independent figure (~1/3 of the
dataset near-duplicated) suggests the truth is serious but far below 98.57%.

**Next action:** run `step8_eda.py` (CPU-only, no GPU needed) to get the real
confirmed-pair count, then `build_groups_from_pairs()` → `SPLIT_MODE="group"`.

---

## 5. Other open items

- **Component ablation** was `feedback.md`'s top-priority ask and is still
  missing for the step6 pipeline: no isolation of class weighting, augmentation
  policy, or RF / XGBoost / SVM individually. `step9`'s four ablation switches
  cover the fusion model only.
- **Raw per-fold data for step6 lives only in the console log.**
  `backbone_sweep.jsonl` is gitignored, and `step7_backbone_figures.py`
  regenerates every figure by regex-matching that log's exact print format — a
  changed f-string in step6 silently breaks figure regeneration.
- `step6`'s `BACKBONE` is hardcoded (currently `"resnet50"`), hand-edited per
  session.
- `step1` uses `TTA_ROUNDS = 8` while every later script uses 5. Harmless —
  step1's numbers are not in the paper — but it means step1 is not quite the
  identical-protocol baseline its name implies.
