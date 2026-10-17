# Findings

Running record of verified experimental results, their provenance, and open
issues. Append to this rather than rewriting it; keep the provenance line on
every claim so anything here can be re-checked later.

Last updated: 2026-10-17 (RRF-Net run #3 fold 1 — v3 fix verified on the
real keras_hub ViT; fusion mechanism works; 5-fold run is the next step)

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

### Fix applied (`RUN_TAG = "v2_vitfix"`) — partial; superseded by §2b

> This fix made the ViT train, but not the configured 4 blocks. See §2b.

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

## 2b. RRF-Net run #2 (`rrf_full@v2_vitfix`, fold 1 only) — ViT alive, but the wrong ViT

Provenance: `rrf_fusion_v2.log` (102 lines, fold 1 of 5), Kaggle T4.
Config as §2 plus `RUN_TAG = "v2_vitfix"`, `FOLDS_TO_RUN = [1]`.

| metric | run #1 fold 1 (ViT frozen) | run #2 fold 1 | step6 best (5-fold mean) |
|---|---|---|---|
| CNN branch | 93.89 | **92.22** | — |
| ViT branch | 90.35 | **98.40** | — |
| Fused | 95.28 | **98.40** | — |
| Fused + TTA | 97.57 | **98.68** | ResNet50 98.37 |
| ECE | 0.0523 | **0.0205** | — |
| NLL | 0.1667 | **0.0652** | — |
| mean entropy | — | 0.0910 | — |
| β̄ (gate weight on CNN) | 0.757 | **0.107** | — |
| route_acc | 0.704 | 0.9495 | — |
| branches disagree | 125/1440 | 99/1440 | — |
| train time | 2130 s | **4754 s** (stage1 464 + stage2 4290) | — |

The §2 diagnosis is confirmed outright: unfreezing the ViT moved its branch
from 90.35% to 98.40% on the identical fold. The frozen-ViT reading of run #1
was correct.

### But the fine-tune was not the configured one

```
[vit] 3 weighted layers nested in 'vit_b16_kerashub':
    vi_t_backbone / vit_patching_and_embedding / vit_encoder
[vit] strategy=tail-fraction  1/3 weighted layers trainable  (85,056,000 params)
```

`VIT_FINE_TUNE_BLOCKS = 4` was set; **all 12 blocks trained.** keras_hub's
`ViTEncoder` is a plain `Layer` that keeps its blocks in a list attribute and
exposes no `.layers`, so the v2 recursive walk stopped at the encoder and saw
three containers, not 38 blocks. `tail-fraction` then unfroze "the last 1 of 3"
— the entire 85M-parameter encoder. The diagnostic line reported this as
*1/3 layers trainable*, which is why it read as conservative.

So run #2 is a valid **full-ViT-fine-tune** data point, not the configured
last-4-blocks one. Label it that way; do not present it as `VIT_FINE_TUNE_BLOCKS = 4`.

### The fusion contributes nothing in this configuration

`Fused = 0.9840` and `ViT = 0.9840` — identical to four decimal places, with
`β̄ = 0.107` (the gate assigns the CNN 11% weight). The model is a ViT with a
decorative CNN attached. `route_acc = 0.9495` looks strong but is not
evidence of good routing: a router that always answers "ViT" scores ~0.95 when
the ViT is right 98% of the time.

This matters more than the headline. Mahmud's whole fusion stack is worth
**+0.64 over ViT alone** (Table 11: 97.46 → 98.10). The equivalent number here
is **+0.00**, and the +0.28 from 98.40 to 98.68 is TTA, not fusion. An ablation
table built on this configuration would show the contribution of every novel
component (N1/N2/N3) as approximately zero.

Root cause is the branch asymmetry, not the routing code: the CNN branch at
92.22% is 5.5 pp below step6's EfficientNetB0 (97.75% TTA) on the same data.
With β̄ = 0.107 almost no gradient reaches the CNN through the fused path, so
it is trained mainly by the `W_AUX = 0.3` auxiliary head — and the weaker it
gets, the less the gate uses it. Run #1 had the mirror image of this with the
roles reversed (β̄ = 0.757 toward the CNN).

### Fix applied (`RUN_TAG = f"v3_vit{VIT_FINE_TUNE_BLOCKS}"`)

Validated by `step9_validate_vit_unfreeze.py` (committed, runs on CPU in
~2 min with no GPU and no downloads) against a stub reproducing the
real nesting (a `Model` backbone containing a plain-`Layer` encoder holding
blocks in a list attribute); that stub reproduces the v2 log's "3 weighted
layers" exactly before the fix.

- `_iter_sublayers()` falls back to scanning instance attributes (Layer,
  list/tuple/dict of Layers) when `.layers` is empty, with cycle protection.
  Walk now finds 101 weighted layers instead of 3.
- New `find_transformer_blocks()` locates the block stack **structurally** —
  the longest run of same-class weighted siblings under one parent — instead
  of by regex on layer names. Name matching caused both the v1 and v2 failures
  and differs across keras_hub versions.
- Parameter counts now come from `vit_model.trainable_weights`, which Keras
  de-duplicates; the old per-layer sum double-counted a trainable parent and
  its children (reported 91.8% for a 4-block unfreeze in testing).
- Diagnostic prints trainable params **as a percentage of the ViT**, the
  figure that would have made the v2 bug obvious at a glance.
- Two new guards raise rather than train silently: fewer blocks found than
  requested, and a partial unfreeze exceeding 60% of the model.
- `VIT_FINE_TUNE_BLOCKS` accepts `"all"` for a deliberate full fine-tune.
- `RUN_TAG` now interpolates the setting, so `v3_vit4` and `v3_vitall` rows
  cannot collide in the JSONL.

Verified: last-4-of-12 → 11,817,984 / 36,046,080 params (32.8%), blocks 8–11;
`"all"` → 100%; both guards fire when their conditions are simulated.

---

## 2c. Decision: which ViT configuration gets the 5-fold budget

Two arms are now on the table. Fold 1 is the cheap discriminator, and one arm
is already measured for free.

| arm | tag | ViT trainable | fold-1 Fused+TTA | fusion worth |
|---|---|---|---|---|
| frozen ViT | `rrf_full` (v1) | 0 | 97.57 | +1.39 over best branch |
| full encoder | `rrf_full@v2_vitfix` | 85.1M (100%) | **98.68** | **+0.00** |
| last 4 blocks | `rrf_full@v3_vit4` | 11.8M (32.8%) | **not yet run** | ? |

Neither measured arm is usable as the headline: v1's ViT never trained, and
v2's fusion is inert. The 4-block arm is the one configuration where both
branches could plausibly be strong enough to make routing do work — the ViT
constrained enough not to swamp the CNN, the CNN therefore still receiving
gradient through the fused path.

**Probe:** `FOLDS_TO_RUN = [1]`, `VIT_FINE_TUNE_BLOCKS = 4` → tag
`rrf_full@v3_vit4`. Expect ~45 min (stage 2 trains 11.8M rather than 85M
params, against v2's 4290 s).

**Decision rule, checked against v2 fold 1:**

1. `[vit] strategy=last-4-of-12-blocks  4 units trainable (11,817,984 … 32.8%)`
   — if the strategy or the percentage differs, stop; the structural block
   search did not match the installed keras_hub.
2. `Fused > max(CNN, ViT)` by ≥ 0.3 pp. This is the one thing v2 failed and
   the whole justification for the architecture. If it fails here too, the
   problem is branch asymmetry, not the ViT depth — raise `W_AUX` from 0.3
   before spending five folds.
3. `Fused+TTA ≥ 98.37` (step6's ResNet50 mean, the bar to beat).
4. `β̄` in roughly 0.25–0.75. Outside that the gate has collapsed to one branch
   again and `route_acc` is uninformative.

If 2 and 3 both hold → run folds 2–5 under `v3_vit4` (fold 1 is already in the
JSONL and will be skipped). If 3 holds but 2 fails → the honest paper claim is
"a strong ViT fine-tune", not a fusion contribution; keep v2's full-encoder arm
as the headline and report the fusion ablation as negative. If neither holds →
`W_AUX` and branch balance come first, before any more 5-fold runs.

Keep `rrf_fusion_v2.log` and its JSONL row either way: whichever arm wins, the
other is a legitimate ablation row (full ViT fine-tune vs last-4-blocks), and
v1 remains the frozen-ViT row. Three points on the ViT-capacity axis is a
table nobody else on this dataset has published.

---

## 2d. RRF-Net run #3 (`rrf_full@v3_vit4`, fold 1) — the fix holds and the fusion works

Provenance: `rrf_fusion_v3.log`, Kaggle T4, fold 1 of 5, 2796 s
(stage1 452 + stage2 2345) — 41% faster than v2's 4754 s.

```
[vit] 138 weighted layers nested in 'vit_b16_kerashub'
[vit] strategy=last-4-of-12-blocks  4 units trainable  (28,351,488 of 85,798,656 params, 33.0%)
```

Gate 1 passes on the real model: 12 blocks located, last 4 unfrozen, 33.0% —
against 32.8% predicted on the stub. The dump also settles why the structural
search was necessary: blocks are `tranformer_block_1 … _12` (keras_hub's own
typo, 1-indexed) and their children are `ln_1`, `ln_2`, `dense_1`, `dense_2`.
Any rule grouping by digits in layer names would key on those, not the block
index.

### Three arms on the identical fold

| | CNN br. | ViT br. | Fused | Fused+TTA | ECE | NLL | β̄ | route_acc | fusion gain |
|---|---|---|---|---|---|---|---|---|---|
| v1 frozen ViT | 93.89 | 90.35 | 95.28 | 97.57 | 0.0523 | 0.1667 | 0.757 | 0.704 | **+1.39** |
| v2 all 12 blocks | 92.22 | 98.40 | 98.40 | **98.68** | 0.0205 | 0.0652 | 0.107 | 0.9495 | **+0.00** |
| **v3 last 4 blocks** | 92.15 | 96.67 | 97.08 | 98.26 | 0.0287 | 0.1017 | **0.413** | 0.7327 | **+0.41** |

"fusion gain" = Fused − max(CNN branch, ViT branch).

Monotone in ViT capacity: as the ViT strengthens, β̄ shifts off the CNN and the
fusion contributes less. v3 is the only arm that is simultaneously accurate and
an actual fusion model — β̄ mid-range, and a `route_acc` that a
constant-output router could not have produced (v2's 0.9495 could).

### Correction to the §2c gate 3 comparison

Gate 3 was specified as "Fused+TTA ≥ 98.37", step6's ResNet50 **five-fold
mean**. That is the wrong comparison for a single fold, and fold 1 is step6's
easiest fold for all four backbones. Per-fold TTA, same fold:

| model | fold 1 | 5-fold mean | folds 2–5 |
|---|---|---|---|
| resnet50 | **98.75** | 98.37 | 98.12 98.54 98.26 98.19 |
| RRF v2 all-12 | 98.68 | — | — |
| densenet121 | 98.54 | 98.32 | 98.26 97.92 98.40 98.47 |
| **RRF v3 last-4** | **98.26** | — | — |
| mobilenetv2 | 97.78 | 97.96 | 98.47 97.50 98.06 97.99 |
| efficientnetb0 | 97.64 | 97.75 | 97.50 97.57 98.19 97.85 |
| RRF v1 frozen | 97.57 | — | — |

So v2's 98.68 did **not** clear step6's best; on the shared fold ResNet50 is
98.75. v3 at 98.26 beats mobilenetv2 (+0.48) and efficientnetb0 (+0.62), loses
to densenet121 (−0.28) and resnet50 (−0.49). Nothing is settled on one fold:
resnet50 falls to 98.12–98.54 on folds 2–5, so the mean comparison is open.

### The ~92% CNN branch is not a backbone failure

CNN branch accuracy is ~92–94 in all three arms, far below step6's
efficientnetb0 (97.64 on this fold). This is mostly wiring, not a broken
backbone: the branch heads read the **256-d projected, LayerNormed** tokens at
`W_AUX = 0.3` for 18 epochs, while step6's head reads full 1280-d pooled
features at full loss weight for `FINE_TUNE_EPOCHS = 35`. Treat the branch
accuracies as router diagnostics, not single-backbone baselines. Raising
`W_AUX`, widening the branch heads, or matching step6's 35 epochs are the
levers if the gate needs a stronger CNN input — none of them tested yet.

### Decision: run v3 to five folds

v3 is the arm to spend the budget on. It is the only configuration where the
architecture does what the paper would claim for it, the cost is ~3.9 h for
four more folds, and a Wilcoxon against step6 needs five paired folds
regardless. v2 and v1 become the two other points on the ViT-capacity axis —
an ablation neither competing paper has published.

Under the leakage-delta headline chosen for the paper, RRF-Net does not have to
beat resnet50. It has to be a defensible entry in the grid with a fusion
mechanism that demonstrably works. v3 fold 1 is that; v2 was not.

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
