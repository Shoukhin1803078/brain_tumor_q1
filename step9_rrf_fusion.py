"""
Step 9: RRF-Net -- Reliability-Routed Fusion of a CNN and a ViT branch, under
the same 5-fold protocol as step6_backbone_sweep.py, with an optional
leakage-aware (similarity-group) split mode.

Why this script exists
-----------------------
step6 compared four CNN backbones under one protocol. Two 2025/2026 papers on
this same Kaggle dataset (masoudnickparvar) instead fuse a CNN with a Vision
Transformer:

  - Jayaraman et al., Sci. Reports 2025 ("CAFNet"): MobileNetV2 + ViT, one-way
    cross-attention (ViT supplies queries, CNN supplies keys/values), then a
    fixed Concat(Q, Z)W_o. 96.41% on a single image-level 80/10/10 split.
  - Mahmud et al., Informatics and Health 2026: EfficientNetV2-B0 + ViT-B/16,
    both projected to a shared 512-d space, bidirectional cross-attention, then
    an adaptive gate alpha = sigmoid(W_g[C;V] + b_g), F = alpha*C + (1-alpha)*V.
    98.10% test under a similarity-group split. Their own ablation prices the
    whole fusion stack at +0.64 over ViT alone.

In both, the gate (or its absence) is a *global, feature-only* decision: one
blend weight for the entire image, inferred implicitly from the classification
loss. RRF-Net changes that in three ways, each independently ablatable:

  N1. RELIABILITY ROUTING. The global router sees each branch's normalized
      predictive entropy, not just its features. The intuition is that the
      useful question is not "what do these two feature vectors look like" but
      "which branch is in a position to be trusted on this image". Entropies
      enter the router under stop_gradient, so the branches cannot lower the
      routing loss by manufacturing confidence.

  N2. SPATIAL ROUTING. Instead of one alpha per image, a gate is computed at
      each of the 7x7 CNN grid locations from [C, C_tilde, |C - C_tilde|], so
      a slice can lean on local CNN texture inside the lesion and on routed ViT
      context elsewhere. Mahmud's formulation cannot express that.

  N3. SUPERVISED ROUTING LOSS. The router is trained directly: on every image
      where exactly one branch is correct, the gate is pushed toward that
      branch via a binary cross-entropy term. Images where both branches agree
      carry zero routing weight, so the term only speaks where it has evidence.

None of these is a reskin of either paper's module -- N1 and N3 have no
counterpart in them at all, and N2 changes the gate's domain rather than its
parameterization.

Protocol
--------
Identical to step6 wherever it can be: same image list, same 224x224 uint8
cache, same StratifiedKFold(5, shuffle=True, random_state=42), same TTA, same
per-fold JSONL append. So RRF-Net rows drop straight into the step7 figures and
the Wilcoxon tests alongside the four CNN backbones.

SPLIT_MODE adds the leakage experiment. "image" reproduces step6 exactly.
"group" reads similarity groups (built from step8_eda.py's confirmed
near-duplicate pairs via connected components) and uses StratifiedGroupKFold so
no near-duplicate pair straddles a fold. Running both modes on the same model
gives the leakage delta directly -- the number neither paper above reports,
since CAFNet never grouped and Mahmud et al. never published the ungrouped
counterfactual.

Beyond accuracy this logs ECE, NLL and mean predictive entropy per fold, since
"is the confidence meaningful" is the question the routing design is built
around.

Running on Kaggle
-----------------
  1. Enable GPU *and* internet (the ViT weights are downloaded at build time).
  2. pip install the ViT source -- see load_vit_backbone() below.
  3. Set SPLIT_MODE, and FOLDS_TO_RUN if you want to split folds across
     sessions. Rows append to RESULTS_PATH and completed folds are skipped on
     restart, so a killed kernel costs you at most the fold in progress.
  4. SMOKE_TEST = True first. It runs 1 fold on a small subset for 1+1 epochs
     and should finish in a few minutes. Do that before spending a GPU session.

Budget note: ViT-B/16 + EfficientNetV2-B0 is far heavier than any step6
backbone. Expect roughly 1-1.5 h per fold on a T4 at these epoch counts, so one
session holds about 5 folds with no margin. Running 2-3 folds per session is
the safer pattern.
"""

import os
import gc
import glob
import json
import time
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
RESULTS_PATH = "/kaggle/working/rrf_fusion.jsonl"

# "image" -> StratifiedKFold, reproduces step6 exactly (leakage-permissive).
# "group" -> StratifiedGroupKFold over similarity groups (leakage-aware).
SPLIT_MODE = "image"
# Only used when SPLIT_MODE == "group". Produced by build_groups_from_pairs().
GROUPS_PATH = "/kaggle/working/similarity_groups.npy"
# step8_eda.py's confirmed near-duplicate pairs, used to build the groups file.
PAIRS_CSV = "/kaggle/working/results/eda_a_near_duplicates.csv"

IMG_SIZE = 224
BATCH_SIZE = 16
N_FOLDS = 5
FOLDS_TO_RUN = [1, 2, 3, 4, 5]     # fold 1 done under rrf_full@v3_vit4 and
                                   # skipped automatically if its JSONL row is
                                   # present; folds 2-5 are ~3.9 h total
SEED = 42

HEAD_EPOCHS = 5                     # stage 1: both backbones frozen
FINE_TUNE_EPOCHS = 18               # stage 2: top of each backbone unfrozen
EARLY_STOP_PATIENCE = 5
CNN_FINE_TUNE_AT = 0.8              # unfreeze top 80% of the CNN, as in step6
VIT_FINE_TUNE_BLOCKS = 4            # last N transformer blocks, or "all"

FUSION_DIM = 256                    # shared projection width (Mahmud used 512)
ATTN_HEADS = 4
DROPOUT = 0.3

FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
W_AUX = 0.3                         # deep supervision on each branch head
W_ROUTE = 0.2                       # N3: supervised routing loss

TTA_ROUNDS = 5
ECE_BINS = 15

# Ablation switches -- flip these to produce the ablation table.
USE_SPATIAL_ROUTING = True          # N2; False -> single global gate
USE_RELIABILITY_ROUTING = True      # N1; False -> router sees features only
USE_ROUTE_LOSS = True               # N3; False -> router learns implicitly only
USE_CROSS_ATTENTION = True          # False -> no cross-attention at all

# Bump this whenever the code changes in a way that invalidates earlier rows.
# completed_folds() keys on it, so a fixed re-run will NOT be skipped as
# "already done", and old/new rows stay separable in the JSONL.
# v1  = ViT silently frozen (0 params) -- invalid, kept as an ablation row.
# v2  = whole ViT encoder trained (85M) because the walk stopped at ViTEncoder.
# v3  = blocks located structurally; the tag records what was actually trained.
RUN_TAG = f"v3_vit{VIT_FINE_TUNE_BLOCKS}"

SMOKE_TEST = False                  # tiny subset, 1+1 epochs, 1 fold
SMOKE_N_PER_CLASS = 60

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Data (identical to step4/step6 so folds line up exactly)
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
            f"Check DATA_DIR and that the class folders match {CLASS_NAMES}."
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


def build_groups_from_pairs(paths, pairs_csv, out_path):
    """Turn step8_eda.py's CONFIRMED near-duplicate pairs into group labels via
    connected components (union-find), so StratifiedGroupKFold can keep every
    near-duplicate cluster inside a single fold.

    Transitivity is deliberate: if a~b and b~c, all three go to one fold. That
    over-groups slightly, which is the safe direction for a leakage audit --
    it can only make the reported accuracy more conservative, never less.

    Images in no confirmed pair become singleton groups. Writes an int array
    aligned with `paths` and returns it.
    """
    index_of = {os.path.basename(p): i for i, p in enumerate(paths)}
    parent = list(range(len(paths)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    import csv as _csv
    n_pairs = 0
    with open(pairs_csv) as f:
        for row in _csv.DictReader(f):
            # step8 writes the two image paths; fall back to positional columns.
            keys = list(row.keys())
            a = row.get("path_i") or row.get("image_i") or row[keys[0]]
            b = row.get("path_j") or row.get("image_j") or row[keys[1]]
            ia, ib = index_of.get(os.path.basename(a)), index_of.get(os.path.basename(b))
            if ia is not None and ib is not None:
                union(ia, ib)
                n_pairs += 1

    roots = np.array([find(i) for i in range(len(paths))])
    _, groups = np.unique(roots, return_inverse=True)
    np.save(out_path, groups)

    sizes = np.bincount(groups)
    print(f"Groups from {n_pairs} confirmed pairs: {groups.max() + 1} groups "
          f"for {len(paths)} images (largest {sizes.max()}, "
          f"{(sizes > 1).sum()} non-singleton, "
          f"{len(paths) - (groups.max() + 1)} images absorbed into a cluster)")
    return groups


# ---------------------------------------------------------------------------
# 2. ViT backbone loader
# ---------------------------------------------------------------------------
def load_vit_backbone():
    """Return (model mapping [0,255] float input -> (B, N, D) patch tokens,
    source_name).

    Kaggle images vary in what ViT packages they carry, so this tries the two
    realistic sources in order rather than hard-coding one. The class token is
    dropped where present -- the fusion operates on patch tokens only, since a
    class token has no spatial position for the cross-attention to route to.

        pip install keras-hub          # preferred
        pip install vit-keras          # fallback

    If vit-keras raises on `tensorflow_addons`, prefer keras-hub; tfa is
    archived and does not build against recent TF.
    """
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="vit_raw_input")

    # --- Option A: keras-hub ---
    try:
        import keras_hub
        backbone = keras_hub.models.ViTBackbone.from_preset(
            "vit_base_patch16_224_imagenet"
        )
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="vit_rescale")(inp)
        tokens = backbone(x)
        if len(tokens.shape) != 3:
            raise RuntimeError(f"expected token sequence, got {tokens.shape}")
        if tokens.shape[1] in (197, 577):          # drop the class token
            tokens = layers.Lambda(lambda t: t[:, 1:, :], name="vit_drop_cls")(tokens)
        return keras.Model(inp, tokens, name="vit_b16_kerashub"), "keras_hub"
    except Exception as e:                          # noqa: BLE001 - want any failure
        print(f"[vit] keras_hub unavailable ({type(e).__name__}: {e}); trying vit_keras")

    # --- Option B: vit-keras ---
    try:
        from vit_keras import vit
        base = vit.vit_b16(
            image_size=IMG_SIZE, include_top=False, pretrained=True,
            pretrained_top=False,
        )
        # vit_keras' include_top=False still extracts the class token, so reach
        # back to the final encoder LayerNorm for the full token sequence.
        try:
            tok = base.get_layer("Transformer/encoder_norm").output
        except ValueError:
            tok = None
            for layer in reversed(base.layers):
                if len(getattr(layer, "output_shape", ()) or ()) == 3:
                    tok = layer.output
                    break
            if tok is None:
                raise RuntimeError("no 3-D layer found in vit_keras model")
        seq = keras.Model(base.input, tok)
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="vit_rescale")(inp)
        tokens = seq(x)
        if tokens.shape[1] in (197, 577):
            tokens = layers.Lambda(lambda t: t[:, 1:, :], name="vit_drop_cls")(tokens)
        return keras.Model(inp, tokens, name="vit_b16_vitkeras"), "vit_keras"
    except Exception as e:                          # noqa: BLE001
        raise RuntimeError(
            "Could not load a pretrained ViT-B/16. Install one of:\n"
            "    pip install keras-hub\n"
            "    pip install vit-keras\n"
            "and make sure Kaggle internet is enabled so the weights download.\n"
            f"Last error: {type(e).__name__}: {e}"
        )


_VIT_STRUCTURE_PRINTED = False


def _iter_sublayers(layer, _seen=None):
    """Depth-first walk over a layer and everything nested inside it.

    Two levels of nesting have to be crossed, and they hide sub-layers in
    different places:

    1. keras_hub returns the entire ViT as ONE nested sub-model, so the wrapper
       built in load_vit_backbone() has only 4 top-level layers (Input,
       Rescaling, ViTBackbone, drop-cls Lambda) -- iterating `vit_model.layers`
       never reaches a transformer block.
    2. `ViTEncoder` is a plain Layer, not a Model: it keeps its 12 transformer
       blocks in a list attribute and exposes NO `.layers`. Recursing on
       `.layers` alone therefore stops at the encoder and reports it as a
       single 85M-parameter "layer" -- which is how run v2 unfroze the whole
       encoder while the log said "1/3 layers trainable".

    So: follow `.layers` when it exists, and otherwise fall back to scanning
    the instance attributes for Layers and lists/tuples/dicts of Layers.
    """
    if _seen is None:
        _seen = set()
    subs = list(getattr(layer, "layers", None) or [])
    if not subs:
        for attr in vars(layer).values():
            if isinstance(attr, keras.layers.Layer):
                subs.append(attr)
            elif isinstance(attr, (list, tuple)):
                subs.extend(a for a in attr if isinstance(a, keras.layers.Layer))
            elif isinstance(attr, dict):
                subs.extend(a for a in attr.values()
                            if isinstance(a, keras.layers.Layer))
    for sub in subs:
        if id(sub) in _seen:
            continue
        _seen.add(id(sub))
        yield sub
        yield from _iter_sublayers(sub, _seen)


def _direct_sublayers(layer):
    """The layers one level below `layer`, whether exposed via `.layers`
    (Models) or held in instance attributes (plain Layers like ViTEncoder)."""
    subs = list(getattr(layer, "layers", None) or [])
    if subs:
        return subs
    seen, out = set(), []
    for attr in vars(layer).values():
        items = []
        if isinstance(attr, keras.layers.Layer):
            items = [attr]
        elif isinstance(attr, (list, tuple)):
            items = [a for a in attr if isinstance(a, keras.layers.Layer)]
        elif isinstance(attr, dict):
            items = [a for a in attr.values() if isinstance(a, keras.layers.Layer)]
        for a in items:
            if id(a) not in seen:
                seen.add(id(a))
                out.append(a)
    return out


def find_transformer_blocks(vit_model, min_blocks=6):
    """Locate the ordered list of transformer blocks *structurally*.

    Layer naming differs between keras_hub versions and vit_keras, so matching
    on names is brittle -- that brittleness is what produced both the v1
    (0 params) and v2 (whole encoder) failures. Instead: find the container
    whose direct children are a long run of same-class weighted siblings.
    That run is the transformer stack, in depth order.
    """
    best = []
    for parent in [vit_model] + list(_iter_sublayers(vit_model)):
        kids = [k for k in _direct_sublayers(parent) if k.weights]
        if len(kids) < min_blocks:
            continue
        by_cls = {}
        for k in kids:
            by_cls.setdefault(k.__class__.__name__, []).append(k)
        run = max(by_cls.values(), key=len)
        if len(run) >= min_blocks and len(run) > len(best):
            best = run
    return best


def set_vit_trainable(vit_model, n_blocks):
    """Freeze the ViT, then unfreeze the last `n_blocks` transformer blocks
    (or everything, when n_blocks == "all").

    Parameter counts come from `vit_model.trainable_weights`, which Keras
    already de-duplicates -- summing per-layer counts double-counts a trainable
    parent and its children and can report >100% of the model.
    """
    global _VIT_STRUCTURE_PRINTED
    vit_model.trainable = True
    flat = list(_iter_sublayers(vit_model))
    weighted = [l for l in flat if l.weights]
    if not weighted:
        raise RuntimeError(
            f"No weighted layers found inside '{vit_model.name}' -- the ViT "
            "branch carries no parameters at all. Check load_vit_backbone()."
        )
    if not _VIT_STRUCTURE_PRINTED:
        debug_vit_structure(vit_model)
        _VIT_STRUCTURE_PRINTED = True

    total_params = int(sum(np.prod(w.shape) for w in vit_model.weights))

    if n_blocks == "all":
        strategy, n_train = "all", len(weighted)
    else:
        for l in flat:
            l.trainable = False
        blocks = find_transformer_blocks(vit_model)
        if len(blocks) < n_blocks:
            raise RuntimeError(
                f"Found only {len(blocks)} transformer blocks inside "
                f"'{vit_model.name}' but {n_blocks} were requested. Inspect "
                "the debug_vit_structure() output above and either lower "
                'VIT_FINE_TUNE_BLOCKS or set it to "all".'
            )
        strategy = f"last-{n_blocks}-of-{len(blocks)}-blocks"
        for b in blocks[-n_blocks:]:
            b.trainable = True
        n_train = n_blocks

    trainable_params = int(sum(np.prod(w.shape) for w in vit_model.trainable_weights))
    pct = 100.0 * trainable_params / max(total_params, 1)
    print(f"[vit] strategy={strategy}  {n_train} units trainable  "
          f"({trainable_params:,} of {total_params:,} params, {pct:.1f}%)")
    if trainable_params == 0:
        raise RuntimeError(
            "ViT fine-tuning would train 0 parameters -- the branch would stay "
            "a frozen feature extractor. Inspect the layer names printed by "
            "debug_vit_structure() and fix the unfreezing rule before running."
        )
    # A partial unfreeze landing on nearly the whole network means the block
    # list was not really the block list. v2 reported "1/3 layers" while
    # training all 85M encoder params; fail loudly instead of repeating that.
    if strategy != "all" and pct > 60.0:
        raise RuntimeError(
            f"Asked to unfreeze {n_blocks} blocks but {pct:.1f}% of the ViT "
            "became trainable -- find_transformer_blocks() did not isolate the "
            'blocks. Check debug_vit_structure(). Set VIT_FINE_TUNE_BLOCKS = '
            '"all" if full fine-tuning is intended.'
        )
    return trainable_params


def debug_vit_structure(vit_model, limit=40):
    """Print the nested layer names so the unfreezing rule can be checked
    against what the installed keras_hub / vit_keras version actually builds."""
    flat = [l for l in _iter_sublayers(vit_model) if l.weights]
    print(f"[vit] {len(flat)} weighted layers nested in '{vit_model.name}':")
    for l in flat[:limit]:
        print(f"    {l.name:50s} {l.__class__.__name__}")
    if len(flat) > limit:
        print(f"    ... and {len(flat) - limit} more")


# ---------------------------------------------------------------------------
# 3. RRF-Net
# ---------------------------------------------------------------------------
def normalized_entropy(logits):
    """Predictive entropy scaled to [0, 1] -- 0 = fully confident, 1 = uniform."""
    p = tf.nn.softmax(logits, axis=-1)
    ent = -tf.reduce_sum(p * tf.math.log(p + 1e-8), axis=-1, keepdims=True)
    return ent / math.log(NUM_CLASSES)


def build_rrf_net():
    """Functional core. Outputs, in order:
        fused_logits (B,K), cnn_logits (B,K), vit_logits (B,K),
        beta (B,1) global route weight, gate_map (B,49,1) spatial gate.
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

    # -- CNN branch. EfficientNetV2 carries its own normalization, so feed it
    #    raw 0-255; the ViT branch rescales to [-1,1] internally. Using one
    #    branch's preprocessing for both is the silent-accuracy-loss trap that
    #    step6's backbone registry exists to avoid.
    cnn = keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    cnn.trainable = False
    c_map = cnn(keras.applications.efficientnet_v2.preprocess_input(inputs),
                training=False)                                      # (B,7,7,1280)
    n_cnn_tokens = c_map.shape[1] * c_map.shape[2]
    C = layers.Reshape((n_cnn_tokens, c_map.shape[-1]), name="cnn_tokens")(c_map)
    C = layers.Dense(FUSION_DIM, name="cnn_proj")(C)                 # (B,49,d)
    C = layers.LayerNormalization(name="cnn_proj_ln")(C)

    # -- ViT branch.
    vit_model, vit_source = load_vit_backbone()
    vit_model.trainable = False
    V = vit_model(inputs)                                            # (B,196,768)
    V = layers.Dense(FUSION_DIM, name="vit_proj")(V)                 # (B,196,d)
    V = layers.LayerNormalization(name="vit_proj_ln")(V)

    # -- Cross-attention: each CNN location queries the ViT tokens, so global
    #    context is routed onto the convolutional grid. Keeping the CNN grid as
    #    the canonical frame is what makes a per-location gate well defined.
    if USE_CROSS_ATTENTION:
        attn = layers.MultiHeadAttention(
            num_heads=ATTN_HEADS, key_dim=FUSION_DIM // ATTN_HEADS,
            dropout=0.1, name="cross_attn",
        )(query=C, value=V, key=V)
        C_tilde = layers.LayerNormalization(name="cross_attn_ln")(
            layers.Add(name="cross_attn_res")([C, attn])
        )
    else:
        # Ablation: broadcast a pooled ViT summary to every location instead.
        v_pool = layers.GlobalAveragePooling1D(name="vit_pool_abl")(V)
        C_tilde = layers.RepeatVector(n_cnn_tokens, name="vit_bcast")(v_pool)

    # -- N2: spatial routing. One gate per grid location, from the local
    #    feature, the routed feature, and their absolute disagreement.
    if USE_SPATIAL_ROUTING:
        diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]), name="route_diff")([C, C_tilde])
        gate_in = layers.Concatenate(name="route_concat")([C, C_tilde, diff])
        gate_map = layers.Dense(1, activation="sigmoid", name="spatial_gate")(gate_in)
    else:
        # Ablation: Mahmud-style single global gate, broadcast to all locations.
        g_in = layers.Concatenate(name="global_gate_concat")([
            layers.GlobalAveragePooling1D()(C),
            layers.GlobalAveragePooling1D()(C_tilde),
        ])
        g = layers.Dense(1, activation="sigmoid", name="global_gate")(g_in)
        gate_map = layers.Lambda(
            lambda t: tf.tile(tf.expand_dims(t, 1), [1, n_cnn_tokens, 1]),
            name="global_gate_bcast",
        )(g)

    fused_tokens = layers.Lambda(
        lambda t: t[0] * t[1] + (1.0 - t[0]) * t[2], name="route_blend",
    )([gate_map, C, C_tilde])
    f = layers.GlobalAveragePooling1D(name="fused_pool")(fused_tokens)   # (B,d)

    # -- Per-branch heads. These do real work: deep supervision, and they are
    #    the only place branch-specific correctness is observable for N1/N3.
    c_vec = layers.GlobalAveragePooling1D(name="cnn_pool")(C)
    v_vec = layers.GlobalAveragePooling1D(name="vit_pool")(V)
    cnn_logits = layers.Dense(NUM_CLASSES, name="cnn_logits")(
        layers.Dropout(DROPOUT)(c_vec))
    vit_logits = layers.Dense(NUM_CLASSES, name="vit_logits")(
        layers.Dropout(DROPOUT)(v_vec))

    # -- N1: reliability routing. The router sees how uncertain each branch is,
    #    under stop_gradient so a branch cannot game the routing loss by
    #    inflating its own confidence.
    if USE_RELIABILITY_ROUTING:
        u = layers.Lambda(
            lambda t: tf.stop_gradient(tf.concat([
                normalized_entropy(t[0]),
                normalized_entropy(t[1]),
                normalized_entropy(t[0]) - normalized_entropy(t[1]),
            ], axis=-1)),
            name="branch_entropy",
        )([cnn_logits, vit_logits])
        route_in = layers.Concatenate(name="route_in")([f, u])
    else:
        route_in = f

    beta = layers.Dense(1, activation="sigmoid", name="beta")(
        layers.Dense(64, activation="relu", name="route_hidden")(route_in))

    branch_mix = layers.Lambda(
        lambda t: t[0] * t[1] + (1.0 - t[0]) * t[2], name="branch_mix",
    )([beta, c_vec, v_vec])

    z = layers.Concatenate(name="head_concat")([f, branch_mix])
    z = layers.Dense(256, activation="relu", name="head_dense")(z)
    z = layers.BatchNormalization(name="head_bn")(z)
    z = layers.Dropout(DROPOUT, name="head_drop")(z)
    fused_logits = layers.Dense(NUM_CLASSES, name="fused_logits")(z)

    core = keras.Model(
        inputs, [fused_logits, cnn_logits, vit_logits, beta, gate_map],
        name="rrf_net",
    )
    core.vit_source = vit_source
    return core, cnn, vit_model


# ---------------------------------------------------------------------------
# 4. Losses and the training wrapper
# ---------------------------------------------------------------------------
def focal_ls_loss(y_true_oh, logits, gamma=FOCAL_GAMMA, eps=LABEL_SMOOTHING):
    """Focal loss on label-smoothed targets.

    Smoothing is applied to the targets while the focal modulation uses the
    true-class probability, so the two regularizers do not cancel: smoothing
    caps confidence, focal re-weights toward hard examples.
    """
    y_s = y_true_oh * (1.0 - eps) + eps / NUM_CLASSES
    logp = tf.nn.log_softmax(logits, axis=-1)
    p = tf.exp(logp)
    p_t = tf.reduce_sum(y_true_oh * p, axis=-1, keepdims=True)
    focal = tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(-tf.reduce_sum(y_s * logp, axis=-1, keepdims=True) * focal)


class RRFTrainer(keras.Model):
    """Wraps the functional core so the composite objective can be computed in
    one place. Keras' built-in multi-output loss plumbing cannot express the
    routing term, which needs both branches' correctness and the label at once.
    """

    def __init__(self, core, **kw):
        super().__init__(**kw)
        self.core = core
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.route_tracker = keras.metrics.Mean(name="route_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker, self.route_tracker]

    def call(self, x, training=False):
        return self.core(x, training=training)

    def _compute(self, y, outputs):
        fused_logits, cnn_logits, vit_logits, beta, _ = outputs
        y_oh = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)

        loss = focal_ls_loss(y_oh, fused_logits)
        loss += W_AUX * focal_ls_loss(y_oh, cnn_logits)
        loss += W_AUX * focal_ls_loss(y_oh, vit_logits)

        # -- N3: supervised routing. Target 1 where only the CNN is right,
        #    0 where only the ViT is right; weight 0 where they agree, so the
        #    term stays silent unless the branches actually disagree.
        yi = tf.cast(y, tf.int64)
        cnn_ok = tf.cast(tf.equal(tf.argmax(cnn_logits, axis=-1), yi), tf.float32)
        vit_ok = tf.cast(tf.equal(tf.argmax(vit_logits, axis=-1), yi), tf.float32)
        route_target = tf.expand_dims(cnn_ok * (1.0 - vit_ok), -1)
        route_w = tf.expand_dims(tf.abs(cnn_ok - vit_ok), -1)

        b = tf.clip_by_value(beta, 1e-6, 1.0 - 1e-6)
        bce = -(route_target * tf.math.log(b) + (1.0 - route_target) * tf.math.log(1.0 - b))
        route_loss = tf.reduce_sum(bce * route_w) / (tf.reduce_sum(route_w) + 1e-6)
        if USE_ROUTE_LOSS:
            loss += W_ROUTE * route_loss

        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(fused_logits, axis=-1), yi), tf.float32))
        return loss, acc, route_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self.core(x, training=True)
            loss, acc, route_loss = self._compute(y, outputs)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            (g, v) for g, v in zip(grads, self.trainable_variables) if g is not None
        )
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.route_tracker.update_state(route_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        outputs = self.core(x, training=False)
        loss, acc, route_loss = self._compute(y, outputs)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.route_tracker.update_state(route_loss)
        return {m.name: m.result() for m in self.metrics}


# ---------------------------------------------------------------------------
# 5. Augmentation, inference, calibration
# ---------------------------------------------------------------------------
def make_augmenter():
    # Same policy as step4/step6, so the comparison isolates the architecture.
    return keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
    ], name="augmenter")


def predict_probs(core, X, batch_size=BATCH_SIZE):
    out = core.predict(X, batch_size=batch_size, verbose=0)
    fused_logits, cnn_logits, vit_logits, beta, _ = out
    return (tf.nn.softmax(fused_logits).numpy(),
            tf.nn.softmax(cnn_logits).numpy(),
            tf.nn.softmax(vit_logits).numpy(),
            beta.reshape(-1))


def tta_predict(core, augmenter, X, rounds=TTA_ROUNDS, batch_size=BATCH_SIZE):
    probs = predict_probs(core, X, batch_size)[0]
    for _ in range(rounds):
        X_aug = augmenter(X, training=True).numpy()
        probs = probs + predict_probs(core, X_aug, batch_size)[0]
    return probs / (rounds + 1)


def expected_calibration_error(probs, y_true, n_bins=ECE_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def nll(probs, y_true):
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-12)))


def mean_entropy(probs):
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return float(np.mean(ent / math.log(NUM_CLASSES)))


# ---------------------------------------------------------------------------
# 6. Per-fold training
# ---------------------------------------------------------------------------
def train_one_fold(X_train, y_train, X_val, y_val, augmenter):
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED)
                .batch(BATCH_SIZE)
                .map(lambda a, b: (augmenter(a, training=True), b),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    core, cnn, vit_model = build_rrf_net()
    model = RRFTrainer(core)

    # -- Stage 1: both backbones frozen, train projections / fusion / heads.
    model.compile(optimizer=keras.optimizers.Adam(1e-3))
    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds,
              epochs=1 if SMOKE_TEST else HEAD_EPOCHS, verbose=2)
    t_stage1 = time.perf_counter() - t0

    # -- Stage 2: unfreeze the top of each backbone.
    cnn.trainable = True
    freeze_until = int(len(cnn.layers) * (1 - CNN_FINE_TUNE_AT))
    for layer in cnn.layers[:freeze_until]:
        layer.trainable = False
    # BatchNorm in a partially-frozen CNN must stay in inference mode, or the
    # running statistics get rewritten by small batches and accuracy collapses.
    for layer in cnn.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    set_vit_trainable(vit_model, VIT_FINE_TUNE_BLOCKS)

    model.compile(optimizer=keras.optimizers.Adam(1e-5))
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                      patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds,
              epochs=1 if SMOKE_TEST else FINE_TUNE_EPOCHS,
              callbacks=cbs, verbose=2)
    t_stage2 = time.perf_counter() - t0

    return model, core, t_stage1, t_stage2


def append_result(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def completed_folds():
    if not os.path.exists(RESULTS_PATH):
        return set()
    done = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("split_mode") == SPLIT_MODE and r.get("variant") == variant_tag():
                done.add(r["fold"])
    return done


def variant_tag():
    if all([USE_SPATIAL_ROUTING, USE_RELIABILITY_ROUTING,
            USE_ROUTE_LOSS, USE_CROSS_ATTENTION]):
        return f"rrf_full@{RUN_TAG}"
    bits = []
    if not USE_CROSS_ATTENTION:
        bits.append("noattn")
    if not USE_SPATIAL_ROUTING:
        bits.append("nospatial")
    if not USE_RELIABILITY_ROUTING:
        bits.append("norelia")
    if not USE_ROUTE_LOSS:
        bits.append("noroute")
    return "rrf_" + "_".join(bits) + f"@{RUN_TAG}"


# ---------------------------------------------------------------------------
# 7. Driver
# ---------------------------------------------------------------------------
def main():
    assert SPLIT_MODE in ("image", "group"), "SPLIT_MODE must be 'image' or 'group'"
    print(f"RRF-Net | variant={variant_tag()} | split_mode={SPLIT_MODE}")

    paths, y = list_image_paths(DATA_DIR)
    print(f"Found {len(paths)} images across {NUM_CLASSES} classes")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == i).sum()}")

    if SMOKE_TEST:
        keep = np.concatenate([
            np.where(y == c)[0][:SMOKE_N_PER_CLASS] for c in range(NUM_CLASSES)
        ])
        paths = [paths[i] for i in keep]
        y = y[keep]
        print(f"[smoke] reduced to {len(paths)} images")

    groups = None
    if SPLIT_MODE == "group":
        if os.path.exists(GROUPS_PATH):
            groups = np.load(GROUPS_PATH)
            print(f"Loaded {groups.max() + 1} similarity groups from {GROUPS_PATH}")
        elif os.path.exists(PAIRS_CSV):
            groups = build_groups_from_pairs(paths, PAIRS_CSV, GROUPS_PATH)
        else:
            raise FileNotFoundError(
                f"SPLIT_MODE='group' needs either {GROUPS_PATH} or {PAIRS_CSV}. "
                "Run step8_eda.py first to produce the confirmed near-duplicate "
                "pairs, then rerun this script."
            )
        if len(groups) != len(paths):
            raise ValueError(
                f"groups has {len(groups)} entries but there are {len(paths)} "
                "images -- the groups file was built for a different image list."
            )

    print("\nLoading + preprocessing images...")
    X = load_and_preprocess(paths)
    gc.collect()

    if SPLIT_MODE == "group":
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X, y)

    done = completed_folds()
    if done:
        print(f"Already completed folds for this variant/mode: {sorted(done)} -- skipping")

    augmenter = make_augmenter()
    summary = {"cnn": [], "vit": [], "fused": [], "fused_tta": []}

    for fold_idx, (trainval_idx, test_idx) in enumerate(split_iter):
        fold = fold_idx + 1
        if fold not in FOLDS_TO_RUN or fold in done:
            continue
        print(f"\n{'=' * 70}\nRRF-Net [{variant_tag()}/{SPLIT_MODE}] -- "
              f"FOLD {fold}/{N_FOLDS}\n{'=' * 70}")

        if SPLIT_MODE == "group":
            # Keep the validation carve-out group-clean too, otherwise early
            # stopping is selected on near-duplicates of the training images.
            g_tv = groups[trainval_idx]
            inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
            tr_sub, val_sub = next(inner.split(trainval_idx, y[trainval_idx], groups=g_tv))
            train_idx, val_idx = trainval_idx[tr_sub], trainval_idx[val_sub]
            assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        else:
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=0.15, random_state=SEED,
                stratify=y[trainval_idx])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

        model, core, t_stage1, t_stage2 = train_one_fold(
            X_train, y_train, X_val, y_val, augmenter)

        t0 = time.perf_counter()
        p_fused, p_cnn, p_vit, beta = predict_probs(core, X_test)
        t_infer = time.perf_counter() - t0
        ms_per_image = 1000.0 * t_infer / len(X_test)

        t0 = time.perf_counter()
        p_tta = tta_predict(core, augmenter, X_test)
        ms_per_image_tta = 1000.0 * (time.perf_counter() - t0) / len(X_test)

        acc_cnn = accuracy_score(y_test, p_cnn.argmax(1))
        acc_vit = accuracy_score(y_test, p_vit.argmax(1))
        acc_fused = accuracy_score(y_test, p_fused.argmax(1))
        acc_tta = accuracy_score(y_test, p_tta.argmax(1))

        summary["cnn"].append(acc_cnn)
        summary["vit"].append(acc_vit)
        summary["fused"].append(acc_fused)
        summary["fused_tta"].append(acc_tta)

        # How often did routing actually matter, and did it route correctly?
        cnn_ok = (p_cnn.argmax(1) == y_test)
        vit_ok = (p_vit.argmax(1) == y_test)
        disagree = cnn_ok != vit_ok
        route_acc = (float(np.mean((beta[disagree] > 0.5) == cnn_ok[disagree]))
                     if disagree.sum() > 0 else None)

        print(f"\nFold {fold} [{variant_tag()}/{SPLIT_MODE}] results: "
              f"CNN={acc_cnn:.4f}  ViT={acc_vit:.4f}  "
              f"Fused={acc_fused:.4f} ({ms_per_image:.2f} ms/img)  "
              f"Fused+TTA={acc_tta:.4f} ({ms_per_image_tta:.2f} ms/img)  "
              f"train_time={t_stage1 + t_stage2:.1f}s "
              f"(stage1={t_stage1:.1f}s, stage2={t_stage2:.1f}s)")
        print(f"  ECE={expected_calibration_error(p_fused, y_test):.4f}  "
              f"NLL={nll(p_fused, y_test):.4f}  "
              f"entropy={mean_entropy(p_fused):.4f}  "
              f"beta_mean={float(beta.mean()):.3f}  "
              f"branches_disagree={int(disagree.sum())}/{len(y_test)}  "
              f"route_acc={route_acc if route_acc is None else round(route_acc, 4)}")
        print("\n-- Fused+TTA classification report --")
        print(classification_report(y_test, p_tta.argmax(1), target_names=CLASS_NAMES))
        print(confusion_matrix(y_test, p_tta.argmax(1)))

        append_result({
            "model": "rrf_net",
            "variant": variant_tag(),
            "split_mode": SPLIT_MODE,
            "vit_source": getattr(core, "vit_source", "unknown"),
            "fold": fold,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "train_time_stage1_s": round(t_stage1, 2),
            "train_time_stage2_s": round(t_stage2, 2),
            "train_time_total_s": round(t_stage1 + t_stage2, 2),
            "cnn_branch_acc": round(acc_cnn, 4),
            "vit_branch_acc": round(acc_vit, 4),
            "fused_acc": round(acc_fused, 4),
            "fused_ms_per_image": round(ms_per_image, 3),
            "tta_acc": round(acc_tta, 4),
            "tta_ms_per_image": round(ms_per_image_tta, 3),
            "ece": round(expected_calibration_error(p_fused, y_test), 5),
            "nll": round(nll(p_fused, y_test), 5),
            "mean_entropy": round(mean_entropy(p_fused), 5),
            "beta_mean": round(float(beta.mean()), 4),
            "branches_disagree": int(disagree.sum()),
            "route_acc": route_acc,
        })

        del model, core
        keras.backend.clear_session()
        gc.collect()

    if summary["fused"]:
        print(f"\n{'=' * 70}\nSUMMARY -- {variant_tag()} / {SPLIT_MODE} "
              f"({len(summary['fused'])} folds this session)\n{'=' * 70}")
        for key, label in [("cnn", "CNN branch"), ("vit", "ViT branch"),
                           ("fused", "RRF fused"), ("fused_tta", "RRF fused + TTA")]:
            a = np.array(summary[key])
            print(f"{label:24s}: {a.mean() * 100:.2f}% +/- {a.std() * 100:.2f}%   "
                  f"(folds: {[f'{v * 100:.2f}' for v in a]})")
    print(f"\nRows appended to {RESULTS_PATH} -- download it and keep the raw "
          f"per-fold rows for the mean/std and the Wilcoxon tests.")


if __name__ == "__main__":
    main()
