"""Reproduce the v2 'whole-encoder' bug against a stub shaped like the REAL
keras_hub ViT (ViTEncoder is a plain Layer holding blocks in a list attribute,
exposing no .layers), then confirm the fixed walk reaches the blocks."""
import sys, re
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, "/home/tasnia/brain_tumor_q1")
keras.applications.EfficientNetV2B0 = lambda **kw: keras.Model(
    layers.Input(shape=(224, 224, 3)), layers.Conv2D(4, 3)(layers.Input(shape=(224, 224, 3))))
import step9_rrf_fusion as s9


class ViTBlock(layers.Layer):
    def __init__(self, i, **kw):
        super().__init__(name=f"transformer_layer_{i}", **kw)
        self.attn = layers.MultiHeadAttention(12, 64, name=f"tl{i}_attn")
        self.ln = layers.LayerNormalization(name=f"tl{i}_ln")
        self.mlp = layers.Dense(768, name=f"tl{i}_mlp")
    def call(self, x):
        return self.mlp(self.ln(x + self.attn(x, x)))


class ViTEncoder(layers.Layer):
    """Plain Layer: blocks live in a LIST ATTRIBUTE, no .layers property."""
    def __init__(self, **kw):
        super().__init__(name="vit_encoder", **kw)
        self.encoder_layers = [ViTBlock(i) for i in range(12)]
        self.norm = layers.LayerNormalization(name="encoder_final_ln")
    def call(self, x):
        for b in self.encoder_layers:
            x = b(x)
        return self.norm(x)


class ViTPatching(layers.Layer):
    def __init__(self, **kw):
        super().__init__(name="vit_patching_and_embedding", **kw)
        self.proj = layers.Conv2D(768, 16, strides=16, name="patch_proj")
        self.rs = layers.Reshape((196, 768))
    def call(self, x):
        return self.rs(self.proj(x))


class ViTBackbone(keras.Model):
    def __init__(self, **kw):
        super().__init__(name="vi_t_backbone", **kw)
        self.patch = ViTPatching()
        self.enc = ViTEncoder()
    def call(self, x):
        return self.enc(self.patch(x))


def nested_vit():
    inp = layers.Input(shape=(224, 224, 3), name="vit_raw_input")
    y = layers.Rescaling(1 / 127.5, offset=-1.0, name="vit_rescale")(inp)
    y = ViTBackbone()(y)
    y = layers.Lambda(lambda t: t, name="vit_drop_cls")(y)
    return keras.Model(inp, y, name="vit_b16_kerashub")


m = nested_vit()
m(np.zeros((1, 224, 224, 3), "float32"))
print(f"top-level layers: {[l.name for l in m.layers]}")

# --- OLD walk (.layers only) ---
def old_iter(layer):
    for sub in getattr(layer, "layers", []):
        yield sub
        yield from old_iter(sub)
old_weighted = [l for l in old_iter(m) if l.weights]
print(f"OLD walk -> {len(old_weighted)} weighted layers: {[l.name for l in old_weighted]}")
assert len(old_weighted) == 3, f"stub should reproduce the v2 log's 3 weighted layers, got {len(old_weighted)}"

# --- NEW walk ---
s9.debug_vit_structure(m, limit=8)
new_weighted = [l for l in s9._iter_sublayers(m) if l.weights]
print(f"NEW walk -> {len(new_weighted)} weighted layers")
assert len(new_weighted) > 30, len(new_weighted)

# --- 4 blocks ---
m2 = nested_vit(); m2(np.zeros((1, 224, 224, 3), "float32"))
s9._VIT_STRUCTURE_PRINTED = True
got = s9.set_vit_trainable(m2, 4)
total = int(sum(np.prod(w.shape) for w in m2.weights))
print(f"  -> {got:,} / {total:,} = {100*got/total:.1f}%")
blocks = s9.find_transformer_blocks(m2)
print(f"  blocks found: {len(blocks)} -> {[b.name for b in blocks[:3]]} ... {[b.name for b in blocks[-2:]]}")
assert len(blocks) == 12, len(blocks)
idxs = sorted(int(re.findall(r"(\d+)", b.name)[-1]) for b in blocks if b.trainable)
print(f"  unfrozen block indices: {idxs}")
assert idxs == [8, 9, 10, 11], idxs
assert 0 < got < total * 0.6

# --- "all" ---
m3 = nested_vit(); m3(np.zeros((1, 224, 224, 3), "float32"))
g3 = s9.set_vit_trainable(m3, "all")
print(f"  all -> {g3:,} / {int(sum(np.prod(w.shape) for w in m3.weights)):,}")

# --- structural search survives the v2-style .layers-only walk ---
real_iter = s9._iter_sublayers
s9._iter_sublayers = old_iter
m4 = nested_vit(); m4(np.zeros((1, 224, 224, 3), "float32"))
g4 = s9.set_vit_trainable(m4, 4)
assert g4 == got, (g4, got)
print(f"  robust to the v2 walk: still {g4:,} params (same as fixed walk)")
s9._iter_sublayers = real_iter

# --- >60% guard fires if block isolation ever regresses ---
real_find = s9.find_transformer_blocks
m5 = nested_vit(); m5(np.zeros((1, 224, 224, 3), "float32"))
enc = [l for l in s9._iter_sublayers(m5) if l.name == "vit_encoder"][0]
s9.find_transformer_blocks = lambda m, **kw: [enc] * 12   # grabs the whole encoder
try:
    s9.set_vit_trainable(m5, 4)
    print("GUARD DID NOT FIRE -- bad"); raise SystemExit(1)
except RuntimeError as e:
    print(f"  >60% guard fires: {str(e)[:70]}...")

# --- too-few-blocks guard ---
s9.find_transformer_blocks = lambda m, **kw: []
m6 = nested_vit(); m6(np.zeros((1, 224, 224, 3), "float32"))
try:
    s9.set_vit_trainable(m6, 4)
    print("COUNT GUARD DID NOT FIRE -- bad"); raise SystemExit(1)
except RuntimeError as e:
    print(f"  block-count guard fires: {str(e)[:70]}...")
s9.find_transformer_blocks = real_find

print("\nFIX v3 VALIDATED")
