#!/usr/bin/env python3
"""OpenAI Privacy Filter → ANE-native CoreML converter.

Converts the HuggingFace safetensors checkpoint into a CoreML model optimized
for Apple Neural Engine inference. The model is a bidirectional token classifier
(NOT autoregressive), so there is no KV cache — the entire sequence is processed
in a single forward pass.

Architecture:
  - 8 transformer blocks with bidirectional banded attention (window=257)
  - Sparse MoE FFN: 128 experts, top-4 routing per token
  - GQA: 14 query heads, 2 KV heads, head_dim=64
  - d_model=640, classification head → 33 classes (BIOES PII labels)
  - YaRN RoPE scaling

ANE optimizations:
  - All attention projections as Conv2d(1×1) for ANE matmul engine
  - Float16 throughout (ANE native dtype)
  - Fixed input shape to avoid ANE replanning
  - MoE expert routing handled via batched gather + matmul
  - Optional int8 weight quantization

Usage:
  python3 privacy_filter_to_ane.py [--seq-len S] [--quantize] [--checkpoint DIR]

  By default downloads from HuggingFace openai/privacy-filter.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants ───────────────────────────────────────────────────────────────

NUM_LABELS = 33
D_MODEL = 640
N_HEADS = 14
N_KV_HEADS = 2
HEAD_DIM = 64
N_LAYERS = 8
N_EXPERTS = 128
EXPERTS_PER_TOKEN = 4
INTERMEDIATE_SIZE = 640  # per expert
SLIDING_WINDOW = 128  # left=128, right=128, total band=257
RMS_NORM_EPS = 1e-5
ROPE_THETA = 150000.0
ROPE_SCALING_FACTOR = 32.0
ROPE_NTK_ALPHA = 1.0
ROPE_NTK_BETA = 32.0
INITIAL_CONTEXT_LENGTH = 4096
VOCAB_SIZE = 200064
SWIGLU_LIMIT = 7.0


def load_weights(checkpoint_dir):
    """Load weights from safetensors as PyTorch tensors (handles bfloat16)."""
    from safetensors import safe_open

    weights = {}
    st_path = Path(checkpoint_dir) / "model.safetensors"
    if not st_path.exists():
        import glob
        st_files = sorted(glob.glob(str(Path(checkpoint_dir) / "model-*.safetensors")))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
    else:
        st_files = [str(st_path)]

    for f in st_files:
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                weights[key] = sf.get_tensor(key)
    return weights


def download_model():
    """Download the model from HuggingFace if not already cached."""
    from huggingface_hub import hf_hub_download

    files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']
    path = None
    for f in files:
        p = hf_hub_download("openai/privacy-filter", f)
        if path is None:
            path = str(Path(p).parent)
    print(f"Model at: {path}")
    return path


def build_rope_tables(max_seq_len):
    """Build YaRN-scaled RoPE cos/sin tables."""
    d_half = HEAD_DIM // 2
    freq = ROPE_THETA ** (
        np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM
    )

    if ROPE_SCALING_FACTOR > 1.0:
        concentration = 0.1 * math.log(ROPE_SCALING_FACTOR) + 1.0
        d_half_f = HEAD_DIM / 2
        low = (d_half_f * math.log(INITIAL_CONTEXT_LENGTH / (ROPE_NTK_BETA * 2 * math.pi))
               / math.log(ROPE_THETA))
        high = (d_half_f * math.log(INITIAL_CONTEXT_LENGTH / (ROPE_NTK_ALPHA * 2 * math.pi))
                / math.log(ROPE_THETA))

        interpolation = 1.0 / (ROPE_SCALING_FACTOR * freq)
        extrapolation = 1.0 / freq

        ramp = (np.arange(d_half_f, dtype=np.float32) - low) / (high - low)
        mask = 1.0 - np.clip(ramp, 0, 1)

        inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
        concentration = 1.0
        inv_freq = 1.0 / freq

    positions = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    rope_cos = (np.cos(freqs) * concentration).astype(np.float16)
    rope_sin = (np.sin(freqs) * concentration).astype(np.float16)
    return rope_cos, rope_sin


# ─── PyTorch Model for ANE ───────────────────────────────────────────────────

class RMSNormANE(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.eps = RMS_NORM_EPS
        self.weight = nn.Parameter(weight.float().reshape(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        # x: (B, C, 1, T)
        xf = x.float()
        variance = xf.pow(2).mean(dim=1, keepdim=True)
        x_normed = xf * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).half()


class AttentionLayerANE(nn.Module):
    def __init__(self, layer_idx, weights, seq_len):
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.nh = N_HEADS
        self.nkv = N_KV_HEADS
        self.dh = HEAD_DIM
        self.hpk = N_HEADS // N_KV_HEADS  # 7
        self.seq_len = seq_len

        # Norm
        self.norm = RMSNormANE(weights[f"{prefix}.input_layernorm.weight"])

        # Fused QKV as Conv2d(1×1) — q:(896,640), k:(128,640), v:(128,640)
        q_w = weights[f"{prefix}.self_attn.q_proj.weight"].half()  # (896, 640)
        k_w = weights[f"{prefix}.self_attn.k_proj.weight"].half()  # (128, 640)
        v_w = weights[f"{prefix}.self_attn.v_proj.weight"].half()  # (128, 640)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)  # (1152, 640)
        qkv_dim = qkv_w.shape[0]

        q_b = weights[f"{prefix}.self_attn.q_proj.bias"].half()
        k_b = weights[f"{prefix}.self_attn.k_proj.bias"].half()
        v_b = weights[f"{prefix}.self_attn.v_proj.bias"].half()
        qkv_b = torch.cat([q_b, k_b, v_b])

        self.qkv_conv = nn.Conv2d(D_MODEL, qkv_dim, 1, bias=True)
        self.qkv_conv.weight = nn.Parameter(qkv_w.reshape(qkv_dim, D_MODEL, 1, 1), requires_grad=False)
        self.qkv_conv.bias = nn.Parameter(qkv_b, requires_grad=False)

        # Output proj: (640, 896) as Conv2d
        o_w = weights[f"{prefix}.self_attn.o_proj.weight"].half()  # (640, 896)
        o_b = weights[f"{prefix}.self_attn.o_proj.bias"].half()
        self.out_conv = nn.Conv2d(N_HEADS * HEAD_DIM, D_MODEL, 1, bias=True)
        self.out_conv.weight = nn.Parameter(o_w.reshape(D_MODEL, N_HEADS * HEAD_DIM, 1, 1), requires_grad=False)
        self.out_conv.bias = nn.Parameter(o_b, requires_grad=False)

        # Sinks (precompute sink_scores * ln2 in fp16)
        sinks_raw = weights[f"{prefix}.self_attn.sinks"].float()
        self.register_buffer("sink_scores", (sinks_raw * math.log(2.0)).half())

    def forward(self, x, rope_cos, rope_sin, attn_mask):
        """
        x: (1, D, 1, T), rope_cos/sin: (T, d_half), attn_mask: (T, T)
        """
        T = self.seq_len
        residual = x

        normed = self.norm(x)
        qkv = self.qkv_conv(normed)  # (1, 1152, 1, T)
        qkv = qkv.squeeze(2).permute(0, 2, 1)  # (1, T, 1152)

        q_dim = self.nh * self.dh  # 896
        kv_dim = self.nkv * self.dh  # 128
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim:q_dim + kv_dim]
        v = qkv[:, :, q_dim + kv_dim:]

        # RoPE — interleaved format (x[..., ::2], x[..., 1::2])
        d_half = self.dh // 2

        def apply_rope(x_flat, n_heads):
            x_r = x_flat.reshape(1, T, n_heads, self.dh)
            x1 = x_r[..., ::2]
            x2 = x_r[..., 1::2]
            cos = rope_cos[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_half)
            sin = rope_sin[:T].unsqueeze(0).unsqueeze(2)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.stack((o1, o2), dim=-1).reshape(1, T, n_heads * self.dh)

        q = apply_rope(q, self.nh)
        k = apply_rope(k, self.nkv)

        # QK scaling (split: qk_scale = 1/sqrt(sqrt(dh)))
        qk_scale = 1.0 / math.sqrt(math.sqrt(self.dh))
        q = q * qk_scale
        k = k * qk_scale

        # GQA attention
        q = q.reshape(1, T, self.nkv, self.hpk, self.dh)
        k = k.reshape(1, T, self.nkv, self.dh)
        v = v.reshape(1, T, self.nkv, self.dh)

        # scores: (1, nkv, hpk, T, T)
        q_p = q.permute(0, 2, 3, 1, 4)  # (1, nkv, hpk, T, dh)
        k_p = k.permute(0, 2, 3, 1).unsqueeze(2)  # (1, nkv, 1, dh, T)
        scores = torch.matmul(q_p, k_p)  # (1, nkv, hpk, T, T)

        # Apply mask
        scores = scores + attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Sink: append extra column (precomputed sink_scores in fp16)
        sink_col = self.sink_scores.reshape(1, self.nkv, self.hpk, 1, 1).expand(1, -1, -1, T, 1)
        scores_with_sink = torch.cat([scores.float(), sink_col.float()], dim=-1)

        attn_w = torch.softmax(scores_with_sink, dim=-1)
        attn_w = attn_w[..., :-1].half()  # drop sink, (1, nkv, hpk, T, T)

        # Weighted sum
        v_p = v.permute(0, 2, 1, 3)  # (1, nkv, T, dh)
        attn_out = torch.matmul(attn_w, v_p.unsqueeze(2))  # (1, nkv, hpk, T, dh)

        # Reshape to (1, T, nh*dh) then (1, nh*dh, 1, T)
        attn_out = attn_out.permute(0, 3, 1, 2, 4).reshape(1, T, self.nh * self.dh)
        attn_out = attn_out.permute(0, 2, 1).unsqueeze(2)  # (1, 896, 1, T)

        proj = self.out_conv(attn_out)  # (1, 640, 1, T)
        return residual + proj


class MoEBlockANE(nn.Module):
    def __init__(self, layer_idx, weights, seq_len):
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.seq_len = seq_len

        self.norm = RMSNormANE(weights[f"{prefix}.post_attention_layernorm.weight"])

        # Router
        gate_w = weights[f"{prefix}.mlp.router.weight"].half()  # (128, 640)
        gate_b = weights[f"{prefix}.mlp.router.bias"].half()    # (128,)
        self.register_buffer("gate_weight", gate_w)
        self.register_buffer("gate_bias", gate_b)

        # Experts: gate_up_proj (128, 640, 1280), down_proj (128, 640, 640)
        self.register_buffer("mlp1_weight",
            weights[f"{prefix}.mlp.experts.gate_up_proj"].half())  # (128, 640, 1280)
        self.register_buffer("mlp1_bias",
            weights[f"{prefix}.mlp.experts.gate_up_proj_bias"].half())  # (128, 1280)
        self.register_buffer("mlp2_weight",
            weights[f"{prefix}.mlp.experts.down_proj"].half())  # (128, 640, 640)
        self.register_buffer("mlp2_bias",
            weights[f"{prefix}.mlp.experts.down_proj_bias"].half())  # (128, 640)

    def forward(self, x):
        """x: (1, D, 1, T) → (1, D, 1, T)

        Sparse MoE: sort router logits, take top-4, gather expert weights,
        run matmuls only for selected experts. Uses index_select which CoreML supports.
        """
        T = self.seq_len
        D = D_MODEL
        residual = x

        normed = self.norm(x)
        t = normed.squeeze(2).permute(0, 2, 1).reshape(T, D)  # (T, D)

        # Router (fp16 throughout to avoid CoreML cast overflow)
        g = F.linear(t, self.gate_weight, self.gate_bias)  # (T, 128)
        sorted_vals, sorted_idx = torch.sort(g, dim=-1, descending=True)
        top_vals = sorted_vals[:, :EXPERTS_PER_TOKEN]  # (T, 4)
        top_idx = sorted_idx[:, :EXPERTS_PER_TOKEN]    # (T, 4)
        expert_weights = torch.softmax(top_vals.float(), dim=-1).half()  # (T, 4)

        # Gather selected expert weights
        flat_idx = top_idx.reshape(-1)  # (T*4,)
        mlp1_w_sel = torch.index_select(self.mlp1_weight, 0, flat_idx)  # (T*4, D, 2I)
        mlp1_b_sel = torch.index_select(self.mlp1_bias, 0, flat_idx)    # (T*4, 2I)
        mlp2_w_sel = torch.index_select(self.mlp2_weight, 0, flat_idx)  # (T*4, I, D)
        mlp2_b_sel = torch.index_select(self.mlp2_bias, 0, flat_idx)    # (T*4, D)

        # Reshape for batched matmul
        mlp1_w_sel = mlp1_w_sel.reshape(T, EXPERTS_PER_TOKEN, D, 2 * INTERMEDIATE_SIZE)
        mlp1_b_sel = mlp1_b_sel.reshape(T, EXPERTS_PER_TOKEN, 2 * INTERMEDIATE_SIZE)
        mlp2_w_sel = mlp2_w_sel.reshape(T, EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE, D)
        mlp2_b_sel = mlp2_b_sel.reshape(T, EXPERTS_PER_TOKEN, D)

        # MLP1: (T, 1, 1, D) @ (T, 4, D, 2I) -> (T, 4, 1, 2I)
        t_exp = t.unsqueeze(1).unsqueeze(2)  # (T, 1, 1, D)
        h = torch.matmul(t_exp, mlp1_w_sel)  # (T, 4, 1, 2I)
        h = h.squeeze(2) + mlp1_b_sel  # (T, 4, 2I)

        # SwiGLU (use float for sigmoid stability only)
        h_glu = h[..., :INTERMEDIATE_SIZE].clamp(max=SWIGLU_LIMIT)
        h_linear = h[..., INTERMEDIATE_SIZE:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h_act = (h_glu * torch.sigmoid(1.702 * h_glu.float()).half() * (h_linear + 1.0))  # (T, 4, I)

        # MLP2: (T, 4, 1, I) @ (T, 4, I, D) -> (T, 4, 1, D)
        h_act_exp = h_act.unsqueeze(2)  # (T, 4, 1, I)
        out = torch.matmul(h_act_exp, mlp2_w_sel)  # (T, 4, 1, D)
        out = out.squeeze(2) + mlp2_b_sel  # (T, 4, D)

        # Weighted sum
        out = (out * expert_weights.unsqueeze(-1)).sum(dim=1)  # (T, D)

        # Reshape back
        out = out.reshape(1, T, D).permute(0, 2, 1).unsqueeze(2)
        return residual + out


class PrivacyFilterANE(nn.Module):
    def __init__(self, weights, seq_len):
        super().__init__()
        self.seq_len = seq_len

        # Embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.embedding.weight = nn.Parameter(
            weights["model.embed_tokens.weight"].half(), requires_grad=False)

        # Transformer blocks
        self.attn_layers = nn.ModuleList()
        self.moe_layers = nn.ModuleList()
        for i in range(N_LAYERS):
            print(f"  Layer {i}/{N_LAYERS}...")
            self.attn_layers.append(AttentionLayerANE(i, weights, seq_len))
            self.moe_layers.append(MoEBlockANE(i, weights, seq_len))

        # Final norm + classification head
        self.output_norm = RMSNormANE(weights["model.norm.weight"])

        cls_w = weights["score.weight"].half()  # (33, 640)
        cls_b = weights["score.bias"].half()    # (33,)
        self.cls_conv = nn.Conv2d(D_MODEL, NUM_LABELS, 1, bias=True)
        self.cls_conv.weight = nn.Parameter(
            cls_w.reshape(NUM_LABELS, D_MODEL, 1, 1), requires_grad=False)
        self.cls_conv.bias = nn.Parameter(cls_b, requires_grad=False)

    def forward(self, token_ids, rope_cos, rope_sin, attn_mask):
        """
        token_ids: (1, T) int32
        rope_cos:  (T, d_half) fp16
        rope_sin:  (T, d_half) fp16
        attn_mask: (T, T) fp16
        Returns: (1, NUM_LABELS, T) fp16
        """
        T = self.seq_len
        x = self.embedding(token_ids)  # (1, T, D)
        x = x.permute(0, 2, 1).unsqueeze(2)  # (1, D, 1, T)

        for i in range(N_LAYERS):
            x = self.attn_layers[i](x, rope_cos, rope_sin, attn_mask)
            x = self.moe_layers[i](x)

        x = self.output_norm(x)
        logits = self.cls_conv(x)  # (1, 33, 1, T)
        return logits.squeeze(2)   # (1, 33, T)


# ─── Main Build ─────────────────────────────────────────────────────────────

def build_model(checkpoint_dir, max_seq_len=512, quantize=False):
    import coremltools as ct

    print(f"Loading checkpoint: {checkpoint_dir}")
    weights = load_weights(checkpoint_dir)
    print(f"  Loaded {len(weights)} tensors")

    print(f"Config: {N_LAYERS}L, d={D_MODEL}, nh={N_HEADS}, nkv={N_KV_HEADS}, "
          f"dh={HEAD_DIM}, experts={N_EXPERTS}, top-{EXPERTS_PER_TOKEN}")
    print(f"Seq len: {max_seq_len}, Float16, bidirectional banded attention")

    # RoPE tables
    rope_cos, rope_sin = build_rope_tables(max_seq_len)

    # Banded attention mask
    print("Building banded attention mask...")
    mask = np.full((max_seq_len, max_seq_len), -1e4, dtype=np.float16)
    for i in range(max_seq_len):
        left = max(0, i - SLIDING_WINDOW)
        right = min(max_seq_len, i + SLIDING_WINDOW + 1)
        mask[i, left:right] = 0.0
    attn_mask_tensor = torch.tensor(mask, dtype=torch.float16)

    # Build model
    print(f"\nBuilding PrivacyFilterANE ({N_LAYERS}L, fp16)...")
    model = PrivacyFilterANE(weights, max_seq_len)
    model.half()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for b in model.buffers())
    print(f"  {n_params:,} parameters")

    # Free raw weights
    del weights

    # Trace
    d_half = HEAD_DIM // 2
    token_ex = torch.randint(0, VOCAB_SIZE, (1, max_seq_len), dtype=torch.int32)
    cos_ex = torch.tensor(rope_cos, dtype=torch.float16)
    sin_ex = torch.tensor(rope_sin, dtype=torch.float16)

    print("\nTracing...")
    with torch.no_grad():
        outputs = model(token_ex, cos_ex, sin_ex, attn_mask_tensor)
        print(f"  Output: {outputs.shape}, dtype: {outputs.dtype}")

    traced = torch.jit.trace(model, (token_ex, cos_ex, sin_ex, attn_mask_tensor))

    # Convert to CoreML
    print(f"\nConverting to CoreML...")
    ct_inputs = [
        ct.TensorType(name="token_ids", shape=(1, max_seq_len), dtype=np.int32),
        ct.TensorType(name="rope_cos", shape=(max_seq_len, d_half), dtype=np.float16),
        ct.TensorType(name="rope_sin", shape=(max_seq_len, d_half), dtype=np.float16),
        ct.TensorType(name="attn_mask", shape=(max_seq_len, max_seq_len), dtype=np.float16),
    ]
    ct_outputs = [ct.TensorType(name="logits", dtype=np.float16)]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        # FLOAT32 compute required: fp16 produces all-zeros for 8-layer model
        # due to CoreML MIL optimization pass bug with large gather+matmul graphs.
        # Weights are still stored as fp16; only intermediate compute uses fp32.
        compute_precision=ct.precision.FLOAT32,
    )

    if quantize:
        print("\nQuantizing weights to int8...")
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
        )
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=opt_config)

    # Save
    prefix = f"PrivacyFilterANE_{max_seq_len}"
    pkg_path = f"{prefix}.mlpackage"
    mlmodel.save(pkg_path)
    print(f"\n  Saved {pkg_path}")

    # Metadata
    meta = {
        "model": "openai/privacy-filter",
        "task": "token-classification",
        "num_labels": NUM_LABELS,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "n_layers": N_LAYERS,
        "n_experts": N_EXPERTS,
        "experts_per_token": EXPERTS_PER_TOKEN,
        "intermediate_size": INTERMEDIATE_SIZE,
        "sliding_window": SLIDING_WINDOW,
        "max_seq_len": max_seq_len,
        "vocab_size": VOCAB_SIZE,
        "dtype": "float16",
        "quantized": quantize,
    }
    meta_path = f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta: {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Model:      OpenAI Privacy Filter")
    print(f"  Format:     CoreML (iOS18+, CPU_AND_NE)")
    print(f"  Seq len:    {max_seq_len} (fixed)")
    print(f"  Attention:  Bidirectional banded (window={2*SLIDING_WINDOW+1})")
    print(f"  FFN:        MoE ({N_EXPERTS} experts, top-{EXPERTS_PER_TOKEN})")
    print(f"  Output:     {NUM_LABELS} PII classes")
    print(f"{'='*60}")
    return pkg_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenAI Privacy Filter to ANE-native CoreML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint dir (default: download from HF)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Fixed sequence length (default: 512)")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply int8 weight quantization")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint
    if checkpoint_dir is None:
        print("No checkpoint specified, downloading from HuggingFace...")
        checkpoint_dir = download_model()

    build_model(checkpoint_dir, max_seq_len=args.seq_len, quantize=args.quantize)


if __name__ == "__main__":
    main()
