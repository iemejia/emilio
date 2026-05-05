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


def load_safetensors_weights(checkpoint_dir):
    """Load weights from safetensors checkpoint."""
    from safetensors import safe_open

    weights = {}
    st_path = Path(checkpoint_dir) / "model.safetensors"
    if not st_path.exists():
        # Try multi-file sharded format
        import glob
        st_files = sorted(glob.glob(str(Path(checkpoint_dir) / "model-*.safetensors")))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
    else:
        st_files = [str(st_path)]

    for f in st_files:
        with safe_open(f, framework="numpy") as sf:
            for key in sf.keys():
                weights[key] = sf.get_tensor(key)
    return weights


def download_model():
    """Download the model from HuggingFace if not already cached."""
    from huggingface_hub import snapshot_download

    path = snapshot_download("openai/privacy-filter")
    print(f"Model downloaded to: {path}")
    return path


def build_rope_tables(max_seq_len):
    """Build YaRN-scaled RoPE cos/sin tables."""
    d_half = HEAD_DIM // 2
    freq = ROPE_THETA ** (
        np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM
    )

    # YaRN NTK-by-parts scaling
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


def build_model(checkpoint_dir, max_seq_len=512, quantize=False):
    """Build CoreML model from Privacy Filter checkpoint."""
    import coremltools as ct
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print(f"Loading checkpoint: {checkpoint_dir}")
    weights = load_safetensors_weights(checkpoint_dir)

    print(f"Config: {N_LAYERS}L, d={D_MODEL}, nh={N_HEADS}, nkv={N_KV_HEADS}, "
          f"dh={HEAD_DIM}, experts={N_EXPERTS}, top-{EXPERTS_PER_TOKEN}")
    print(f"Seq len: {max_seq_len}, Float16, bidirectional banded attention")

    # ── RoPE tables ──
    rope_cos, rope_sin = build_rope_tables(max_seq_len)

    # ── PyTorch model ────────────────────────────────────────────────────

    class RMSNormANE(nn.Module):
        """RMSNorm operating on (B, C, 1, T) tensor layout for ANE."""
        def __init__(self, weight):
            super().__init__()
            self.eps = RMS_NORM_EPS
            # weight: (C, 1, 1) for broadcasting over (B, C, 1, T)
            self.weight = nn.Parameter(
                torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # x: (B, C, 1, T)
            xf = x.float()
            variance = xf.pow(2).mean(dim=1, keepdim=True)
            x_normed = xf * torch.rsqrt(variance + self.eps)
            return (x_normed * self.weight.float()).half()

    class AttentionLayerANE(nn.Module):
        """Bidirectional banded attention with GQA. Conv2d projections."""
        def __init__(self, layer_idx, weights_dict):
            super().__init__()
            prefix = f"block.{layer_idx}.attn"
            self.d = D_MODEL
            self.nh = N_HEADS
            self.nkv = N_KV_HEADS
            self.dh = HEAD_DIM
            self.hpk = N_HEADS // N_KV_HEADS  # 7
            self.scale = 1.0 / HEAD_DIM  # qk_scale^2

            # Norm
            norm_w = weights_dict[f"{prefix}.norm.scale"]
            self.norm = RMSNormANE(norm_w)

            # QKV projection as Conv2d(1×1)
            qkv_dim = HEAD_DIM * (N_HEADS + 2 * N_KV_HEADS)  # 64*(14+4) = 1152
            qkv_w = weights_dict[f"{prefix}.qkv.weight"]  # (qkv_dim, d_model)
            qkv_b = weights_dict[f"{prefix}.qkv.bias"]    # (qkv_dim,)
            self.qkv_conv = nn.Conv2d(D_MODEL, qkv_dim, 1, bias=True)
            self.qkv_conv.weight = nn.Parameter(
                torch.tensor(qkv_w, dtype=torch.float16).reshape(qkv_dim, D_MODEL, 1, 1),
                requires_grad=False)
            self.qkv_conv.bias = nn.Parameter(
                torch.tensor(qkv_b, dtype=torch.float16), requires_grad=False)

            # Output projection as Conv2d(1×1)
            out_w = weights_dict[f"{prefix}.out.weight"]  # (d_model, nh*dh)
            out_b_key = f"{prefix}.out.bias"
            self.out_conv = nn.Conv2d(N_HEADS * HEAD_DIM, D_MODEL, 1,
                                      bias=out_b_key in weights_dict)
            self.out_conv.weight = nn.Parameter(
                torch.tensor(out_w, dtype=torch.float16).reshape(D_MODEL, N_HEADS * HEAD_DIM, 1, 1),
                requires_grad=False)
            if out_b_key in weights_dict:
                self.out_conv.bias = nn.Parameter(
                    torch.tensor(weights_dict[out_b_key], dtype=torch.float16),
                    requires_grad=False)

            # Sink logits
            sink_w = weights_dict[f"{prefix}.sinks"]  # (n_heads,)
            self.register_buffer("sinks",
                torch.tensor(sink_w, dtype=torch.float32))

        def forward(self, x, rope_cos, rope_sin, attn_mask):
            """
            x: (B, D, 1, T) fp16
            rope_cos: (T, d_half) fp16
            rope_sin: (T, d_half) fp16
            attn_mask: (T, T) fp16 — banded mask (0=valid, -inf=masked)
            Returns: (B, D, 1, T) fp16
            """
            B = x.shape[0]
            T = x.shape[3]
            residual = x

            normed = self.norm(x)
            # QKV: (B, qkv_dim, 1, T)
            qkv = self.qkv_conv(normed)
            qkv = qkv.squeeze(2)  # (B, qkv_dim, T)
            qkv = qkv.permute(0, 2, 1)  # (B, T, qkv_dim)

            q = qkv[:, :, :self.nh * self.dh]  # (B, T, nh*dh)
            k = qkv[:, :, self.nh * self.dh:(self.nh + self.nkv) * self.dh]
            v = qkv[:, :, (self.nh + self.nkv) * self.dh:]

            # Apply RoPE
            d_half = self.dh // 2
            def apply_rope(x_flat, n_heads):
                # x_flat: (B, T, n_heads*dh)
                x_r = x_flat.reshape(B, T, n_heads, self.dh)
                x1 = x_r[..., ::2]   # (B, T, nh, d_half)
                x2 = x_r[..., 1::2]
                cos = rope_cos[:T, :].unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_half)
                sin = rope_sin[:T, :].unsqueeze(0).unsqueeze(2)
                o1 = x1 * cos - x2 * sin
                o2 = x2 * cos + x1 * sin
                out = torch.stack((o1, o2), dim=-1).reshape(B, T, n_heads, self.dh)
                return out.reshape(B, T, n_heads * self.dh)

            q = apply_rope(q, self.nh)
            k = apply_rope(k, self.nkv)

            # Scale Q and K (qk_scale = 1/sqrt(sqrt(dh)))
            qk_scale = 1.0 / math.sqrt(math.sqrt(self.dh))
            q = q * qk_scale
            k = k * qk_scale

            # Reshape for GQA attention
            # q: (B, T, nkv, hpk, dh), k: (B, T, nkv, dh), v: (B, T, nkv, dh)
            q = q.reshape(B, T, self.nkv, self.hpk, self.dh)
            k = k.reshape(B, T, self.nkv, self.dh)
            v = v.reshape(B, T, self.nkv, self.dh)

            # Compute attention scores: Q*K^T
            # scores: (B, T, nkv, hpk, T)
            scores = torch.einsum("btghd,bsghd->btghs",
                                  q, k.unsqueeze(3).expand(-1, -1, -1, self.hpk, -1))
            # Wait — more efficient: einsum("btnhd,bsnd->btnhs", q_reshaped, k_reshaped)
            # Let me redo: q(B,T,nkv,hpk,dh) @ k(B,S,nkv,dh) -> (B,nkv,hpk,T,S)
            q_p = q.permute(0, 2, 3, 1, 4)  # (B, nkv, hpk, T, dh)
            k_p = k.permute(0, 2, 1, 4).unsqueeze(2)  # (B, nkv, 1, dh, T) -- wait
            # Actually: k is (B, T, nkv, dh) -> (B, nkv, T, dh) -> transpose -> (B, nkv, dh, T)
            k_p = k.permute(0, 2, 3, 1)  # (B, nkv, dh, T)
            scores = torch.matmul(q_p, k_p)  # (B, nkv, hpk, T, T)

            # Apply banded attention mask + sink
            scores = scores.float()
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # Add sink scores
            sink_scores = (self.sinks * math.log(2.0))  # (n_heads,)
            sink_scores = sink_scores.reshape(self.nkv, self.hpk, 1, 1)
            # Append sink as extra "position"
            sink_expanded = sink_scores.expand(B, -1, -1, T, 1)
            scores_with_sink = torch.cat([scores, sink_expanded], dim=-1)

            attn_w = torch.softmax(scores_with_sink, dim=-1)
            attn_w = attn_w[..., :-1].half()  # drop sink column

            # Weighted sum of values
            v_p = v.permute(0, 2, 1, 3)  # (B, nkv, T, dh)
            # attn_w: (B, nkv, hpk, T, T) @ v_p: (B, nkv, T, dh)
            # -> need v_p as (B, nkv, 1, T, dh)
            attn_out = torch.matmul(attn_w, v_p.unsqueeze(2))  # (B, nkv, hpk, T, dh)

            # Reshape: (B, T, nh*dh)
            attn_out = attn_out.permute(0, 3, 1, 2, 4).reshape(B, T, self.nh * self.dh)

            # Output projection: need (B, nh*dh, 1, T)
            attn_out = attn_out.permute(0, 2, 1).unsqueeze(2)  # (B, nh*dh, 1, T)
            proj = self.out_conv(attn_out)
            return residual + proj

    class MoEBlockANE(nn.Module):
        """Sparse MoE FFN block. Top-4 routing from 128 experts."""
        def __init__(self, layer_idx, weights_dict):
            super().__init__()
            prefix = f"block.{layer_idx}.mlp"

            norm_w = weights_dict[f"{prefix}.norm.scale"]
            self.norm = RMSNormANE(norm_w)

            # Gate/router
            gate_w = weights_dict[f"{prefix}.gate.weight"]  # (n_experts, d_model)
            gate_b = weights_dict.get(f"{prefix}.gate.bias")
            self.gate_weight = nn.Parameter(
                torch.tensor(gate_w, dtype=torch.float16), requires_grad=False)
            if gate_b is not None:
                self.gate_bias = nn.Parameter(
                    torch.tensor(gate_b, dtype=torch.float16), requires_grad=False)
            else:
                self.gate_bias = None

            # Expert weights: mlp1 (up+gate), mlp2 (down)
            mlp1_w = weights_dict[f"{prefix}.mlp1_weight"]  # (n_experts, d_model, 2*intermediate)
            mlp1_b = weights_dict[f"{prefix}.mlp1_bias"]    # (n_experts, 2*intermediate)
            mlp2_w = weights_dict[f"{prefix}.mlp2_weight"]  # (n_experts, intermediate, d_model)
            mlp2_b = weights_dict[f"{prefix}.mlp2_bias"]    # (n_experts, d_model)

            self.register_buffer("mlp1_weight",
                torch.tensor(mlp1_w, dtype=torch.float16))
            self.register_buffer("mlp1_bias",
                torch.tensor(mlp1_b, dtype=torch.float16))
            self.register_buffer("mlp2_weight",
                torch.tensor(mlp2_w, dtype=torch.float16))
            self.register_buffer("mlp2_bias",
                torch.tensor(mlp2_b, dtype=torch.float16))

        def forward(self, x):
            """
            x: (B, D, 1, T) fp16
            Returns: (B, D, 1, T) fp16
            """
            B, D, _, T = x.shape
            residual = x

            normed = self.norm(x)  # (B, D, 1, T)
            # Reshape to (B*T, D) for routing
            t = normed.squeeze(2).permute(0, 2, 1).reshape(B * T, D)  # (BT, D)

            # Router
            g = F.linear(t.float(), self.gate_weight.float(),
                        self.gate_bias.float() if self.gate_bias is not None else None)
            top_vals, top_idx = torch.topk(g, k=EXPERTS_PER_TOKEN, dim=-1)
            expert_weights = torch.softmax(top_vals, dim=-1) / EXPERTS_PER_TOKEN

            # Batched expert computation
            # Gather expert weights for selected experts
            # mlp1_weight[top_idx]: (BT, 4, D, 2*I)
            mlp1_w = self.mlp1_weight[top_idx.reshape(-1)]  # (BT*4, D, 2*I)
            mlp1_b = self.mlp1_bias[top_idx.reshape(-1)]    # (BT*4, 2*I)

            t_expanded = t.unsqueeze(1).expand(-1, EXPERTS_PER_TOKEN, -1)  # (BT, 4, D)
            t_flat = t_expanded.reshape(B * T * EXPERTS_PER_TOKEN, 1, D)  # (BT*4, 1, D)

            # MLP1: (BT*4, 1, D) @ (BT*4, D, 2*I) -> (BT*4, 1, 2*I)
            h = torch.bmm(t_flat.float(), mlp1_w.float())
            h = h.squeeze(1) + mlp1_b.float()  # (BT*4, 2*I)

            # SwiGLU
            h_glu = h[..., :INTERMEDIATE_SIZE]
            h_linear = h[..., INTERMEDIATE_SIZE:]
            h_glu = h_glu.clamp(max=SWIGLU_LIMIT)
            h_linear = h_linear.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            h_act = h_glu * torch.sigmoid(1.702 * h_glu) * (h_linear + 1.0)

            # MLP2
            mlp2_w = self.mlp2_weight[top_idx.reshape(-1)]  # (BT*4, I, D)
            mlp2_b = self.mlp2_bias[top_idx.reshape(-1)]    # (BT*4, D)
            h_act = h_act.unsqueeze(1)  # (BT*4, 1, I)
            out = torch.bmm(h_act, mlp2_w.float())  # (BT*4, 1, D)
            out = out.squeeze(1) + mlp2_b.float()   # (BT*4, D)

            # Weighted sum
            out = out.reshape(B * T, EXPERTS_PER_TOKEN, D)
            expert_weights_expanded = expert_weights.unsqueeze(-1)  # (BT, 4, 1)
            out = (out * expert_weights_expanded).sum(dim=1)  # (BT, D)
            out = out * EXPERTS_PER_TOKEN  # rescale

            # Reshape back
            out = out.half().reshape(B, T, D).permute(0, 2, 1).unsqueeze(2)  # (B, D, 1, T)
            return residual + out

    class PrivacyFilterANE(nn.Module):
        """Full Privacy Filter model for ANE."""
        def __init__(self, weights_dict):
            super().__init__()

            # Token embedding
            embd_w = weights_dict["embedding.weight"]  # (vocab, d_model)
            self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.embedding.weight = nn.Parameter(
                torch.tensor(embd_w, dtype=torch.float16), requires_grad=False)

            # Transformer blocks
            self.attn_layers = nn.ModuleList()
            self.moe_layers = nn.ModuleList()
            for i in range(N_LAYERS):
                print(f"  Loading layer {i}/{N_LAYERS}...")
                self.attn_layers.append(AttentionLayerANE(i, weights_dict))
                self.moe_layers.append(MoEBlockANE(i, weights_dict))

            # Final norm
            out_norm_w = weights_dict["norm.scale"]
            self.output_norm = RMSNormANE(out_norm_w)

            # Classification head (not LM head)
            cls_w = weights_dict["unembedding.weight"]  # (num_labels, d_model)
            self.cls_conv = nn.Conv2d(D_MODEL, NUM_LABELS, 1, bias=False)
            self.cls_conv.weight = nn.Parameter(
                torch.tensor(cls_w, dtype=torch.float16).reshape(NUM_LABELS, D_MODEL, 1, 1),
                requires_grad=False)

        def forward(self, token_ids, rope_cos, rope_sin, attn_mask):
            """
            token_ids: (1, T) int32
            rope_cos:  (T, d_half) fp16
            rope_sin:  (T, d_half) fp16
            attn_mask:  (T, T) fp16
            Returns: (1, NUM_LABELS, T) fp16 — per-token logits
            """
            # Embedding lookup
            x = self.embedding(token_ids)  # (1, T, D)
            # Reshape to ANE layout: (1, D, 1, T)
            x = x.permute(0, 2, 1).unsqueeze(2)

            for i in range(N_LAYERS):
                x = self.attn_layers[i](x, rope_cos, rope_sin, attn_mask)
                x = self.moe_layers[i](x)

            x = self.output_norm(x)
            logits = self.cls_conv(x)  # (1, NUM_LABELS, 1, T)
            return logits.squeeze(2)   # (1, NUM_LABELS, T)

    # ── Build attention mask ─────────────────────────────────────────────

    print("Building banded attention mask...")
    mask = np.full((max_seq_len, max_seq_len), -1e4, dtype=np.float16)
    for i in range(max_seq_len):
        left = max(0, i - SLIDING_WINDOW)
        right = min(max_seq_len, i + SLIDING_WINDOW + 1)
        mask[i, left:right] = 0.0
    attn_mask_tensor = torch.tensor(mask, dtype=torch.float16)

    # ── Map weight names ─────────────────────────────────────────────────

    print("Mapping weight names...")
    mapped_weights = {}
    for key, val in weights.items():
        # HF transformers format -> our internal format
        new_key = key
        # Common prefix removals for HF format
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        # Map "layers.X.self_attn.*" -> "block.X.attn.*"
        new_key = new_key.replace("layers.", "block.")
        new_key = new_key.replace("self_attn.", "attn.")
        new_key = new_key.replace("input_layernorm.weight", "attn.norm.scale")
        new_key = new_key.replace("post_attention_layernorm.weight", "mlp.norm.scale")
        new_key = new_key.replace("attn.q_proj.", "attn.qkv_q.")
        new_key = new_key.replace("attn.k_proj.", "attn.qkv_k.")
        new_key = new_key.replace("attn.v_proj.", "attn.qkv_v.")
        new_key = new_key.replace("attn.o_proj.", "attn.out.")
        new_key = new_key.replace("embed_tokens.weight", "embedding.weight")
        new_key = new_key.replace("model_norm.weight", "norm.scale")
        new_key = new_key.replace("classifier.weight", "unembedding.weight")
        # MoE mappings
        new_key = new_key.replace("block_sparse_moe.gate.", "mlp.gate.")
        new_key = new_key.replace("block_sparse_moe.", "mlp.")
        mapped_weights[new_key] = val

    # Check what keys we actually have
    print(f"  Found {len(mapped_weights)} weight tensors")
    sample_keys = sorted(mapped_weights.keys())[:20]
    print(f"  Sample keys: {sample_keys}")

    # ── Fuse QKV if separate ─────────────────────────────────────────────

    for i in range(N_LAYERS):
        q_key = f"block.{i}.attn.qkv_q.weight"
        k_key = f"block.{i}.attn.qkv_k.weight"
        v_key = f"block.{i}.attn.qkv_v.weight"
        fused_key = f"block.{i}.attn.qkv.weight"

        if q_key in mapped_weights and fused_key not in mapped_weights:
            q_w = mapped_weights.pop(q_key)
            k_w = mapped_weights.pop(k_key)
            v_w = mapped_weights.pop(v_key)
            mapped_weights[fused_key] = np.concatenate([q_w, k_w, v_w], axis=0)

            # Bias
            q_b_key = f"block.{i}.attn.qkv_q.bias"
            k_b_key = f"block.{i}.attn.qkv_k.bias"
            v_b_key = f"block.{i}.attn.qkv_v.bias"
            if q_b_key in mapped_weights:
                q_b = mapped_weights.pop(q_b_key)
                k_b = mapped_weights.pop(k_b_key)
                v_b = mapped_weights.pop(v_b_key)
                mapped_weights[f"block.{i}.attn.qkv.bias"] = np.concatenate(
                    [q_b, k_b, v_b])

    # ── Build model ──────────────────────────────────────────────────────

    print(f"\nBuilding PrivacyFilterANE ({N_LAYERS}L, fp16)...")
    model = PrivacyFilterANE(mapped_weights)
    model.half()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for b in model.buffers())
    print(f"  {n_params:,} parameters (fp16)")

    # ── Trace ────────────────────────────────────────────────────────────

    d_half = HEAD_DIM // 2
    token_ex = torch.randint(0, VOCAB_SIZE, (1, max_seq_len), dtype=torch.int32)
    cos_ex = torch.tensor(rope_cos[:max_seq_len], dtype=torch.float16)
    sin_ex = torch.tensor(rope_sin[:max_seq_len], dtype=torch.float16)

    print("\nTracing...")
    with torch.no_grad():
        outputs = model(token_ex, cos_ex, sin_ex, attn_mask_tensor)
        print(f"  logits: {outputs.shape}, dtype: {outputs.dtype}")

    traced = torch.jit.trace(model, (token_ex, cos_ex, sin_ex, attn_mask_tensor))

    # ── Convert to CoreML ────────────────────────────────────────────────

    print(f"\nConverting to CoreML (fp16, {N_LAYERS}L)...")

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
        compute_precision=ct.precision.FLOAT16,
    )

    # ── Post-training int8 weight quantization ───────────────────────────

    if quantize:
        print("\nQuantizing weights to int8...")
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=opt_config)
        print("  Weights quantized to int8")

    # ── Save ─────────────────────────────────────────────────────────────

    prefix = f"PrivacyFilterANE_{max_seq_len}"
    pkg_path = f"{prefix}.mlpackage"
    mlmodel.save(pkg_path)
    print(f"\n  Saved {pkg_path}")

    # ── Metadata ─────────────────────────────────────────────────────────

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
        "id2label": {
            "0": "O",
            "1": "B-account_number", "2": "I-account_number",
            "3": "E-account_number", "4": "S-account_number",
            "5": "B-private_address", "6": "I-private_address",
            "7": "E-private_address", "8": "S-private_address",
            "9": "B-private_date", "10": "I-private_date",
            "11": "E-private_date", "12": "S-private_date",
            "13": "B-private_email", "14": "I-private_email",
            "15": "E-private_email", "16": "S-private_email",
            "17": "B-private_person", "18": "I-private_person",
            "19": "E-private_person", "20": "S-private_person",
            "21": "B-private_phone", "22": "I-private_phone",
            "23": "E-private_phone", "24": "S-private_phone",
            "25": "B-private_url", "26": "I-private_url",
            "27": "E-private_url", "28": "S-private_url",
            "29": "B-secret", "30": "I-secret",
            "31": "E-secret", "32": "S-secret",
        },
    }
    meta_path = f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta: {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Model:      OpenAI Privacy Filter")
    print(f"  Task:       Token Classification (PII detection)")
    print(f"  Format:     CoreML mlprogram (iOS18+)")
    print(f"  Dtype:      Float16 {'+ int8 weights' if quantize else ''}")
    print(f"  Seq len:    {max_seq_len} (fixed, single pass)")
    print(f"  Attention:  Bidirectional banded (window={2*SLIDING_WINDOW+1})")
    print(f"  FFN:        Sparse MoE ({N_EXPERTS} experts, top-{EXPERTS_PER_TOKEN})")
    print(f"  Output:     {NUM_LABELS} classes (BIOES PII labels)")
    print(f"  Target:     ANE (CPU_AND_NE)")
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
