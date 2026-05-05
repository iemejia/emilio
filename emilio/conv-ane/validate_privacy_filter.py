#!/usr/bin/env python3
"""Validate CoreML Privacy Filter against HuggingFace reference.

Compares per-token predictions and logit values between the CoreML ANE model
and the HuggingFace transformers reference implementation.

Requirements: transformers, torch, coremltools, safetensors, numpy
Use with Python 3.12 venv (.venv312).

Usage:
  python3 validate_privacy_filter.py \
    --model PrivacyFilterANE_32.mlpackage \
    --checkpoint /path/to/openai-privacy-filter \
    --seq-len 32
"""

import argparse
import json
import math
import sys

import numpy as np


def build_rope_tables(seq_len, head_dim=64, theta=150000.0, scaling_factor=32.0,
                      ntk_alpha=1.0, ntk_beta=32.0, initial_ctx=4096):
    """Build YaRN-scaled RoPE cos/sin tables (matches privacy_filter_to_ane.py)."""
    d_half = head_dim // 2
    freq = theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)

    concentration = 0.1 * math.log(scaling_factor) + 1.0
    d_half_f = head_dim / 2
    low = d_half_f * math.log(initial_ctx / (ntk_beta * 2 * math.pi)) / math.log(theta)
    high = d_half_f * math.log(initial_ctx / (ntk_alpha * 2 * math.pi)) / math.log(theta)

    interpolation = 1.0 / (scaling_factor * freq)
    extrapolation = 1.0 / freq
    ramp = (np.arange(d_half_f, dtype=np.float32) - low) / (high - low)
    mask = 1.0 - np.clip(ramp, 0, 1)
    inv_freq = interpolation * (1 - mask) + extrapolation * mask

    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    rope_cos = (np.cos(freqs) * concentration).astype(np.float16)
    rope_sin = (np.sin(freqs) * concentration).astype(np.float16)
    return rope_cos, rope_sin


def build_attn_mask(seq_len, real_len=None, window=128):
    """Build banded attention mask with padding.
    
    Positions j >= real_len are treated as padding and masked with -10000.
    If real_len is None, no padding mask is applied (all positions are valid).
    """
    mask = np.full((seq_len, seq_len), -10000.0, dtype=np.float16)
    if real_len is None:
        real_len = seq_len
    for i in range(seq_len):
        left = max(0, i - window)
        right = min(seq_len, i + window + 1)
        for j in range(left, right):
            if j < real_len:
                mask[i, j] = 0.0
    return mask


ID2LABEL = {
    0: "O",
    1: "B-account_number", 2: "I-account_number", 3: "E-account_number", 4: "S-account_number",
    5: "B-private_address", 6: "I-private_address", 7: "E-private_address", 8: "S-private_address",
    9: "B-private_date", 10: "I-private_date", 11: "E-private_date", 12: "S-private_date",
    13: "B-private_email", 14: "I-private_email", 15: "E-private_email", 16: "S-private_email",
    17: "B-private_person", 18: "I-private_person", 19: "E-private_person", 20: "S-private_person",
    21: "B-private_phone", 22: "I-private_phone", 23: "E-private_phone", 24: "S-private_phone",
    25: "B-private_url", 26: "I-private_url", 27: "E-private_url", 28: "S-private_url",
    29: "B-secret", 30: "I-secret", 31: "E-secret", 32: "S-secret",
}


def run_hf_reference(checkpoint_dir, text, seq_len):
    """Run HuggingFace reference model and return token_ids + logits."""
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_dir, dtype=torch.float32)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                       max_length=seq_len, truncation=True)
    token_ids = inputs["input_ids"].numpy().astype(np.int32)

    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0].numpy()  # (seq_len, 33)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return token_ids, logits, tokens


def run_coreml(model_path, token_ids, seq_len, compute_units="CPU_AND_NE"):
    """Run CoreML model and return logits."""
    import coremltools as ct

    units_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL,
    }
    model = ct.models.MLModel(model_path, compute_units=units_map[compute_units])

    rope_cos, rope_sin = build_rope_tables(seq_len)
    
    # Determine real token count (non-padding)
    # Padding token is 199999 for this model
    real_len = seq_len
    for i in range(seq_len):
        if token_ids[0, i] == 199999:
            real_len = i
            break
    
    attn_mask = build_attn_mask(seq_len, real_len=real_len)

    inputs = {
        "token_ids": token_ids,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "attn_mask": attn_mask,
    }

    out = model.predict(inputs)
    logits = out["logits"][0]  # (33, seq_len) — CoreML output is transposed
    return logits.T  # -> (seq_len, 33) to match HF


def compare_outputs(ref_logits, coreml_logits, tokens, seq_len):
    """Compare reference and CoreML outputs."""
    ref_preds = ref_logits.argmax(axis=-1)
    coreml_preds = coreml_logits.argmax(axis=-1)

    # Label agreement
    match = (ref_preds == coreml_preds).sum()
    print(f"\nLabel agreement: {match}/{seq_len} ({100*match/seq_len:.1f}%)")

    # Cosine similarity
    ref_norm = np.linalg.norm(ref_logits)
    coreml_norm = np.linalg.norm(coreml_logits)
    if ref_norm > 0 and coreml_norm > 0:
        cos_sim = np.sum(ref_logits * coreml_logits) / (ref_norm * coreml_norm)
        print(f"Logits cosine similarity: {cos_sim:.6f}")

    # Value ranges
    print(f"Reference logits: [{ref_logits.min():.2f}, {ref_logits.max():.2f}]")
    print(f"CoreML logits:    [{coreml_logits.min():.2f}, {coreml_logits.max():.2f}]")

    # Per-token comparison for PII tokens
    print(f"\nPII detections:")
    print(f"{'Pos':>4} {'Token':<20} {'Reference':<25} {'CoreML':<25} {'Match':>5}")
    print("-" * 80)
    for i in range(seq_len):
        ref_label = ID2LABEL.get(ref_preds[i], "?")
        coreml_label = ID2LABEL.get(coreml_preds[i], "?")
        if ref_label != "O" or coreml_label != "O":
            tok = tokens[i] if i < len(tokens) else "<pad>"
            match_str = "✓" if ref_label == coreml_label else "✗"
            print(f"{i:4d} {tok:<20} {ref_label:<25} {coreml_label:<25} {match_str:>5}")

    # Summary statistics
    ref_pii = sum(1 for p in ref_preds if p != 0)
    coreml_pii = sum(1 for p in coreml_preds if p != 0)
    print(f"\nTotal PII tokens: ref={ref_pii}, coreml={coreml_pii}")

    return match / seq_len


def main():
    parser = argparse.ArgumentParser(description="Validate CoreML Privacy Filter")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .mlpackage")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to HF checkpoint directory")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--text", type=str,
                        default="My name is Alice Smith and my email is alice.smith@example.com")
    parser.add_argument("--compute-units", type=str, default="CPU_AND_NE",
                        choices=["CPU_AND_NE", "CPU_ONLY", "ALL"])
    args = parser.parse_args()

    print(f"Text: \"{args.text}\"")
    print(f"Seq len: {args.seq_len}")
    print(f"Compute units: {args.compute_units}")

    # Run HF reference
    print("\n--- Running HuggingFace reference ---")
    token_ids, ref_logits, tokens = run_hf_reference(
        args.checkpoint, args.text, args.seq_len)
    print(f"Token IDs (first 15): {token_ids[0, :15].tolist()}")

    # Run CoreML
    print(f"\n--- Running CoreML ({args.model}) ---")
    coreml_logits = run_coreml(args.model, token_ids, args.seq_len, args.compute_units)

    # Compare
    print("\n--- Comparison ---")
    accuracy = compare_outputs(ref_logits, coreml_logits, tokens, args.seq_len)

    if accuracy >= 0.95:
        print(f"\n✓ PASS: {accuracy*100:.1f}% label agreement")
    elif accuracy >= 0.8:
        print(f"\n~ PARTIAL: {accuracy*100:.1f}% label agreement (fp16 approximation)")
    else:
        print(f"\n✗ FAIL: {accuracy*100:.1f}% label agreement")
        sys.exit(1)


if __name__ == "__main__":
    main()
