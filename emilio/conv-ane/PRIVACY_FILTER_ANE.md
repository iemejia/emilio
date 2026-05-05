# OpenAI Privacy Filter → ANE: Design & Implementation Notes

## Overview

This document describes the work done to bring [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) to Apple Neural Engine (ANE) inference via CoreML, integrated into the emilio project's `conv-ane/` pipeline.

## Model Architecture

OpenAI Privacy Filter is a **bidirectional token-classification model** for PII detection. It is fundamentally different from the existing Qwen2.5-0.5B autoregressive model that emilio already supports on ANE.

### Key specs (from `config.json`)

| Parameter | Value |
|-----------|-------|
| Task | Token classification (NER/PII) |
| d_model | 640 |
| Layers | 8 |
| Query heads | 14 |
| KV heads | 2 (GQA, 7 queries per KV group) |
| Head dim | 64 |
| FFN type | Sparse MoE |
| Experts | 128 total, top-4 routed per token |
| Intermediate size | 640 per expert |
| Attention | Bidirectional banded (sliding window=128 each side, total band=257) |
| RoPE | YaRN-scaled (theta=150k, factor=32) |
| Vocab size | 200,064 |
| Output classes | 33 (BIOES tags × 8 PII categories + background "O") |
| Total params | ~1.5B |
| Active params/token | ~50M (due to MoE sparsity) |

### Architecture source

The model implementation lives in the [openai/privacy-filter](https://github.com/openai/privacy-filter) repo under `opf/_model/model.py`. Key classes:

- `Transformer` — embedding → N blocks → norm → classification head
- `AttentionBlock` — RMSNorm, fused QKV linear, RoPE, GQA with sink logits, banded attention via windowed SDPA
- `MLPBlock` — RMSNorm, router gate, top-k expert selection, batched MLP1 (up+gate), SwiGLU, MLP2 (down), weighted sum
- `RotaryEmbedding` — YaRN NTK-by-parts scaling

### Why this is different from Qwen2

| | Qwen2 (existing emilio ANE) | Privacy Filter |
|---|---|---|
| Inference mode | Autoregressive (token-by-token) | Single forward pass (all tokens) |
| KV cache | Required (stateful, on-chip MLState) | Not needed |
| Attention type | Causal (unidirectional) | Bidirectional banded (±128 window) |
| FFN | Dense SwiGLU (gate+up → down) | Sparse MoE (128 experts, top-4 routing) |
| Output | Next-token logits (151k vocab) | Per-token class logits (33 labels) |
| Decoding | Autoregressive sampling | Viterbi span decoding |

## Design Decisions

### 1. No KV cache

Since this is an encoder (bidirectional, single-pass), there is no need for the stateful KV cache machinery that dominates the Qwen2 ANE converter. The entire sequence is processed at once and per-token logits are returned.

### 2. ANE layout: Conv2d(1×1) for projections

Following the same principle as the Qwen2 converter (`gguf_to_ane.py`), all linear projections (QKV, output, classification head) are expressed as Conv2d(1×1) operations. The tensor layout is `(B, C, 1, T)` — batch, channels, height=1, sequence length.

### 3. Sparse MoE via sort + index_select

The MoE routing was the hardest part to get working in CoreML. The challenges:

**What doesn't work in CoreML/torch.jit.trace:**
- `torch.topk` — produces dynamic shapes that CoreML cannot trace
- Dense all-experts approach (compute all 128) — works for conversion but model is 2.6GB and prediction hangs (128 × 640 × 1280 = 105M params per layer × 8 layers of matmuls)

**What works:**
- `torch.sort` (descending) + slice top-4 — CoreML supports sort
- `torch.index_select` to gather the selected expert weights — CoreML supports this
- Batched matmul on gathered weights: reshape to (T, 4, D, 2I) then `torch.matmul`

The final sparse MoE pattern:
```python
# Router: sort to get top-4
sorted_vals, sorted_idx = torch.sort(router_logits, dim=-1, descending=True)
top_idx = sorted_idx[:, :4]  # (T, 4)
expert_weights = softmax(sorted_vals[:, :4])  # (T, 4)

# Gather selected expert weights
flat_idx = top_idx.reshape(-1)  # (T*4,)
mlp1_w_sel = torch.index_select(all_expert_weights, 0, flat_idx)  # (T*4, D, 2I)

# Batched matmul
mlp1_w_sel = mlp1_w_sel.reshape(T, 4, D, 2*I)
h = torch.matmul(input.unsqueeze(1).unsqueeze(2), mlp1_w_sel)  # (T, 4, 1, 2I)
```

### 4. Fixed input shapes

The ANE replans execution graphs when input shapes change (~6ms penalty). By fixing sequence length at conversion time, we avoid this. The user specifies `--seq-len` (default 512) and inputs are padded. The banded attention mask ensures padding tokens don't affect results.

### 5. Banded attention mask

Precomputed `(T, T)` mask: 0.0 for valid positions (within ±128 window), -10000.0 for masked positions. Passed as a fixed input constant.

### 6. Sink logits

The model uses "sink" attention — an extra virtual position in softmax that acts as a learned bias. Preserved by appending a precomputed scalar (sink_score × ln2) to attention scores before softmax, then discarding the corresponding output column. Sink scores are precomputed in fp16 at model load time to avoid runtime float32 casts.

### 7. YaRN RoPE

Precomputed cos/sin tables with NTK-by-parts interpolation, concentration scaling. Passed as fixed inputs.

### 8. SwiGLU activation

Custom SwiGLU: `sigmoid(1.702 * x) * x * (linear + 1.0)` with clamp at ±7.0. The sigmoid is computed in float32 for numerical stability, cast back to fp16 immediately.

## Benchmark Results

Measured on Apple Silicon (M-series), `compute_units=CPU_AND_NE`:

| Config | Seq Len | Mean Latency | Throughput | Notes |
|--------|---------|-------------|------------|-------|
| fp32 compute (correct) | 32 | 444 ms | 72 tok/s | 100% label agreement with HF reference |
| fp16 compute (broken) | 32 | 145 ms | 221 tok/s | Outputs zeros due to CoreML MIL bug |
| fp16 compute (broken) | 128 | 920 ms | 139 tok/s | Same zeros issue |

### Verification Against Reference

**100% label agreement** verified between CoreML model and HuggingFace transformers reference:
- Test text: `"My name is Alice Smith and my email is alice.smith@example.com"`
- HF predictions: `[O, O, O, B-person, E-person, O, O, O, O, B-email, I-email, I-email, I-email, E-email]`
- CoreML predictions: identical
- Logits: HF [-8.23, 24.31], CoreML [-7.84, 24.22] (minor fp16 weight rounding)

**Critical:** The attention mask must incorporate padding (positions where no real tokens exist should be masked with `-10000`). Without this, the model predicts PII everywhere. This matches the HF model behavior with `attention_mask`.

## Known Issues

### fp16 Compute Precision Produces Zeros (8 layers)

**Symptom:** The full 8-layer model with `compute_precision=FLOAT16` produces all-zero logits. Models with ≤2 layers work correctly.

**Root Cause:** CoreML's MIL optimization passes apply constant folding and fp16 casts. With 8 layers of `index_select` + large tensor reshapes, some intermediate computation overflows or gets incorrectly optimized. The `RuntimeWarning: overflow encountered in cast` during conversion is the telltale.

**Status:** Unresolved. Using `compute_precision=FLOAT32` as workaround (weights still fp16, only compute is fp32). This gives 72 tok/s which is sufficient for real-time PII detection.

**Impact:** ~3× throughput loss vs theoretical fp16 performance (72 vs 221 tok/s).

### Attention Mask Must Include Padding

The model requires padding tokens to be masked in the attention mask. The `attn_mask` input should be constructed as:
```python
mask[i, j] = 0.0    # if j is a real token AND within banded window
mask[i, j] = -10000  # if j is padding OR outside banded window
```

Without proper padding masking, the model predicts PII labels on every token.

### Model Size

The full model is ~2.6GB in fp16 due to the MoE expert weights:
- Per layer: 128 experts × (640×1280 + 1280 + 640×640 + 640) = ~157M params
- 8 layers of MoE: ~1.26B params in experts alone
- Plus attention (~19M), embeddings (~128M), etc.

**Mitigation:** Use `--quantize` flag for int8 weight quantization (~1.3GB) or int4 palettization (~650MB).

## Files

### `privacy_filter_to_ane.py`

Python converter that:
1. Downloads or loads the `openai/privacy-filter` safetensors checkpoint
2. Maps HuggingFace weight names to internal format
3. Fuses Q/K/V projections into single Conv2d(1×1)
4. Builds PyTorch model with sparse MoE (sort + index_select + batched matmul)
5. Traces with `torch.jit.trace` and converts to CoreML via `coremltools`
6. Optionally quantizes weights to int8
7. Exports metadata JSON

**Usage:**
```bash
# Requires Python 3.12 (coremltools incompatible with 3.14)
source .venv312/bin/activate

# Basic conversion (seq_len=32 for testing)
python3 privacy_filter_to_ane.py \
  --checkpoint /path/to/openai-privacy-filter \
  --seq-len 32

# With int8 quantization (halves model size)
python3 privacy_filter_to_ane.py \
  --checkpoint /path/to/openai-privacy-filter \
  --seq-len 128 --quantize
```

**Outputs:**
- `PrivacyFilterANE_{seq_len}.mlpackage` — CoreML model
- `PrivacyFilterANE_{seq_len}_meta.json` — Architecture metadata

### `privacy_filter_ane.swift`

Swift inference runner that:
1. Loads the compiled CoreML model targeting ANE
2. Tokenizes input text (simplified byte-level; production should use BPE from `tokenizer.json`)
3. Builds RoPE tables and banded attention mask
4. Runs single-pass inference
5. Decodes per-token predictions into BIOES spans
6. Prints detected PII entities

**Usage:**
```bash
swiftc -O -o privacy_filter_ane privacy_filter_ane.swift \
  -framework CoreML -framework Foundation
./privacy_filter_ane PrivacyFilterANE_128.mlpackage \
  PrivacyFilterANE_128_meta.json \
  "My name is Alice Smith and my email is alice@example.com"
```

### `validate_privacy_filter.py`

Validation script (see below) that compares CoreML output against HuggingFace reference.

## Development Environment

- **Python 3.12** required (coremltools broken on 3.14 — missing `libmilstoragepython`)
- Virtual env: `.venv312/` with torch, coremltools, safetensors, transformers
- HF checkpoint cached at: `~/.cache/huggingface/hub/models--openai--privacy-filter/`
- Disk requirement: ~6GB free for conversion (2.6GB model + temp files)

## Next Steps

1. **Int8 quantization:** Apply post-conversion quantization to reduce from 2.6GB → 1.3GB
2. **Full BPE tokenizer:** Implement proper tokenizer in Swift using `tokenizer.json`
3. **Viterbi decoding:** Add constrained Viterbi span decoding for coherent BIOES labels
4. **Production seq_len:** Convert at seq_len=128 or 256 (balance between coverage and latency)
5. **fp16 investigation:** File CoreML bug / try coremltools nightly for the MIL overflow issue
6. **Benchmark on-device:** Profile ANE vs CPU utilization with Instruments
