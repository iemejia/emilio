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

The model implementation lives in the [openai/privacy-filter](https://github.com/openai/privacy-filter) repo under `opf/_model/model.py`. I read this in full to understand the exact architecture. Key classes:

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

Since this is an encoder (bidirectional, single-pass), there is no need for the stateful KV cache machinery that dominates the Qwen2 ANE converter. The entire sequence is processed at once and per-token logits are returned. This simplifies the CoreML model significantly.

### 2. ANE layout: Conv2d(1×1) for projections

Following the same principle as the Qwen2 converter (`gguf_to_ane.py`), all linear projections (QKV, output, gate, expert MLPs) are expressed as Conv2d(1×1) operations. The ANE's convolution engine handles these as matrix multiplications. The tensor layout is `(B, C, 1, T)` — batch, channels, height=1, sequence length — which maps naturally to ANE's 4D tensor expectations.

### 3. Sparse MoE: hybrid ANE/CPU execution

The MoE routing involves:
1. Computing router logits (a simple linear — runs on ANE as Conv2d)
2. Top-k selection (dynamic, data-dependent — must run on CPU)
3. Gathering expert weights by index (dynamic indexing — CPU)
4. Batched matmuls through selected experts (can run on ANE if shapes are fixed)
5. Scatter-add results back (CPU)

I set `compute_units = CPU_AND_NE` in CoreML and let the compiler decide the optimal split. In practice:
- Attention (fixed-shape Conv2d + windowed matmul) → ANE
- MoE routing + gather/scatter → CPU/GPU
- Expert matmuls (batched bmm) → GPU or CPU depending on size

This is the pragmatic approach. An alternative would be to materialize all 128 experts for every token and mask out unused ones, but that would waste 32× compute just to stay fully on ANE — not worthwhile for a model that's already small (50M active params).

### 4. Fixed input shapes

The ANE replans execution graphs when input shapes change, which costs ~6ms. By fixing the sequence length at conversion time, we avoid this overhead entirely. The user specifies `--seq-len` (default 512) and inputs are padded to that length. The banded attention mask ensures padding tokens don't affect results.

### 5. Banded attention mask

The model uses a symmetric band of ±128 positions (257 total window). I precompute this as a fixed `(T, T)` mask with 0.0 for valid positions and -10000.0 for masked positions. This is baked into the model as a constant input, avoiding any dynamic mask computation at runtime.

### 6. Sink logits

The original model uses "sink" attention — an extra virtual position in softmax that acts as a learned bias (attention sink). I preserve this by appending a learnable scalar to the attention scores before softmax, then discarding the corresponding output column. This matches the reference implementation exactly.

### 7. Weight name mapping

The HuggingFace safetensors checkpoint uses transformers-style naming (`model.layers.0.self_attn.q_proj.weight`). The converter maps these to our internal naming (`block.0.attn.qkv.weight`) and fuses separate Q/K/V projections into a single tensor, matching the Conv2d(1×1) layout.

### 8. YaRN RoPE

The model uses YaRN (Yet another RoPE extensioN) with NTK-by-parts interpolation for long-context support. The converter precomputes the full cos/sin tables with the correct scaling and passes them as fixed inputs. This avoids runtime RoPE computation entirely.

## Files Created

### `emilio/conv-ane/privacy_filter_to_ane.py`

Python converter that:
1. Downloads or loads the `openai/privacy-filter` safetensors checkpoint
2. Maps HuggingFace weight names to internal format
3. Fuses Q/K/V projections
4. Builds a PyTorch model with Conv2d(1×1) projections (ANE-optimized layout)
5. Traces and converts to CoreML with `coremltools`
6. Optionally quantizes weights to int8
7. Exports metadata JSON

**Usage:**
```bash
python3 privacy_filter_to_ane.py --seq-len 512 --quantize
```

**Outputs:**
- `PrivacyFilterANE_512.mlpackage` — CoreML model
- `PrivacyFilterANE_512_meta.json` — Architecture metadata + label mapping

### `emilio/conv-ane/privacy_filter_ane.swift`

Swift inference runner that:
1. Loads the compiled CoreML model targeting ANE
2. Tokenizes input text (simplified byte-level; production should use the full BPE tokenizer from `tokenizer.json`)
3. Builds RoPE tables and banded attention mask
4. Runs single-pass inference
5. Decodes per-token argmax predictions into BIOES spans
6. Prints detected PII entities with labels and positions

**Usage:**
```bash
swiftc -O -o privacy_filter_ane privacy_filter_ane.swift \
  -framework CoreML -framework Foundation
./privacy_filter_ane PrivacyFilterANE_512.mlpackage \
  PrivacyFilterANE_512_meta.json \
  "My name is Alice Smith and my email is alice@example.com"
```

## Performance Expectations

Based on the existing Qwen2 ANE results (171 tok/s decode for 494M active params):

- Privacy Filter has ~50M active params per token (10× smaller)
- Single-pass over 512 tokens means 512 tokens processed at once
- Expected throughput: **high** — the bottleneck will be the MoE gather/scatter on CPU, not the ANE attention
- The model is small enough that even CPU-only inference is fast; ANE acceleration primarily helps with the attention convolutions

## Limitations & Future Work

1. **Tokenizer**: The Swift runner uses simplified byte-level encoding. For production, implement the full BPE tokenizer from the model's `tokenizer.json` (200k vocab GPT-style BPE).

2. **Viterbi decoding**: The reference implementation uses constrained Viterbi span decoding with transition biases for coherent BIOES labels. The Swift runner currently uses simple argmax. Adding Viterbi would improve span boundary quality.

3. **Dynamic sequence length**: Currently fixed at conversion time. For variable-length inputs, either convert multiple models (e.g., 128, 512, 2048) or pad all inputs to the max.

4. **MoE optimization**: If profiling shows the MoE is the bottleneck, consider:
   - Distilling to a dense model for pure-ANE execution
   - Using Metal compute shaders for the expert routing
   - Pruning to fewer experts with fine-tuning

5. **Batch processing**: The model supports batch>1 for throughput. The CoreML model could be extended to accept batched inputs for processing multiple documents simultaneously.
