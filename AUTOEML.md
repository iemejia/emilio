# AutoEML — Autonomous EML Graph Optimization

AutoEML is an autonomous optimization loop for EML (exp-minus-ln) evaluation
strategies, inspired by [AutoKernel](https://github.com/RightNow-AI/autokernel).
It optimizes the hot inner loops of EML-native matrix operations by iteratively
editing a single kernel file, benchmarking, and keeping or reverting changes.

## How It Works

### Architecture

```
autoeml_kernel.rs   ← THE file the agent edits (one kernel at a time)
autoeml_reference.rs ← Correctness oracles (DO NOT MODIFY)
autoeml_main.rs      ← CLI: profile / bench / verify (DO NOT MODIFY)
autoeml_program.md   ← Agent playbook (optimization tiers, constraints)
results.tsv          ← Experiment log
```

### The Loop

```
 ┌─────────────┐
 │  PROFILE    │  Analytical breakdown: which ops dominate?
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  HYPOTHESIZE│  Pick an optimization from the playbook
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  EDIT       │  Modify autoeml_kernel.rs
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  BENCH      │  5-stage correctness + throughput measurement
 └─────┬───────┘
       ▼
 ┌──────┴──────┐
 │  KEEP or    │  Faster & correct → git commit
 │  REVERT     │  Slower or wrong → git checkout
 └─────────────┘
       │
       └──→ repeat
```

### Constraints

- Every multiply **must** go through `exp(ln(a) + ln(b))` — no raw float ops.
- `c_exp()` and `c_ln()` are the **only** transcendental primitives.
- Results must match the reference implementation to within 1e-6 relative error.
- We don't cheat: no replacing EML with GPU/Metal ops.

### Benchmarking

The bench harness runs 5 stages before performance measurement:

1. **Smoke test** — 4×4 matmul
2. **Shape sweep** — 8×8 through model-size
3. **Numerical stability** — near-zero, large, negative, mixed-sign values
4. **Determinism** — same input → same output
5. **EML purity** — transcendental count matches expectation

Then it measures median latency over N iterations on a model-sized matmul
`(1, 896) × (896, 896)` — the QKV projection shape for Qwen2.5-0.5B.

## Profiling Results

Target model: **Qwen2.5-0.5B-Instruct** (single-token generation)

| Operation | % of Budget |
|-----------|-------------|
| FFN matmuls (gate+up+down) | 63.5% |
| LM head | 27.5% |
| QKV + output projections | 8.9% |
| SiLU + gate mul | ~0.1% |

Matmul dominates. All optimization effort targeted the CSE matmul kernel.

## Experiments

15 experiments were run. 7 were kept, 8 were reverted.

Then 3 more book-driven experiments (15–17) were run, 2 kept, 1 reverted.
**Total: 18 experiments, 9 kept, 9 reverted.**

### Kept (cumulative, in order applied)

| # | Optimization | Latency (μs) | Δ from baseline | Key insight |
|---|-------------|-------------|-----------------|-------------|
| 0 | Baseline CSE matmul | 17,238 | — | ln(A) + ln(B) shared across dot products |
| 2 | Transpose ln(B) | 16,494 | −4.3% | Cache-friendly sequential k-loop access |
| 4 | Activation sharing | — | −1,792 ln/QKV | Precompute ln(X) once for Q, K, V projections |
| 5 | Pre-transposed weights | 7,082 | −59.7% | One-time transpose at model load, not per call |
| 6 | Real-valued exp bypass | 4,264 | −75.3% | f64 exp + sign from im/π parity, skip Complex64 trig |
| 9 | 4-wide loop unroll | 4,065 | −76.4% | Independent accumulators for instruction-level parallelism |
| 10 | Batched atomic counter | 3,995 | −76.8% | One atomic add per matmul instead of per element |
| 11 | Branchless sign | 3,917 | −77.3% | `1.0 - 2.0*(n&1)` instead of if/else |
| 16 | Rayon j-parallelism | ~710 | −95.9% | Work-stealing over 896 columns [Iverson APL] |
| 17 | Zero-copy borrow + par_iter_mut | ~456 | −97.4% | Eliminate 12MB clone; write directly to result |

### Reverted

| # | Optimization | Latency (μs) | Why reverted |
|---|-------------|-------------|-------------|
| 1 | Rayon parallel outer loop | ~52,000 | 3× slower — thread pool overhead on M=1 |
| 3 | Two-phase batched eval | 16,913 | Slight regression — per-element Vec allocation |
| 7 | SoA re/im split | 5,485 | 28% slower — extra allocation + cache pressure from 4 arrays |
| 8 | Precomputed signs | 4,675 | 9.6% slower — sign extraction overhead > savings |
| 12 | Truncation vs round | ~3,921 | Noise — not worth the complexity |
| 13 | 8-wide loop unroll | 4,235 | Register pressure on Apple Silicon ARM |
| 14 | Pre-extracted re/sign arrays | 5,719 | 46% slower — 4-array cache pressure dominates |
| 15 | Compact f64+u8 format | ~3,834 | No improvement — compute-bound on exp(), not memory-bound |

## Final Results

Benchmark: `(1, 896) × (896, 896)` matmul with transposed precomputed weights.

| Metric | Baseline | After Exp 11 | After Exp 17 | Total Improvement |
|--------|----------|-------------|-------------|-------------------|
| **Latency** | 17,238 μs | 3,917 μs | **~456 μs** | **37.8× faster** |
| **Throughput** | 51,978 elem/s | ~225,000 elem/s | **~1,970,000 elem/s** | **37.9× higher** |
| **Transcendentals** | 1,606,528 | 803,712 | **803,712** | **50% reduction** |
| **ln calls** | 803,712 | 896 | **896** | **99.9% reduction** |

At ~456 μs we are at the theoretical hardware limit:
802,816 exp calls × ~4.7 ns / 8 cores ≈ 472 μs.

All 11 verification tests pass. Correctness confirmed across all shapes and
numerical edge cases.

## Key Learnings

1. **The biggest single win was avoiding Complex64 in the hot loop** (exp 6).
   Since ln(real) produces imaginary parts of exactly 0 or π, we replace
   `Complex64::exp()` (which computes cos+sin) with `f64::exp()` + a sign bit.
   This alone was ~40% faster.

2. **Memory layout beats algorithmic cleverness.** Transposing weight matrices
   at load time (exp 5) gave a larger speedup than any loop-level trick.

3. **SoA hurts when both components are accessed together.** Splitting Complex64
   into separate re/im arrays (exp 7) increased cache pressure without benefit —
   the AoS layout is correct when re and im are consumed as a pair.

4. **ARM register pressure limits unrolling.** 4-wide is the sweet spot on
   Apple Silicon; 8-wide causes register spills and is 8% slower.

5. **Diminishing returns at the loop level.** After exp 6, each remaining
   single-threaded improvement was single-digit percent.

6. **Parallelism unlocked the next frontier** (exp 16–17). Rayon work-stealing
   over the j dimension gave 5.3× on 8 cores. Eliminating the 12MB clone
   (zero-copy borrow of precomputed weights) added another 1.5×. Combined:
   ~8× over the single-threaded optimized kernel. Source: Iverson's APL model
   of treating inner products as atomic parallel operations.

7. **We hit the hardware wall.** At ~456 μs, latency matches the theoretical
   minimum of 802,816 exp calls × 4.7 ns / 8 cores ≈ 472 μs. Further gains
   require reducing exp call count (algebraic pruning of negligible terms) or
   using approximate exp.

## Commit History

```
<pending> .eml compiled format: precomputed ln(W) as Complex64, --compile CLI, automation script
e663c44 emilio: precomputed ln(weights) + optimized matmul — 0.49 → 4.3 tok/s (8.8×)
02b0dca autoeml exp 15-17: rayon j-parallelism + zero-copy — 456 μs (37.8× faster)
34dd3e7 autoeml exp 11: branchless sign — ~3,917 μs (77.7% faster than baseline)
31072d7 autoeml exp 10: batched atomic counter — ~3,995 μs (77.3% faster than baseline)
cc618f6 autoeml exp 9: 4-wide loop unroll — 4,065 μs (76.9% faster than baseline)
1d7a4e3 autoeml exp 6: real-valued exp bypass — 4,264 μs (75.8% faster than baseline)
8b49923 autoeml exp 5: pre-transposed weight precompute — 7,082 μs (59.7% faster than baseline)
a494232 autoeml exp 4: activation sharing — precompute ln(X) for Q/K/V reuse
a48ff74 autoeml exp 2: transpose ln(B) for cache-friendly k-loop
97e9a58 autoeml: autonomous EML graph optimization agent
```

## Emilio Integration: Kernel → Inference Engine

The autoeml micro-benchmark optimizes a single `(1, 896) × (896, 896)` matmul
in isolation. Emilio is the full Qwen2.5-0.5B inference engine that executes
24 transformer layers per token, each containing 7 matmuls plus RMSNorm, SiLU,
RoPE, and softmax — all through pure EML arithmetic.

### The Gap

Emilio originally used `build_matmul_cse()` from `eml_optimizer.rs`, which:
- Used full `Complex64::exp()` in the inner loop (cos + sin computation)
- Recomputed `ln(B)` for every matmul call (weights never change)
- Transposed `ln(B)` on every call (redundant work)
- Used `Complex64::ln()` element-wise (2× slower than f64 path)

This gave **0.49 tok/s** — each token required ~168 matmuls (7 per layer × 24
layers) plus the LM head (the largest matmul: 896 × 151,936).

### What Changed

**Step 1: Optimized `build_matmul_cse`** — Rewrote the function body to use
all autoeml kernel optimizations without atomic counters (which cause massive
contention under rayon):
- Real-exp bypass: `f64::exp(re)` + branchless sign from `im/π` parity
- 4-wide loop unroll with independent accumulators
- Rayon `par_iter_mut` over the j dimension (column parallelism)
- Parallel `ln(A)`, `ln(B)`, and transpose

**Step 2: Precomputed `ln(weights)` at load time** — Added
`build_matmul_cse_precomp()` that accepts pre-computed `ln(B)` in transposed
layout, skipping both `ln()` and transpose per call. Key insight: GGUF stores
weights as `(out_dim, in_dim)`, and emilio transposes them to `(in_dim, out_dim)`
before matmul, which internally transposes `ln(B)` to `(out_dim, in_dim)`.
The double-transpose cancels — so element-wise `ln()` of the original GGUF
layout gives the correct `ln_b_t` directly.

**Step 3: Wired all 13 matmul call sites** — Every matmul in emilio
(QKV projections, output projection, gate/up/down FFN, and LM head) now uses
`build_matmul_cse_precomp` with layer-specific precomputed `ln(W)` fields.

### Results

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| **Token generation** | 0.49 tok/s | **~4.3 tok/s** | **8.8×** |
| **Per-token time** | ~2.05 s | ~0.23 s | 8.8× |
| **Model load time** | 1.24 s | 2.69 s | +1.45 s (one-time) |
| **Output quality** | ✓ correct | ✓ identical | — |

The load-time increase (~1.45s) pays for precomputing `ln()` over ~494M weight
elements (168 weight matrices × avg ~2.9M elements each). This is amortized
over all generated tokens — break-even at ~1 token.

### Where the Time Goes (per token, ~230 ms)

At ~4.3 tok/s, each token takes ~230 ms across 24 layers:
- **FFN matmuls** (gate + up + down): ~63% — 3 matmuls per layer, the gate/up
  projections are (1, 896) × (896, 4864) = ~4.4M exp each
- **LM head**: ~28% — (1, 896) × (896, 151936) = ~136M exp, runs once per token
- **QKV + output projections**: ~9% — 4 matmuls per layer but smaller
- **RMSNorm, SiLU, RoPE, softmax**: <1% — dominated by matmul cost

## Running AutoEML

```bash
cd eml_rust

# Profile transcendental budget
cargo run --bin autoeml --release -- profile

# Benchmark current kernel
cargo run --bin autoeml --release -- bench
cargo run --bin autoeml --release -- bench --precomputed
cargo run --bin autoeml --release -- bench --transposed

# Verify all operations
cargo run --bin autoeml --release -- verify

# Run emilio inference (Qwen2.5-0.5B)
cargo run --bin emilio --release -- ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --generate "The capital of France is"

# Compile and run from .eml format
cargo run --bin emilio --release -- ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --compile ../models/qwen2.5-0.5b-instruct.eml
cargo run --bin emilio --release -- ../models/qwen2.5-0.5b-instruct.eml \
  --chat "What is 2+2?"

# Or use the automation script
./compile_model.sh models/qwen2.5-0.5b-instruct-q8_0.gguf
```

## Compiled .eml Format

### Motivation

The EML inference pipeline has an *offline* stage and a *runtime* stage:

- **Offline (compile time):** dequantize GGUF weights → compute `ln(W)` for each
  weight matrix → store as `Complex64` in `(cols, inner)` layout. This is a
  pure function of the model weights and only needs to run once per model.

- **Runtime (inference):** load precomputed `ln(W)`, compute `exp(ln(A) + ln_W_t)`
  for each matmul. No dequantization, no `ln()` of weights needed.

The `.eml` compiled format persists the offline stage output, making model
distribution self-contained and independent of GGUF tooling.

### Format Specification (EML v1)

```
┌─────────────────────────────────────────────────────┐
│ Header                                              │
│   magic: "EML1" (4 bytes)                           │
│   version: u32                                      │
│   config: vocab_size, n_layers, n_heads, n_kv_heads,│
│           d_model, d_ff (u32 × 6)                   │
│           rope_freq_base, rms_norm_eps (f64 × 2)    │
│           max_seq_len, d_head (u32 × 2)             │
├─────────────────────────────────────────────────────┤
│ Tokenizer                                           │
│   vocab_size, merges_count, bos_id, eos_id (u32 × 4)│
│   vocab[]: length-prefixed UTF-8 strings            │
│   merges[]: (left, right) string pairs, rank-ordered│
├─────────────────────────────────────────────────────┤
│ Global Weights                                      │
│   token_embd: f64[] (vocab × d_model)               │
│   output_norm: f64[] (d_model)                      │
│   ln_output: Complex64[] (vocab × d_model)          │
├─────────────────────────────────────────────────────┤
│ Per-Layer Weights (× n_layers)                      │
│   ln_q, ln_k, ln_v, ln_o: Complex64[]              │
│   ln_gate, ln_up, ln_down: Complex64[]              │
│   q_bias, k_bias, v_bias: f64[]                     │
│   attn_norm, ffn_norm: f64[]                        │
└─────────────────────────────────────────────────────┘
```

All multi-byte values are little-endian. Arrays are length-prefixed with `u64`
element count. Complex64 values are stored as `(re: f64, im: f64)` pairs.

**Key design decision:** Raw f64 weight matrices are NOT stored in the .eml
format. They are only needed to compute `ln(W)`, which is done at compile time.
During inference, only `ln_*` (Complex64), biases, norms, and embeddings are
accessed — confirmed by exhaustive grep of all weight field references in
`emilio.rs`.

### Benchmark: GGUF vs .eml

| Metric | GGUF (Q8_0) | .eml (EML v1) | Notes |
|--------|-------------|---------------|-------|
| **File size** | ~530 MB | 8,580 MB | 16× larger (Complex64 = 16 bytes vs Q8 = 1 byte) |
| **Load time** | ~1.9 s | ~2.6 s | .eml is I/O bound on 8.6 GB read |
| **Inference** | ~6.8 tok/s | ~7.1 tok/s | Equivalent (same runtime path) |
| **Dependencies** | GGUF parser + dequant | Pure binary read | .eml is self-contained |

**Honest assessment:** The current .eml format does not improve load time over
GGUF Q8. The GGUF file is 16× smaller because Q8 quantization stores 1 byte per
weight vs 16 bytes for Complex64 (2 × f64). The smaller I/O footprint of GGUF +
runtime dequant+ln computation is faster than reading the 8.6 GB .eml file.

**Where .eml wins:**
1. **Self-contained:** No GGUF parser, no dequantization code, no quantization
   format knowledge needed at runtime.
2. **Pre-verified:** The `ln(W)` values are computed once and can be validated
   before distribution.
3. **mmap-ready:** The flat binary layout enables memory-mapped I/O (future work),
   which would give near-instant "load" times — the OS pages in data on demand.
4. **Format for f16/f32 models:** For models already stored as f16/f32 in GGUF
   (no quantization compression), the .eml format would be competitive in size
   while eliminating the `ln(W)` computation.

### Future Work: Memory-Mapped I/O

The .eml v1 format uses length-prefixed arrays, which requires sequential reading.
A v2 format with fixed-offset header (all tensor offsets precomputed) would enable:

```rust
let mmap = unsafe { memmap2::Mmap::map(&file)? };
let ln_q: &[Complex64] = bytemuck::cast_slice(&mmap[offset..offset + size]);
```

Expected load time with mmap: **< 1 ms** (virtual memory mapping only, actual
data read deferred to first access). This is the canonical approach used by GGML,
safetensors, and other ML tensor formats.

### Paper Notes

The .eml compiled format demonstrates the **offline/online separation** inherent
in the EML formulation. Because `eml(x, y) = exp(x) − ln(y)` decomposes
multiplication into `exp(ln(a) + ln(b))`, and model weights are constants during
inference, the `ln(W)` computation can be moved entirely to compile time. This is
analogous to ahead-of-time (AOT) compilation in traditional computing: the
compiler pays a one-time cost to enable faster execution.

The trade-off between file size (quantized vs precomputed) and load time
(compute vs I/O) is an empirical question that depends on hardware I/O bandwidth
vs CPU throughput. On Apple Silicon M-series with NVMe (7 GB/s read), the
crossover point is approximately where `ln_compute_time > file_size_delta / io_bandwidth`.

Reference: Odrzywołek (2026) arXiv:2603.21852, Section 3 on EML graph reduction.

