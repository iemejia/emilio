//! emilio — EML inference engine.
//!
//! Compiles a GGUF model into CSE-optimized EML expression DAGs,
//! then executes them through exp()/ln() only.
//!
//! Architecture support: Qwen2 (RMSNorm, SwiGLU, RoPE, GQA)
//!
//! INVARIANT: every numerical result flows through eml(x,y) = exp(x) - ln(y).
//! The only non-EML operations are:
//!   - Token lookup (discrete indexing)
//!   - Argmax / sampling (comparison)
//!   - RoPE angle computation (constant, precomputed)

use crate::eml_ops::*;
use crate::eml_optimizer::*;
use crate::gguf::GGUFFile;
use num_complex::Complex64;
use rayon::prelude::*;

// ─── Model config (extracted from GGUF metadata) ───────────────────────────

#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub rope_freq_base: f64,
    pub rms_norm_eps: f64,
    pub max_seq_len: usize,
    pub d_head: usize,
}

impl QwenConfig {
    pub fn from_gguf(gguf: &GGUFFile) -> Self {
        let arch = gguf.meta_str("general.architecture").unwrap_or("qwen2");
        let d_model = gguf.meta_u32(&format!("{arch}.embedding_length")).unwrap_or(896) as usize;
        let n_heads = gguf.meta_u32(&format!("{arch}.attention.head_count")).unwrap_or(14) as usize;
        let n_kv_heads = gguf.meta_u32(&format!("{arch}.attention.head_count_kv")).unwrap_or(2) as usize;

        QwenConfig {
            vocab_size: gguf.meta_u32(&format!("{arch}.vocab_size"))
                .or_else(|| gguf.meta_u32("tokenizer.ggml.vocab_size"))
                .unwrap_or(151936) as usize,
            n_layers: gguf.meta_u32(&format!("{arch}.block_count")).unwrap_or(24) as usize,
            n_heads,
            n_kv_heads,
            d_model,
            d_ff: gguf.meta_u32(&format!("{arch}.feed_forward_length")).unwrap_or(4864) as usize,
            rope_freq_base: gguf.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(1000000.0) as f64,
            rms_norm_eps: gguf.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f64,
            max_seq_len: gguf.meta_u32(&format!("{arch}.context_length")).unwrap_or(32768) as usize,
            d_head: d_model / n_heads,
        }
    }

    pub fn print(&self) {
        println!("  vocab_size:     {}", self.vocab_size);
        println!("  n_layers:       {}", self.n_layers);
        println!("  n_heads:        {}", self.n_heads);
        println!("  n_kv_heads:     {}", self.n_kv_heads);
        println!("  d_model:        {}", self.d_model);
        println!("  d_head:         {}", self.d_head);
        println!("  d_ff:           {}", self.d_ff);
        println!("  rope_freq_base: {}", self.rope_freq_base);
        println!("  rms_norm_eps:   {}", self.rms_norm_eps);
        println!("  max_seq_len:    {}", self.max_seq_len);
    }
}

// ─── Layer weights ─────────────────────────────────────────────────────────

pub struct LayerWeights {
    // Attention
    pub q_weight: Vec<f64>,  // (d_model, n_heads * d_head)
    pub k_weight: Vec<f64>,  // (d_model, n_kv_heads * d_head)
    pub v_weight: Vec<f64>,  // (d_model, n_kv_heads * d_head)
    pub q_bias: Vec<f64>,    // (n_heads * d_head,)
    pub k_bias: Vec<f64>,    // (n_kv_heads * d_head,)
    pub v_bias: Vec<f64>,    // (n_kv_heads * d_head,)
    pub o_weight: Vec<f64>,  // (n_heads * d_head, d_model)

    // Attention norm
    pub attn_norm: Vec<f64>, // (d_model,)

    // FFN (SwiGLU)
    pub gate_weight: Vec<f64>,  // (d_model, d_ff)
    pub up_weight: Vec<f64>,    // (d_model, d_ff)
    pub down_weight: Vec<f64>,  // (d_ff, d_model)

    // FFN norm
    pub ffn_norm: Vec<f64>,  // (d_model,)
}

pub struct ModelWeights {
    pub config: QwenConfig,
    pub token_embd: Vec<f64>,   // (vocab, d_model)
    pub output_norm: Vec<f64>,  // (d_model,)
    pub output_weight: Vec<f64>, // (d_model, vocab) — may be tied to token_embd
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    /// Load all weights from a GGUF file, dequantizing to f64.
    pub fn from_gguf(gguf: &GGUFFile) -> std::io::Result<Self> {
        let config = QwenConfig::from_gguf(gguf);
        println!("  Loading weights ({} layers)...", config.n_layers);

        // Token embeddings
        let token_embd = load_tensor(gguf, "token_embd.weight")?;

        // Output norm
        let output_norm = load_tensor(gguf, "output_norm.weight")?;

        // Output weight — may be tied to token_embd
        // GGUF stores as (vocab, d_model); we need (d_model, vocab) for matmul
        let output_weight = if gguf.tensor_info("output.weight").is_some() {
            let raw = load_tensor(gguf, "output.weight")?;
            // raw is (vocab, d_model), transpose to (d_model, vocab)
            let v = config.vocab_size;
            let d = config.d_model;
            let mut out = vec![0.0f64; d * v];
            for i in 0..v {
                for j in 0..d {
                    out[j * v + i] = raw[i * d + j];
                }
            }
            out
        } else {
            // Tied embeddings: output = token_embd^T
            // token_embd is (vocab, d_model), we need (d_model, vocab)
            let v = config.vocab_size;
            let d = config.d_model;
            let mut out = vec![0.0f64; d * v];
            for i in 0..v {
                for j in 0..d {
                    out[j * v + i] = token_embd[i * d + j];
                }
            }
            out
        };

        // Layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let pfx = format!("blk.{i}");
            let layer = LayerWeights {
                q_weight: load_tensor(gguf, &format!("{pfx}.attn_q.weight"))?,
                k_weight: load_tensor(gguf, &format!("{pfx}.attn_k.weight"))?,
                v_weight: load_tensor(gguf, &format!("{pfx}.attn_v.weight"))?,
                q_bias: load_tensor(gguf, &format!("{pfx}.attn_q.bias"))?,
                k_bias: load_tensor(gguf, &format!("{pfx}.attn_k.bias"))?,
                v_bias: load_tensor(gguf, &format!("{pfx}.attn_v.bias"))?,
                o_weight: load_tensor(gguf, &format!("{pfx}.attn_output.weight"))?,
                attn_norm: load_tensor(gguf, &format!("{pfx}.attn_norm.weight"))?,
                gate_weight: load_tensor(gguf, &format!("{pfx}.ffn_gate.weight"))?,
                up_weight: load_tensor(gguf, &format!("{pfx}.ffn_up.weight"))?,
                down_weight: load_tensor(gguf, &format!("{pfx}.ffn_down.weight"))?,
                ffn_norm: load_tensor(gguf, &format!("{pfx}.ffn_norm.weight"))?,
            };
            layers.push(layer);
            if (i + 1) % 8 == 0 || i == config.n_layers - 1 {
                println!("    Loaded layer {}/{}", i + 1, config.n_layers);
            }
        }

        Ok(ModelWeights { config, token_embd, output_norm, output_weight, layers })
    }
}

fn load_tensor(gguf: &GGUFFile, name: &str) -> std::io::Result<Vec<f64>> {
    let info = gguf.tensor_info(name)
        .ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("tensor not found: {name}"),
        ))?;
    gguf.load_tensor_f64(info)
}

// ─── EML RMSNorm ────────────────────────────────────────────────────────────
//
// RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
//
// In EML with CSE:
//   - x² = exp(2 * ln(x))  → share ln(x)
//   - mean = sum / N        → sum is 0-cost adds, div via log domain
//   - sqrt = exp(0.5 * ln(..))
//   - final = x * gamma / std = exp(ln(x) + ln(gamma) - ln(std))
//     → ln(x) already cached, ln(gamma) constant, ln(std) computed once

pub fn eml_rms_norm(x: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let nc = Complex64::new(n as f64, 0.0);

    // Cache ln(x_i) — shared between squaring and final division
    let ln_x: Vec<Complex64> = x.iter()
        .map(|&v| Complex64::new(v, 0.0).ln())
        .collect();

    // x² = exp(2 * ln(x))
    let two = Complex64::new(2.0, 0.0);
    let sq_sum: Complex64 = ln_x.iter()
        .map(|&lx| (two * lx).exp())
        .fold(Complex64::new(0.0, 0.0), |a, b| a + b);  // add is 0-cost

    // mean(x²) = sq_sum / N via log domain
    let mean_sq = (sq_sum.ln() - nc.ln()).exp();

    // std = sqrt(mean_sq + eps) = exp(0.5 * ln(mean_sq + eps))
    let half = Complex64::new(0.5, 0.0);
    let ln_std = half * (mean_sq + Complex64::new(eps, 0.0)).ln();

    // Cache ln(gamma)
    let ln_gamma: Vec<Complex64> = gamma.iter()
        .map(|&g| Complex64::new(g, 0.0).ln())
        .collect();

    // result_i = x_i * gamma_i / std
    //          = exp(ln(x_i) + ln(gamma_i) - ln(std))
    // All ln's already cached!
    (0..n)
        .map(|i| {
            (ln_x[i] + ln_gamma[i] - ln_std).exp().re
        })
        .collect()
}

// ─── EML RoPE ───────────────────────────────────────────────────────────────
//
// Rotary position embedding applies a rotation:
//   q'[2i]   = q[2i]   * cos(θ) - q[2i+1] * sin(θ)
//   q'[2i+1] = q[2i]   * sin(θ) + q[2i+1] * cos(θ)
//
// where θ_i = pos / (base ^ (2i/d_head))
//
// In EML: mul,add,sub are all EML. cos/sin can be expressed as:
//   cos(θ) = Re(exp(iθ)) — but we're in the real domain, so we precompute
//   the cos/sin tables (they're constants, not data-dependent).
//
// The actual rotation mul/add/sub is all EML.

pub struct RopeCache {
    /// cos[pos * d_half + i] and sin[pos * d_half + i]
    pub cos: Vec<f64>,
    pub sin: Vec<f64>,
    pub d_half: usize,
    pub max_len: usize,
}

impl RopeCache {
    pub fn new(d_head: usize, max_len: usize, base: f64) -> Self {
        let d_half = d_head / 2;
        let mut cos = vec![0.0; max_len * d_half];
        let mut sin = vec![0.0; max_len * d_half];

        for pos in 0..max_len {
            for i in 0..d_half {
                let freq = 1.0 / base.powf(2.0 * i as f64 / d_head as f64);
                let angle = pos as f64 * freq;
                cos[pos * d_half + i] = angle.cos();
                sin[pos * d_half + i] = angle.sin();
            }
        }

        RopeCache { cos, sin, d_half, max_len }
    }

    /// Apply RoPE to a vector of length d_head at position pos.
    /// Uses EML mul/add/sub for the rotation.
    pub fn apply(&self, x: &mut [f64], pos: usize) {
        let d_half = self.d_half;
        for i in 0..d_half {
            let cos_v = self.cos[pos * d_half + i];
            let sin_v = self.sin[pos * d_half + i];
            let x0 = x[i];
            let x1 = x[i + d_half];

            // x'[i]        = x[i] * cos - x[i+d_half] * sin
            // x'[i+d_half] = x[i] * sin + x[i+d_half] * cos
            // All via EML mul/add/sub
            let x0c = to_c(x0);
            let x1c = to_c(x1);
            let cc = to_c(cos_v);
            let sc = to_c(sin_v);

            x[i] = to_r(eml_sub(eml_mul(x0c, cc), eml_mul(x1c, sc)));
            x[i + d_half] = to_r(eml_add(eml_mul(x0c, sc), eml_mul(x1c, cc)));
        }
    }
}

// ─── EML SwiGLU FFN ─────────────────────────────────────────────────────────
//
// SwiGLU(x) = (x @ gate_w) * silu(x @ up_w)
// silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
//
// In EML:
//   sigmoid(z) = inv(1 + exp(-z)) = exp(-ln(1 + exp(-z)))
//   silu(z) = z * sigmoid(z) = exp(ln(z) + ln(sigmoid(z)))
//   final = gate * silu(up) = exp(ln(gate) + ln(silu(up)))
//
// CSE: ln(gate) and ln(silu(up)) cached per element.

pub fn eml_silu(x: &[f64]) -> Vec<f64> {
    x.par_iter()
        .map(|&v| {
            let xc = Complex64::new(v, 0.0);
            let one = Complex64::new(1.0, 0.0);
            // sigmoid(x) = 1 / (1 + exp(-x))
            let sig = one / (one + (-xc).exp());
            // silu(x) = x * sigmoid(x) via log domain
            (xc.ln() + sig.ln()).exp().re
        })
        .collect()
}

// ─── KV Cache ───────────────────────────────────────────────────────────────

/// Per-layer KV cache: stores projected, RoPE'd K and V for all past positions.
pub struct LayerKVCache {
    /// k_cache[pos * kv_dim + h * d_head + d] — cached key vectors
    pub k: Vec<f64>,
    /// v_cache[pos * kv_dim + h * d_head + d] — cached value vectors
    pub v: Vec<f64>,
    pub len: usize,  // number of positions cached
    pub kv_dim: usize,
}

impl LayerKVCache {
    pub fn new(kv_dim: usize, max_len: usize) -> Self {
        LayerKVCache {
            k: vec![0.0; max_len * kv_dim],
            v: vec![0.0; max_len * kv_dim],
            len: 0,
            kv_dim,
        }
    }

    /// Append K,V for one position.
    pub fn append(&mut self, k: &[f64], v: &[f64]) {
        let off = self.len * self.kv_dim;
        self.k[off..off + self.kv_dim].copy_from_slice(k);
        self.v[off..off + self.kv_dim].copy_from_slice(v);
        self.len += 1;
    }
}

/// Full KV cache for all layers.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(cfg: &QwenConfig, max_len: usize) -> Self {
        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let layers = (0..cfg.n_layers)
            .map(|_| LayerKVCache::new(kv_dim, max_len))
            .collect();
        KVCache { layers }
    }
}

// ─── Full forward pass (prefill: process all tokens at once) ────────────────

pub fn emilio_forward(
    token_ids: &[usize],
    weights: &ModelWeights,
    rope: &RopeCache,
) -> Vec<f64> {
    let cfg = &weights.config;
    let t = token_ids.len();
    let d = cfg.d_model;

    // 1. Token embedding (discrete lookup — not EML)
    let mut x = vec![0.0f64; t * d];
    for (i, &tok) in token_ids.iter().enumerate() {
        x[i * d..(i + 1) * d].copy_from_slice(
            &weights.token_embd[tok * d..(tok + 1) * d]
        );
    }

    // 2. Transformer layers
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        x = transformer_layer(&x, layer, cfg, rope, t, layer_idx);
    }

    // 3. Final RMSNorm
    let mut normed = Vec::with_capacity(t * d);
    for i in 0..t {
        normed.extend(eml_rms_norm(
            &x[i * d..(i + 1) * d],
            &weights.output_norm,
            cfg.rms_norm_eps,
        ));
    }

    // 4. LM head: (T, d_model) @ (d_model, vocab)
    // output_weight stored as (d_model, vocab) if tied
    let logits = build_matmul_cse(
        &normed, &weights.output_weight,
        t, d, cfg.vocab_size,
    );

    logits
}

// ─── Single-token forward pass with KV cache ───────────────────────────────

/// Forward pass for a single new token, using KV cache for past context.
/// Returns logits for the new token only (vocab_size elements).
pub fn emilio_forward_one(
    token_id: usize,
    pos: usize,
    weights: &ModelWeights,
    rope: &RopeCache,
    kv_cache: &mut KVCache,
) -> Vec<f64> {
    let cfg = &weights.config;
    let d = cfg.d_model;

    // 1. Token embedding
    let mut x: Vec<f64> = weights.token_embd[token_id * d..(token_id + 1) * d].to_vec();

    // 2. Transformer layers (single token)
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        x = transformer_layer_one(&x, layer, cfg, rope, pos,
                                   &mut kv_cache.layers[layer_idx]);
    }

    // 3. Final RMSNorm
    let normed = eml_rms_norm(&x, &weights.output_norm, cfg.rms_norm_eps);

    // 4. LM head: (1, d_model) @ (d_model, vocab)
    build_matmul_cse(&normed, &weights.output_weight, 1, d, cfg.vocab_size)
}

fn transformer_layer(
    x: &[f64],
    layer: &LayerWeights,
    cfg: &QwenConfig,
    rope: &RopeCache,
    t: usize,
    _layer_idx: usize,
) -> Vec<f64> {
    let d = cfg.d_model;

    // ── Pre-attention RMSNorm ──────────────────────────────────────
    let mut normed = Vec::with_capacity(t * d);
    for i in 0..t {
        normed.extend(eml_rms_norm(
            &x[i * d..(i + 1) * d],
            &layer.attn_norm,
            cfg.rms_norm_eps,
        ));
    }

    // ── Attention ──────────────────────────────────────────────────
    let attn_out = eml_gqa_attention(&normed, layer, cfg, rope, t);

    // Residual add (0-cost in EML)
    let mut x2 = eml_add_vec(x, &attn_out);

    // ── Pre-FFN RMSNorm ────────────────────────────────────────────
    let mut normed2 = Vec::with_capacity(t * d);
    for i in 0..t {
        normed2.extend(eml_rms_norm(
            &x2[i * d..(i + 1) * d],
            &layer.ffn_norm,
            cfg.rms_norm_eps,
        ));
    }

    // ── SwiGLU FFN ─────────────────────────────────────────────────
    let ffn_out = eml_swiglu_ffn(&normed2, layer, cfg, t);

    // Residual add
    x2 = eml_add_vec(&x2, &ffn_out);
    x2
}

// ─── Single-token transformer layer with KV cache ──────────────────────────

fn transformer_layer_one(
    x: &[f64],
    layer: &LayerWeights,
    cfg: &QwenConfig,
    rope: &RopeCache,
    pos: usize,
    kv: &mut LayerKVCache,
) -> Vec<f64> {
    let d = cfg.d_model;

    // Pre-attention RMSNorm
    let normed = eml_rms_norm(x, &layer.attn_norm, cfg.rms_norm_eps);

    // Attention (single query, cached KV)
    let attn_out = eml_gqa_attention_one(&normed, layer, cfg, rope, pos, kv);

    // Residual
    let mut x2 = eml_add_vec(x, &attn_out);

    // Pre-FFN RMSNorm
    let normed2 = eml_rms_norm(&x2, &layer.ffn_norm, cfg.rms_norm_eps);

    // SwiGLU FFN (single token)
    let ffn_out = eml_swiglu_ffn(&normed2, layer, cfg, 1);

    // Residual
    x2 = eml_add_vec(&x2, &ffn_out);
    x2
}

// ─── Single-token GQA attention with KV cache ──────────────────────────────

fn eml_gqa_attention_one(
    x: &[f64],          // (d_model,) — single normalized token
    layer: &LayerWeights,
    cfg: &QwenConfig,
    rope: &RopeCache,
    pos: usize,
    kv: &mut LayerKVCache,
) -> Vec<f64> {
    let d = cfg.d_model;
    let d_head = cfg.d_head;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let heads_per_kv = n_heads / n_kv_heads;
    let q_dim = n_heads * d_head;
    let kv_dim = n_kv_heads * d_head;

    // QKV projections for single token: (1, d) @ (d, out_dim)
    let q_wt = transpose(&layer.q_weight, q_dim, d);
    let k_wt = transpose(&layer.k_weight, kv_dim, d);
    let v_wt = transpose(&layer.v_weight, kv_dim, d);

    let mut q = build_matmul_cse(x, &q_wt, 1, d, q_dim);
    let mut k_new = build_matmul_cse(x, &k_wt, 1, d, kv_dim);
    let mut v_new = build_matmul_cse(x, &v_wt, 1, d, kv_dim);

    // Add bias
    for j in 0..q_dim {
        q[j] += layer.q_bias[j];
    }
    for j in 0..kv_dim {
        k_new[j] += layer.k_bias[j];
        v_new[j] += layer.v_bias[j];
    }

    // Apply RoPE
    for h in 0..n_heads {
        rope.apply(&mut q[h * d_head..(h + 1) * d_head], pos);
    }
    for h in 0..n_kv_heads {
        rope.apply(&mut k_new[h * d_head..(h + 1) * d_head], pos);
    }

    // Store K,V in cache
    kv.append(&k_new, &v_new);
    let t = kv.len; // total sequence length including this token

    // Attention: single query against all cached K,V
    let mut out = vec![0.0f64; d];
    let scale = to_r(eml_inv(eml_sqrt(to_c(d_head as f64))));

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;

        // Compute scores[j] = dot(Q[h,:], K_cached[kv_h, j, :]) / sqrt(d_head)
        let mut scores = Vec::with_capacity(t);
        for j in 0..t {
            let mut dot = Complex64::new(0.0, 0.0);
            for dd in 0..d_head {
                let qv = q[h * d_head + dd];
                let kv_val = kv.k[j * kv_dim + kv_h * d_head + dd];
                dot = eml_add(dot, eml_mul(to_c(qv), to_c(kv_val)));
            }
            scores.push(to_r(eml_mul(dot, to_c(scale))));
        }

        // Softmax
        let attn_w = build_softmax_cse(&scores);

        // Weighted sum over cached V
        for dd in 0..d_head {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..t {
                let vv = kv.v[j * kv_dim + kv_h * d_head + dd];
                acc = eml_add(acc, eml_mul(to_c(attn_w[j]), to_c(vv)));
            }
            out[h * d_head + dd] = to_r(acc);
        }
    }

    // Output projection
    let o_wt = transpose(&layer.o_weight, d, q_dim);
    build_matmul_cse(&out, &o_wt, 1, q_dim, d)
}

// ─── GQA Attention (full sequence, for prefill) ─────────────────────────────

fn eml_gqa_attention(
    x: &[f64],
    layer: &LayerWeights,
    cfg: &QwenConfig,
    rope: &RopeCache,
    t: usize,
) -> Vec<f64> {
    let d = cfg.d_model;
    let d_head = cfg.d_head;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let heads_per_kv = n_heads / n_kv_heads;

    // QKV projections via CSE matmul
    // GGUF stores weights as (out_features, in_features)
    // So q_weight is (n_heads*d_head, d_model), we need x @ q_weight^T
    // which is (T, d_model) @ (d_model, n_heads*d_head)
    // We need to transpose the weight matrices
    let q_dim = n_heads * d_head;
    let kv_dim = n_kv_heads * d_head;

    // Transpose weights for matmul: (out, in) → (in, out)
    let q_wt = transpose(&layer.q_weight, q_dim, d);
    let k_wt = transpose(&layer.k_weight, kv_dim, d);
    let v_wt = transpose(&layer.v_weight, kv_dim, d);

    let mut q = build_matmul_cse(x, &q_wt, t, d, q_dim);
    let mut k = build_matmul_cse(x, &k_wt, t, d, kv_dim);
    let mut v = build_matmul_cse(x, &v_wt, t, d, kv_dim);

    // Add bias (0-cost EML adds)
    for i in 0..t {
        for j in 0..q_dim {
            q[i * q_dim + j] += layer.q_bias[j];
        }
        for j in 0..kv_dim {
            k[i * kv_dim + j] += layer.k_bias[j];
            v[i * kv_dim + j] += layer.v_bias[j];
        }
    }

    // Apply RoPE
    for pos in 0..t {
        for h in 0..n_heads {
            let start = pos * q_dim + h * d_head;
            rope.apply(&mut q[start..start + d_head], pos);
        }
        for h in 0..n_kv_heads {
            let start = pos * kv_dim + h * d_head;
            rope.apply(&mut k[start..start + d_head], pos);
        }
    }

    // Attention scores + softmax + weighted sum, per head
    let mut out = vec![0.0f64; t * d];
    let scale = to_r(eml_inv(eml_sqrt(to_c(d_head as f64))));

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;

        for i in 0..t {
            // Compute scores[j] = dot(Q[h,i,:], K[kv_h,j,:]) * scale
            let mut scores = Vec::with_capacity(t);
            for j in 0..t {
                if j > i {
                    scores.push(-1e9_f64); // causal mask
                } else {
                    let mut dot = Complex64::new(0.0, 0.0);
                    for dd in 0..d_head {
                        let qv = q[i * q_dim + h * d_head + dd];
                        let kv = k[j * kv_dim + kv_h * d_head + dd];
                        dot = eml_add(dot, eml_mul(to_c(qv), to_c(kv)));
                    }
                    scores.push(to_r(eml_mul(dot, to_c(scale))));
                }
            }

            // Softmax (CSE-optimized)
            let attn_w = build_softmax_cse(&scores);

            // Weighted sum over V
            for dd in 0..d_head {
                let mut acc = Complex64::new(0.0, 0.0);
                for j in 0..t {
                    let vv = v[j * kv_dim + kv_h * d_head + dd];
                    acc = eml_add(acc, eml_mul(to_c(attn_w[j]), to_c(vv)));
                }
                out[i * d + h * d_head + dd] = to_r(acc);
            }
        }
    }

    // Output projection
    let o_wt = transpose(&layer.o_weight, d, q_dim); // (q_dim, d) → (d, q_dim) wait...
    // o_weight is (d_model, n_heads*d_head) in GGUF = (out, in)
    // We need out @ o_weight^T = (T, n_heads*d_head) @ (n_heads*d_head, d_model)
    let o_wt2 = transpose(&layer.o_weight, d, q_dim);
    // o_weight: (d_model, q_dim) → transpose → (q_dim, d_model)
    build_matmul_cse(&out, &o_wt2, t, q_dim, d)
}

fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ─── SwiGLU FFN ─────────────────────────────────────────────────────────────

fn eml_swiglu_ffn(
    x: &[f64],
    layer: &LayerWeights,
    cfg: &QwenConfig,
    t: usize,
) -> Vec<f64> {
    let d = cfg.d_model;
    let d_ff = cfg.d_ff;

    // gate and up projections: x @ W^T
    // gate_weight, up_weight: (d_ff, d_model) in GGUF → transpose to (d_model, d_ff)
    let gate_wt = transpose(&layer.gate_weight, d_ff, d);
    let up_wt = transpose(&layer.up_weight, d_ff, d);

    let gate = build_matmul_cse(x, &gate_wt, t, d, d_ff);
    let up = build_matmul_cse(x, &up_wt, t, d, d_ff);

    // SwiGLU: silu(gate) * up
    let gate_activated = eml_silu(&gate);

    // Element-wise mul via EML
    let hidden = eml_mul_vec(&gate_activated, &up);

    // Down projection
    let down_wt = transpose(&layer.down_weight, d, d_ff);
    // down_weight: (d_model, d_ff) → transpose → (d_ff, d_model)
    build_matmul_cse(&hidden, &down_wt, t, d_ff, d)
}

// ─── Generation ─────────────────────────────────────────────────────────────

// ─── Generation with KV cache ───────────────────────────────────────────────

pub fn emilio_generate(
    prompt: &[usize],
    weights: &ModelWeights,
    rope: &RopeCache,
    max_new: usize,
) -> Vec<usize> {
    let cfg = &weights.config;
    let mut ids = prompt.to_vec();
    let max_len = cfg.max_seq_len.min(prompt.len() + max_new + 16);
    let mut kv_cache = KVCache::new(cfg, max_len);

    // Prefill: process all prompt tokens one-by-one through KV cache path.
    // This is simpler than a batched prefill + cache population.
    eprintln!("  Prefilling {} prompt tokens...", prompt.len());
    let mut _last_logits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        _last_logits = emilio_forward_one(tok, i, weights, rope, &mut kv_cache);
        if (i + 1) % 10 == 0 || i == prompt.len() - 1 {
            eprint!("\r  Prefilled {}/{}", i + 1, prompt.len());
        }
    }
    eprintln!();

    // Decode: generate new tokens one at a time
    for step in 0..max_new {
        let logits = if step == 0 {
            // First decode step: we already have logits from last prefill token
            _last_logits.clone()
        } else {
            // Process the most recently appended token
            let last_tok = *ids.last().unwrap();
            let pos = ids.len() - 1; // position of this token in the sequence
            emilio_forward_one(last_tok, pos, weights, rope, &mut kv_cache)
        };

        // Greedy argmax
        let next_token = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        ids.push(next_token);

        eprint!("\r  Generated {}/{} tokens", step + 1, max_new);

        // Stop on EOS
        if next_token == cfg.vocab_size - 1 { break; } // fallback EOS
    }
    eprintln!();

    ids
}
