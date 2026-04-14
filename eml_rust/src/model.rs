//! Tiny EML Transformer — port of eml_model.py.
//!
//! Architecture: 1 layer, 2 heads, d_model=16, d_ff=32, vocab=64.
//! Every mathematical operation is pure EML (fused).
//! Weights are random — this is architecture, not a trained model.

use crate::eml_ops::*;
use rayon::prelude::*;

// ─── Config ─────────────────────────────────────────────────────────────────

pub const VOCAB: usize = 64;
pub const D_MODEL: usize = 16;
pub const N_HEADS: usize = 2;
pub const D_HEAD: usize = D_MODEL / N_HEADS; // 8
pub const D_FF: usize = 32;
pub const MAX_SEQ: usize = 16;

// ─── 2-D matrix helper (row-major) ─────────────────────────────────────────

#[derive(Clone)]
pub struct Mat {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn ones(cols: usize) -> Self {
        Self {
            data: vec![1.0; cols],
            rows: 1,
            cols,
        }
    }

    pub fn zeros_1d(cols: usize) -> Self {
        Self {
            data: vec![0.0; cols],
            rows: 1,
            cols,
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Slice rows [start..end) — returns a new Mat (cheap copy for small dims).
    pub fn slice_rows(&self, start: usize, end: usize) -> Mat {
        let rows = end - start;
        Mat {
            data: self.data[start * self.cols..end * self.cols].to_vec(),
            rows,
            cols: self.cols,
        }
    }

    /// Slice columns [start..end) from every row.
    pub fn slice_cols(&self, start: usize, end: usize) -> Mat {
        let new_cols = end - start;
        let mut out = Vec::with_capacity(self.rows * new_cols);
        for r in 0..self.rows {
            for c in start..end {
                out.push(self.get(r, c));
            }
        }
        Mat {
            data: out,
            rows: self.rows,
            cols: new_cols,
        }
    }

    /// Transpose.
    pub fn t(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Flat slice ref.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

pub struct Weights {
    pub wte: Mat,      // (VOCAB, D_MODEL)
    pub wpe: Mat,      // (MAX_SEQ, D_MODEL)
    pub attn_qkv: Mat, // (D_MODEL, 3*D_MODEL)
    pub attn_out: Mat,  // (D_MODEL, D_MODEL)
    pub ln1_g: Vec<f64>,
    pub ln1_b: Vec<f64>,
    pub ff1: Mat,      // (D_MODEL, D_FF)
    pub ff2: Mat,      // (D_FF, D_MODEL)
    pub ln2_g: Vec<f64>,
    pub ln2_b: Vec<f64>,
    pub lnf_g: Vec<f64>,
    pub lnf_b: Vec<f64>,
    pub lm_head: Mat,  // (D_MODEL, VOCAB)
}

/// Simple xoshiro256** PRNG — matches numpy's default_rng(42) bit patterns
/// only in structure, not in exact values. That's fine: we just need
/// reproducible random weights, not identical to Python's.
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to initialise the state from a single seed
        let mut s = seed;
        let mut state = [0u64; 4];
        for st in &mut state {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *st = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
            let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
            if u1 > 0.0 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos(); // EML_AUDIT:OK — Box-Muller weight init (not inference path)
            }
        }
    }

    fn normal_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next_normal() * scale).collect()
    }

    /// Public access to the PRNG for benchmarks.
    pub fn next_normal_pub(&mut self) -> f64 {
        self.next_normal()
    }
}

impl Weights {
    pub fn init(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let scale = 0.02;

        let wte_data = rng.normal_vec(VOCAB * D_MODEL, scale);
        let wpe_data = rng.normal_vec(MAX_SEQ * D_MODEL, scale);
        let attn_qkv_data = rng.normal_vec(D_MODEL * 3 * D_MODEL, scale);
        let attn_out_data = rng.normal_vec(D_MODEL * D_MODEL, scale);
        let ff1_data = rng.normal_vec(D_MODEL * D_FF, scale);
        let ff2_data = rng.normal_vec(D_FF * D_MODEL, scale);

        // lm_head = wte^T
        let wte = Mat {
            data: wte_data,
            rows: VOCAB,
            cols: D_MODEL,
        };
        let lm_head = wte.t();

        Weights {
            wte,
            wpe: Mat { data: wpe_data, rows: MAX_SEQ, cols: D_MODEL },
            attn_qkv: Mat { data: attn_qkv_data, rows: D_MODEL, cols: 3 * D_MODEL },
            attn_out: Mat { data: attn_out_data, rows: D_MODEL, cols: D_MODEL },
            ln1_g: vec![1.0; D_MODEL],
            ln1_b: vec![0.0; D_MODEL],
            ff1: Mat { data: ff1_data, rows: D_MODEL, cols: D_FF },
            ff2: Mat { data: ff2_data, rows: D_FF, cols: D_MODEL },
            ln2_g: vec![1.0; D_MODEL],
            ln2_b: vec![0.0; D_MODEL],
            lnf_g: vec![1.0; D_MODEL],
            lnf_b: vec![0.0; D_MODEL],
            lm_head,
        }
    }
}

// ─── EML Attention ──────────────────────────────────────────────────────────

fn eml_attention(x: &Mat, w: &Weights) -> Mat {
    let t = x.rows;

    // 1. QKV projection: (T, D) @ (D, 3D)
    let qkv = mat_mul(x, &w.attn_qkv);

    // Split Q, K, V
    let q_flat = qkv.slice_cols(0, D_MODEL);
    let k_flat = qkv.slice_cols(D_MODEL, 2 * D_MODEL);
    let v_flat = qkv.slice_cols(2 * D_MODEL, 3 * D_MODEL);

    // Reshape to (N_HEADS, T, D_HEAD) — stored as flat [h][i][d]
    // q_flat is (T, D_MODEL) where D_MODEL = N_HEADS * D_HEAD
    // q[h, i, d] = q_flat[i, h*D_HEAD + d]

    // 2. Scaled dot-product attention
    let scale = eml_sqrt(to_c(D_HEAD as f64));
    let scale_inv = eml_inv(scale);

    // scores[h, i, j] = dot(Q[h,i,:], K[h,j,:]) / sqrt(D_HEAD)
    // Parallelise over (h, i, j) triples
    let total = N_HEADS * t * t;
    let scores_flat: Vec<f64> = (0..total)
        .into_par_iter()
        .map(|idx| {
            let h = idx / (t * t);
            let rem = idx % (t * t);
            let i = rem / t;
            let j = rem % t;
            let mut acc = eml_mul(
                to_c(q_flat.get(i, h * D_HEAD)),
                to_c(k_flat.get(j, h * D_HEAD)),
            );
            for d in 1..D_HEAD {
                acc = eml_add(
                    acc,
                    eml_mul(
                        to_c(q_flat.get(i, h * D_HEAD + d)),
                        to_c(k_flat.get(j, h * D_HEAD + d)),
                    ),
                );
            }
            to_r(eml_mul(acc, scale_inv))
        })
        .collect();

    // 3. Causal mask + softmax
    let large_neg = -1e9_f64;
    let mut attn_weights = vec![0.0_f64; N_HEADS * t * t];
    for h in 0..N_HEADS {
        for i in 0..t {
            // Apply causal mask
            let mut row = Vec::with_capacity(t);
            for j in 0..t {
                let s = scores_flat[h * t * t + i * t + j];
                if j > i {
                    row.push(s + large_neg);
                } else {
                    row.push(s);
                }
            }
            // Softmax this row
            let sm = eml_softmax(&row);
            for j in 0..t {
                attn_weights[h * t * t + i * t + j] = sm[j];
            }
        }
    }

    // 4. Weighted sum: out[h,i,d] = sum_j attn[h,i,j] * V[h,j,d]
    let out_total = N_HEADS * t * D_HEAD;
    let out_flat: Vec<f64> = (0..out_total)
        .into_par_iter()
        .map(|idx| {
            let h = idx / (t * D_HEAD);
            let rem = idx % (t * D_HEAD);
            let i = rem / D_HEAD;
            let d = rem % D_HEAD;
            let mut acc = eml_mul(
                to_c(attn_weights[h * t * t + i * t]),
                to_c(v_flat.get(0, h * D_HEAD + d)),
            );
            for j in 1..t {
                acc = eml_add(
                    acc,
                    eml_mul(
                        to_c(attn_weights[h * t * t + i * t + j]),
                        to_c(v_flat.get(j, h * D_HEAD + d)),
                    ),
                );
            }
            to_r(acc)
        })
        .collect();

    // 5. Merge heads: (N_HEADS, T, D_HEAD) → (T, D_MODEL)
    // out_flat[h * T * D_HEAD + i * D_HEAD + d] → merged[i, h * D_HEAD + d]
    let mut merged = Mat::zeros(t, D_MODEL);
    for h in 0..N_HEADS {
        for i in 0..t {
            for d in 0..D_HEAD {
                merged.set(i, h * D_HEAD + d, out_flat[h * t * D_HEAD + i * D_HEAD + d]);
            }
        }
    }

    // 6. Output projection
    mat_mul(&merged, &w.attn_out)
}

// ─── EML FFN ────────────────────────────────────────────────────────────────

fn eml_ffn(x: &Mat, w: &Weights) -> Mat {
    let h = mat_mul(x, &w.ff1); // (T, D_FF)
    let h_gelu = Mat {
        data: eml_gelu_vec(h.as_slice()),
        rows: h.rows,
        cols: h.cols,
    };
    mat_mul(&h_gelu, &w.ff2) // (T, D_MODEL)
}

// ─── Layer norm ─────────────────────────────────────────────────────────────

fn layer_norm(x: &Mat, gamma: &[f64], beta: &[f64]) -> Mat {
    let result = eml_layer_norm(x.as_slice(), gamma, beta, x.rows, x.cols, 1e-5);
    Mat {
        data: result,
        rows: x.rows,
        cols: x.cols,
    }
}

// ─── Matrix multiply wrapper ────────────────────────────────────────────────

fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows, "matmul dimension mismatch: ({},{}) @ ({},{})", a.rows, a.cols, b.rows, b.cols);
    let result = eml_matmul(a.as_slice(), b.as_slice(), a.rows, a.cols, b.cols);
    Mat {
        data: result,
        rows: a.rows,
        cols: b.cols,
    }
}

// ─── Element-wise add of two matrices ───────────────────────────────────────

fn mat_add(a: &Mat, b: &Mat) -> Mat {
    Mat {
        data: eml_add_vec(a.as_slice(), b.as_slice()),
        rows: a.rows,
        cols: a.cols,
    }
}

// ─── Transformer layer ─────────────────────────────────────────────────────

fn eml_transformer_layer(x: &Mat, w: &Weights) -> Mat {
    // Pre-attention layer norm
    let x_norm = layer_norm(x, &w.ln1_g, &w.ln1_b);

    // Attention + residual
    let attn_out = eml_attention(&x_norm, w);
    let x2 = mat_add(x, &attn_out);

    // Pre-FFN layer norm
    let x_norm2 = layer_norm(&x2, &w.ln2_g, &w.ln2_b);

    // FFN + residual
    let ffn_out = eml_ffn(&x_norm2, w);
    mat_add(&x2, &ffn_out)
}

// ─── Full forward pass ─────────────────────────────────────────────────────

pub fn eml_forward(token_ids: &[usize], w: &Weights) -> Mat {
    let t = token_ids.len();
    assert!(t <= MAX_SEQ, "Sequence too long: {} > {}", t, MAX_SEQ);

    // Token embeddings (discrete lookup)
    let mut x = Mat::zeros(t, D_MODEL);
    for (i, &tok) in token_ids.iter().enumerate() {
        for d in 0..D_MODEL {
            x.set(i, d, w.wte.get(tok, d));
        }
    }

    // Positional embeddings + add (EML)
    let pos = w.wpe.slice_rows(0, t);
    x = mat_add(&x, &pos);

    // Transformer layer
    x = eml_transformer_layer(&x, w);

    // Final layer norm
    x = layer_norm(&x, &w.lnf_g, &w.lnf_b);

    // LM head: (T, D_MODEL) @ (D_MODEL, VOCAB)
    mat_mul(&x, &w.lm_head)
}

// ─── Generation ─────────────────────────────────────────────────────────────

pub fn eml_generate(prompt: &[usize], w: &Weights, max_new: usize, temperature: f64) -> Vec<usize> {
    let mut ids = prompt.to_vec();

    for _ in 0..max_new {
        let ctx = if ids.len() > MAX_SEQ {
            &ids[ids.len() - MAX_SEQ..]
        } else {
            &ids
        };

        let logits = eml_forward(ctx, w);
        let t = logits.rows;

        // Last position logits
        let mut next_logits: Vec<f64> = (0..VOCAB).map(|v| logits.get(t - 1, v)).collect();

        // Temperature scaling
        if (temperature - 1.0).abs() > 1e-12 {
            let inv_temp = to_r(eml_inv(to_c(temperature)));
            next_logits = eml_mul_vec(
                &next_logits,
                &vec![inv_temp; VOCAB],
            );
        }

        // Softmax → argmax (greedy)
        let probs = eml_softmax(&next_logits);
        let next_token = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        ids.push(next_token);
    }

    ids
}
