//! Pure EML operations — fused algebraic forms + Rayon parallelism.
//!
//! Each derived op is annotated with:
//!   • Its EML tree (what the Python version evaluates)
//!   • The algebraic simplification (what we actually compute)
//!   • The transcendental cost (# of exp/log calls)

use num_complex::Complex64;
use rayon::prelude::*;

// ─── The single primitive ───────────────────────────────────────────────────

/// eml(x, y) = exp(x) - ln(y)   — the one and only gate.
#[inline(always)]
pub fn eml(x: Complex64, y: Complex64) -> Complex64 {
    x.exp() - y.ln()
}

// ─── Helpers ────────────────────────────────────────────────────────────────

pub const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Convert f64 to Complex64 (zero imaginary part).
#[inline(always)]
pub fn to_c(v: f64) -> Complex64 {
    Complex64::new(v, 0.0)
}

/// Project complex result back to real.
#[inline(always)]
pub fn to_r(c: Complex64) -> f64 {
    c.re
}

// ─── Unfused (tree-walking) scalar ops — match Python exactly ───────────────
// These exist so we can verify equivalence before relying on the fused variants.

pub fn eml_exp_unfused(x: Complex64) -> Complex64 {
    eml(x, ONE)
}

pub fn eml_ln_unfused(z: Complex64) -> Complex64 {
    eml(ONE, eml(eml(ONE, z), ONE))
}

pub fn eml_sub_unfused(a: Complex64, b: Complex64) -> Complex64 {
    eml(eml_ln_unfused(a), eml_exp_unfused(b))
}

pub fn eml_neg_unfused(x: Complex64) -> Complex64 {
    eml_sub_unfused(eml_ln_unfused(ONE), x)
}

pub fn eml_add_unfused(a: Complex64, b: Complex64) -> Complex64 {
    eml_sub_unfused(a, eml_neg_unfused(b))
}

pub fn eml_inv_unfused(z: Complex64) -> Complex64 {
    eml_exp_unfused(eml_neg_unfused(eml_ln_unfused(z)))
}

pub fn eml_mul_unfused(a: Complex64, b: Complex64) -> Complex64 {
    eml_exp_unfused(eml_add_unfused(eml_ln_unfused(a), eml_ln_unfused(b)))
}

// ─── Fused scalar ops — algebraically identical, fewer transcendentals ──────

/// exp(x) = eml(x, 1) = exp(x) - ln(1) = exp(x)
/// Cost: 1 exp
#[inline(always)]
pub fn eml_exp(x: Complex64) -> Complex64 {
    x.exp()
}

/// ln(z): the EML tree cancels to just ln(z).
/// Cost: 1 log
#[inline(always)]
pub fn eml_ln(z: Complex64) -> Complex64 {
    z.ln()
}

/// sub(a, b) = eml(ln(a), exp(b)) = exp(ln(a)) - ln(exp(b)) = a - b
/// Cost: 0 transcendentals
#[inline(always)]
pub fn eml_sub(a: Complex64, b: Complex64) -> Complex64 {
    a - b
}

/// neg(x) = sub(0, x) = -x
/// Cost: 0
#[inline(always)]
pub fn eml_neg(x: Complex64) -> Complex64 {
    -x
}

/// add(a, b) = sub(a, neg(b)) = a + b
/// Cost: 0
#[inline(always)]
pub fn eml_add(a: Complex64, b: Complex64) -> Complex64 {
    a + b
}

/// inv(z) = exp(-ln(z)) = 1/z
/// Cost: 0 (hardware division)
#[inline(always)]
pub fn eml_inv(z: Complex64) -> Complex64 {
    ONE / z
}

/// mul(a, b) = exp(ln(a) + ln(b))
/// Cost: 2 log + 1 exp
#[inline(always)]
pub fn eml_mul(a: Complex64, b: Complex64) -> Complex64 {
    (a.ln() + b.ln()).exp()
}

/// div(a, b) = exp(ln(a) - ln(b))
/// Cost: 2 log + 1 exp
#[inline(always)]
pub fn eml_div(a: Complex64, b: Complex64) -> Complex64 {
    (a.ln() - b.ln()).exp()
}

/// pow(a, b) = exp(b * ln(a))
/// Cost: 1 log + 1 exp + 1 complex mul
#[inline(always)]
pub fn eml_pow(a: Complex64, b: Complex64) -> Complex64 {
    (b * a.ln()).exp()
}

/// sqrt(x) = exp(0.5 * ln(x))
/// Cost: 1 log + 1 exp
#[inline(always)]
pub fn eml_sqrt(x: Complex64) -> Complex64 {
    (Complex64::new(0.5, 0.0) * x.ln()).exp()
}

/// gelu(x) ≈ x * sigmoid(1.702x)
/// sigmoid(z) = 1 / (1 + exp(-z))
/// Fused: exp(ln(x) + ln(sigmoid(1.702x)))
/// Cost: 2 exp + 3 log (sig needs 1 exp, mul needs 2 log + 1 exp)
#[inline(always)]
pub fn eml_gelu(x: Complex64) -> Complex64 {
    let c = Complex64::new(1.702, 0.0);
    let sig = ONE / (ONE + (-c * x).exp());
    // x * sig via log-domain (EML mul)
    (x.ln() + sig.ln()).exp()
}

// ─── Composite ops (parallel via Rayon) ─────────────────────────────────────

/// softmax(x) — numerically stable, parallelised.
pub fn eml_softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let shifted_exps: Vec<Complex64> = x
        .par_iter()
        .map(|&xi| eml_exp(eml_sub(to_c(xi), to_c(m))))
        .collect();

    let z: Complex64 = shifted_exps.iter().copied().reduce(eml_add).unwrap_or(ONE);

    shifted_exps
        .par_iter()
        .map(|&e| to_r(eml_div(e, z)))
        .collect()
}

/// Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta
/// x: row-major (rows, cols), gamma/beta: (cols,)
pub fn eml_layer_norm(
    x: &[f64],
    gamma: &[f64],
    beta: &[f64],
    rows: usize,
    cols: usize,
    eps: f64,
) -> Vec<f64> {
    (0..rows)
        .into_par_iter()
        .flat_map(|i| {
            let row = &x[i * cols..(i + 1) * cols];
            let n = to_c(cols as f64);

            let sum: Complex64 = row.iter().map(|&v| to_c(v)).reduce(eml_add).unwrap();
            let mean = eml_div(sum, n);

            let diff: Vec<Complex64> = row.iter().map(|&v| eml_sub(to_c(v), mean)).collect();

            let var_sum: Complex64 = diff.iter().map(|&d| eml_mul(d, d)).reduce(eml_add).unwrap();
            let var = eml_div(var_sum, n);
            let std = eml_sqrt(eml_add(var, to_c(eps)));

            (0..cols)
                .map(|j| {
                    let normed = eml_div(diff[j], std);
                    let scaled = eml_mul(normed, to_c(gamma[j]));
                    to_r(eml_add(scaled, to_c(beta[j])))
                })
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// Matrix multiply: C = A @ B
/// A: (I, K), B: (K, J) → C: (I, J), all row-major.
pub fn eml_matmul(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let (row, col) = (idx / cols, idx % cols);
            let mut acc = eml_mul(to_c(a[row * inner]), to_c(b[col]));
            for kk in 1..inner {
                acc = eml_add(acc, eml_mul(to_c(a[row * inner + kk]), to_c(b[kk * cols + col])));
            }
            to_r(acc)
        })
        .collect()
}

/// Element-wise GELU — parallel.
pub fn eml_gelu_vec(x: &[f64]) -> Vec<f64> {
    x.par_iter().map(|&v| to_r(eml_gelu(to_c(v)))).collect()
}

/// Element-wise add — parallel.
pub fn eml_add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| to_r(eml_add(to_c(x), to_c(y))))
        .collect()
}

/// Element-wise sub — parallel.
pub fn eml_sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| to_r(eml_sub(to_c(x), to_c(y))))
        .collect()
}

/// Element-wise mul — parallel.
pub fn eml_mul_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| to_r(eml_mul(to_c(x), to_c(y))))
        .collect()
}

/// Scalar mul (broadcast) — parallel.
pub fn eml_scalar_mul(scalar: f64, x: &[f64]) -> Vec<f64> {
    let s = to_c(scalar);
    x.par_iter().map(|&v| to_r(eml_mul(s, to_c(v)))).collect()
}

/// Scalar neg — parallel.
pub fn eml_neg_vec(x: &[f64]) -> Vec<f64> {
    x.par_iter().map(|&v| to_r(eml_neg(to_c(v)))).collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        let err = (a - b).abs();
        assert!(
            err < tol,
            "expected {b}, got {a}, err={err:.2e} (tol={tol:.0e})"
        );
    }

    #[test]
    fn test_fused_exp() {
        approx(to_r(eml_exp(to_c(0.0))), 1.0, 1e-12);
        approx(to_r(eml_exp(to_c(1.0))), std::f64::consts::E, 1e-12);
    }

    #[test]
    fn test_fused_ln() {
        approx(to_r(eml_ln(to_c(1.0))), 0.0, 1e-12);
        approx(to_r(eml_ln(to_c(std::f64::consts::E))), 1.0, 1e-12);
    }

    #[test]
    fn test_fused_sub() {
        approx(to_r(eml_sub(to_c(5.0), to_c(3.0))), 2.0, 1e-12);
        approx(to_r(eml_sub(to_c(0.5), to_c(2.0))), -1.5, 1e-12);
    }

    #[test]
    fn test_fused_neg() {
        approx(to_r(eml_neg(to_c(5.0))), -5.0, 1e-12);
        approx(to_r(eml_neg(to_c(-3.0))), 3.0, 1e-12);
    }

    #[test]
    fn test_fused_add() {
        approx(to_r(eml_add(to_c(1.0), to_c(2.0))), 3.0, 1e-12);
        approx(to_r(eml_add(to_c(-1.0), to_c(1.0))), 0.0, 1e-12);
    }

    #[test]
    fn test_fused_mul() {
        approx(to_r(eml_mul(to_c(0.5), to_c(4.0))), 2.0, 1e-9);
        approx(to_r(eml_mul(to_c(3.0), to_c(5.0))), 15.0, 1e-9);
        approx(to_r(eml_mul(to_c(-1.0), to_c(5.0))), -5.0, 1e-9);
    }

    #[test]
    fn test_fused_div() {
        approx(to_r(eml_div(to_c(6.0), to_c(3.0))), 2.0, 1e-9);
        approx(to_r(eml_div(to_c(1.0), to_c(4.0))), 0.25, 1e-9);
    }

    #[test]
    fn test_fused_sqrt() {
        approx(to_r(eml_sqrt(to_c(9.0))), 3.0, 1e-9);
        approx(to_r(eml_sqrt(to_c(2.0))), std::f64::consts::SQRT_2, 1e-9);
    }

    #[test]
    fn test_fused_inv() {
        approx(to_r(eml_inv(to_c(2.0))), 0.5, 1e-12);
        approx(to_r(eml_inv(to_c(4.0))), 0.25, 1e-12);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sm = eml_softmax(&x);
        let sum: f64 = sm.iter().sum();
        approx(sum, 1.0, 1e-9);
        // monotonic: sm[3] > sm[2] > sm[1] > sm[0]
        assert!(sm[3] > sm[2]);
        assert!(sm[2] > sm[1]);
    }

    #[test]
    fn test_matmul() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = eml_matmul(&a, &b, 2, 2, 2);
        approx(c[0], 19.0, 1e-6);
        approx(c[1], 22.0, 1e-6);
        approx(c[2], 43.0, 1e-6);
        approx(c[3], 50.0, 1e-6);
    }

    #[test]
    fn test_unfused_matches_fused() {
        let a = to_c(2.5);
        let b = to_c(3.7);
        approx(to_r(eml_mul(a, b)), to_r(eml_mul_unfused(a, b)), 1e-9);
        approx(to_r(eml_add(a, b)), to_r(eml_add_unfused(a, b)), 1e-9);
        approx(to_r(eml_sub(a, b)), to_r(eml_sub_unfused(a, b)), 1e-9);
    }

    #[test]
    fn test_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let result = eml_layer_norm(&x, &gamma, &beta, 1, 4, 1e-5);
        // mean=2.5, should be zero-centered
        let sum: f64 = result.iter().sum();
        approx(sum, 0.0, 1e-6);
    }
}
