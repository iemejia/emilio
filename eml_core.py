"""
EML Core: All math from eml(x, y) = exp(x) - ln(y), constant 1.

Every function here is derived from the paper:
  Odrzywołek, "All elementary functions from a single binary operator" (2026)
  arXiv:2603.21852

Bootstrapping chain (from the paper's EML compiler):
  exp  → ln  → sub → zero → neg → add → inv → mul → div → pow → sqrt

The single key identity that breaks all circularity:
  sub(a, b) = eml(ln(a), exp(b)) = exp(ln(a)) - ln(exp(b)) = a - b  ✓

We work in complex128 numpy internally (the paper says: "computations must be
done in the complex domain" for generating π, i, trig functions etc.).
The eml() primitive uses numpy only for exp() and log() — the hardware
implementation of the one primitive gate.  Every derived operation above is
built purely from eml() calls.
"""

import numpy as np

# ─── The single primitive ────────────────────────────────────────────────────

def eml(x, y):
    """The one and only primitive: exp(x) - ln(y)

    Operates in complex128 per the paper: "computations must be done in the
    complex domain".  This is exactly how the paper's NumPy test harness
    defines eml.  The imaginary parts cancel in complete EML expressions
    for real-valued functions, so callers take .real when a real result is
    needed.
    """
    with np.errstate(all="ignore"):
        x = np.complex128(x)
        y = np.complex128(y)
    return np.exp(x) - np.log(y)

# ─── Constants (from 1 alone) ─────────────────────────────────────────────────

ONE = 1.0

# ─── Level 0: exp and ln — direct from paper abstract ─────────────────────────

def eml_exp(x):
    """exp(x) = eml(x, 1)"""
    return eml(x, ONE)

def eml_ln(z):
    """
    ln(z) = eml(1, eml(eml(1, z), 1))
    Proof:
      inner = eml(1, z) = e - ln(z)
      eml(inner, 1) = exp(e - ln(z)) - 0 = e^e / z
      eml(1, e^e/z) = e - ln(e^e/z) = e - (e - ln z) = ln(z)  ✓
    """
    return eml(ONE, eml(eml(ONE, z), ONE))

# ─── Level 1: sub — the key identity that breaks circularity ──────────────────

def eml_sub(a, b):
    """
    sub(a, b) = a - b = eml(ln(a), exp(b))
    Proof:
      eml(ln(a), exp(b)) = exp(ln(a)) - ln(exp(b)) = a - b  ✓

    From the paper's official EML compiler (eml_compiler_v4.py).
    This is the foundation — it needs only eml_ln and eml_exp, no circularity.
    """
    return eml(eml_ln(a), eml_exp(b))

# ─── Level 2: zero, neg ───────────────────────────────────────────────────────

def const_zero():
    """0 = ln(1) = eml_ln(1)"""
    return eml_ln(ONE)

def eml_neg(x):
    """neg(x) = 0 - x = sub(0, x)"""
    return eml_sub(const_zero(), x)

# ─── Level 3: add ─────────────────────────────────────────────────────────────

def eml_add(a, b):
    """add(a, b) = a + b = sub(a, neg(b)) = a - (-b)"""
    return eml_sub(a, eml_neg(b))

# ─── Level 4: inv (reciprocal) ────────────────────────────────────────────────

def eml_inv(z):
    """inv(z) = 1/z = exp(-ln(z)) = exp(neg(ln(z)))"""
    return eml_exp(eml_neg(eml_ln(z)))

# ─── Level 5: mul ─────────────────────────────────────────────────────────────

def eml_mul(a, b):
    """mul(a, b) = a * b = exp(ln(a) + ln(b)) = exp(add(ln(a), ln(b)))"""
    return eml_exp(eml_add(eml_ln(a), eml_ln(b)))

# ─── Level 6: div ─────────────────────────────────────────────────────────────

def eml_div(a, b):
    """div(a, b) = a / b = mul(a, inv(b))"""
    return eml_mul(a, eml_inv(b))

# ─── Level 7: pow ─────────────────────────────────────────────────────────────

def eml_pow(a, b):
    """pow(a, b) = a^b = exp(b * ln(a)) = exp(mul(b, ln(a)))"""
    return eml_exp(eml_mul(b, eml_ln(a)))

# ─── Level 8: sqrt ────────────────────────────────────────────────────────────

def eml_sqrt(x):
    """sqrt(x) = x^(1/2) = exp(mul(1/2, ln(x)))
    1/2 = inv(add(1, 1)) — pure EML constant
    """
    two = eml_add(ONE, ONE)
    half = eml_inv(two)
    return eml_exp(eml_mul(half, eml_ln(x)))

# ─── Derived constants ────────────────────────────────────────────────────────

def const_e():
    """e = exp(1) = eml(1, 1)"""
    return eml_exp(ONE)

def const_neg_one():
    """-1 = neg(1)"""
    return eml_neg(ONE)

# ─── Composite ops for the transformer ────────────────────────────────────────

def eml_softmax(x):
    """
    softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

    Numerically stable: subtract max first.
    max is a comparison op (not EML-derivable, not continuous).
    We use it only as a meta-operation for numerical stability.

    All arithmetic (exp, sub, add, div, ln) is pure EML.
    Results are projected back to reals (.real) since softmax outputs
    probabilities.
    """
    m = np.max(np.real(x))  # stability shift — discrete comparison

    # exp(x_i - m) for each i — pure EML per element
    shifted_exps = np.array([eml_exp(eml_sub(xi, m)) for xi in x.flat])
    shifted_exps = shifted_exps.reshape(x.shape)

    # Z = sum of exps — via repeated eml_add
    Z = shifted_exps.flat[0]
    for val in shifted_exps.flat[1:]:
        Z = eml_add(Z, val)

    # softmax_i = exp(x_i - m) / Z = div(exp_i, Z)
    result = np.array([np.real(eml_div(e, Z)) for e in shifted_exps.flat])
    return result.reshape(x.shape)


def eml_layer_norm(x, gamma, beta, eps=1e-5):
    """
    LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta

    All ops are pure EML: add, sub, mul, div, sqrt.
    """
    N = x.shape[-1]
    n_val = float(N)

    # mean = sum(x) / N along last axis
    def _sum_last_axis(arr):
        if arr.ndim == 1:
            acc = arr[0]
            for i in range(1, len(arr)):
                acc = eml_add(acc, arr[i])
            return np.array([acc])
        result = []
        for row in arr:
            acc = row[0]
            for i in range(1, len(row)):
                acc = eml_add(acc, row[i])
            result.append(acc)
        return np.array(result).reshape(arr.shape[:-1] + (1,))

    x_sum = _sum_last_axis(x)
    mean = np.vectorize(lambda s: eml_div(s, n_val))(x_sum)

    # diff = x - mean (broadcast)
    diff = np.vectorize(eml_sub)(x, mean)

    # var = sum(diff^2) / N
    diff_sq = np.vectorize(lambda d: eml_mul(d, d))(diff)
    var_sum = _sum_last_axis(diff_sq)
    var = np.vectorize(lambda s: eml_div(s, n_val))(var_sum)

    # std = sqrt(var + eps)
    std = np.vectorize(lambda v: eml_sqrt(eml_add(v, eps)))(var)

    # result = (x - mean) / std * gamma + beta — project to real at end
    normed = np.vectorize(eml_div)(diff, std)
    scaled = np.vectorize(eml_mul)(normed, gamma)
    result = np.vectorize(lambda a, b: np.real(eml_add(a, b)))(scaled, beta)
    return result.astype(np.float64)


def eml_matmul(A, B):
    """
    Matrix multiply: C[i,j] = sum_k mul(A[i,k], B[k,j])

    Every mul and add is pure EML. This is slow but honest.
    """
    if A.ndim == 1 and B.ndim == 1:
        acc = eml_mul(A[0], B[0])
        for k in range(1, len(A)):
            acc = eml_add(acc, eml_mul(A[k], B[k]))
        return acc

    I_ = A.shape[0]
    K_ = A.shape[1]
    J_ = B.shape[1]
    C = np.empty((I_, J_), dtype=np.float64)
    for i in range(I_):
        for j in range(J_):
            acc = eml_mul(A[i, 0], B[0, j])
            for k in range(1, K_):
                acc = eml_add(acc, eml_mul(A[i, k], B[k, j]))
            C[i, j] = np.real(acc)
    return C


def eml_gelu(x):
    """
    GELU(x) ≈ x * sigmoid(1.702 * x)
    sigmoid(z) = 1 / (1 + exp(-z))

    All ops pure EML: mul, neg, exp, add, inv.
    """
    c = 1.702
    z = eml_mul(c, x)                        # 1.702 * x
    neg_z = eml_neg(z)                        # -z
    exp_neg_z = eml_exp(neg_z)                # exp(-z)
    one_plus = eml_add(ONE, exp_neg_z)        # 1 + exp(-z)
    sig = eml_inv(one_plus)                   # 1 / (1 + exp(-z)) = sigmoid(z)
    return eml_mul(x, sig)                    # x * sigmoid(z)


def eml_relu(x):
    """ReLU(x) = max(0, x) — not continuous, can't be EML-derived for arbitrary x.
    The paper covers continuous elementary functions.
    GELU or SiLU are the honest EML-compatible activations.
    """
    raise NotImplementedError("ReLU is not EML-derivable (discontinuous)")
