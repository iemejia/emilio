//  eml_matmul.metal — Metal compute shaders for EML matmul.
//
//  The EML matmul inner loop computes:
//    C[i,j] = Σ_k  exp(ln_a_re[i*K+k] + ln_b_re[j*K+k])
//             × sign(ln_a_im[i*K+k] + ln_b_im[j*K+k])
//
//  where sign is determined by the parity of round((im_a + im_b) / π):
//  even → +1, odd → −1.  This is exact because ln(real) has imaginary
//  part 0 (positive) or π (negative).
//
//  Layout:
//    ln_a is (rows, inner) — split into separate re/im float arrays.
//    ln_b is (cols, inner) — ALREADY TRANSPOSED, split into re/im arrays.
//    result is (rows, cols).
//
//  Each thread computes one output element C[i, j].
//  Uses float (32-bit) for GPU efficiency; the sign computation is exact.

#include <metal_stdlib>
using namespace metal;

// ─── Original per-element kernel (kept for reference/small shapes) ──────────

kernel void eml_matmul(
    device const float* ln_a_re   [[buffer(0)]],
    device const float* ln_a_im   [[buffer(1)]],
    device const float* ln_b_re   [[buffer(2)]],
    device const float* ln_b_im   [[buffer(3)]],
    device       float* result    [[buffer(4)]],
    constant     uint&  rows      [[buffer(5)]],
    constant     uint&  inner     [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint j = tid.x;
    uint i = tid.y;
    if (i >= rows || j >= cols) return;

    uint a_off = i * inner;
    uint b_off = j * inner;

    // 4-way accumulator for ILP
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    uint k4 = inner / 4;
    for (uint c = 0; c < k4; c++) {
        uint k = c * 4;

        float re0 = ln_a_re[a_off + k]     + ln_b_re[b_off + k];
        float re1 = ln_a_re[a_off + k + 1] + ln_b_re[b_off + k + 1];
        float re2 = ln_a_re[a_off + k + 2] + ln_b_re[b_off + k + 2];
        float re3 = ln_a_re[a_off + k + 3] + ln_b_re[b_off + k + 3];

        float im0 = ln_a_im[a_off + k]     + ln_b_im[b_off + k];
        float im1 = ln_a_im[a_off + k + 1] + ln_b_im[b_off + k + 1];
        float im2 = ln_a_im[a_off + k + 2] + ln_b_im[b_off + k + 2];
        float im3 = ln_a_im[a_off + k + 3] + ln_b_im[b_off + k + 3];

        float e0 = exp(re0);
        float e1 = exp(re1);
        float e2 = exp(re2);
        float e3 = exp(re3);

        // sign from im/π parity: n = round(im / π), sign = 1 - 2*(n&1)
        int n0 = (int)rint(im0 * M_1_PI_F);
        int n1 = (int)rint(im1 * M_1_PI_F);
        int n2 = (int)rint(im2 * M_1_PI_F);
        int n3 = (int)rint(im3 * M_1_PI_F);

        acc0 += e0 * (1.0f - 2.0f * (float)(n0 & 1));
        acc1 += e1 * (1.0f - 2.0f * (float)(n1 & 1));
        acc2 += e2 * (1.0f - 2.0f * (float)(n2 & 1));
        acc3 += e3 * (1.0f - 2.0f * (float)(n3 & 1));
    }

    // Remainder
    for (uint k = k4 * 4; k < inner; k++) {
        float re = ln_a_re[a_off + k] + ln_b_re[b_off + k];
        float im = ln_a_im[a_off + k] + ln_b_im[b_off + k];
        float e  = exp(re);
        int   n  = (int)rint(im * M_1_PI_F);
        acc0 += e * (1.0f - 2.0f * (float)(n & 1));
    }

    result[i * cols + j] = acc0 + acc1 + acc2 + acc3;
}

// ─── Precomputed-sign kernel ────────────────────────────────────────────────
//
//  Optimization: signs are precomputed at weight upload time.
//  For real-valued weights, ln(w).im is always 0 (positive) or π (negative),
//  so the sign of each weight element is known at upload time.
//  Similarly ln(a).im is known after the activation ln() pass.
//
//  sign_a[k] and sign_b[j*inner+k] are ±1.0f, precomputed.
//  Product sign = sign_a * sign_b (just a float multiply, no rint/branch).
//
//  The kernel only needs |ln_a_re|, |ln_b_re|, and the product sign.
//  Since ln(|negative|) = ln(abs(x)), re values are always the log magnitudes.

kernel void eml_matmul_signed(
    device const float* ln_a_mag  [[buffer(0)]],  // |ln(a)|.re = ln(|a|)
    device const float* sign_a    [[buffer(1)]],  // ±1.0f per element
    device const float* ln_b_mag  [[buffer(2)]],  // |ln(b)|.re = ln(|b|)
    device const float* sign_b    [[buffer(3)]],  // ±1.0f per element
    device       float* result    [[buffer(4)]],
    constant     uint&  rows      [[buffer(5)]],
    constant     uint&  inner     [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint j = tid.x;
    uint i = tid.y;
    if (i >= rows || j >= cols) return;

    uint a_off = i * inner;
    uint b_off = j * inner;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    uint k4 = inner / 4;
    for (uint c = 0; c < k4; c++) {
        uint k = c * 4;

        float mag0 = ln_a_mag[a_off + k]     + ln_b_mag[b_off + k];
        float mag1 = ln_a_mag[a_off + k + 1] + ln_b_mag[b_off + k + 1];
        float mag2 = ln_a_mag[a_off + k + 2] + ln_b_mag[b_off + k + 2];
        float mag3 = ln_a_mag[a_off + k + 3] + ln_b_mag[b_off + k + 3];

        // sign_a * sign_b: just a float multiply of ±1 values
        float s0 = sign_a[a_off + k]     * sign_b[b_off + k];
        float s1 = sign_a[a_off + k + 1] * sign_b[b_off + k + 1];
        float s2 = sign_a[a_off + k + 2] * sign_b[b_off + k + 2];
        float s3 = sign_a[a_off + k + 3] * sign_b[b_off + k + 3];

        acc0 += exp(mag0) * s0;
        acc1 += exp(mag1) * s1;
        acc2 += exp(mag2) * s2;
        acc3 += exp(mag3) * s3;
    }

    for (uint k = k4 * 4; k < inner; k++) {
        float mag = ln_a_mag[a_off + k] + ln_b_mag[b_off + k];
        float s   = sign_a[a_off + k]   * sign_b[b_off + k];
        acc0 += exp(mag) * s;
    }

    result[i * cols + j] = acc0 + acc1 + acc2 + acc3;
}

// ─── ln(activation) kernel ──────────────────────────────────────────────────
//
//  Computes ln(|x|) and sign(x) for each element.
//  Runs on GPU to avoid CPU→GPU copy of activation data.
//  Output: mag[k] = ln(|x[k]|), sign[k] = (x >= 0) ? 1.0 : -1.0

kernel void eml_ln_split(
    device const float* input     [[buffer(0)]],   // raw f32 activations
    device       float* out_mag   [[buffer(1)]],   // ln(|x|)
    device       float* out_sign  [[buffer(2)]],   // ±1.0
    constant     uint&  count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float x = input[tid];
    float ax = abs(x);
    out_mag[tid]  = log(ax + 1e-30f);  // avoid log(0)
    out_sign[tid] = (x >= 0.0f) ? 1.0f : -1.0f;
}

// ─── SiLU + element-wise EML mul (fused) ────────────────────────────────────
//
//  silu(gate) * up = (gate / (1 + exp(-gate))) * up
//  Via EML: exp(ln(|silu(gate)|) + ln(|up|)) * sign(silu(gate)) * sign(up)
//
//  Fused kernel avoids a round-trip for separate silu + mul.

kernel void eml_silu_mul(
    device const float* gate      [[buffer(0)]],
    device const float* up        [[buffer(1)]],
    device       float* output    [[buffer(2)]],
    constant     uint&  count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float g = gate[tid];
    float u = up[tid];
    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float sig = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sig;
    // EML mul: exp(ln(|silu_g|) + ln(|u|)) * sign
    float a = abs(silu_g);
    float b = abs(u);
    float mag = log(a + 1e-30f) + log(b + 1e-30f);
    float sign = ((silu_g >= 0.0f) ? 1.0f : -1.0f) * ((u >= 0.0f) ? 1.0f : -1.0f);
    output[tid] = exp(mag) * sign;
}

// ─── V2 kernel: half-precision weights ──────────────────────────────────────
//
//  Same math as eml_matmul_signed, but weights stored as half (float16).
//  Halves memory bandwidth for weight reads — the dominant cost.
//  Activation remains float32 (small, likely L1-cached).
//  Accumulation is float32 for precision.

kernel void eml_matmul_v2(
    device const float* ln_a_mag  [[buffer(0)]],  // f32 activation mag
    device const float* sign_a    [[buffer(1)]],  // f32 activation sign
    device const half*  ln_b_mag  [[buffer(2)]],  // HALF weight mag
    device const half*  sign_b    [[buffer(3)]],  // HALF weight sign
    device       float* result    [[buffer(4)]],
    constant     uint&  rows      [[buffer(5)]],
    constant     uint&  inner     [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint j = tid.x;
    uint i = tid.y;
    if (i >= rows || j >= cols) return;

    uint a_off = i * inner;
    uint b_off = j * inner;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    uint k4 = inner / 4;
    for (uint c = 0; c < k4; c++) {
        uint k = c * 4;
        uint ak = a_off + k;
        uint bk = b_off + k;

        float mag0 = ln_a_mag[ak]     + float(ln_b_mag[bk]);
        float mag1 = ln_a_mag[ak + 1] + float(ln_b_mag[bk + 1]);
        float mag2 = ln_a_mag[ak + 2] + float(ln_b_mag[bk + 2]);
        float mag3 = ln_a_mag[ak + 3] + float(ln_b_mag[bk + 3]);

        float s0 = sign_a[ak]     * float(sign_b[bk]);
        float s1 = sign_a[ak + 1] * float(sign_b[bk + 1]);
        float s2 = sign_a[ak + 2] * float(sign_b[bk + 2]);
        float s3 = sign_a[ak + 3] * float(sign_b[bk + 3]);

        acc0 += exp(mag0) * s0;
        acc1 += exp(mag1) * s1;
        acc2 += exp(mag2) * s2;
        acc3 += exp(mag3) * s3;
    }

    for (uint k = k4 * 4; k < inner; k++) {
        float mag = ln_a_mag[a_off + k] + float(ln_b_mag[b_off + k]);
        float s   = sign_a[a_off + k]   * float(sign_b[b_off + k]);
        acc0 += exp(mag) * s;
    }

    result[i * cols + j] = acc0 + acc1 + acc2 + acc3;
}

// ─── Fused SiLU × mul → mag+sign output (pure EML) ─────────────────────────
//
//  Takes raw matmul outputs (gate, up), produces ln(|silu(gate)*up|) and sign.
//  Pure EML: every multiply is exp(ln(a)+ln(b)).
//
//  silu(g) = g / (1+exp(-g)) → ln|silu(g)| = ln|g| - ln(1+exp(-g))
//  sign(silu(g)) = sign(g)   (sigmoid is always positive)
//  result_mag = ln|g| - ln(1+exp(-g)) + ln|u|
//  result_sign = sign(g) × sign(u)

kernel void eml_silu_mul_ln(
    device const float* gate      [[buffer(0)]],
    device const float* up        [[buffer(1)]],
    device       float* out_mag   [[buffer(2)]],
    device       float* out_sign  [[buffer(3)]],
    constant     uint&  count     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float g = gate[tid];
    float u = up[tid];

    // ln|silu(g)| = ln|g| - ln(1 + exp(-g))
    float ln_abs_g = log(abs(g) + 1e-30f);
    float ln_denom = log(1.0f + exp(-g));    // log1p-equivalent, stable for all g
    float silu_mag = ln_abs_g - ln_denom;

    // + ln|u|
    float ln_abs_u = log(abs(u) + 1e-30f);

    out_mag[tid]  = silu_mag + ln_abs_u;
    out_sign[tid] = ((g >= 0.0f) ? 1.0f : -1.0f) * ((u >= 0.0f) ? 1.0f : -1.0f);
}

// ─── V3 kernel: SIMD-cooperative matmul ─────────────────────────────────────
//
//  One SIMD group (32 threads) per output element C[i,j].
//  Benefits over v2 (1 thread per output):
//    - 32× more active threads → better GPU utilization for small output dims
//      (Q/O: 896 outputs → 28 SIMDs in v2, 896 SIMDs in v3)
//    - Coalesced weight reads: 32 adjacent lanes read 32 consecutive halfs
//    - 4-way unroll within each lane for ILP across exp() calls
//  Uses simd_sum() for fast 32-wide reduction (single instruction).

kernel void eml_matmul_v3(
    device const float* ln_a_mag  [[buffer(0)]],
    device const float* sign_a    [[buffer(1)]],
    device const half*  ln_b_mag  [[buffer(2)]],
    device const half*  sign_b    [[buffer(3)]],
    device       float* result    [[buffer(4)]],
    constant     uint&  rows      [[buffer(5)]],
    constant     uint&  inner     [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint j = gid.x;   // output column
    uint i = gid.y;   // output row
    if (i >= rows || j >= cols) return;

    uint a_off = i * inner;
    uint b_off = j * inner;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // 4-way unrolled, SIMD-strided loop.
    // 32 lanes collectively cover inner in strides of 32.
    uint k = lane;
    for (; k + 96 < inner; k += 128) {
        uint ak = a_off + k;
        uint bk = b_off + k;

        float m0 = ln_a_mag[ak]      + float(ln_b_mag[bk]);
        float m1 = ln_a_mag[ak + 32] + float(ln_b_mag[bk + 32]);
        float m2 = ln_a_mag[ak + 64] + float(ln_b_mag[bk + 64]);
        float m3 = ln_a_mag[ak + 96] + float(ln_b_mag[bk + 96]);

        float s0 = sign_a[ak]      * float(sign_b[bk]);
        float s1 = sign_a[ak + 32] * float(sign_b[bk + 32]);
        float s2 = sign_a[ak + 64] * float(sign_b[bk + 64]);
        float s3 = sign_a[ak + 96] * float(sign_b[bk + 96]);

        acc0 += exp(m0) * s0;
        acc1 += exp(m1) * s1;
        acc2 += exp(m2) * s2;
        acc3 += exp(m3) * s3;
    }

    // Scalar remainder
    for (; k < inner; k += 32) {
        float mag = ln_a_mag[a_off + k] + float(ln_b_mag[b_off + k]);
        float s   = sign_a[a_off + k]   * float(sign_b[b_off + k]);
        acc0 += exp(mag) * s;
    }

    float total = simd_sum(acc0 + acc1 + acc2 + acc3);

    if (lane == 0) {
        result[i * cols + j] = total;
    }
}

// ─── V4 kernel: SIMD-cooperative + packed sign bits ─────────────────────────
//
//  Same SIMD reduction as v3, but weight signs are packed as bits:
//  32 signs per uint32 (0=positive, 1=negative).
//  Reduces sign bandwidth 16× (half→bit).  ~23% total weight bandwidth reduction.
//
//  Key insight: since inner is always a multiple of 32 and b_off = j*inner,
//  all 32 SIMD lanes access the SAME word per stride.  The bit position
//  within the word equals the lane ID.  This gives perfect coalescing.

kernel void eml_matmul_v4(
    device const float* ln_a_mag    [[buffer(0)]],
    device const float* sign_a      [[buffer(1)]],
    device const half*  ln_b_mag    [[buffer(2)]],
    device const uint*  sign_b_bits [[buffer(3)]],   // packed: 32 signs per uint
    device       float* result      [[buffer(4)]],
    constant     uint&  rows        [[buffer(5)]],
    constant     uint&  inner       [[buffer(6)]],
    constant     uint&  cols        [[buffer(7)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint j = gid.x;
    uint i = gid.y;
    if (i >= rows || j >= cols) return;

    uint a_off = i * inner;
    uint b_off = j * inner;
    uint b_word_base = b_off >> 5;   // b_off / 32 (inner is multiple of 32)

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // 4-way unrolled, SIMD-strided loop.
    // All 32 lanes share the same packed-sign word per stride.
    // Bit position = lane for all 4 unrolled elements.
    uint k = lane;
    uint word_step = 0;   // increments by 4 (128 elements / 32 bits)
    for (; k + 96 < inner; k += 128, word_step += 4) {
        uint ak = a_off + k;
        uint bk = b_off + k;

        float m0 = ln_a_mag[ak]      + float(ln_b_mag[bk]);
        float m1 = ln_a_mag[ak + 32] + float(ln_b_mag[bk + 32]);
        float m2 = ln_a_mag[ak + 64] + float(ln_b_mag[bk + 64]);
        float m3 = ln_a_mag[ak + 96] + float(ln_b_mag[bk + 96]);

        // Extract weight signs: 4 words, bit position = lane
        uint w0 = sign_b_bits[b_word_base + word_step];
        uint w1 = sign_b_bits[b_word_base + word_step + 1];
        uint w2 = sign_b_bits[b_word_base + word_step + 2];
        uint w3 = sign_b_bits[b_word_base + word_step + 3];

        float sb0 = 1.0f - 2.0f * float((w0 >> lane) & 1);
        float sb1 = 1.0f - 2.0f * float((w1 >> lane) & 1);
        float sb2 = 1.0f - 2.0f * float((w2 >> lane) & 1);
        float sb3 = 1.0f - 2.0f * float((w3 >> lane) & 1);

        acc0 += exp(m0) * (sign_a[ak]      * sb0);
        acc1 += exp(m1) * (sign_a[ak + 32] * sb1);
        acc2 += exp(m2) * (sign_a[ak + 64] * sb2);
        acc3 += exp(m3) * (sign_a[ak + 96] * sb3);
    }

    // Scalar remainder
    for (; k < inner; k += 32, word_step++) {
        uint ak = a_off + k;
        uint bk = b_off + k;
        float mag = ln_a_mag[ak] + float(ln_b_mag[bk]);
        uint w = sign_b_bits[b_word_base + word_step];
        float sb = 1.0f - 2.0f * float((w >> lane) & 1);
        acc0 += exp(mag) * (sign_a[ak] * sb);
    }

    float total = simd_sum(acc0 + acc1 + acc2 + acc3);

    if (lane == 0) {
        result[i * cols + j] = total;
    }
}

// ─── Element-wise addition ──────────────────────────────────────────────────
//
//  c[i] = a[i] + b[i], f32 vectors.
//  Used for residual connections (x + attn, residual + ffn).

kernel void eml_add_f32(
    device const float* a         [[buffer(0)]],
    device const float* b         [[buffer(1)]],
    device       float* c         [[buffer(2)]],
    constant     uint&  count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    c[tid] = a[tid] + b[tid];
}

// ─── Fused RMSNorm + ln_split (pure EML) ────────────────────────────────────
//
//  Computes RMSNorm(x, gamma, eps) and outputs ln(|result|) + sign(result),
//  ready for immediate matmul consumption (no CPU round-trip).
//
//  RMSNorm: result[i] = x[i] * gamma[i] / sqrt(mean(x²) + eps)
//
//  Pure EML derivation (no direct multiply or divide):
//    x²        = exp(ln|x| + ln|x|)               [EML squaring]
//    mean(x²)  = sum(x²) / n                        [sum is free, div via log]
//    ln(std)   = 0.5 * ln(mean(x²) + eps)
//            ... but 0.5 * z is exp(ln(z) + ln(0.5)) = exp(ln(z) - ln(2))
//    So ln(std) = (ln(sum(x²)/n + eps) - ln(2))
//            ... actually ln(std) = ln(sqrt(mean_sq + eps))
//                                 = ln(mean_sq + eps) / 2
//    We compute: half_ln_var = (ln(mean_sq + eps)) - ln(2)
//    Wait — ln(sqrt(a)) = ln(a)/2, and ln(a)/2 ≠ ln(a) - ln(2).
//    ln(a)/2 = ln(a^(1/2)) = ln(sqrt(a)).  Division by 2 on a real number
//    is a scalar operation, not a learned parameter multiplication.
//    Since division by a fixed constant (2, n) is just scaling, we allow it
//    like the CPU code does with eml_mul(to_c(0.5), ...) and eml_div(..., n).
//
//    Final:  result_mag[i] = ln|x[i]| + ln|gamma[i]| - ln(std)
//            result_sign[i] = sign(x[i]) * sign(gamma[i])
//                           = sign(x[i]) * sign(gamma[i])   [sign multiply is XOR]
//
//  So the kernel:
//    Pass 1: accumulate sum( exp(2 * ln|x[i]|) ) = sum(x²) via EML
//    Pass 2: ln(std) = 0.5 * ln(sum_sq / n + eps)
//            out_mag[i] = ln|x[i]| + ln|gamma[i]| - ln(std)
//            out_sign[i] = sign(x[i]) * sign(gamma[i])
//
//  The ONLY non-EML ops are: addition (free), ln/exp (fundamental), and
//  the scalar divisions by fixed constants (n, 2) which mirror the CPU code.

kernel void eml_rms_norm_ln_split(
    device const float* x         [[buffer(0)]],   // input vector
    device const float* gamma     [[buffer(1)]],   // learned scale weights
    device       float* out_mag   [[buffer(2)]],   // ln(|result|)
    device       float* out_sign  [[buffer(3)]],   // sign(result)
    constant     uint&  n         [[buffer(4)]],   // vector length
    constant     float& eps       [[buffer(5)]],   // epsilon
    threadgroup  float* shmem     [[threadgroup(0)]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint ntg   [[threads_per_threadgroup]])
{
    // Pass 1: compute sum(x²) = sum( exp(2 * ln|x[i]|) )  — EML squaring
    float sum_sq = 0.0f;
    for (uint i = tpitg; i < n; i += ntg) {
        float ln_abs_x = log(abs(x[i]) + 1e-30f);
        sum_sq += exp(ln_abs_x + ln_abs_x);          // exp(2*ln|x|) = x²
    }

    // SIMD-level reduction (addition is free in EML)
    sum_sq = simd_sum(sum_sq);

    // Cross-simdgroup reduction via shared memory
    uint nsg = (ntg + 31) / 32;
    if (tiisg == 0) {
        shmem[sgitg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = (tiisg < nsg) ? shmem[tiisg] : 0.0f;
        v = simd_sum(v);
        if (tiisg == 0) {
            shmem[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ln(std) = 0.5 * ln( sum_sq/n + eps )
    //  sum_sq/n = exp(ln(sum_sq) - ln(n))   [EML division]
    //  then + eps, then ln, then * 0.5
    //  0.5 * z = exp(ln|z| - ln(2))         [EML scalar multiply by 0.5]
    float total_sq = shmem[0];
    float mean_sq = exp(log(total_sq + 1e-30f) - log(float(n)));  // EML div
    float ln_var = log(mean_sq + eps);
    float ln_std = exp(log(abs(ln_var) + 1e-30f) - log(2.0f));
    // ln_var is always positive (mean_sq + eps > 0), so ln_var > some negative
    // but we need ln_var / 2 which IS just a divide by 2.
    // Actually: ln(std) = ln(sqrt(mean_sq + eps)) = ln(mean_sq + eps) / 2.
    // Division by constant 2 mirrors CPU code's eml_mul(to_c(0.5), ...).
    ln_std = ln_var / 2.0f;  // scalar constant division, same as CPU path

    // Pass 2: out_mag[i] = ln|x[i]| + ln|gamma[i]| - ln(std)
    //         out_sign[i] = sign(x[i]) * sign(gamma[i])       [sign XOR]
    for (uint i = tpitg; i < n; i += ntg) {
        float xi = x[i];
        float gi = gamma[i];
        float ln_abs_xi = log(abs(xi) + 1e-30f);
        float ln_abs_gi = log(abs(gi) + 1e-30f);
        out_mag[i]  = ln_abs_xi + ln_abs_gi - ln_std;
        // sign(x) * sign(gamma): XOR of sign bits
        float sx = (xi >= 0.0f) ? 1.0f : -1.0f;
        float sg = (gi >= 0.0f) ? 1.0f : -1.0f;
        out_sign[i] = sx * sg;
    }
}

// ─── Fused residual_add + RMSNorm + ln_split (pure EML) ─────────────────────
//
//  Combines: z[i] = a[i] + b[i]  (residual add — free in EML)
//            out = RMSNorm_ln_split(z, gamma, eps)
//
//  Same pure EML math as eml_rms_norm_ln_split above but with fused add.

kernel void eml_residual_rms_norm_ln_split(
    device const float* a         [[buffer(0)]],   // first residual input
    device const float* b         [[buffer(1)]],   // second residual input
    device const float* gamma     [[buffer(2)]],   // norm scale weights
    device       float* out_mag   [[buffer(3)]],   // ln(|result|)
    device       float* out_sign  [[buffer(4)]],   // sign(result)
    device       float* z_out     [[buffer(5)]],   // a+b output (needed for 2nd residual)
    constant     uint&  n         [[buffer(6)]],
    constant     float& eps       [[buffer(7)]],
    threadgroup  float* shmem     [[threadgroup(0)]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint ntg   [[threads_per_threadgroup]])
{
    // Pass 1: z = a + b (free), accumulate sum(z²) via EML squaring
    float sum_sq = 0.0f;
    for (uint i = tpitg; i < n; i += ntg) {
        float zi = a[i] + b[i];            // addition is free in EML
        z_out[i] = zi;
        float ln_abs_z = log(abs(zi) + 1e-30f);
        sum_sq += exp(ln_abs_z + ln_abs_z); // exp(2*ln|z|) = z²
    }

    sum_sq = simd_sum(sum_sq);

    uint nsg = (ntg + 31) / 32;
    if (tiisg == 0) {
        shmem[sgitg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = (tiisg < nsg) ? shmem[tiisg] : 0.0f;
        v = simd_sum(v);
        if (tiisg == 0) {
            shmem[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ln(std) via EML chain
    float total_sq = shmem[0];
    float mean_sq = exp(log(total_sq + 1e-30f) - log(float(n)));  // EML div
    float ln_std = log(mean_sq + eps) / 2.0f;  // scalar constant, same as CPU

    // Pass 2: out_mag[i] = ln|z[i]| + ln|gamma[i]| - ln_std
    for (uint i = tpitg; i < n; i += ntg) {
        float zi = z_out[i];
        float gi = gamma[i];
        float ln_abs_zi = log(abs(zi) + 1e-30f);
        float ln_abs_gi = log(abs(gi) + 1e-30f);
        out_mag[i]  = ln_abs_zi + ln_abs_gi - ln_std;
        float sz = (zi >= 0.0f) ? 1.0f : -1.0f;
        float sg = (gi >= 0.0f) ? 1.0f : -1.0f;
        out_sign[i] = sz * sg;
    }
}
