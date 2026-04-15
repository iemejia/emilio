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
