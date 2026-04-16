/*
 * eml_metal.h -- Metal GPU bridge for tag-system matmul
 *
 * Provides GPU-accelerated sign+magnitude matmul using the
 * SIMD-cooperative v4 kernel from eml_matmul.metal.
 *
 * On Apple Silicon, uses unified memory for zero-copy buffer access.
 * Falls back to CPU if Metal is unavailable.
 */

#ifndef EML_METAL_H
#define EML_METAL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the Metal GPU engine.
 * Compiles shaders, creates command queue.
 * Returns 1 on success, 0 on failure (Metal not available).
 */
int metal_init(void);

/*
 * Upload sign+magnitude weights to GPU.
 * Converts double magnitudes to half-precision, packs signs to bits.
 * inner must be a multiple of 32 (required by the SIMD kernel).
 * Returns a weight ID (>= 0) on success, -1 on failure.
 */
int metal_upload_weights(const double *magnitudes, const double *signs,
                         int inner, int cols);

/*
 * Register pre-converted GPU-native weights (zero-copy from mmap).
 * mag_f16: pointer to half-precision magnitudes (inner × cols × 2 bytes)
 * sign_bits: pointer to packed u32 sign bits (ceil(inner*cols/32) × 4 bytes)
 * The pointers must remain valid for the lifetime of the engine.
 * Returns a weight ID (>= 0) on success, -1 on failure.
 */
int metal_register_weights_nocopy(const void *mag_f16, const void *sign_bits,
                                  int inner, int cols);

/*
 * GPU-accelerated sign+magnitude matmul.
 * result[i*cols+j] = sum_k a[i*inner+k] * w[j*inner+k]
 * computed via exp(ln|a|+ln|w|) * sign on GPU.
 * weight_id must have been returned by metal_upload_weights.
 */
void metal_matmul(double *result, const double *a,
                  int weight_id, int rows, int inner, int cols);

/* Returns 1 if Metal engine is initialized and available. */
int metal_available(void);

/* Return GPU device name (valid after metal_init). */
const char *metal_device_name(void);

/* Shutdown the Metal engine and free GPU resources. */
void metal_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* EML_METAL_H */
