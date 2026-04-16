/*
 * eml_metal.m -- Metal GPU bridge for tag-system matmul
 *
 * SIMD-cooperative sign+magnitude matmul on Apple Silicon GPU.
 * Uses unified memory for zero-copy access to weight and activation buffers.
 *
 * The v4 kernel: one SIMD group (32 threads) per output element.
 * Each thread processes inner/32 elements and contributes to a
 * simd_sum reduction.  4-way unrolled for ILP.
 *
 * Weight storage: half-precision magnitudes + packed sign bits.
 * ~23% bandwidth reduction vs float magnitudes + float signs.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "eml_metal.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ─── f32 → f16 conversion ────────────────────────────────────────────── */

static uint16_t f32_to_f16(float val)
{
    uint32_t b;
    memcpy(&b, &val, 4);
    uint16_t sign = (b >> 16) & 0x8000;
    int exp_bits = (int)((b >> 23) & 0xFF);
    uint32_t man = b & 0x007FFFFF;

    if (exp_bits == 0) return sign;
    if (exp_bits == 0xFF) return sign | 0x7C00 | (man ? 0x200 : 0);

    int new_exp = exp_bits - 127 + 15;
    if (new_exp >= 31) return sign | 0x7C00;
    if (new_exp <= 0) {
        if (new_exp >= -10) {
            uint32_t shifted = (man | 0x00800000) >> (14 - new_exp);
            return sign | (uint16_t)shifted;
        }
        return sign;
    }
    return sign | ((uint16_t)new_exp << 10) | (uint16_t)(man >> 13);
}

/* ─── Metal shader source (v4 kernel) ─────────────────────────────────── */

static NSString *metalShaderSource =
@"#include <metal_stdlib>\n"
 "using namespace metal;\n"
 "\n"
 "kernel void matmul_sm(\n"
 "    device const float* ln_a_mag    [[buffer(0)]],\n"
 "    device const float* sign_a      [[buffer(1)]],\n"
 "    device const half*  ln_b_mag    [[buffer(2)]],\n"
 "    device const uint*  sign_b_bits [[buffer(3)]],\n"
 "    device       float* result      [[buffer(4)]],\n"
 "    constant     uint&  rows        [[buffer(5)]],\n"
 "    constant     uint&  inner       [[buffer(6)]],\n"
 "    constant     uint&  cols        [[buffer(7)]],\n"
 "    uint2 gid  [[threadgroup_position_in_grid]],\n"
 "    uint  lane [[thread_index_in_simdgroup]])\n"
 "{\n"
 "    uint j = gid.x;\n"
 "    uint i = gid.y;\n"
 "    if (i >= rows || j >= cols) return;\n"
 "\n"
 "    uint a_off = i * inner;\n"
 "    uint b_off = j * inner;\n"
 "    uint b_word_base = b_off >> 5;\n"
 "\n"
 "    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
 "\n"
 "    uint k = lane;\n"
 "    uint word_step = 0;\n"
 "    for (; k + 96 < inner; k += 128, word_step += 4) {\n"
 "        uint ak = a_off + k;\n"
 "        uint bk = b_off + k;\n"
 "        float m0 = ln_a_mag[ak]      + float(ln_b_mag[bk]);\n"
 "        float m1 = ln_a_mag[ak + 32] + float(ln_b_mag[bk + 32]);\n"
 "        float m2 = ln_a_mag[ak + 64] + float(ln_b_mag[bk + 64]);\n"
 "        float m3 = ln_a_mag[ak + 96] + float(ln_b_mag[bk + 96]);\n"
 "        uint w0 = sign_b_bits[b_word_base + word_step];\n"
 "        uint w1 = sign_b_bits[b_word_base + word_step + 1];\n"
 "        uint w2 = sign_b_bits[b_word_base + word_step + 2];\n"
 "        uint w3 = sign_b_bits[b_word_base + word_step + 3];\n"
 "        float sb0 = 1.0f - 2.0f * float((w0 >> lane) & 1);\n"
 "        float sb1 = 1.0f - 2.0f * float((w1 >> lane) & 1);\n"
 "        float sb2 = 1.0f - 2.0f * float((w2 >> lane) & 1);\n"
 "        float sb3 = 1.0f - 2.0f * float((w3 >> lane) & 1);\n"
 "        acc0 += exp(m0) * (sign_a[ak]      * sb0);\n"
 "        acc1 += exp(m1) * (sign_a[ak + 32] * sb1);\n"
 "        acc2 += exp(m2) * (sign_a[ak + 64] * sb2);\n"
 "        acc3 += exp(m3) * (sign_a[ak + 96] * sb3);\n"
 "    }\n"
 "    for (; k < inner; k += 32, word_step++) {\n"
 "        uint ak = a_off + k;\n"
 "        uint bk = b_off + k;\n"
 "        float mag = ln_a_mag[ak] + float(ln_b_mag[bk]);\n"
 "        uint w = sign_b_bits[b_word_base + word_step];\n"
 "        float sb = 1.0f - 2.0f * float((w >> lane) & 1);\n"
 "        acc0 += exp(mag) * (sign_a[ak] * sb);\n"
 "    }\n"
 "\n"
 "    float total = simd_sum(acc0 + acc1 + acc2 + acc3);\n"
 "    if (lane == 0) {\n"
 "        result[i * cols + j] = total;\n"
 "    }\n"
 "}\n";

/* ─── Engine state ────────────────────────────────────────────────────── */

#define MAX_GPU_WEIGHTS 128

@interface EmlMetalEngine : NSObject {
@public
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeMatmul;

    /* Weight storage: half-precision magnitudes + packed sign bits */
    id<MTLBuffer> weightMagBufs[MAX_GPU_WEIGHTS];
    id<MTLBuffer> weightSignBufs[MAX_GPU_WEIGHTS];
    int nWeights;

    /* Scratch buffers (reused across calls) */
    id<MTLBuffer> aMagBuf;
    id<MTLBuffer> aSignBuf;
    id<MTLBuffer> resultBuf;
    int maxASize;
    int maxResultSize;

    /* GPU name for banner */
    char deviceName[128];
}
@end

@implementation EmlMetalEngine

- (instancetype)init
{
    self = [super init];
    if (!self) return nil;

    device = MTLCreateSystemDefaultDevice();
    if (!device) return nil;

    /* Copy device name */
    const char *name = [[device name] UTF8String];
    strncpy(deviceName, name ? name : "Unknown", sizeof(deviceName) - 1);
    deviceName[sizeof(deviceName) - 1] = '\0';

    queue = [device newCommandQueue];
    if (!queue) return nil;

    /* Compile shader */
    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:metalShaderSource
                                              options:opts
                                                error:&err];
    if (!lib) {
        fprintf(stderr, "  Metal shader compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        return nil;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_sm"];
    if (!fn) {
        fprintf(stderr, "  Metal: matmul_sm function not found\n");
        return nil;
    }

    pipeMatmul = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipeMatmul) {
        fprintf(stderr, "  Metal pipeline error: %s\n",
                [[err localizedDescription] UTF8String]);
        return nil;
    }

    nWeights = 0;
    maxASize = 0;
    maxResultSize = 0;

    return self;
}

- (void)ensureScratchA:(int)aLen result:(int)rLen
{
    if (aLen > maxASize) {
        maxASize = aLen;
        aMagBuf = [device newBufferWithLength:(NSUInteger)aLen * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        aSignBuf = [device newBufferWithLength:(NSUInteger)aLen * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    }
    if (rLen > maxResultSize) {
        maxResultSize = rLen;
        resultBuf = [device newBufferWithLength:(NSUInteger)rLen * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    }
}

@end

static EmlMetalEngine *g_engine = nil;

/* ─── C API ───────────────────────────────────────────────────────────── */

int metal_init(void)
{
    @autoreleasepool {
        g_engine = [[EmlMetalEngine alloc] init];
        if (!g_engine || !g_engine->pipeMatmul) {
            g_engine = nil;
            return 0;
        }
        return 1;
    }
}

int metal_available(void)
{
    return g_engine != nil ? 1 : 0;
}

const char *metal_device_name(void)
{
    if (!g_engine) return "none";
    return g_engine->deviceName;
}

int metal_upload_weights(const double *magnitudes, const double *signs,
                         int inner, int cols)
{
    @autoreleasepool {
        if (!g_engine) return -1;
        if (g_engine->nWeights >= MAX_GPU_WEIGHTS) return -1;
        if (inner % 32 != 0) return -1;  /* SIMD kernel requires 32-alignment */

        int n = cols * inner;

        /* Convert double magnitudes → half */
        uint16_t *mag_h = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
        for (int i = 0; i < n; i++) {
            mag_h[i] = f32_to_f16((float)magnitudes[i]);
        }

        /* Pack signs: bit=1 means negative */
        int n_words = (n + 31) / 32;
        uint32_t *sign_bits = (uint32_t *)calloc((size_t)n_words, sizeof(uint32_t));
        for (int i = 0; i < n; i++) {
            if (signs[i] < 0.0) {
                sign_bits[i >> 5] |= (1u << (i & 31));
            }
        }

        id<MTLBuffer> magBuf = [g_engine->device
            newBufferWithBytes:mag_h
                        length:(NSUInteger)n * sizeof(uint16_t)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> signBuf = [g_engine->device
            newBufferWithBytes:sign_bits
                        length:(NSUInteger)n_words * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];

        free(mag_h);
        free(sign_bits);

        if (!magBuf || !signBuf) return -1;

        int wid = g_engine->nWeights++;
        g_engine->weightMagBufs[wid] = magBuf;
        g_engine->weightSignBufs[wid] = signBuf;

        return wid;
    }
}

int metal_register_weights_nocopy(const void *mag_f16, const void *sign_bits,
                                  int inner, int cols)
{
    @autoreleasepool {
        if (!g_engine) return -1;
        if (g_engine->nWeights >= MAX_GPU_WEIGHTS) return -1;

        int n = cols * inner;
        int n_words = (n + 31) / 32;

        /* Zero-copy: wrap mmap'd pointers as Metal buffers.
         * On Apple Silicon unified memory, this avoids any data copy.
         * The deallocator is nil because the mmap owns the memory. */
        id<MTLBuffer> magBuf = [g_engine->device
            newBufferWithBytesNoCopy:(void *)mag_f16
                              length:(NSUInteger)n * sizeof(uint16_t)
                             options:MTLResourceStorageModeShared
                         deallocator:nil];
        id<MTLBuffer> signBuf = [g_engine->device
            newBufferWithBytesNoCopy:(void *)sign_bits
                              length:(NSUInteger)n_words * sizeof(uint32_t)
                             options:MTLResourceStorageModeShared
                         deallocator:nil];

        if (!magBuf || !signBuf) {
            /* newBufferWithBytesNoCopy requires page-aligned pointers.
             * Fall back to copy if alignment doesn't meet requirements. */
            magBuf = [g_engine->device
                newBufferWithBytes:(void *)mag_f16
                            length:(NSUInteger)n * sizeof(uint16_t)
                           options:MTLResourceStorageModeShared];
            signBuf = [g_engine->device
                newBufferWithBytes:(void *)sign_bits
                            length:(NSUInteger)n_words * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];
            if (!magBuf || !signBuf) return -1;
        }

        int wid = g_engine->nWeights++;
        g_engine->weightMagBufs[wid] = magBuf;
        g_engine->weightSignBufs[wid] = signBuf;

        return wid;
    }
}

void metal_matmul(double *result, const double *a,
                  int weight_id, int rows, int inner, int cols)
{
    @autoreleasepool {
        EmlMetalEngine *e = g_engine;
        int a_len = rows * inner;
        int r_len = rows * cols;

        /* Grow scratch buffers if needed */
        [e ensureScratchA:a_len result:r_len];

        /* Compute ln|a| and sign(a) on CPU → write to shared GPU buffers */
        float *a_mag  = (float *)[e->aMagBuf contents];
        float *a_sign = (float *)[e->aSignBuf contents];
        for (int i = 0; i < a_len; i++) {
            float v = (float)a[i];
            float av = fabsf(v);
            a_mag[i]  = logf(av + 1e-30f);
            a_sign[i] = (v >= 0.0f) ? 1.0f : -1.0f;
        }

        /* Dispatch GPU matmul */
        uint32_t u_rows  = (uint32_t)rows;
        uint32_t u_inner = (uint32_t)inner;
        uint32_t u_cols  = (uint32_t)cols;

        id<MTLCommandBuffer> cmdBuf = [e->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:e->pipeMatmul];
        [enc setBuffer:e->aMagBuf                    offset:0 atIndex:0];
        [enc setBuffer:e->aSignBuf                   offset:0 atIndex:1];
        [enc setBuffer:e->weightMagBufs[weight_id]   offset:0 atIndex:2];
        [enc setBuffer:e->weightSignBufs[weight_id]  offset:0 atIndex:3];
        [enc setBuffer:e->resultBuf                  offset:0 atIndex:4];
        [enc setBytes:&u_rows  length:sizeof(u_rows)  atIndex:5];
        [enc setBytes:&u_inner length:sizeof(u_inner) atIndex:6];
        [enc setBytes:&u_cols  length:sizeof(u_cols)  atIndex:7];

        /* One SIMD group (32 threads) per output element */
        MTLSize threadgroups = MTLSizeMake((NSUInteger)cols,
                                           (NSUInteger)rows, 1);
        MTLSize tgSize = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:tgSize];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* Read results: float → double */
        float *gpu_result = (float *)[e->resultBuf contents];
        for (int i = 0; i < r_len; i++) {
            result[i] = (double)gpu_result[i];
        }
    }
}

void metal_shutdown(void)
{
    g_engine = nil;
}
