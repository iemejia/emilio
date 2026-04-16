/*
 * eml2tag.c -- Convert .eml v2 model to .tag format
 *
 * .tag format: GPU-native, mmap-friendly model for the tag-system engine.
 *
 * Layout:
 *   Header (fixed size):
 *     magic       4B   "TAG1"
 *     version     4B   u32 = 1
 *     config     48B   10 × u32 + 2 × f64 (same fields as EmlConfig)
 *     n_tensors   4B   u32 (total SmTensor count = n_layers * 4 + 1)
 *     vocab_off   8B   u64 offset to tokenizer section
 *     tensor_dir_off 8B u64 offset to tensor directory
 *     data_off    8B   u64 offset to tensor data (page-aligned)
 *
 *   Tokenizer section:
 *     vocab_size  4B   u32
 *     merges_count 4B  u32
 *     bos_id      4B   u32
 *     eos_id      4B   u32
 *     vocab strings: [u32 len, chars...] × vocab_size
 *     merge pairs:   [u32 left_len, chars..., u32 right_len, chars...] × merges_count
 *
 *   Dense f64 arrays (token_embd, output_norm, biases, norms):
 *     For each: [u64 count, f64 × count]
 *
 *   Tensor directory (per SmTensor):
 *     inner       4B   u32
 *     cols        4B   u32
 *     mag_off     8B   u64 (offset from data_off to f16 magnitudes)
 *     sign_off    8B   u64 (offset from data_off to packed u32 sign bits)
 *
 *   Tensor data (page-aligned, 16KB boundary):
 *     f16 magnitudes (inner × cols × 2 bytes, 32-aligned)
 *     u32 packed signs (ceil(inner × cols / 32) × 4 bytes)
 *     ... repeated for each tensor
 *
 * The tensor data section is page-aligned so that mmap can hand
 * pointers directly to Metal newBufferWithBytesNoCopy.
 */

#include "eml_tag.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Stub Metal functions — converter doesn't use GPU */
int metal_init(void) { return 0; }
int metal_available(void) { return 0; }
const char *metal_device_name(void) { return "none"; }
int metal_upload_weights(const double *m, const double *s, int i, int c)
{ (void)m; (void)s; (void)i; (void)c; return -1; }
int metal_register_weights_nocopy(const void *m, const void *s, int i, int c)
{ (void)m; (void)s; (void)i; (void)c; return -1; }
void metal_matmul(double *r, const double *a, int w, int ro, int in, int co)
{ (void)r; (void)a; (void)w; (void)ro; (void)in; (void)co; }
void metal_shutdown(void) {}

/* ── f32 → f16 ────────────────────────────────────────────────────────── */

static unsigned short f32_to_f16(float val)
{
    unsigned int b;
    memcpy(&b, &val, 4);
    unsigned short sign = (b >> 16) & 0x8000;
    int exp_bits = (int)((b >> 23) & 0xFF);
    unsigned int man = b & 0x007FFFFF;

    if (exp_bits == 0) return sign;
    if (exp_bits == 0xFF) return sign | 0x7C00 | (man ? 0x200 : 0);

    int new_exp = exp_bits - 127 + 15;
    if (new_exp >= 31) return sign | 0x7C00;
    if (new_exp <= 0) {
        if (new_exp >= -10) {
            unsigned int shifted = (man | 0x00800000) >> (14 - new_exp);
            return sign | (unsigned short)shifted;
        }
        return sign;
    }
    return sign | ((unsigned short)new_exp << 10) | (unsigned short)(man >> 13);
}

/* ── Write helpers ─────────────────────────────────────────────────────── */

static void w_u32(FILE *f, unsigned int v)
{
    unsigned char buf[4];
    buf[0] = v & 0xFF; buf[1] = (v >> 8) & 0xFF;
    buf[2] = (v >> 16) & 0xFF; buf[3] = (v >> 24) & 0xFF;
    fwrite(buf, 1, 4, f);
}

static void w_u64(FILE *f, unsigned long long v)
{
    unsigned char buf[8];
    int i;
    for (i = 0; i < 8; i++) buf[i] = (v >> (i * 8)) & 0xFF;
    fwrite(buf, 1, 8, f);
}

static void w_f64(FILE *f, double v)
{
    fwrite(&v, 8, 1, f);
}

static void w_string(FILE *f, const char *s)
{
    unsigned int len = s ? (unsigned int)strlen(s) : 0;
    w_u32(f, len);
    if (len > 0) fwrite(s, 1, len, f);
}

static void w_f64_array(FILE *f, const double *data, int count)
{
    w_u64(f, (unsigned long long)count);
    fwrite(data, sizeof(double), (size_t)count, f);
}

/* Pad file to alignment boundary */
static void pad_to(FILE *f, unsigned long long alignment)
{
    long pos = ftell(f);
    unsigned long long rem = (unsigned long long)pos % alignment;
    if (rem != 0) {
        unsigned long long pad = alignment - rem;
        unsigned char zero = 0;
        unsigned long long i;
        for (i = 0; i < pad; i++) fwrite(&zero, 1, 1, f);
    }
}

/* ── Tensor directory entry ────────────────────────────────────────────── */

typedef struct {
    int inner;
    int cols;
    unsigned long long mag_offset;   /* relative to data section start */
    unsigned long long sign_offset;  /* relative to data section start */
} TensorDirEntry;

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *eml_path, *tag_path;
    V2ModelWeights *w;
    Tokenizer tok;
    FILE *out;
    int i, n_tensors;
    TensorDirEntry *dir;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.eml output.tag\n", argv[0]);
        return 1;
    }

    eml_path = argv[1];
    tag_path = argv[2];

    fprintf(stderr, "Loading %s ...\n", eml_path);
    memset(&tok, 0, sizeof(Tokenizer));
    w = load_eml_v2(eml_path, &tok);
    fprintf(stderr, "  Model: %d layers, d_model=%d, vocab=%d\n",
            w->n_layers, w->config.d_model, w->config.vocab_size);

    n_tensors = w->n_layers * 4 + 1;  /* 4 per layer + sm_output */
    dir = (TensorDirEntry *)calloc((size_t)n_tensors, sizeof(TensorDirEntry));

    /* Precompute tensor dimensions */
    {
        int d = w->config.d_model;
        int q_dim = w->config.n_heads * w->config.d_head;
        int kv_dim = w->config.n_kv_heads * w->config.d_head;
        int qkv_dim = q_dim + 2 * kv_dim;
        int d_ff = w->config.d_ff;
        int t = 0;

        for (i = 0; i < w->n_layers; i++) {
            dir[t].inner = d;       dir[t].cols = qkv_dim;   t++;  /* qkv */
            dir[t].inner = q_dim;   dir[t].cols = d;         t++;  /* o */
            dir[t].inner = d;       dir[t].cols = 2 * d_ff;  t++;  /* gate_up */
            dir[t].inner = d_ff;    dir[t].cols = d;         t++;  /* down */
        }
        dir[t].inner = d; dir[t].cols = w->config.vocab_size; t++;  /* output */
    }

    out = fopen(tag_path, "wb");
    if (!out) { fprintf(stderr, "Error: cannot create %s\n", tag_path); return 1; }

    /* ── Header ──────────────────────────────────────────────────────── */
    fwrite("TAG1", 1, 4, out);
    w_u32(out, 1);  /* version */

    /* Config: 10 × u32 + 2 × f64 = 56 bytes */
    w_u32(out, (unsigned int)w->config.vocab_size);
    w_u32(out, (unsigned int)w->config.n_layers);
    w_u32(out, (unsigned int)w->config.n_heads);
    w_u32(out, (unsigned int)w->config.n_kv_heads);
    w_u32(out, (unsigned int)w->config.d_model);
    w_u32(out, (unsigned int)w->config.d_ff);
    w_f64(out, w->config.rope_freq_base);
    w_f64(out, w->config.rms_norm_eps);
    w_u32(out, (unsigned int)w->config.max_seq_len);
    w_u32(out, (unsigned int)w->config.d_head);

    w_u32(out, (unsigned int)n_tensors);

    /* Placeholders for offsets (filled later) */
    long vocab_off_pos = ftell(out);
    w_u64(out, 0);  /* vocab_off */
    long tdir_off_pos = ftell(out);
    w_u64(out, 0);  /* tensor_dir_off */
    long data_off_pos = ftell(out);
    w_u64(out, 0);  /* data_off */

    /* ── Tokenizer section ───────────────────────────────────────────── */
    {
        unsigned long long vocab_off = (unsigned long long)ftell(out);
        fseek(out, vocab_off_pos, SEEK_SET);
        w_u64(out, vocab_off);
        fseek(out, 0, SEEK_END);
    }

    w_u32(out, (unsigned int)tok.vocab_size);
    w_u32(out, (unsigned int)tok.merges_count);
    w_u32(out, (unsigned int)tok.bos_id);
    w_u32(out, (unsigned int)tok.eos_id);

    for (i = 0; i < tok.vocab_size; i++) {
        w_string(out, tok.vocab[i]);
    }
    for (i = 0; i < tok.merges_count; i++) {
        w_string(out, tok.merges[i].left);
        w_string(out, tok.merges[i].right);
    }

    /* ── Dense f64 arrays ────────────────────────────────────────────── */
    /* token_embd: vocab_size × d_model */
    w_f64_array(out, w->token_embd,
                w->config.vocab_size * w->config.d_model);
    /* output_norm: d_model */
    w_f64_array(out, w->output_norm, w->config.d_model);

    /* Per-layer biases and norms */
    for (i = 0; i < w->n_layers; i++) {
        V2LayerWeights *lw = &w->layers[i];
        int q_dim = w->config.n_heads * w->config.d_head;
        int kv_dim = w->config.n_kv_heads * w->config.d_head;

        w_f64_array(out, lw->q_bias,    q_dim);
        w_f64_array(out, lw->k_bias,    kv_dim);
        w_f64_array(out, lw->v_bias,    kv_dim);
        w_f64_array(out, lw->attn_norm, w->config.d_model);
        w_f64_array(out, lw->ffn_norm,  w->config.d_model);
    }

    /* ── Tensor directory ────────────────────────────────────────────── */
    {
        unsigned long long tdir_off = (unsigned long long)ftell(out);
        fseek(out, tdir_off_pos, SEEK_SET);
        w_u64(out, tdir_off);
        fseek(out, 0, SEEK_END);
    }

    /* Write placeholder directory (filled after computing offsets) */
    long dir_file_pos = ftell(out);
    for (i = 0; i < n_tensors; i++) {
        w_u32(out, (unsigned int)dir[i].inner);
        w_u32(out, (unsigned int)dir[i].cols);
        w_u64(out, 0);  /* mag_off placeholder */
        w_u64(out, 0);  /* sign_off placeholder */
    }

    /* ── Tensor data (page-aligned: 16KB) ────────────────────────────── */
    pad_to(out, 16384);

    unsigned long long data_start = (unsigned long long)ftell(out);
    {
        fseek(out, data_off_pos, SEEK_SET);
        w_u64(out, data_start);
        fseek(out, 0, SEEK_END);
    }

    /* Write tensors: f16 magnitudes + packed u32 signs */
    fprintf(stderr, "Writing %d GPU tensors ...\n", n_tensors);
    {
        int t = 0;
        unsigned long long cur_offset = 0;

        /* Helper: write one SmTensor's GPU data */
        #define WRITE_SM_TENSOR(sm, inner_dim, cols_dim) do {            \
            int _n = (inner_dim) * (cols_dim);                            \
            int _n_words = (_n + 31) / 32;                                \
            int _k;                                                       \
                                                                          \
            /* Align to 64 bytes for cache-line alignment */              \
            pad_to(out, 64);                                              \
            cur_offset = (unsigned long long)ftell(out) - data_start;     \
            dir[t].mag_offset = cur_offset;                               \
                                                                          \
            /* f16 magnitudes */                                          \
            for (_k = 0; _k < _n; _k++) {                                \
                unsigned short h = f32_to_f16((float)(sm).magnitudes[_k]);\
                fwrite(&h, 2, 1, out);                                    \
            }                                                             \
                                                                          \
            /* Align signs to 4-byte boundary */                          \
            pad_to(out, 4);                                               \
            cur_offset = (unsigned long long)ftell(out) - data_start;     \
            dir[t].sign_offset = cur_offset;                              \
                                                                          \
            /* packed u32 sign bits (bit=1 → negative) */                 \
            {                                                             \
                unsigned int *bits = (unsigned int *)calloc(              \
                    (size_t)_n_words, sizeof(unsigned int));              \
                for (_k = 0; _k < _n; _k++) {                            \
                    if ((sm).signs[_k] < 0.0)                             \
                        bits[_k >> 5] |= (1u << (_k & 31));              \
                }                                                         \
                fwrite(bits, sizeof(unsigned int),                        \
                       (size_t)_n_words, out);                            \
                free(bits);                                               \
            }                                                             \
            t++;                                                          \
        } while(0)

        for (i = 0; i < w->n_layers; i++) {
            V2LayerWeights *lw = &w->layers[i];
            int d = w->config.d_model;
            int q_dim = w->config.n_heads * w->config.d_head;
            int kv_dim = w->config.n_kv_heads * w->config.d_head;
            int qkv_dim = q_dim + 2 * kv_dim;
            int d_ff = w->config.d_ff;

            WRITE_SM_TENSOR(lw->sm_qkv,     d,     qkv_dim);
            WRITE_SM_TENSOR(lw->sm_o,        q_dim, d);
            WRITE_SM_TENSOR(lw->sm_gate_up,  d,     2 * d_ff);
            WRITE_SM_TENSOR(lw->sm_down,     d_ff,  d);

            if ((i + 1) % 8 == 0 || i == w->n_layers - 1)
                fprintf(stderr, "  Layer %d/%d\n", i + 1, w->n_layers);
        }

        WRITE_SM_TENSOR(w->sm_output, w->config.d_model, w->config.vocab_size);
        #undef WRITE_SM_TENSOR
    }

    /* ── Backpatch tensor directory ──────────────────────────────────── */
    fseek(out, dir_file_pos, SEEK_SET);
    for (i = 0; i < n_tensors; i++) {
        w_u32(out, (unsigned int)dir[i].inner);
        w_u32(out, (unsigned int)dir[i].cols);
        w_u64(out, dir[i].mag_offset);
        w_u64(out, dir[i].sign_offset);
    }

    fseek(out, 0, SEEK_END);
    {
        long total = ftell(out);
        fprintf(stderr, "Written %s: %.1f MB (%d tensors)\n",
                tag_path, (double)total / (1024.0 * 1024.0), n_tensors);
    }

    fclose(out);
    free(dir);
    free_v2_weights(w);
    tokenizer_free(&tok);

    return 0;
}
