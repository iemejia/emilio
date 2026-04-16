/*
 * eml_tag.h -- Emilio tag-system inference engine (header)
 *
 * Defines:
 *   1. Shared types (EmlConfig, SmTensor, etc.) compatible with eml_mov.h
 *   2. Tag-system-specific types (TagKind, TagState, TagSystem)
 *
 * Computational model:
 *   A 2-tag system (Post, 1943) extended with parameterized productions.
 *   The alphabet is finite: {operation_kinds} × {layer_indices} ∪ {STATE}.
 *   Productions are determined by model weights (fixed after loading).
 *   The word (FIFO queue) has constant length 2: [OP, STATE].
 *   Each tag step advances one neural network sub-operation.
 *
 * Reference: Post (1943), Cocke & Minsky (1964)
 */

#ifndef EML_TAG_H
#define EML_TAG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==== Libc compatibility (matches eml_mov.h interface) ================== */

typedef size_t eml_size_t;
#define EML_NULL NULL
typedef FILE EML_FILE;

/* ==== Constants ========================================================= */

#define EML_NEG_INF   (-(1.0/0.0))
#define EML_POS_INF   (1.0/0.0)
#define EML_PI        3.14159265358979323846
#define EML_LN2       0.69314718055994530942
#define EML_INV_LN2   1.44269504088896340736

#define EML_EOS_ID    151643
#define EML_EOT_ID    151645

/* ==== Model Configuration =============================================== */

typedef struct {
    int vocab_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int d_model;
    int d_ff;
    double rope_freq_base;
    double rms_norm_eps;
    int max_seq_len;
    int d_head;
} EmlConfig;

/* ==== Sign+Magnitude Tensor ============================================ */

typedef struct {
    double *magnitudes;   /* ln|w_k| per element */
    double *signs;        /* +1.0 or -1.0 per element */
    int len;
    int gpu_weight_id;    /* Metal GPU weight handle (-1 = CPU only) */
} SmTensor;

/* ==== V2 Layer Weights ================================================== */

typedef struct {
    SmTensor sm_qkv;
    SmTensor sm_o;
    SmTensor sm_gate_up;
    SmTensor sm_down;
    double *q_bias;
    double *k_bias;
    double *v_bias;
    double *attn_norm;
    double *ffn_norm;
} V2LayerWeights;

/* ==== V2 Model Weights ================================================== */

typedef struct {
    EmlConfig config;
    double *token_embd;
    double *output_norm;
    SmTensor sm_output;
    V2LayerWeights *layers;
    int n_layers;
} V2ModelWeights;

/* ==== RoPE Cache ======================================================== */

typedef struct {
    double *cos_cache;
    double *sin_cache;
    int d_half;
    int max_len;
} RopeCache;

/* ==== KV Cache ========================================================== */

typedef struct {
    double *k;
    double *v;
    int len;
    int max_len;
    int kv_dim;
} LayerKVCache;

typedef struct {
    LayerKVCache *layers;
    int n_layers;
} KVCache;

/* ==== Tokenizer ========================================================= */

typedef struct {
    char *str;
    int id;
} TokenEntry;

typedef struct {
    char *left;
    char *right;
    int rank;
} MergeEntry;

typedef struct {
    char **vocab;
    int vocab_size;
    TokenEntry *sorted;
    int sorted_count;
    MergeEntry *merges;
    int merges_count;
    int byte_to_unicode[256];
    int unicode_to_byte[512];
    int bos_id;
    int eos_id;
    int im_start_id;
    int im_end_id;
} Tokenizer;

/* ==== Tokenizer functions (defined in eml_tokenizer.c) ================== */

void tokenizer_init_tables(Tokenizer *tok);
int *tokenizer_encode(const Tokenizer *tok, const char *text, int *out_len);
char *tokenizer_decode(const Tokenizer *tok, const int *ids, int n);
int *tokenizer_encode_chat(const Tokenizer *tok, const char *msg, int *out_len);
void tokenizer_free(Tokenizer *tok);

/* ========================================================================
 * TAG SYSTEM TYPES
 *
 * The 2-tag system alphabet:
 *   Each symbol is (kind, layer_index).  The 'kind' selects the
 *   production rule; the layer_index parameterises it with the
 *   corresponding weight tensors.
 *
 *   The second symbol in every pair is STATE, which carries the
 *   mutable computation state (activation vectors, residuals).
 *
 * Word invariant:
 *   |word| = 2 at every step:  [OP_symbol, STATE_symbol]
 *   After 315 steps per token, the system halts.
 * ======================================================================== */

typedef enum {
    TAG_HALT = 0,       /* Halting symbol -- computation complete           */
    TAG_STATE,          /* Data-carrier symbol (activation + residual)      */
    /* --- Transformer operations (one production rule each) --- */
    TAG_EMBED,          /* P: look up token embedding                      */
    TAG_RMSNORM_ATTN,   /* P: pre-attention RMSNorm + save residual        */
    TAG_QKV_MATMUL,     /* P: fused QKV projection (sign+mag matmul)       */
    TAG_BIAS_ADD,       /* P: add Q, K, V biases                           */
    TAG_ROPE,           /* P: apply rotary position embedding              */
    TAG_KV_STORE,       /* P: append K, V to layer cache                   */
    TAG_ATTENTION,      /* P: GQA attention (scores, softmax, weighted V)  */
    TAG_OUT_PROJ,       /* P: output projection matmul                     */
    TAG_RESIDUAL_ATTN,  /* P: add attention residual                       */
    TAG_FFN_NORM,       /* P: pre-FFN RMSNorm + save residual              */
    TAG_GATE_UP,        /* P: fused gate+up matmul                         */
    TAG_SILU_MUL,       /* P: SiLU(gate) ⊙ up                             */
    TAG_DOWN_PROJ,      /* P: down projection matmul                       */
    TAG_RESIDUAL_FFN,   /* P: add FFN residual → advance layer             */
    TAG_FINAL_NORM,     /* P: final RMSNorm                                */
    TAG_LM_HEAD,        /* P: LM head matmul → logits                     */
    TAG_NUM_KINDS
} TagKind;

/*
 * TagState: the mutable payload carried by the STATE symbol.
 *
 * At each tag step the OP symbol's production reads this state,
 * performs one neural-network sub-operation, and writes the
 * transformed state into the new STATE symbol appended to the tail.
 *
 * The buffers are pre-allocated once to the maximum needed size
 * and reused across all tag steps.
 */
typedef struct {
    double *x;          /* current activation vector                      */
    int     x_len;      /* current length of x                            */
    double *residual;   /* saved vector for residual connection            */
    int     has_residual;
    int     token_id;   /* input token (for EMBED)                        */
    int     pos;        /* sequence position (for RoPE / KV cache)        */
    int     layer;      /* current transformer layer index                */
} TagState;

/*
 * TagSymbol: one symbol in the tag-system word.
 */
typedef struct {
    TagKind kind;
    int     layer;      /* layer index (meaningful for OP symbols)        */
} TagSymbol;

/*
 * TagSystem: the complete tag-system machine.
 *
 * word[0] = current OP symbol
 * word[1] = STATE symbol (carries TagState)
 *
 * The FIFO queue is logically unbounded but our invariant keeps it
 * at exactly 2 symbols, so we store it inline.
 */
typedef struct {
    /* --- FIFO queue (constant length 2) --- */
    TagSymbol word[2];

    /* --- Mutable state (the STATE symbol's payload) --- */
    TagState state;

    /* --- Tag system parameters --- */
    int deletion_number;    /* m = 2 (always) */
    int steps;              /* total tag steps executed */
    int halted;

    /* --- Model and caches (production rule parameters) --- */
    V2ModelWeights *model;
    KVCache        *kv_cache;
    RopeCache      *rope_cache;

    /* --- Output --- */
    double *logits;         /* written by TAG_LM_HEAD production */
    int     lm_head_argmax; /* set by TAG_LM_HEAD when using argmax_matmul_sm */
} TagSystem;

/* ==== Tag system API ==================================================== */

TagSystem *tag_system_new(V2ModelWeights *model, RopeCache *rope, KVCache *kv);
void       tag_system_free(TagSystem *ts);
void       tag_step(TagSystem *ts);
void       tag_run(TagSystem *ts);
void       tag_forward_one(TagSystem *ts, int token_id, int pos);

/* ==== Model I/O ========================================================= */

V2ModelWeights *load_eml_v2(const char *path, Tokenizer *tok);
void free_v2_weights(V2ModelWeights *w);

/* ==== Utility ops (shared with MOV engine interface) ==================== */

void build_matmul_sm(double *result, const double *a,
                     const SmTensor *w, int rows, int inner, int cols);
void eml_rms_norm(double *out, const double *x, const double *gamma,
                  int n, double eps);
void eml_silu_vec(double *out, const double *x, int n);
void build_softmax_cse(double *out, const double *x, int n);
void eml_add_vec(double *out, const double *a, const double *b, int n);

RopeCache *rope_cache_new(int d_head, int max_len, double base);
void rope_cache_free(RopeCache *r);
void rope_apply(const RopeCache *r, double *x, int pos);

KVCache *kv_cache_new(const EmlConfig *cfg, int max_len);
void kv_cache_free(KVCache *kv);

#endif /* EML_TAG_H */
