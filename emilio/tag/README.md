# emilio tag-system inference engine

Neural network inference as a **2-tag system** (Post, 1943).

## Computational Model

A [tag system](https://en.wikipedia.org/wiki/Tag_system) is a triple *(m, A, P)* where:

- **m = 2** — the deletion number
- **A** — a finite alphabet of operation symbols + one STATE symbol  
- **P: A → A\*** — production rules parameterised by model weights

At each step the machine:
1. Reads the leftmost symbol of the word
2. Appends the production P(symbol) to the right end
3. Deletes the leftmost *m = 2* symbols

The word is always exactly `[OP, STATE]`: an operation symbol paired with
a state-carrier symbol. The word length is invariant at 2.

### Alphabet (16 operation symbols + STATE + HALT)

| Symbol | Production | Neural Op |
|--------|-----------|-----------|
| `EMBED` | embedding lookup | `x = embd[token_id]` |
| `RMSNORM_ATTN` | pre-attention norm | `x = RMSNorm(x, γ_attn)` |
| `QKV_MATMUL` | fused Q+K+V projection | `qkv = x @ W_qkv` |
| `BIAS_ADD` | add Q,K,V biases | `q += b_q; k += b_k; v += b_v` |
| `ROPE` | rotary position encoding | `q,k = RoPE(q,k,pos)` |
| `KV_STORE` | cache K,V | `cache.append(k,v)` |
| `ATTENTION` | grouped-query attention | `x = softmax(QK^T/√d)V` |
| `OUT_PROJ` | output projection | `x = attn @ W_o` |
| `RESIDUAL_ATTN` | attention residual | `x = residual + x` |
| `FFN_NORM` | pre-FFN norm | `x = RMSNorm(x, γ_ffn)` |
| `GATE_UP` | fused gate+up matmul | `gu = x @ W_gate_up` |
| `SILU_MUL` | SiLU activation | `x = SiLU(gate) ⊙ up` |
| `DOWN_PROJ` | down projection | `x = hidden @ W_down` |
| `RESIDUAL_FFN` | FFN residual + next layer | `x = residual + x` |
| `FINAL_NORM` | final RMSNorm | `x = RMSNorm(x, γ_out)` |
| `LM_HEAD` | logit projection | `logits = x @ W_output` |
| `STATE` | data carrier | (carries activation vectors) |
| `HALT` | halting symbol | (computation complete) |

### Tag Steps Per Token

For a model with *L* layers:

```
1 (EMBED) + L × 13 (layer ops) + 1 (FINAL_NORM) + 1 (LM_HEAD) = 13L + 3
```

- **Tiny test model** (1 layer): 16 tag steps/token
- **Qwen 2.5 0.5B** (24 layers): 315 tag steps/token

## Relation to Post's Original Formulation

In Post's 1943 formulation, the word grows and shrinks as computation
proceeds. In our formulation, the word length is *invariant* at 2
because each production replaces the deleted pair with exactly one
new `[OP, STATE]` pair. This is a valid 2-tag system where every
production has length exactly 2.

The weights are encoded in the production rules: `P(QKV_MATMUL, layer=k)`
uses the QKV weight matrix of layer *k*. The alphabet is finite
(|ops| × |layers| + STATE + HALT); the productions are fixed once
the model is loaded.

## Turing-Completeness

2-tag systems are Turing-complete (Cocke & Minsky, 1964). This
engine demonstrates that Turing-completeness concretely — running
a real 0.5B parameter language model through 315 tag steps per token.

## Build

```bash
./build_tag.sh              # optimized build
./build_tag.sh --debug      # debug build with sanitizers
```

## Run

```bash
# Tiny model (pipeline test)
python3 ../mov/make_tiny_model.py build/tiny_model.eml
./build/emilio_tag build/tiny_model.eml "hello" 4

# Qwen 0.5B
./build/emilio_tag ../../models/qwen2.5-0.5b-instruct-v2.eml "What is a tag system?" 32
```

## Files

| File | Description |
|------|-------------|
| `eml_tag.h` | Tag system types + shared model types |
| `eml_tag.c` | Complete engine: interpreter, productions, math ops, loader |
| `eml_mov.h` | Compatibility shim so `eml_tokenizer.c` compiles |
| `build_tag.sh` | Build script |
| `test_tag.sh` | Test script (tiny model + optional Qwen) |

## References

- Post, E. L. (1943). "Formal reductions of the combinatorial decision problem." *Am. J. Math.* 65(2):197–215.
- Cocke, J. & Minsky, M. (1964). "Universality of Tag Systems with P=2." *JACM* 11:15–20.
- Odrzywołek (2026). arXiv:2603.21852 (EML algebraic inference)
