# Book-Driven AutoEML Experiment Design

**Date**: 2026-04-14  
**Purpose**: Mine 9 classic CS/math books for new AutoEML kernel optimization ideas.  
**Context**: After 15 experiments (7 kept, 8 reverted), the kernel is at 3,917 μs / 803,712 transcendentals.  
We need fundamentally new strategies, not incremental tuning.

---

## Books Surveyed

| # | Book | Author(s) | Key Chapters Studied |
|---|------|-----------|---------------------|
| a | *Concrete Mathematics* | Graham, Knuth, Patashnik | Ch. 2 (Summation), Ch. 7 (Generating Functions), Ch. 9 (Asymptotics) |
| b | *The Art of Computer Programming* | Knuth | Vol. 2 Ch. 4 (Arithmetic), Vol. 4A–4B (Combinatorial), Fascicles 5–7 (Backtracking, SAT, Constraint Satisfaction) |
| c | *Elements of Programming* | Stepanov, McJones | Foundations, Associative Operations, Semigroups, Orbits |
| d | *A Programming Language* | Iverson | Array operators, Inner/Outer product, Reduction operators |
| e | *Thinking Forth* | Brodie | Factoring, stack discipline, composition-as-optimization |
| f | *Compilers: Principles, Techniques, and Tools* (Dragon Book) | Aho, Lam, Sethi, Ullman | Code optimization, peephole optimization, data flow analysis, register allocation |
| g | *Elements of Automata Theory* | Sakarovitch | Weighted automata, transducers, semirings |
| h | *Types and Programming Languages* (TAPL) | Pierce | Type inference, System F, subtyping, polymorphism |
| i | *Constraint Processing* | Dechter | Arc consistency (AC-3), constraint propagation, backtracking, CSP formulation |

---

## Experiment Ideas

### Experiment 16: Log-Sum-Exp Peephole Rewrite

**Sources**: Concrete Mathematics Ch. 9 (Asymptotics) + Dragon Book (Peephole Optimization)

**Mathematical basis**:
The matmul accumulator currently does: `acc = ln(exp(acc) + exp(new_term))` — that's
2 exps + 1 ln per accumulation step. The log-sum-exp identity rewrites this as:

    ln(e^a + e^b) = max(a,b) + ln(1 + e^{-|a-b|})

This is 1 exp + 1 ln + 1 add — saving 1 transcendental per accumulation step.

**Provenance**:
- The identity itself is standard in numerical computing, but the *framing* as a 
  peephole rewrite (scan a window of 3–5 EML operations, pattern-match, replace) 
  comes directly from the Dragon Book's treatment of peephole optimization (§8.7 
  in 2nd edition).
- The asymptotic analysis of why this matters at scale (O(K) savings per dot product 
  where K=896) is Concrete Mathematics Ch. 9 thinking.

**Expected impact**: ~50% fewer transcendentals in the accumulation loop.

---

### Experiment 17: Fused APL-Style Inner Product

**Sources**: Iverson's *A Programming Language* — inner product operator `+.×`

**Mathematical basis**:
APL treats `A +.× B` (matmul) as a single fused operator. For EML:

    dot(a, b) = ln(Σ_j exp(ln(a_j) + ln(b_j)))

Using the log-sum-exp trick with a running max:

    m = max_j(ln(a_j) + ln(b_j))
    dot(a, b) = m + ln(Σ_j exp((ln(a_j) + ln(b_j)) - m))

This is K exps + 1 ln for the whole dot product instead of K exps + K lns 
for element-wise accumulation. Cuts lns by factor of K (896).

**Provenance**:
- Iverson's key insight: "think of the whole array operation as a single entity, 
  not a loop over scalars." His inner product operator `+.×` fuses reduction with 
  element-wise application.
- APL idiom recognition (from the APL implementation literature): detect 
  `+/A×B` patterns and evaluate as a single fused operation.
- The running-max numerically-stable variant is standard in ML (used in softmax), 
  but applying it to EML's log-domain matmul accumulation is novel.

**Expected impact**: Potentially reduces lns from O(K) to O(1) per dot product.
Combined with Exp 16, this is the most promising direction.

---

### Experiment 18: Constraint Propagation for Realness

**Sources**: Dechter's *Constraint Processing* (Arc Consistency, AC-3) + TAOCP Vol. 4 
Fascicle 7 (Constraint Satisfaction)

**Formulation as CSP**:
- Variables: each node in the EML computation graph
- Domain: {real, complex}
- Constraints:
  - "final output must be real"
  - "ln(positive_real) → real"
  - "exp(real) → positive_real"
  - "real + real → real"
  - "positive_real × positive_real → positive_real"

Run AC-3 backwards from outputs. Any node proven to be in the "real" domain 
uses f64 ops instead of Complex64.

**Provenance**:
- The CSP formulation maps directly to Dechter's framework: variables = graph nodes, 
  domains = {real, complex}, constraints = type rules.
- AC-3 (arc consistency algorithm 3) from Dechter Ch. 3 is the workhorse: 
  iterate until fixpoint, propagating domain reductions.
- TAOCP Fascicle 7's treatment of constraint satisfaction provides the 
  backtracking framework for cases where AC-3 alone is insufficient.
- This generalizes our best single optimization (Exp 6, real-exp bypass, ~40% speedup) 
  from hand-coded matmul-only to *all* operations (softmax, RMSNorm, SiLU, RoPE).

**Expected impact**: Generalize real-bypass to all ops. Could be significant for 
softmax and RMSNorm which also have known-real intermediate values.

---

### Experiment 19: Balanced Tree Reduction (Semigroup Accumulator)

**Source**: Stepanov & McJones, *Elements of Programming* — Ch. on associative 
operations and semigroups

**Mathematical basis**:
The EML accumulation `ln(exp(a) + exp(b))` is associative (it's addition in log-space, 
i.e., log-sum-exp defines a semigroup). Currently accumulated linearly (depth K, 
zero ILP). A balanced tree of width W has depth log_W(K) and W-1 independent 
pairs at each level.

Linear (current):
```
acc = op(acc, x[0])  // serial chain, depth K
acc = op(acc, x[1])
...
```

Tree (width 8, proposed):
```
t0 = op(x[0], x[1])  // 4 independent pairs → ILP
t1 = op(x[2], x[3])
t2 = op(x[4], x[5])
t3 = op(x[6], x[7])
u0 = op(t0, t1)       // 2 independent pairs
u1 = op(t2, t3)
result = op(u0, u1)   // final merge
```

**Provenance**:
- Stepanov's key theorem: for any associative binary operation, the 
  number of operations is fixed but the *depth* (critical path length) can be 
  reduced from n to ceil(log2(n)) via balanced tree evaluation.
- This is distinct from Exp 13 (8-wide linear unroll, which failed from register 
  pressure). Tree reduction changes the *dependency structure*, not just the width.
- The semigroup concept ensures correctness: associativity guarantees any 
  parenthesization gives the same result.

**Expected impact**: Better ILP by reducing dependency chain depth. Different 
failure mode than Exp 13 — register pressure is similar but dependency chains 
are logarithmic instead of linear.

---

### Additional Ideas (Lower Priority, for Future Work)

#### Weighted Automaton Scheduling
**Source**: Sakarovitch, *Elements of Automata Theory* — weighted automata over semirings

Model EML evaluation as a weighted transducer over the (min,+) semiring:
- States = sets of live register values
- Transitions = EML operations
- Weights = operation latency (exp/ln ≈ 10 cycles, add ≈ 1 cycle)

Minimum-weight path = optimal instruction schedule. More principled than manual 
reordering experiments.

#### Forth-Style Factoring
**Source**: Brodie, *Thinking Forth*

Factor the monolithic kernel into small composable "words": `eml_dot_word`, 
`eml_acc_word`, `eml_sign_word`. The composition boundaries become optimization 
boundaries where the Rust compiler can make independent inlining/vectorization 
decisions.

#### TAPL-Inspired Phantom Types
**Source**: Pierce, *TAPL* — type inference, System F

Encode EML value domains (`EmlReal(f64)`, `EmlComplex(Complex64)`, 
`EmlPositiveReal(f64)`) as Rust phantom types. The type system then enforces 
and optimizes domain transitions at compile time — the type-theoretic version 
of Experiment 18, resolved statically.

---

## Experiment Execution Order

| Order | Exp | Rationale |
|-------|-----|-----------|
| 1 | 17 (Fused APL dot) | Highest potential: K→1 ln reduction. Subsumes Exp 16. |
| 2 | 16 (LSE rewrite) | If Exp 17 isn't viable as full fusion, LSE is the fallback. |
| 3 | 18 (Constraint propagation) | Generalize real-bypass. Independent of 16/17. |
| 4 | 19 (Tree reduction) | ILP improvement. Can stack on top of 16/17. |

---

## Citation Notes for ACM Paper

When writing up these experiments, cite:

- **LSE identity**: Standard numerical computing, but frame as peephole optimization 
  per Aho et al. [Dragon Book, §8.7]
- **Fused inner product**: Iverson, K.E. "A Programming Language" (1962). 
  The inner product operator `+.×` as a first-class fused operation.
- **Constraint propagation**: Dechter, R. "Constraint Processing" (2003), Ch. 3 
  (Arc Consistency). Also Mackworth, A.K. "Consistency in Networks of Relations" 
  (1977) for AC-3.
- **Semigroup tree reduction**: Stepanov, A. and McJones, P. "Elements of Programming" 
  (2009), Ch. 4 (Linear Orderings) and Ch. 5 (Ordered Algebraic Structures). 
  Also Blelloch, G. "Prefix Sums and Their Applications" (1990) for parallel 
  tree reduction.
- **EML operator itself**: Odrzywołek, A. "All elementary functions from a single 
  binary operator" (2026), arXiv:2603.21852.
