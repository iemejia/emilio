#!/usr/bin/env bash
# compile_model.sh — Compile a GGUF model to .eml format
#
# Usage:
#   ./compile_model.sh <input.gguf> [output.eml]
#
# Examples:
#   ./compile_model.sh models/qwen2.5-0.5b-instruct-q8_0.gguf
#   ./compile_model.sh models/qwen2.5-0.5b-instruct-q8_0.gguf models/qwen.eml
#
# The .eml format pre-computes ln(W) as Complex64 at compile time,
# so that inference only needs to read the precomputed values — no
# dequantization, no ln() computation at load time.
#
# Reference: Odrzywołek (2026) arXiv:2603.21852
#   eml(x,y) = exp(x) - ln(y) as the universal binary operator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUST_DIR="$SCRIPT_DIR/eml_rust"
BINARY="$RUST_DIR/target/release/emilio"

# ── Args ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input.gguf> [output.eml]"
    echo ""
    echo "Compiles a GGUF model to .eml compiled format."
    echo "The .eml format stores precomputed ln(W) as Complex64,"
    echo "eliminating dequantization and ln() computation at inference time."
    exit 1
fi

INPUT="$1"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

if [[ ! "$INPUT" == *.gguf ]]; then
    echo "Warning: Input file does not have .gguf extension: $INPUT"
fi

# Default output: replace .gguf with .eml
if [[ $# -ge 2 ]]; then
    OUTPUT="$2"
else
    OUTPUT="${INPUT%.gguf}.eml"
fi

# ── Build ─────────────────────────────────────────────────────────
echo "Building emilio (release)..."
cd "$RUST_DIR"
cargo build --bin emilio --release 2>&1 | grep -E "^(Compiling|Finished|error)" || true

if [[ ! -x "$BINARY" ]]; then
    echo "Error: Build failed — emilio binary not found at $BINARY"
    exit 1
fi

# ── Compile ───────────────────────────────────────────────────────
echo ""
echo "Compiling: $INPUT → $OUTPUT"
echo ""

cd "$SCRIPT_DIR"
"$BINARY" "$INPUT" --compile "$OUTPUT"

# ── Verify ────────────────────────────────────────────────────────
if [[ -f "$OUTPUT" ]]; then
    SIZE_MB=$(du -m "$OUTPUT" | cut -f1)
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Compilation complete"
    echo "  Output: $OUTPUT ($SIZE_MB MB)"
    echo "═══════════════════════════════════════════════════"
    echo ""
    echo "  Run inference with:"
    echo "    $BINARY $OUTPUT --chat \"Hello!\""
    echo "    $BINARY $OUTPUT --generate \"Once upon a time\""
else
    echo "Error: Output file not created"
    exit 1
fi
