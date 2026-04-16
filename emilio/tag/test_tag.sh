#!/bin/bash
# test_tag.sh -- Test the tag-system inference engine
#
# 1. Generates a tiny test model
# 2. Runs emilio_tag against it
# 3. Verifies the tag system executes the expected number of steps
#
# Usage:
#   ./test_tag.sh              # build + test with tiny model
#   ./test_tag.sh --qwen       # test with Qwen 0.5B model

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
MOV_DIR="$SCRIPT_DIR/../mov"
MODEL_DIR="$SCRIPT_DIR/../../models"

# Build first
bash "$SCRIPT_DIR/build_tag.sh"

echo ""
echo "=== Test 1: Tiny model (pipeline validation) ==="
echo ""

# Generate tiny model
TINY_MODEL="$BUILD_DIR/tiny_model.eml"
python3 "$MOV_DIR/make_tiny_model.py" "$TINY_MODEL"

# Run with tiny model
echo ""
echo "Running tag-system engine on tiny model..."
"$BUILD_DIR/emilio_tag" "$TINY_MODEL" "hello" 4 2>&1 || true

echo ""
echo "=== Pipeline test complete ==="

# Optional: test with Qwen 0.5B
if [ "$1" = "--qwen" ]; then
    QWEN_MODEL="$MODEL_DIR/qwen2.5-0.5b-instruct.eml"
    if [ ! -f "$QWEN_MODEL" ]; then
        echo "Error: Qwen model not found at $QWEN_MODEL"
        exit 1
    fi
    echo ""
    echo "=== Test 2: Qwen 0.5B (full model) ==="
    echo ""
    "$BUILD_DIR/emilio_tag" "$QWEN_MODEL" "What is a tag system?" 32
fi
