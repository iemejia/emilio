#!/bin/bash
# build_tag.sh -- Build the tag-system inference engine
#
# Compiles eml_tag.c + eml_tokenizer.c (shared from mov/) into emilio_tag.
# Uses standard gcc with libm (no movfuscator needed).
#
# Usage:
#   ./build_tag.sh              # optimized build
#   ./build_tag.sh --debug      # debug build with sanitizers

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
MOV_DIR="$SCRIPT_DIR/../mov"

mkdir -p "$BUILD_DIR"

CFLAGS="-O2 -Wall -Wextra"
DEBUG=0

for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG=1 ;;
    esac
done

if [ "$DEBUG" = "1" ]; then
    CFLAGS="-g -O0 -Wall -Wextra -fsanitize=address,undefined"
    echo "[build_tag] Debug build with ASan/UBSan"
fi

echo "[build_tag] Compiling tag-system engine..."
gcc $CFLAGS \
    -I"$SCRIPT_DIR" \
    -o "$BUILD_DIR/emilio_tag" \
    "$SCRIPT_DIR/eml_tag.c" \
    "$MOV_DIR/eml_tokenizer.c" \
    -lm

echo "[build_tag] Built: $BUILD_DIR/emilio_tag"
echo "[build_tag] Computational model: 2-tag system (Post, 1943)"
ls -lh "$BUILD_DIR/emilio_tag"
