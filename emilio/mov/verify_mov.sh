#!/bin/bash
# verify_mov.sh -- Verify that a binary was compiled with the M/o/Vfuscator
#
# Checks:
#   1. Binary is ELF32 i386 (movfuscator only targets 32-bit x86)
#   2. User code is 100% MOV (excluding PLT stubs and CRT bootstrap)
#   3. Full binary is >90% MOV (including PLT/CRT overhead)
#   4. No standard GCC function prologues (push ebp; mov ebp,esp)
#
# Two metrics are reported:
#   - "User code MOV%"  -- only emilio functions (must be 100%)
#   - "Full binary MOV%" -- includes PLT stubs + CRT setup (~98%)
#
# Usage: ./verify_mov.sh <binary>
#
# Exit code:
#   0 = binary is a valid movfuscator binary (100% MOV user code)
#   1 = binary was NOT compiled with the movfuscator

set -e

BINARY="${1:?Usage: $0 <binary>}"

if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found" >&2
    exit 1
fi

# Check for objdump
if ! command -v objdump &>/dev/null; then
    echo "Error: objdump not found. Install binutils." >&2
    exit 1
fi

echo "=== MOV-Only Verification ==="
echo "Binary: $BINARY"
echo "Size: $(ls -lh "$BINARY" | awk '{print $5}')"
echo ""

# ---- Check 1: ELF class ----
# The movfuscator always produces 32-bit (ELF32 / i386) binaries.
# A 64-bit binary was compiled by GCC or another standard compiler.

ELFCLASS=$(readelf -h "$BINARY" 2>/dev/null | grep 'Class:' | awk '{print $2}')
MACHINE=$(readelf -h "$BINARY" 2>/dev/null | grep 'Machine:' | sed 's/.*Machine:\s*//')

if [ "$ELFCLASS" = "ELF64" ]; then
    echo "FAIL: Binary is 64-bit ($ELFCLASS, $MACHINE)."
    echo "  The M/o/Vfuscator produces 32-bit (ELF32 / i386) binaries."
    echo "  This binary was likely compiled with GCC, not movcc."
    exit 1
fi

echo "Architecture: $ELFCLASS ($MACHINE)"
echo ""

# ---- Check 2: MOV instruction ratio (full binary) ----
# Movfuscator binaries compiled with --no-mov-flow have 100% MOV user code.
# The only non-MOV instructions are in PLT stubs (dynamic libc dispatch)
# and a few CRT bootstrap instructions (signal handler setup).

DISASM=$(objdump -d -j .text "$BINARY" 2>/dev/null || objdump -d "$BINARY")

TOTAL=$(echo "$DISASM" | grep -cE '^\s+[0-9a-f]+:' || true)
echo "Total instructions in .text: $TOTAL"

if [ "$TOTAL" -eq 0 ]; then
    echo "FAIL: No instructions found in .text section"
    exit 1
fi

# Count MOV instructions (all variants: mov, movl, movw, movb, movzbl, movsbl, etc.)
MOV_COUNT=$(echo "$DISASM" | grep -E '^\s+[0-9a-f]+:' | \
    grep -ciE '\bmov[a-z]*\b' || true)
echo "MOV instructions: $MOV_COUNT"

NON_MOV_COUNT=$((TOTAL - MOV_COUNT))
echo "Non-MOV instructions: $NON_MOV_COUNT"

MOV_PCT=$((MOV_COUNT * 100 / TOTAL))
echo "Full binary MOV ratio: ${MOV_PCT}%"
echo ""

# ---- Check 2b: User code MOV ratio (excluding PLT + CRT) ----
# Extract only user-code sections: skip PLT stubs (<*@plt>) and
# CRT symbols (_start, __libc_*, deregister_tm_clones, etc.)

# Filter out PLT section and known CRT/linker functions
USER_DISASM=$(echo "$DISASM" | awk '
    /^[0-9a-f]+ <.*@plt>:/       { in_skip=1; next }
    /^[0-9a-f]+ <_start>:/       { in_skip=1; next }
    /^[0-9a-f]+ <__libc/         { in_skip=1; next }
    /^[0-9a-f]+ <_dl_/           { in_skip=1; next }
    /^[0-9a-f]+ <__do_global/    { in_skip=1; next }
    /^[0-9a-f]+ <deregister_tm/  { in_skip=1; next }
    /^[0-9a-f]+ <register_tm/    { in_skip=1; next }
    /^[0-9a-f]+ <frame_dummy/    { in_skip=1; next }
    /^[0-9a-f]+ <__x86.get_pc/   { in_skip=1; next }
    /^[0-9a-f]+ <_fini>:/        { in_skip=1; next }
    /^[0-9a-f]+ <_init>:/        { in_skip=1; next }
    /^[0-9a-f]+ <\.plt/          { in_skip=1; next }
    /^[0-9a-f]+ </               { in_skip=0 }
    /^$/                          { next }
    { if (!in_skip) print }
')

USER_TOTAL=$(echo "$USER_DISASM" | grep -cE '^\s+[0-9a-f]+:' || true)
USER_MOV=$(echo "$USER_DISASM" | grep -E '^\s+[0-9a-f]+:' | \
    grep -ciE '\bmov[a-z]*\b' || true)
USER_NON_MOV=$((USER_TOTAL - USER_MOV))

if [ "$USER_TOTAL" -gt 0 ]; then
    USER_MOV_PCT=$((USER_MOV * 100 / USER_TOTAL))
else
    USER_MOV_PCT=0
fi

echo "=== User Code Analysis ==="
echo "  User code instructions: $USER_TOTAL"
echo "  User code MOV:          $USER_MOV"
echo "  User code non-MOV:      $USER_NON_MOV"
echo "  User code MOV ratio:    ${USER_MOV_PCT}%"
echo ""

# ---- Check 3: No GCC function prologues in user code ----
# GCC-compiled functions start with "push %ebp; mov %ebp,%esp" (or similar).
# Movfuscator-compiled code never uses push/pop/call/ret in user functions.

GCC_PROLOGUES=$(echo "$USER_DISASM" | grep -cE '^\s+[0-9a-f]+:.*\bpush\s+%ebp\b' || true)

if [ "$GCC_PROLOGUES" -gt 0 ]; then
    echo "WARNING: Found $GCC_PROLOGUES 'push %ebp' in user code."
    echo "  This may indicate GCC-compiled code mixed in."
fi

# ---- Results ----

echo "=== Instruction Breakdown (top 20, user code) ==="
echo "$USER_DISASM" | grep -E '^\s+[0-9a-f]+:' | \
    sed 's/.*:\s\+[0-9a-f ]\+\s\+//' | \
    awk '{print $1}' | sort | uniq -c | sort -rn | head -20
echo ""

# Count PLT stubs (expected non-MOV for libc dispatch)
PLT_NON_MOV=$(echo "$DISASM" | \
    sed -n '/<.*@plt>/,/^$/p' | \
    grep -E '^\s+[0-9a-f]+:' | \
    grep -viE '\bmov[a-z]*\b' | \
    grep -viE '\bnop\b|\bint\b|\bhlt\b' | \
    wc -l || true)

echo "=== Analysis ==="
echo "  User code MOV ratio:                  ${USER_MOV_PCT}% (target: 100%)"
echo "  Full binary MOV ratio:                ${MOV_PCT}%"
echo "  PLT stubs (libc dispatch, expected):  ~$PLT_NON_MOV non-MOV"
echo "  CRT bootstrap (expected):             ~$((NON_MOV_COUNT - PLT_NON_MOV - USER_NON_MOV)) non-MOV"
echo ""

# ---- Pass/Fail ----

# User code must be 100% MOV (the whole point of --no-mov-flow)
USER_THRESHOLD=100
# Full binary allows a few PLT/CRT stubs
FULL_THRESHOLD=90

PASSED=1

if [ "$USER_MOV_PCT" -lt "$USER_THRESHOLD" ] && [ "$USER_TOTAL" -gt 0 ]; then
    echo "FAIL: User code MOV ratio ${USER_MOV_PCT}% < ${USER_THRESHOLD}%"
    echo ""
    echo "  Non-MOV instructions in user code:"
    echo "$USER_DISASM" | grep -E '^\s+[0-9a-f]+:' | \
        grep -viE '\bmov[a-z]*\b' | head -20
    echo ""
    PASSED=0
fi

if [ "$MOV_PCT" -lt "$FULL_THRESHOLD" ]; then
    echo "FAIL: Full binary MOV ratio ${MOV_PCT}% < ${FULL_THRESHOLD}%"
    PASSED=0
fi

if [ "$PASSED" = "1" ]; then
    if [ "$USER_MOV_PCT" -ge 100 ] && [ "$USER_TOTAL" -gt 0 ]; then
        echo "PASS: 100% MOV -- every user instruction is MOV (${USER_TOTAL} instructions)"
        echo "  Full binary: ${MOV_PCT}% (${NON_MOV_COUNT} non-MOV in PLT/CRT only)"
    else
        echo "PASS: Binary is a valid M/o/Vfuscator binary"
        echo "  User code: ${USER_MOV_PCT}% MOV | Full binary: ${MOV_PCT}% MOV"
    fi
    exit 0
else
    exit 1
fi
