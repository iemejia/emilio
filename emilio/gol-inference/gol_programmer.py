#!/usr/bin/env python3
"""
GOL Computer Programmer — programs the scalable GOL computer.
Compiles assembly to binary, computes ROM coordinates, places blockers.
"""

import sys
import os
import math
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(500000)

from hashlife import HashLife


# ═══════════════════════════════════════════════════════════════════
# Scalable Assembler (from assembly.py, no Golly dependency)
# ═══════════════════════════════════════════════════════════════════

INSTRUCTIONS = [
    " ", "++", "-", "+", "or", "and", "xor", "not", ">>", "<<",
    "rr", "rl", "=0", "!=0", "less", "most", "write", "move",
    "rfb", "wfb", "disp", "erase", "print", "*-"
]
ALU1 = ["++", "not", ">>", "<<", "rr", "rl", "=0", "!=0", "less", "most", "move", "*-"]
ALU2 = ["-", "+", "or", "and", "xor"]


def twos_complement(v, M):
    v = bin(-v)[2:].zfill(M)
    v = "".join(["0" if c == "1" else "1" for c in v])
    v = int(v, 2) + 1
    v = bin(v)[2:].zfill(M)
    return v


def preprocess_line(line):
    """Preprocess one assembly line to canonical form: op aw ar1 ar2 data"""
    parts = line.split()
    instruction = parts[0]
    aw = 'a0'
    ar1 = 'a0'
    ar2 = 'a0'
    data = 0

    if instruction in ALU1:
        aw = parts[1]
        ar1 = parts[2]
    elif instruction in ALU2:
        aw = parts[1]
        ar1 = parts[2]
        ar2 = parts[3]
    if instruction == "-":
        aw = parts[1]
        ar1 = parts[3]
        ar2 = parts[2]
    if instruction == "write":
        aw = parts[1]
        data = parts[2]
    if instruction == "rfb":
        aw = parts[1]
        ar2 = parts[2]
    if instruction == 'wfb':
        ar1 = parts[1]
        ar2 = parts[2]
    if instruction == 'disp':
        ar1 = parts[1]
        ar2 = parts[2]
    if instruction == 'print':
        ar1 = parts[1]
    if instruction == 'goto':
        instruction = "write"
        data = parts[1]
    if instruction == 'jump':
        instruction = "+"
        ar1 = parts[1]

    return " ".join([instruction, aw, ar1, ar2, str(data)])


def preprocess(program_text):
    """Preprocess entire program."""
    lines = []
    for line in program_text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(preprocess_line(line))
    return lines


def assemble_line(line, M, Nd):
    """Assemble one preprocessed line to binary string."""
    parts = line.split()
    instruction = INSTRUCTIONS.index(parts[0])
    aw = int(parts[1][1:])
    ar1 = int(parts[2][1:])
    ar2 = int(parts[3][1:])
    value = int(parts[4])

    instr_bin = bin(instruction)[2:].zfill(5)
    aw_bin = bin(aw)[2:].zfill(Nd)
    ar1_bin = bin(ar1)[2:].zfill(Nd)
    ar2_bin = bin(ar2)[2:].zfill(Nd)

    if value >= 0:
        value_bin = bin(value)[2:].zfill(M)[::-1]  # LSB first
    else:
        value_bin = twos_complement(value, M)

    return instr_bin + aw_bin + ar1_bin + ar2_bin + value_bin


def assemble(program_text, N=8, M=8, P=32):
    """Full assembly pipeline: text → binary strings."""
    Nd = int(math.log2(N))

    preprocessed = preprocess(program_text)
    binary = [assemble_line(line, M, Nd) for line in preprocessed]

    bits_per_line = 5 + 3 * Nd + M
    print(f"Assembled {len(binary)} lines, {bits_per_line} bits/line")
    for i, (src, bcode) in enumerate(zip(preprocessed, binary)):
        print(f"  [{i:2d}] {src:40s} → {bcode}")

    return binary


# ═══════════════════════════════════════════════════════════════════
# ROM Coordinate Calculator
# ═══════════════════════════════════════════════════════════════════

def compute_bit_coordinates(N, M, P):
    """Compute Golly (x, y) coordinates for each bit position in the ROM.

    Returns a function: bit_coords(line_idx, bit_idx) → (gx, gy)
    """
    Nd = int(math.log2(N))
    Pd = int(math.log2(P))

    dj2 = 2 * 30      # = 60
    di2 = 60 * Nd
    di = 4 * 30        # = 120 (spacing between bits)
    dj = Pd * 2 * 30   # spacing between lines

    left_input_x = -(3*N-1)*di2 - (3*N-1)*dj2 - 900 - 500 + 123 + (3*Nd)*(-4*30)
    left_input_y = -(3*N-1)*di2 + (3*N-1)*dj2 - 300 + 73 + (3*Nd)*(4*30)

    x = left_input_x - 4*30*Nd
    y = left_input_y + 4*30*Nd

    x_wp = x - 60*Pd*(P-1) - 14274 - 249 - 240
    y_wp = y - 60*Pd*(P-1) - 6556 - 227 + 240

    d2 = 30*30*4 + 4*30*Nd            # offset for address bits
    d3 = (3*N)*dj2 - 1110 + 300 + 900 + 6*30  # offset for data bits

    def bit_coords(il, bit_idx):
        """Get Golly (x, y) for bit at program line il, bit position bit_idx."""
        # bit_idx: 0-4 = instruction, 5-(5+3*Nd-1) = addresses, rest = data
        k = bit_idx
        if k < 5:
            # Instruction bits
            i = k
            bx = x_wp + di*i + il*dj
            by = y_wp - di*i + il*dj
        elif k < 5 + 3*Nd:
            # Address bits
            i = k - 5
            bx = x_wp + d2 + 5*di + di*i + il*dj
            by = y_wp - d2 - 5*di - di*i + il*dj
        else:
            # Data bits
            i = k - 5 - 3*Nd
            bx = x_wp + d2 + d3 + (5 + Nd*3)*di + 17*30*i + il*dj
            by = y_wp - d2 - d3 - (5 + Nd*3)*di - 17*30*i + il*dj
        return bx, by

    return bit_coords


# ═══════════════════════════════════════════════════════════════════
# Programmer: place blockers in Hashlife tree
# ═══════════════════════════════════════════════════════════════════

def load_bit_pattern(hl):
    """Load bit2.mc and return its cells in relative coordinates."""
    bit2_path = os.path.join(os.path.dirname(__file__), 'scalable-ref', 'bit2.mc')
    root, gen = hl.load_mc(bit2_path)
    cells = hl.to_cells(root)
    center = 1 << (root.level - 1)

    # Convert to Golly coordinates
    golly_cells = [(c - center, r - center) for r, c in cells]
    min_x = min(x for x, y in golly_cells)
    min_y = min(y for x, y in golly_cells)

    # Relative to top-left of bounding box (for paste behavior)
    # When Golly pastes, it places the pattern so the bounding box's
    # top-left corner is at (paste_x, paste_y)
    rel_cells = [(x - min_x, y - min_y) for x, y in golly_cells]

    print(f"Bit pattern: {len(rel_cells)} cells, "
          f"Golly bbox: x=[{min_x},{max(x for x,y in golly_cells)}], "
          f"y=[{min_y},{max(y for x,y in golly_cells)}]")

    return rel_cells, min_x, min_y


def program_computer(program_text, N=8, M=8, P=32):
    """Program the GOL computer with the given assembly program.

    Returns:
        (hl, root, gen): Hashlife instance, programmed tree, generation
    """
    Nd = int(math.log2(N))
    bits_per_line = 5 + 3 * Nd + M

    hl = HashLife()

    # Load the blank computer
    pattern_path = os.path.join(
        os.path.dirname(__file__), 'scalable-ref', 'patterns',
        f'computer_{N}_{M}_{P}.mc'
    )
    print(f"Loading computer pattern: {pattern_path}")
    root, gen = hl.load_mc(pattern_path)
    print(f"  Level={root.level}, pop={root.pop:,}, gen={gen:,}")

    # Load the bit blocker pattern
    print("Loading bit pattern...")
    bit_cells, bit_min_x, bit_min_y = load_bit_pattern(hl)

    # Compile the program
    print("\nCompiling program...")
    binary = assemble(program_text, N, M, P)

    # Compute bit coordinates
    bit_coords = compute_bit_coordinates(N, M, P)
    center = 1 << (root.level - 1)

    # Place blockers for each '1' bit
    total_bits_set = 0
    total_cells_placed = 0
    print(f"\nPlacing blockers (center={center})...")

    for il, bits in enumerate(binary):
        for k in range(len(bits)):
            if bits[k] == '1':
                gx, gy = bit_coords(il, k)
                # Golly paste places bounding-box top-left at (gx, gy)
                # But the bit2 pattern has its own offset
                for dx, dy in bit_cells:
                    cell_gx = gx + dx
                    cell_gy = gy + dy
                    # Convert Golly → tree coords: col = x + center, row = y + center
                    tree_row = cell_gy + center
                    tree_col = cell_gx + center
                    root = hl.set_cell(root, tree_row, tree_col, 1)
                    total_cells_placed += 1
                total_bits_set += 1

        if (il + 1) % 5 == 0:
            print(f"  Line {il+1}/{len(binary)}: {total_cells_placed:,} cells placed so far")

    print(f"\nDone: {total_bits_set} bits set, {total_cells_placed:,} cells placed")
    print(f"Final population: {root.pop:,}")

    return hl, root, gen


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

# Matmul element: compute 1*5 + 2*7 = 19
MATMUL_ELEMENT = """
write a3 0
write a4 1
write a1 1
write a2 5
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 10
goto 4
write a1 2
write a2 7
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 18
goto 12
print a3
goto 19
"""


if __name__ == '__main__':
    import time

    print("=" * 60)
    print("GOL Computer Programmer")
    print("=" * 60)
    print()

    # First just test assembly compilation
    print("--- Assembly test ---")
    binary = assemble(MATMUL_ELEMENT, N=8, M=8, P=32)
    print()

    # Test coordinate calculation
    print("--- Coordinate test ---")
    bit_coords = compute_bit_coordinates(N=8, M=8, P=32)
    for il in range(3):
        for k in [0, 5, 14]:
            gx, gy = bit_coords(il, k)
            print(f"  Line {il}, bit {k}: Golly ({gx}, {gy})")
    print()

    # Full programming test
    print("--- Programming GOL computer ---")
    t0 = time.time()
    hl, root, gen = program_computer(MATMUL_ELEMENT, N=8, M=8, P=32)
    t1 = time.time()
    print(f"\nProgramming took {t1-t0:.1f}s")
