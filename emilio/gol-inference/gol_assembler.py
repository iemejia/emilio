#!/usr/bin/env python3
"""
GOL Computer Assembler — compiles assembly programs to binary,
then programs the GOL computer's ROM by modifying Hashlife tree cells.

Based on nicolasloizeau/gol-computer assembly.py, but without Golly dependency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hashlife import HashLife


# ═══════════════════════════════════════════════════════════════════
# Assembler: text → binary
# ═══════════════════════════════════════════════════════════════════

INSTRUCTIONS = {
    "goto": "0000", "move": "0001", "write": "0000", "jumpif": "0010",
    "print": "0011", "add": "1000", "or": "1001", "and": "1010",
    "xor": "1011", "not": "1100", "flat": "1101", "sign": "1110",
    "increment": "1111"
}

ADDRESSES = {
    "a": "111", "b": "110", "c": "101", "d": "100",
    "e": "011", "f": "010", "g": "001", "h": "000"
}


def twos_complement(v):
    """Convert negative value to 8-bit two's complement binary string."""
    v = bin(-v)[2:].zfill(8)
    v = "".join(["0" if c == "1" else "1" for c in v])
    v = int(v, 2) + 1
    v = bin(v)[2:].zfill(8)
    return v


def translate_line(line):
    """Translate one assembly line to 21-bit binary string."""
    bin_instruction = "0000"
    bin_address1 = "000"
    bin_address2 = "000"
    bin_address3 = "000"
    bin_data = "00000000"

    parts = line.split()
    opcode = parts[0]
    bin_instruction = INSTRUCTIONS[opcode]

    if opcode == "write":
        n = int(parts[2])
        if n >= 0:
            bin_data = bin(n)[2:].zfill(8)[::-1]  # LSB first
        else:
            bin_data = twos_complement(-n)[::-1]
        bin_address1 = ADDRESSES[parts[1]]

    elif opcode in ["add", "or", "and", "xor"]:
        bin_address1 = ADDRESSES[parts[3]]
        bin_address2 = ADDRESSES[parts[1]]
        bin_address3 = ADDRESSES[parts[2]]

    elif opcode in ["not", "flat", "sign", "move"]:
        bin_address1 = ADDRESSES[parts[2]]
        bin_address2 = ADDRESSES[parts[1]]

    elif opcode in ["jumpif", "print", "increment"]:
        bin_address1 = ADDRESSES[parts[1]]
        bin_address2 = ADDRESSES[parts[1]]

    elif opcode == "goto":
        if parts[1] == "0":
            bin_data = "11111111"
        else:
            bin_data = bin(int(parts[1]) - 1)[2:].zfill(8)[::-1]

    return bin_instruction + bin_address1 + bin_address2 + bin_address3 + bin_data


def assemble(asm_text):
    """Compile assembly text to list of 21-bit binary strings."""
    lines = [line.strip() for line in asm_text.strip().split("\n") if line.strip()]
    return [translate_line(line) for line in lines]


# ═══════════════════════════════════════════════════════════════════
# ROM Programmer: modify Hashlife tree to encode program
# ═══════════════════════════════════════════════════════════════════

# ROM coordinates in the GOL computer (from assembly.py):
# Bit position (col, row) = (-23639 + i*600 + j*300, 2159 + i*600 - j*300)
# where i = instruction line, j = bit position (0..20)
# Blocker template region: (-25950, 1890) with size (450, 240)

# The GOL computer.mc uses Golly's coordinate system where (x, y) = (col, row)
# with y increasing upward. In my Hashlife tree, (row, col) has row increasing
# downward. The .mc file's quadtree has origin at (0, 0) in the top-left of
# the level-20 grid (2^20 x 2^20 = 1M x 1M).
#
# The Golly coordinates in assembly.py are relative to the center of the grid.
# For a level-20 grid, center is at (2^19, 2^19) = (524288, 524288).
# Golly x → my_col = x + 524288
# Golly y → my_row = 524288 - y (y is inverted)

GRID_CENTER = 1 << 19  # 524288 for level-20 grid

# Blocker template position (Golly coords)
BLOCKER_X = -25950
BLOCKER_Y = 1890
BLOCKER_W = 450
BLOCKER_H = 240


def golly_to_tree(gx, gy, level=20):
    """Convert Golly (x, y) coordinates to Hashlife tree (row, col)."""
    center = 1 << (level - 1)
    col = gx + center
    row = center - gy
    return row, col


def rom_bit_position(line_idx, bit_idx):
    """Get Golly (x, y) coordinates for a ROM bit."""
    x = -23639 + line_idx * 600 + bit_idx * 300
    y = 2159 + line_idx * 600 - bit_idx * 300
    return x, y


def extract_blocker_pattern(hl, root):
    """Extract the blocker pattern from the known template position."""
    row, col = golly_to_tree(BLOCKER_X, BLOCKER_Y, root.level)
    # The blocker is at (col, row) with size (450, 240) in Golly coords
    # In tree coords: rows [row-240, row), cols [col, col+450)
    # Actually Golly's select uses (x, y, w, h) where y increases up
    # So region is x∈[BLOCKER_X, BLOCKER_X+450), y∈[BLOCKER_Y, BLOCKER_Y+240)
    # In tree: col∈[col, col+450), row∈[center-BLOCKER_Y-240, center-BLOCKER_Y)
    cells = []
    center = 1 << (root.level - 1)
    r_start = center - (BLOCKER_Y + BLOCKER_H)
    r_end = center - BLOCKER_Y
    c_start = BLOCKER_X + center
    c_end = c_start + BLOCKER_W

    for r, c in hl.to_cells(root):
        if r_start <= r < r_end and c_start <= c < c_end:
            cells.append((r - r_start, c - c_start))
    return cells


def program_computer(hl, root, binary_program):
    """Program the GOL computer's ROM with the given binary program.
    
    Args:
        hl: HashLife instance
        root: Hashlife tree of the computer
        binary_program: list of 21-bit binary strings from assemble()
    
    Returns:
        Modified root node with new program in ROM
    """
    center = 1 << (root.level - 1)

    # First, extract the blocker pattern
    print("Extracting blocker pattern...")
    blocker_cells = extract_blocker_pattern(hl, root)
    print(f"  Blocker has {len(blocker_cells)} cells")

    # For each program line and bit:
    # - If bit is '1': ensure blocker is present (clear area, place blocker)
    # - If bit is '0': ensure area is clear (remove blocker)
    
    # First clear ALL existing ROM bits (32 lines × 21 bits)
    print("Clearing existing ROM...")
    for i in range(32):
        for j in range(21):
            gx, gy = rom_bit_position(i, j)
            # Clear the 450x240 area at this position
            r_start = center - (gy + BLOCKER_H)
            c_start = gx + center
            root = hl.clear_rect(root, r_start, c_start,
                                 r_start + BLOCKER_H, c_start + BLOCKER_W)

    # Now write the program
    print(f"Writing {len(binary_program)} instruction lines...")
    for i, bits in enumerate(binary_program):
        for j in range(21):
            if int(bits[j]):
                # Place blocker at this position
                gx, gy = rom_bit_position(i, j)
                r_start = center - (gy + BLOCKER_H)
                c_start = gx + center
                for dr, dc in blocker_cells:
                    root = hl.set_cell(root, r_start + dr, c_start + dc, 1)

    print(f"  Done. New population: {root.pop:,}")
    return root


# ═══════════════════════════════════════════════════════════════════
# Test programs
# ═══════════════════════════════════════════════════════════════════

# Multiply via repeated addition: result = a * b
MULTIPLY_ASM = """
write a 3
write b 5
write c 0
write e 1
not e d
add d e d
add c a c
add b d b
jumpif b
goto 6
print c
goto 11
"""

# Fibonacci (original program in computer.mc)
FIBONACCI_ASM = """
write a 1
write b 1
add a b a
print a
add a b b
print b
goto 2
"""


def test_assembler():
    """Test the assembler output matches expected binary."""
    # Test Fibonacci
    fib_binary = assemble(FIBONACCI_ASM)
    print("Fibonacci program binary:")
    for i, bits in enumerate(fib_binary):
        print(f"  Line {i}: {bits}")

    print()

    # Test multiply
    mul_binary = assemble(MULTIPLY_ASM)
    print("Multiply program binary (3 * 5 = 15):")
    for i, bits in enumerate(mul_binary):
        print(f"  Line {i}: {bits}")

    # Verify: write a 3 → opcode=0000, addr1=111(a), addr2=000, addr3=000, data=11000000 (3 in LSB-first)
    print()
    print("Verification:")
    print(f"  'write a 3' → {mul_binary[0]}")
    print(f"    opcode={mul_binary[0][:4]} addr1={mul_binary[0][4:7]} data={mul_binary[0][13:]}")
    expected_data = "11000000"  # 3 = 011 in binary, reversed = 110, padded = 11000000
    print(f"    data should be: {expected_data} (3 in LSB-first 8-bit)")


if __name__ == '__main__':
    test_assembler()
