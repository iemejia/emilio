#!/usr/bin/env python3
"""Export the programmed GOL computer cells to a binary file for Swift consumption.

Format: header + cell coordinates
  - 4 bytes: magic "GOL\0"
  - 4 bytes: uint32 LE — number of cells
  - 4 bytes: int32 LE — row offset (minimum row)
  - 4 bytes: int32 LE — col offset (minimum col)
  - 4 bytes: uint32 LE — generation (low 32 bits)
  - 4 bytes: uint32 LE — generation (high 32 bits)
  - For each cell: 4 bytes int32 LE row, 4 bytes int32 LE col (absolute coords)
"""

import sys
import os
import struct
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gol-inference'))
sys.setrecursionlimit(500000)

from gol_programmer import program_computer, MATMUL_ELEMENT


def export_cells(output_path):
    print("Programming GOL computer for matmul (1*5 + 2*7 = 19)...")
    hl, root, gen = program_computer(MATMUL_ELEMENT, N=8, M=8, P=32)

    # Extract cells
    print("Extracting cells...")
    cells = []
    def extract(node, row_off, col_off, level):
        if node.pop == 0:
            return
        if level == 0:
            cells.append((row_off, col_off))
            return
        half = 1 << (level - 1)
        extract(node.nw, row_off, col_off, level - 1)
        extract(node.ne, row_off, col_off + half, level - 1)
        extract(node.sw, row_off + half, col_off, level - 1)
        extract(node.se, row_off + half, col_off + half, level - 1)

    extract(root, 0, 0, root.level)
    print(f"  {len(cells):,} cells extracted")

    # Find bounding box
    min_r = min(r for r, c in cells)
    max_r = max(r for r, c in cells)
    min_c = min(c for r, c in cells)
    max_c = max(c for r, c in cells)
    print(f"  Rows: [{min_r}, {max_r}] ({max_r - min_r + 1} high)")
    print(f"  Cols: [{min_c}, {max_c}] ({max_c - min_c + 1} wide)")
    print(f"  Generation: {gen:,}")

    # Write binary file
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GOL\0')
        f.write(struct.pack('<I', len(cells)))
        f.write(struct.pack('<i', min_r))
        f.write(struct.pack('<i', min_c))
        f.write(struct.pack('<I', gen & 0xFFFFFFFF))
        f.write(struct.pack('<I', (gen >> 32) & 0xFFFFFFFF))

        # Cell coordinates
        for r, c in cells:
            f.write(struct.pack('<ii', r, c))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Written: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "gol_computer.bin"
    export_cells(out)
