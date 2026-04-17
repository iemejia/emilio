#!/usr/bin/env python3
"""
Parse Golly macrocell (.mc) files into sets of alive cell coordinates.
Handles both simple leaf nodes (level 1 with 4 integers) and RLE-encoded leaf nodes.
"""
import sys

sys.setrecursionlimit(50000)


def parse_rle_leaf(rle_str, size):
    """Parse an RLE-encoded leaf node into alive cell offsets."""
    cells = []
    row, col = 0, 0
    i = 0
    while i < len(rle_str):
        ch = rle_str[i]
        if ch == '$':
            row += 1
            col = 0
            i += 1
        elif ch == '*':
            if row < size and col < size:
                cells.append((row, col))
            col += 1
            i += 1
        elif ch == '.':
            col += 1
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(rle_str) and rle_str[j].isdigit():
                j += 1
            count = int(rle_str[i:j])
            if j < len(rle_str):
                ch2 = rle_str[j]
                if ch2 == '*':
                    for _ in range(count):
                        if row < size and col < size:
                            cells.append((row, col))
                        col += 1
                    j += 1
                elif ch2 == '.':
                    col += count
                    j += 1
                elif ch2 == '$':
                    row += count
                    col = 0
                    j += 1
            i = j
        else:
            i += 1
    return cells


def parse_mc(filename):
    """Parse a macrocell file. Returns (nodes_dict, root_id)."""
    nodes = {}

    with open(filename) as f:
        raw_lines = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('[') or line.startswith('#'):
                continue
            raw_lines.append(line)

    idx = 1
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        
        # Check if this line starts a node definition: "level nw ne sw se"
        parts = line.split()
        is_node_line = False
        if len(parts) == 5:
            try:
                vals = [int(x) for x in parts]
                is_node_line = True
                nodes[idx] = ('node', vals[0], vals[1], vals[2], vals[3], vals[4])
                idx += 1
                i += 1
                continue
            except ValueError:
                pass
        
        # It's a bare RLE leaf. Each bare line is ONE node.
        rle_str = line
        # Determine level from content
        max_row = 0
        max_col = 0
        row, col = 0, 0
        for ch in rle_str:
            if ch == '$':
                max_col = max(max_col, col)
                row += 1
                col = 0
            elif ch in ('*', '.'):
                col += 1
        max_col = max(max_col, col)
        max_row = row + (1 if col > 0 else 0)
        import math
        dim = max(max_row, max_col, 1)
        level = max(1, math.ceil(math.log2(dim))) if dim > 1 else 1
        
        leaf_cells = parse_rle_leaf(rle_str, 2**level)
        nodes[idx] = ('leaf', level, leaf_cells)
        idx += 1
        i += 1

    return nodes, idx - 1


def extract_cells(nodes, node_id, x=0, y=0):
    """Recursively extract alive cells from a parsed macrocell tree."""
    if node_id == 0:
        return []

    node = nodes[node_id]

    if node[0] == 'leaf':
        _, level, leaf_cells = node
        return [(y + r, x + c) for r, c in leaf_cells]

    _, level, nw, ne, sw, se = node

    if level == 1:
        cells = []
        if nw: cells.append((y, x))
        if ne: cells.append((y, x + 1))
        if sw: cells.append((y + 1, x))
        if se: cells.append((y + 1, x + 1))
        return cells

    half = 2 ** (level - 1)
    cells = []
    cells.extend(extract_cells(nodes, nw, x, y))
    cells.extend(extract_cells(nodes, ne, x + half, y))
    cells.extend(extract_cells(nodes, sw, x, y + half))
    cells.extend(extract_cells(nodes, se, x + half, y + half))
    return cells


def normalize_cells(cells):
    """Shift cells so minimum coords are (0, 0)."""
    if not cells:
        return set()
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def load_gate(name):
    """Load a gate pattern from the gol-computer-ref directory."""
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'gol-computer-ref', 'logic gates', f'{name}.mc')
    nodes, root = parse_mc(path)
    cells = extract_cells(nodes, root)
    return normalize_cells(cells)


if __name__ == '__main__':
    for name in ['not', 'and', 'or', 'xor', 'nor', 'bifurcation']:
        try:
            cells = load_gate(name)
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            w = max(cs) - min(cs) + 1 if cells else 0
            h = max(rs) - min(rs) + 1 if cells else 0
            print(f'{name:15s}: {len(cells):5d} cells, {h:4d}x{w:4d}')
        except Exception as e:
            import traceback
            print(f'{name:15s}: ERROR - {e}')
            traceback.print_exc()
