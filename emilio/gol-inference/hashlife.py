#!/usr/bin/env python3
"""
Hashlife implementation for efficiently simulating large Game of Life patterns.

Uses quadtree memoization (Bill Gosper's algorithm) to achieve exponential
speedup for regular patterns like the GOL computer's logic gates and glider streams.

References:
- Gosper, R. W. (1984). Exploiting regularities in large cellular spaces.
- https://www.drdobbs.com/jvm/an-algorithm-for-compressing-space-and-t/184406478
"""

import sys

sys.setrecursionlimit(100000)


class Node:
    """Immutable quadtree node for Hashlife."""
    __slots__ = ('level', 'nw', 'ne', 'sw', 'se', 'pop', '_hash', 'result')

    def __init__(self, level, nw, ne, sw, se):
        self.level = level
        self.nw = nw
        self.ne = ne
        self.sw = sw
        self.se = se
        self.result = None  # Memoized RESULT of advancing 2^(level-2) generations

        if level == 0:
            # Leaf: nw is the cell value (0 or 1), others unused
            self.pop = nw
            self._hash = hash(nw)
        else:
            self.pop = nw.pop + ne.pop + sw.pop + se.pop
            self._hash = hash((id(nw), id(ne), id(sw), id(se)))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self.level == 0:
            return self.nw == other.nw
        return (self.nw is other.nw and self.ne is other.ne and
                self.sw is other.sw and self.se is other.se)


class HashLife:
    """Hashlife simulator using canonical quadtree nodes."""

    def __init__(self):
        # Canonical node cache
        self._cache = {}
        # Leaf nodes (level 0)
        self.off = self._canonical(Node(0, 0, None, None, None))
        self.on = self._canonical(Node(0, 1, None, None, None))
        # Empty node cache by level
        self._empty = {0: self.off}

    def _canonical(self, node):
        """Return the canonical (interned) version of a node."""
        key = (node.level, 
               node.nw if node.level == 0 else id(node.nw),
               None if node.level == 0 else id(node.ne),
               None if node.level == 0 else id(node.sw),
               None if node.level == 0 else id(node.se))
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = node
        return node

    def make_node(self, level, nw, ne, sw, se):
        """Create or retrieve a canonical node."""
        key = (level, id(nw), id(ne), id(sw), id(se))
        if key in self._cache:
            return self._cache[key]
        node = Node(level, nw, ne, sw, se)
        self._cache[key] = node
        return node

    def empty_node(self, level):
        """Return an empty node at the given level."""
        if level in self._empty:
            return self._empty[level]
        sub = self.empty_node(level - 1)
        node = self.make_node(level, sub, sub, sub, sub)
        self._empty[level] = node
        return node

    def _life_4x4(self, node):
        """Compute the inner 2x2 result of a 4x4 (level-2) node after 1 generation."""
        # Extract all 16 cells from the level-2 node
        # Level-2 has 4 level-1 children, each with 4 level-0 leaves
        n = node
        # Row 0: nw.nw, nw.ne, ne.nw, ne.ne
        # Row 1: nw.sw, nw.se, ne.sw, ne.se
        # Row 2: sw.nw, sw.ne, se.nw, se.ne
        # Row 3: sw.sw, sw.se, se.sw, se.se
        cells = [
            [n.nw.nw.nw, n.nw.ne.nw, n.ne.nw.nw, n.ne.ne.nw],
            [n.nw.sw.nw, n.nw.se.nw, n.ne.sw.nw, n.ne.se.nw],
            [n.sw.nw.nw, n.sw.ne.nw, n.se.nw.nw, n.se.ne.nw],
            [n.sw.sw.nw, n.sw.se.nw, n.se.sw.nw, n.se.se.nw],
        ]

        def cell_val(r, c):
            return cells[r][c]

        def next_cell(r, c):
            count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    count += cell_val(r + dr, c + dc)
            alive = cell_val(r, c)
            if count == 3 or (count == 2 and alive):
                return self.on
            return self.off

        # Result is the inner 2x2 (rows 1-2, cols 1-2)
        return self.make_node(1,
                              next_cell(1, 1), next_cell(1, 2),
                              next_cell(2, 1), next_cell(2, 2))

    def step(self, node):
        """Advance a node by 2^(level-2) generations. Returns the level-(level-1) center."""
        if node.result is not None:
            return node.result

        level = node.level

        if level == 2:
            # Base case: compute 1 generation of 4x4 → 2x2
            node.result = self._life_4x4(node)
            return node.result

        # Recursive case: split into 9 overlapping sub-quadrants
        n = node
        # Build the 9 sub-nodes at level (level-1)
        n00 = n.nw
        n01 = self.make_node(level - 1, n.nw.ne, n.ne.nw, n.nw.se, n.ne.sw)
        n02 = n.ne
        n10 = self.make_node(level - 1, n.nw.sw, n.nw.se, n.sw.nw, n.sw.ne)
        n11 = self.make_node(level - 1, n.nw.se, n.ne.sw, n.sw.ne, n.se.nw)
        n12 = self.make_node(level - 1, n.ne.sw, n.ne.se, n.se.nw, n.se.ne)
        n20 = n.sw
        n21 = self.make_node(level - 1, n.sw.ne, n.se.nw, n.sw.se, n.se.sw)
        n22 = n.se

        # Step each of the 9 sub-nodes (each advances 2^(level-3) generations)
        c00 = self.step(n00)
        c01 = self.step(n01)
        c02 = self.step(n02)
        c10 = self.step(n10)
        c11 = self.step(n11)
        c12 = self.step(n12)
        c20 = self.step(n20)
        c21 = self.step(n21)
        c22 = self.step(n22)

        # Combine into 4 overlapping level-(level-1) nodes and step again
        t00 = self.make_node(level - 1, c00, c01, c10, c11)
        t01 = self.make_node(level - 1, c01, c02, c11, c12)
        t10 = self.make_node(level - 1, c10, c11, c20, c21)
        t11 = self.make_node(level - 1, c11, c12, c21, c22)

        node.result = self.make_node(level - 1,
                                     self.step(t00),
                                     self.step(t01),
                                     self.step(t10),
                                     self.step(t11))
        return node.result

    def expand(self, node):
        """Add a border of empty cells around a node (increase level by 1)."""
        empty = self.empty_node(node.level - 1)
        return self.make_node(node.level + 1,
                              self.make_node(node.level, empty, empty, empty, node.nw),
                              self.make_node(node.level, empty, empty, node.ne, empty),
                              self.make_node(node.level, empty, node.sw, empty, empty),
                              self.make_node(node.level, node.se, empty, empty, empty))

    def _center(self, node):
        """Extract the center sub-node (level-1) from a node."""
        return self.make_node(node.level - 1,
                              node.nw.se, node.ne.sw,
                              node.sw.ne, node.se.nw)

    def advance(self, node, generations):
        """Advance a node by exactly `generations` steps.
        
        Decomposes into power-of-2 steps using step() at controlled tree levels.
        """
        if generations == 0:
            return node

        current = node
        remaining = generations

        # Decompose remaining into sum of powers of 2, biggest first
        while remaining > 0:
            # Find the highest set bit
            import math
            bit = 1 << int(math.log2(remaining))

            # We need step_size = bit, so level = log2(bit) + 2
            needed_level = int(math.log2(bit)) + 2

            # Ensure current is at least at needed_level + 1 (for expand + step)
            while current.level < needed_level + 1:
                current = self.expand(current)

            # Expand once, then step advances 2^(level-2) gens
            # We want step to advance exactly `bit` gens
            # Set tree to level = log2(bit) + 2, then step advances 2^(level-2) = bit
            # But current might be bigger. That's OK — just expand and step at the current level
            # will advance more than we want.
            # Instead: rebuild at the exact right level.
            
            # Actually the simplest: expand to have room, step gives 2^(level-2).
            # If that's bigger than remaining, we need to shrink.
            # Shrink by NOT expanding and using a smaller tree.
            
            # Strategy: ensure level = needed_level, expand once, step.
            # To get level = needed_level, we might need to shrink (rebuild).
            if current.level > needed_level:
                # Rebuild at smaller level (this is expensive but correct)
                cells = self.to_cells(current)
                current = self.from_cells(cells)
                while current.level < needed_level:
                    current = self.expand(current)

            # Now current.level should be around needed_level
            # Expand once so step returns a result at current.level
            current = self.expand(current)
            step_size = 1 << (current.level - 2)

            if step_size == bit:
                current = self.step(current)
                remaining -= bit
            else:
                # Shouldn't happen, but if it does, expand more
                current = self.expand(current)
                step_size = 1 << (current.level - 2)
                # Take what we can
                if step_size <= remaining:
                    current = self.step(current)
                    remaining -= step_size
                else:
                    # This means the tree is too big. Rebuild smaller.
                    cells = self.to_cells(current)
                    current = self.from_cells(cells)
                    while current.level < 3:
                        current = self.expand(current)

        return current

    def from_cells(self, cells):
        """Build a quadtree from a set of (row, col) alive cells."""
        if not cells:
            return self.empty_node(1)

        # Find bounding box
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_r = max(r for r, c in cells)
        max_c = max(c for r, c in cells)

        # Normalize to origin
        cells_set = {(r - min_r, c - min_c) for r, c in cells}
        size = max(max_r - min_r + 1, max_c - min_c + 1)

        # Find minimum level
        import math
        level = max(1, math.ceil(math.log2(max(size, 2))))

        # Build bottom-up
        def build(level, offset_r, offset_c):
            if level == 0:
                return self.on if (offset_r, offset_c) in cells_set else self.off
            half = 1 << (level - 1)
            nw = build(level - 1, offset_r, offset_c)
            ne = build(level - 1, offset_r, offset_c + half)
            sw = build(level - 1, offset_r + half, offset_c)
            se = build(level - 1, offset_r + half, offset_c + half)
            return self.make_node(level, nw, ne, sw, se)

        return build(level, 0, 0)

    def to_cells(self, node, offset_r=0, offset_c=0):
        """Extract alive cells from a quadtree node."""
        if node.pop == 0:
            return []
        if node.level == 0:
            return [(offset_r, offset_c)] if node.nw else []
        half = 1 << (node.level - 1)
        cells = []
        cells.extend(self.to_cells(node.nw, offset_r, offset_c))
        cells.extend(self.to_cells(node.ne, offset_r, offset_c + half))
        cells.extend(self.to_cells(node.sw, offset_r + half, offset_c))
        cells.extend(self.to_cells(node.se, offset_r + half, offset_c + half))
        return cells

    def population(self, node):
        return node.pop

    def set_cell(self, node, row, col, value):
        """Set a single cell in the quadtree. Returns a new root node.
        
        row, col are relative to the node's top-left corner (0, 0).
        """
        if node.level == 0:
            return self.on if value else self.off
        
        half = 1 << (node.level - 1)
        if row < half:
            if col < half:
                return self.make_node(node.level,
                    self.set_cell(node.nw, row, col, value),
                    node.ne, node.sw, node.se)
            else:
                return self.make_node(node.level,
                    node.nw,
                    self.set_cell(node.ne, row, col - half, value),
                    node.sw, node.se)
        else:
            if col < half:
                return self.make_node(node.level,
                    node.nw, node.ne,
                    self.set_cell(node.sw, row - half, col, value),
                    node.se)
            else:
                return self.make_node(node.level,
                    node.nw, node.ne, node.sw,
                    self.set_cell(node.se, row - half, col - half, value))

    def get_cell(self, node, row, col):
        """Get the value of a single cell in the quadtree."""
        if node.level == 0:
            return node.nw  # 0 or 1
        half = 1 << (node.level - 1)
        if row < half:
            if col < half:
                return self.get_cell(node.nw, row, col)
            else:
                return self.get_cell(node.ne, row, col - half)
        else:
            if col < half:
                return self.get_cell(node.sw, row - half, col)
            else:
                return self.get_cell(node.se, row - half, col - half)

    def clear_rect(self, node, r0, c0, r1, c1):
        """Clear all cells in rectangle [r0,r1) x [c0,c1). Returns new root."""
        if node.pop == 0:
            return node
        if node.level == 0:
            if r0 <= 0 < r1 and c0 <= 0 < c1:
                return self.off
            return node
        
        half = 1 << (node.level - 1)
        # Check if rectangle overlaps each quadrant
        nw = node.nw
        ne = node.ne
        sw = node.sw
        se = node.se
        
        if r0 < half and c0 < half:
            nw = self.clear_rect(nw, r0, c0, min(r1, half), min(c1, half))
        if r0 < half and c1 > half:
            ne = self.clear_rect(ne, r0, max(c0 - half, 0), min(r1, half), c1 - half)
        if r1 > half and c0 < half:
            sw = self.clear_rect(sw, max(r0 - half, 0), c0, r1 - half, min(c1, half))
        if r1 > half and c1 > half:
            se = self.clear_rect(se, max(r0 - half, 0), max(c0 - half, 0), r1 - half, c1 - half)
        
        return self.make_node(node.level, nw, ne, sw, se)

    def _rle_to_level3(self, rle_str):
        """Parse an RLE string into an 8x8 grid and build a level-3 node.
        
        RLE: '$' = end of row, '.' or 'b' = dead, '*' or 'o' = alive,
        digits before a char = repeat count.
        """
        grid = [[0]*8 for _ in range(8)]
        row, col = 0, 0
        i = 0
        while i < len(rle_str):
            ch = rle_str[i]
            if ch == '$':
                row += 1
                col = 0
                i += 1
            elif ch == '.' or ch == 'b':
                col += 1
                i += 1
            elif ch == '*' or ch == 'o':
                if row < 8 and col < 8:
                    grid[row][col] = 1
                col += 1
                i += 1
            elif ch.isdigit():
                # Read full number
                j = i
                while j < len(rle_str) and rle_str[j].isdigit():
                    j += 1
                count = int(rle_str[i:j])
                if j < len(rle_str):
                    ch2 = rle_str[j]
                    if ch2 == '$':
                        row += count
                        col = 0
                    elif ch2 == '.' or ch2 == 'b':
                        col += count
                    elif ch2 == '*' or ch2 == 'o':
                        for _ in range(count):
                            if row < 8 and col < 8:
                                grid[row][col] = 1
                            col += 1
                    i = j + 1
                else:
                    i = j
            else:
                i += 1
        
        # Build level-3 node from 8x8 grid using the quadtree structure:
        # Level 1: 2x2 cells
        # Level 2: 4x4 = 2x2 of level-1
        # Level 3: 8x8 = 2x2 of level-2
        def make_l1(r, c):
            nw = self.on if grid[r][c] else self.off
            ne = self.on if grid[r][c+1] else self.off
            sw = self.on if grid[r+1][c] else self.off
            se = self.on if grid[r+1][c+1] else self.off
            return self.make_node(1, nw, ne, sw, se)
        
        def make_l2(r, c):
            return self.make_node(2,
                make_l1(r, c), make_l1(r, c+2),
                make_l1(r+2, c), make_l1(r+2, c+2))
        
        return self.make_node(3,
            make_l2(0, 0), make_l2(0, 4),
            make_l2(4, 0), make_l2(4, 4))

    def load_mc(self, filename):
        """Load a Golly macrocell (.mc) file directly into the Hashlife tree.
        
        Handles both 5-integer node lines and RLE-encoded leaf nodes.
        Each bare RLE line becomes a level-3 (8x8) leaf node.
        """
        nodes = {}  # mc_id -> Node
        
        with open(filename) as f:
            idx = 1
            generation = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('['):
                    continue
                if line.startswith('#G'):
                    generation = int(line.split()[1])
                    continue
                if line.startswith('#'):
                    continue
                
                parts = line.split()
                
                # Check if this is a node line (level nw ne sw se)
                is_node_line = False
                if len(parts) == 5:
                    try:
                        int(parts[0])
                        is_node_line = all(p.lstrip('-').isdigit() for p in parts)
                    except ValueError:
                        pass
                
                if not is_node_line:
                    # RLE-encoded leaf node (level 3, 8x8 grid)
                    nodes[idx] = self._rle_to_level3(line)
                    idx += 1
                    continue
                
                level = int(parts[0])
                refs = [int(x) for x in parts[1:]]
                
                if level == 1:
                    # Leaf node: refs are cell values (0 or 1)
                    nw = self.on if refs[0] else self.off
                    ne = self.on if refs[1] else self.off
                    sw = self.on if refs[2] else self.off
                    se = self.on if refs[3] else self.off
                    nodes[idx] = self.make_node(1, nw, ne, sw, se)
                else:
                    # Interior node: refs are indices into nodes dict, 0 = empty
                    def resolve(ref, lvl):
                        if ref == 0:
                            return self.empty_node(lvl)
                        return nodes[ref]
                    
                    child_level = level - 1
                    nw = resolve(refs[0], child_level)
                    ne = resolve(refs[1], child_level)
                    sw = resolve(refs[2], child_level)
                    se = resolve(refs[3], child_level)
                    nodes[idx] = self.make_node(level, nw, ne, sw, se)
                
                idx += 1
        
        root = nodes[idx - 1]
        return root, generation


def test_hashlife():
    """Quick test: verify Gosper gun produces gliders."""
    from gol_matmul import GOSPER_GUN, GOLGrid

    hl = HashLife()

    # Test with Gosper gun
    gun_cells = [(r, c) for r, c in GOSPER_GUN]
    root = hl.from_cells(gun_cells)

    print(f"Gosper gun: {hl.population(root)} cells, level {root.level}")

    # Advance 30 generations (one glider period)
    root = hl.expand(root)
    root = hl.expand(root)
    result = hl.advance(root, 30)
    result_cells = hl.to_cells(result)
    print(f"After 30 gens: {len(result_cells)} cells")

    # Advance 120 generations
    root2 = hl.from_cells(gun_cells)
    root2 = hl.expand(root2)
    root2 = hl.expand(root2)
    root2 = hl.expand(root2)
    result2 = hl.advance(root2, 120)
    result2_cells = hl.to_cells(result2)
    print(f"After 120 gens: {len(result2_cells)} cells")

    # Compare with naive simulator
    naive = GOLGrid()
    for r, c in GOSPER_GUN:
        naive.set_cell(r, c)
    naive.run(120)
    print(f"Naive after 120: {naive.population()} cells")
    print(f"Match: {len(result2_cells) == naive.population()}")

    # Speed test: advance lots of generations
    import time
    root3 = hl.from_cells(gun_cells)
    for _ in range(10):
        root3 = hl.expand(root3)
    t0 = time.time()
    result3 = hl.advance(root3, 10000)
    t1 = time.time()
    print(f"10000 gens via Hashlife: {hl.population(result3)} cells, {t1-t0:.3f}s")

    t0 = time.time()
    result4 = hl.advance(root3, 1000000)
    t1 = time.time()
    print(f"1M gens via Hashlife: {hl.population(result4)} cells, {t1-t0:.3f}s")


if __name__ == '__main__':
    test_hashlife()
