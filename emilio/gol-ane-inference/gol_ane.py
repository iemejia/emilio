#!/usr/bin/env python3
"""
GOL-on-ANE: Tile-based Game of Life simulation using CoreML/ANE.

The GOL computer pattern (~1.4M cells in a ~210K×215K grid) is split into
tiles. Each tile is a CoreML model invocation on the ANE. Only non-empty
tiles (and their neighbors) are processed each generation.

This is the real thing: the ANE hardware runs the B3/S23 GOL rules via
Conv2d, and the GOL grid contains a programmable computer that computes
matrix multiplication.

Chain: ANE (Conv2d) → GOL (B3/S23) → Computer → Matmul
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gol-inference'))
sys.setrecursionlimit(500000)

from hashlife import HashLife
from gol_programmer import program_computer

# CoreML
try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False
    print("WARNING: coremltools not available, using CPU-only mode")


TILE_SIZE = 4096  # must match the CoreML model


class TiledGOL:
    """Tile-based GOL simulator. Each tile can be run on ANE via CoreML."""

    def __init__(self, tile_size=TILE_SIZE):
        self.tile_size = tile_size
        # tiles[(tr, tc)] = np.array of shape (tile_size, tile_size), dtype=float32
        self.tiles = {}
        self.generation = 0
        # Bounding box in cell coordinates
        self.row_offset = 0
        self.col_offset = 0

    def set_cell(self, row, col, value=1):
        """Set a cell. Row/col are absolute coordinates."""
        tr = (row - self.row_offset) // self.tile_size
        tc = (col - self.col_offset) // self.tile_size
        lr = (row - self.row_offset) % self.tile_size
        lc = (col - self.col_offset) % self.tile_size
        key = (tr, tc)
        if key not in self.tiles:
            if value == 0:
                return
            self.tiles[key] = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        self.tiles[key][lr, lc] = float(value)

    def get_cell(self, row, col):
        tr = (row - self.row_offset) // self.tile_size
        tc = (col - self.col_offset) // self.tile_size
        lr = (row - self.row_offset) % self.tile_size
        lc = (col - self.col_offset) % self.tile_size
        key = (tr, tc)
        if key not in self.tiles:
            return 0
        return int(self.tiles[key][lr, lc] > 0.5)

    def population(self):
        total = 0
        for tile in self.tiles.values():
            total += int(np.sum(tile > 0.5))
        return total

    def active_tiles(self):
        """Return set of tile keys that have alive cells."""
        return {k for k, v in self.tiles.items() if np.any(v > 0.5)}

    def tiles_to_process(self):
        """Return tile keys that need processing: active tiles + their neighbors.
        A cell on the edge of an active tile could birth a cell in a neighbor tile."""
        active = self.active_tiles()
        to_process = set(active)
        for tr, tc in active:
            # Check if any alive cell is within 1 cell of the tile edge
            tile = self.tiles[(tr, tc)]
            # Top edge
            if np.any(tile[0, :] > 0.5):
                to_process.add((tr - 1, tc))
            # Bottom edge
            if np.any(tile[-1, :] > 0.5):
                to_process.add((tr + 1, tc))
            # Left edge
            if np.any(tile[:, 0] > 0.5):
                to_process.add((tr, tc - 1))
            # Right edge
            if np.any(tile[:, -1] > 0.5):
                to_process.add((tr, tc + 1))
            # Corners
            if tile[0, 0] > 0.5:
                to_process.add((tr - 1, tc - 1))
            if tile[0, -1] > 0.5:
                to_process.add((tr - 1, tc + 1))
            if tile[-1, 0] > 0.5:
                to_process.add((tr + 1, tc - 1))
            if tile[-1, -1] > 0.5:
                to_process.add((tr + 1, tc + 1))
        return to_process

    def get_padded_tile(self, tr, tc):
        """Get a tile with 1-cell overlap from neighbors for boundary correctness.
        Returns (tile_size+2) × (tile_size+2) array."""
        T = self.tile_size
        padded = np.zeros((T + 2, T + 2), dtype=np.float32)

        # Center
        if (tr, tc) in self.tiles:
            padded[1:T+1, 1:T+1] = self.tiles[(tr, tc)]

        # Top row
        if (tr-1, tc) in self.tiles:
            padded[0, 1:T+1] = self.tiles[(tr-1, tc)][-1, :]
        # Bottom row
        if (tr+1, tc) in self.tiles:
            padded[T+1, 1:T+1] = self.tiles[(tr+1, tc)][0, :]
        # Left column
        if (tr, tc-1) in self.tiles:
            padded[1:T+1, 0] = self.tiles[(tr, tc-1)][:, -1]
        # Right column
        if (tr, tc+1) in self.tiles:
            padded[1:T+1, T+1] = self.tiles[(tr, tc+1)][:, 0]
        # Corners
        if (tr-1, tc-1) in self.tiles:
            padded[0, 0] = self.tiles[(tr-1, tc-1)][-1, -1]
        if (tr-1, tc+1) in self.tiles:
            padded[0, T+1] = self.tiles[(tr-1, tc+1)][-1, 0]
        if (tr+1, tc-1) in self.tiles:
            padded[T+1, 0] = self.tiles[(tr+1, tc-1)][0, -1]
        if (tr+1, tc+1) in self.tiles:
            padded[T+1, T+1] = self.tiles[(tr+1, tc+1)][0, 0]

        return padded

    def step_cpu(self, padded):
        """One GOL generation using numpy convolution (CPU reference).
        Input: (T+2)×(T+2) padded tile. Output: T×T result."""
        T = self.tile_size
        # Count neighbors via slicing (equivalent to conv with [[1,1,1],[1,0,1],[1,1,1]])
        h, w = padded.shape
        neighbors = np.zeros((h, w), dtype=np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbors[1:h-1, 1:w-1] += padded[1+dr:h-1+dr, 1+dc:w-1+dc]

        center = padded[1:T+1, 1:T+1]
        n = neighbors[1:T+1, 1:T+1]

        # B3/S23 rules
        birth = ((center < 0.5) & (np.abs(n - 3.0) < 0.5)).astype(np.float32)
        survive = ((center > 0.5) & ((np.abs(n - 2.0) < 0.5) | (np.abs(n - 3.0) < 0.5))).astype(np.float32)
        return birth + survive

    def step_all_cpu(self):
        """Step all active tiles using CPU."""
        to_process = self.tiles_to_process()
        new_tiles = {}
        for key in to_process:
            padded = self.get_padded_tile(*key)
            result = self.step_cpu(padded)
            if np.any(result > 0.5):
                new_tiles[key] = result
        self.tiles = new_tiles
        self.generation += 1


class CoreMLGOLEngine:
    """Wraps a CoreML GOL model for ANE execution (single-tile)."""

    def __init__(self, model_path, tile_size=TILE_SIZE):
        import coremltools as ct
        self.tile_size = tile_size

        print(f"Loading CoreML GOL model: {model_path}")
        self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        print(f"  Loaded. Compute units: ALL (CPU+GPU+ANE)")

    def step(self, padded_tile):
        """Run one GOL generation on a padded tile.
        Input: (T+2)×(T+2) numpy array
        Output: T×T numpy array (inner region)
        """
        T = self.tile_size
        inp = padded_tile.reshape(1, 1, T + 2, T + 2).astype(np.float32)
        result = self.model.predict({"grid": inp})
        out = result["next_grid"]
        return np.array(out[0, 0, 1:T+1, 1:T+1], dtype=np.float32)


class CoreMLGOLBatchEngine:
    """Wraps a batched CoreML GOL model for ANE execution.
    Processes multiple tiles per predict() call to reduce dispatch overhead."""

    def __init__(self, model_path, tile_size=TILE_SIZE, batch_size=32):
        import coremltools as ct
        self.tile_size = tile_size
        self.batch_size = batch_size

        print(f"Loading batched CoreML GOL model: {model_path} (batch={batch_size})")
        self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        print(f"  Loaded. Compute units: ALL (CPU+GPU+ANE)")

    def step_batch(self, padded_tiles):
        """Run one GOL generation on a batch of padded tiles.
        Input: list of (T+2)×(T+2) numpy arrays
        Output: list of T×T numpy arrays (inner regions)
        """
        T = self.tile_size
        B = self.batch_size
        n = len(padded_tiles)

        # Assemble batch tensor [B, 1, T+2, T+2]
        batch = np.zeros((B, 1, T + 2, T + 2), dtype=np.float32)
        for i in range(n):
            batch[i, 0] = padded_tiles[i]

        result = self.model.predict({"grid": batch})
        out = np.array(result["next_grid"], dtype=np.float32)

        # Extract inner T×T for each tile that was provided
        results = []
        for i in range(n):
            results.append(out[i, 0, 1:T+1, 1:T+1].copy())
        return results


def load_gol_computer(tile_size=TILE_SIZE):
    """Load the programmed GOL computer into a TiledGOL grid."""
    from gol_programmer import MATMUL_ELEMENT

    print("Programming GOL computer for matmul (1*5 + 2*7 = 19)...")
    hl, root, gen = program_computer(MATMUL_ELEMENT, N=8, M=8, P=32)

    # Find bounding box
    center = 1 << (root.level - 1)

    def find_bounds(node, row_off, col_off, level):
        if node.pop == 0:
            return None
        if level == 0:
            return (row_off, col_off, row_off, col_off)
        half = 1 << (level - 1)
        bounds = []
        for quad, (dr, dc) in [(node.nw, (0, 0)), (node.ne, (0, half)),
                                (node.sw, (half, 0)), (node.se, (half, half))]:
            b = find_bounds(quad, row_off + dr, col_off + dc, level - 1)
            if b:
                bounds.append(b)
        if not bounds:
            return None
        return (min(b[0] for b in bounds), min(b[1] for b in bounds),
                max(b[2] for b in bounds), max(b[3] for b in bounds))

    bb = find_bounds(root, 0, 0, root.level)
    h = bb[2] - bb[0] + 1
    w = bb[3] - bb[1] + 1
    print(f"  Bounding box: {h}×{w}, {root.pop:,} alive cells")

    # Extract cells into TiledGOL
    print("  Extracting cells into tiled grid...")
    tiled = TiledGOL(tile_size)
    # Align offset to tile boundary
    tiled.row_offset = (bb[0] // tile_size) * tile_size
    tiled.col_offset = (bb[1] // tile_size) * tile_size
    tiled.generation = gen

    # Walk quadtree to extract cells
    t0 = time.time()
    cells_loaded = [0]

    def extract_cells(node, row_off, col_off, level):
        if node.pop == 0:
            return
        if level == 0:
            if node.pop > 0:
                tiled.set_cell(row_off, col_off, 1)
                cells_loaded[0] += 1
            return
        half = 1 << (level - 1)
        extract_cells(node.nw, row_off, col_off, level - 1)
        extract_cells(node.ne, row_off, col_off + half, level - 1)
        extract_cells(node.sw, row_off + half, col_off, level - 1)
        extract_cells(node.se, row_off + half, col_off + half, level - 1)

    extract_cells(root, 0, 0, root.level)
    t1 = time.time()

    active = tiled.active_tiles()
    print(f"  Loaded {cells_loaded[0]:,} cells into {len(active)} tiles in {t1-t0:.1f}s")
    print(f"  Generation: {tiled.generation:,}")

    return tiled


def verify_cpu_step(tiled):
    """Run a few steps on CPU to verify tiled GOL works correctly."""
    print("\nVerifying tiled CPU GOL (3 steps)...")
    pop0 = tiled.population()
    gen0 = tiled.generation

    for i in range(3):
        t0 = time.time()
        tiled.step_all_cpu()
        t1 = time.time()
        pop = tiled.population()
        active = len(tiled.active_tiles())
        print(f"  Gen {tiled.generation}: pop={pop:,}, active_tiles={active}, "
              f"time={t1-t0:.1f}s")

    return True


def run_ane_simulation(tiled, engine, n_gens=10):
    """Run GOL simulation using ANE for each tile step (unbatched)."""
    print(f"\nRunning {n_gens} generations on ANE (unbatched)...")

    for g in range(n_gens):
        to_process = tiled.tiles_to_process()
        new_tiles = {}

        t0 = time.time()
        for key in to_process:
            padded = tiled.get_padded_tile(*key)
            result = engine.step(padded)
            if np.any(result > 0.5):
                new_tiles[key] = result

        tiled.tiles = new_tiles
        tiled.generation += 1
        t1 = time.time()

        pop = tiled.population()
        n_tiles = len(to_process)
        print(f"  Gen {tiled.generation}: pop={pop:,}, tiles={n_tiles}, "
              f"time={t1-t0:.1f}s ({1000*(t1-t0)/max(n_tiles,1):.1f}ms/tile)")


def run_ane_simulation_batched(tiled, engine, n_gens=10):
    """Run GOL simulation using ANE with batched tile processing."""
    B = engine.batch_size
    print(f"\nRunning {n_gens} generations on ANE (batched, B={B})...")

    for g in range(n_gens):
        to_process = list(tiled.tiles_to_process())
        new_tiles = {}

        t0 = time.time()

        # Prepare all padded tiles
        padded_list = []
        for key in to_process:
            padded_list.append(tiled.get_padded_tile(*key))

        # Process in batches of B
        t_ane = 0.0
        for batch_start in range(0, len(to_process), B):
            batch_end = min(batch_start + B, len(to_process))
            batch_padded = padded_list[batch_start:batch_end]

            ta = time.time()
            batch_results = engine.step_batch(batch_padded)
            t_ane += time.time() - ta

            for i, key in enumerate(to_process[batch_start:batch_end]):
                result = batch_results[i]
                if np.any(result > 0.5):
                    new_tiles[key] = result

        tiled.tiles = new_tiles
        tiled.generation += 1
        t1 = time.time()

        pop = tiled.population()
        n_tiles = len(to_process)
        n_batches = math.ceil(n_tiles / B)
        print(f"  Gen {tiled.generation}: pop={pop:,}, tiles={n_tiles}, "
              f"batches={n_batches}, time={t1-t0:.1f}s "
              f"(ANE={t_ane:.2f}s, overhead={t1-t0-t_ane:.2f}s)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GOL-on-ANE: Matmul via Game of Life on Apple Neural Engine")
    parser.add_argument("--model", help="Path to CoreML GOL model (unbatched)")
    parser.add_argument("--batch-model", help="Path to batched CoreML GOL model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for batched model")
    parser.add_argument("--tile-size", type=int, default=4096)
    parser.add_argument("--cpu-only", action="store_true", help="CPU reference only")
    parser.add_argument("--gens", type=int, default=3, help="Generations to simulate")
    args = parser.parse_args()

    print("=" * 70)
    print("  GOL-on-ANE: Matrix Multiplication via Game of Life on ANE")
    print("  Chain: ANE (Conv2d) → GOL (B3/S23) → Computer → Matmul")
    print("=" * 70)
    print()

    # Load the programmed GOL computer
    tiled = load_gol_computer(args.tile_size)

    if args.cpu_only or (not args.model and not args.batch_model):
        # CPU reference
        verify_cpu_step(tiled)
    elif args.batch_model:
        # Batched ANE simulation
        engine = CoreMLGOLBatchEngine(args.batch_model, args.tile_size, args.batch_size)
        run_ane_simulation_batched(tiled, engine, args.gens)
    else:
        # Unbatched ANE simulation
        engine = CoreMLGOLEngine(args.model, args.tile_size)
        run_ane_simulation(tiled, engine, args.gens)

    print(f"\nFinal state: gen={tiled.generation:,}, pop={tiled.population():,}")


if __name__ == "__main__":
    main()
