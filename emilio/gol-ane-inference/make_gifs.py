#!/usr/bin/env python3
"""Generate animated GIFs for the GOL-on-ANE paper.

GIFs:
  1. gol_conv_step.gif    — GOL evolution showing the Conv2d(3×3) kernel in action
  2. tiled_computer.gif   — zoomed-out view of 880 tiles being processed
  3. pipeline_scaling.gif — animated bar chart: 1T→2T→4T→8T throughput
  4. chain_animation.gif  — the full chain: Conv2d → B3/S23 → Computer → Matmul
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import struct
import math

OUT_DIR = os.path.join(os.path.dirname(__file__), "gifs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors (matching the HTML paper) ─────────────────────────────
BG       = (13, 17, 23)       # --color-canvas-default
BG2      = (22, 27, 34)       # --color-canvas-subtle
BORDER   = (48, 54, 61)       # --color-border-default
FG       = (230, 237, 243)    # --color-fg-default
FG_MUTED = (125, 133, 144)    # --color-fg-muted
ACCENT   = (88, 166, 255)     # --color-accent-fg
GREEN    = (63, 185, 80)      # --color-success-fg
YELLOW   = (210, 153, 34)     # --color-attention-fg
RED      = (248, 81, 73)      # --color-danger-fg
PURPLE   = (163, 113, 247)    # --color-done-fg

ALIVE_CLR = (63, 185, 80)     # bright green
DEAD_CLR  = (30, 35, 42)      # just slightly above BG
KERNEL_CLR = (88, 166, 255, 80)  # accent with alpha


def try_load_font(size):
    candidates = [
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/Library/Fonts/Courier New.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_SM = try_load_font(13)
FONT_MD = try_load_font(16)
FONT_LG = try_load_font(24)
FONT_XL = try_load_font(32)


# ═══════════════════════════════════════════════════════════════════
# GOL simulation
# ═══════════════════════════════════════════════════════════════════

def gol_step(grid):
    padded = np.pad(grid, 1, mode='constant')
    n = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
         padded[1:-1, :-2] +                     padded[1:-1, 2:] +
         padded[2:, :-2]   + padded[2:, 1:-1]   + padded[2:, 2:])
    return ((n == 3) | ((grid == 1) & (n == 2))).astype(np.uint8)


def gosper_glider_gun():
    grid = np.zeros((64, 80), dtype=np.uint8)
    gun = [
        (5,1),(5,2),(6,1),(6,2),
        (3,13),(3,14),(4,12),(4,16),(5,11),(5,17),
        (6,11),(6,15),(6,17),(6,18),(7,11),(7,17),
        (8,12),(8,16),(9,13),(9,14),
        (1,25),(2,23),(2,25),(3,21),(3,22),(4,21),(4,22),
        (5,21),(5,22),(6,23),(6,25),(7,25),
        (3,35),(3,36),(4,35),(4,36),
    ]
    for r, c in gun:
        grid[r, c] = 1
    return grid


def r_pentomino():
    """R-pentomino: classic chaotic pattern."""
    grid = np.zeros((80, 100), dtype=np.uint8)
    mid_r, mid_c = 40, 50
    # R-pentomino
    for r, c in [(0,1),(0,2),(1,0),(1,1),(2,1)]:
        grid[mid_r + r, mid_c + c] = 1
    return grid


# ═══════════════════════════════════════════════════════════════════
# GIF 1: GOL evolution with kernel visualization
# ═══════════════════════════════════════════════════════════════════

def render_gol_kernel_frame(grid, gen, kernel_pos, cell_size=8):
    """Render GOL grid showing the 3×3 kernel sliding across."""
    h, w = grid.shape
    img_w = w * cell_size + 1
    img_h = h * cell_size + 80
    img = Image.new('RGB', (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)

    y_off = 55

    # Header
    draw.text((10, 8), f"Conv2d(3×3) → B3/S23", fill=ACCENT, font=FONT_LG)
    draw.text((10, 34), f"Generation {gen}  ·  "
              f"Kernel: [[1,1,1],[1,9,1],[1,1,1]]", fill=FG_MUTED, font=FONT_SM)

    # Draw cells
    for r in range(h):
        for c in range(w):
            x0 = c * cell_size
            y0 = r * cell_size + y_off
            color = ALIVE_CLR if grid[r, c] else DEAD_CLR
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                          fill=color)

    # Draw kernel overlay (3×3 box at kernel_pos)
    if kernel_pos is not None:
        kr, kc = kernel_pos
        x0 = (kc - 1) * cell_size
        y0 = (kr - 1) * cell_size + y_off
        x1 = (kc + 2) * cell_size
        y1 = (kr + 2) * cell_size + y_off
        # Semi-transparent overlay via rectangle outline
        for offset in range(2):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                          outline=ACCENT)
        # Center cell highlight
        cx0 = kc * cell_size
        cy0 = kr * cell_size + y_off
        draw.rectangle([cx0, cy0, cx0 + cell_size - 1, cy0 + cell_size - 1],
                       outline=YELLOW, width=2)

        # Show sum value
        s_val = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = kr + dr, kc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    weight = 9 if (dr == 0 and dc == 0) else 1
                    s_val += grid[nr, nc] * weight
        draw.text((x1 + 6, y0 + 2), f"s={s_val}", fill=YELLOW, font=FONT_SM)

        # Show result
        if s_val == 3:
            draw.text((x1 + 6, y0 + 18), "birth!", fill=GREEN, font=FONT_SM)
        elif s_val in (11, 12):
            draw.text((x1 + 6, y0 + 18), "survive", fill=GREEN, font=FONT_SM)
        elif s_val >= 9:
            draw.text((x1 + 6, y0 + 18), "die", fill=RED, font=FONT_SM)

    # Population count
    pop = int(grid.sum())
    draw.text((img_w - 130, 8), f"pop: {pop}", fill=GREEN, font=FONT_MD)

    return img


def make_gol_conv_gif():
    """GIF 1: GOL evolution with Conv2d kernel visualization."""
    print("GIF 1: GOL evolution with kernel...")
    grid = gosper_glider_gun()
    frames = []
    cell_size = 8

    # Find interesting alive cells to position kernel near
    for gen in range(200):
        if gen % 20 == 0:
            print(f"  gen {gen}/200")

        # Place kernel near an alive cell (scan for one)
        alive_cells = list(zip(*np.where(grid == 1)))
        if alive_cells and gen % 2 == 0:
            # Pick a cell near the action (bottom of glider stream)
            idx = min(len(alive_cells) - 1, gen // 2 % len(alive_cells))
            kr, kc = alive_cells[idx]
            # Clamp to visible range
            kr = max(1, min(kr, grid.shape[0] - 2))
            kc = max(1, min(kc, grid.shape[1] - 2))
            kernel_pos = (kr, kc)
        else:
            kernel_pos = None

        frames.append(render_gol_kernel_frame(grid, gen, kernel_pos, cell_size))
        grid = gol_step(grid)

    path = os.path.join(OUT_DIR, "gol_conv_step.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
# GIF 2: Tiled computer — 880 tiles being processed generation by gen
# ═══════════════════════════════════════════════════════════════════

def load_tile_map():
    """Load gol_computer.bin and compute which tiles are occupied."""
    bin_path = os.path.join(os.path.dirname(__file__), "gol_computer.bin")
    if not os.path.exists(bin_path):
        print("  [!] gol_computer.bin not found, generating synthetic tile map")
        return generate_synthetic_tiles()

    with open(bin_path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GOL\0'
        n_cells = struct.unpack('<I', f.read(4))[0]
        row_off = struct.unpack('<i', f.read(4))[0]
        col_off = struct.unpack('<i', f.read(4))[0]
        gen_lo = struct.unpack('<I', f.read(4))[0]
        gen_hi = struct.unpack('<I', f.read(4))[0]

        tiles = set()
        for _ in range(n_cells):
            r = struct.unpack('<i', f.read(4))[0]
            c = struct.unpack('<i', f.read(4))[0]
            tr = (r - row_off) // 1024
            tc = (c - col_off) // 1024
            tiles.add((tr, tc))

    return tiles, n_cells, row_off, col_off


def generate_synthetic_tiles():
    """Generate a synthetic tile map resembling the GOL computer."""
    tiles = set()
    # ~210 tiles wide, ~210 tiles tall, ~880 non-empty
    # Roughly a rectangular region with some gaps
    for tr in range(210):
        for tc in range(210):
            # Create a pattern that looks like a computer circuit
            if (10 <= tr <= 200 and 10 <= tc <= 200 and
                ((tr + tc) % 7 != 0) and
                not (80 <= tr <= 130 and 80 <= tc <= 130)):
                if len(tiles) < 880:
                    tiles.add((tr, tc))
    return tiles, 1426998, 233472, 74752


def render_tile_frame(tiles_set, active_tiles, gen, processed_count,
                      total_tiles, elapsed_ms, width=800, height=520):
    """Render the tiled GOL computer view."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((15, 8), "GOL Computer — Tiled Simulation",
              fill=FG, font=FONT_LG)
    draw.text((15, 34), f"Generation {gen:,}  ·  "
              f"1,426,998 cells  ·  880 tiles × 1024²",
              fill=FG_MUTED, font=FONT_SM)

    # Find bounding box of tiles
    if not tiles_set:
        return img
    min_r = min(t[0] for t in tiles_set)
    max_r = max(t[0] for t in tiles_set)
    min_c = min(t[1] for t in tiles_set)
    max_c = max(t[1] for t in tiles_set)

    tile_range_r = max_r - min_r + 1
    tile_range_c = max_c - min_c + 1

    # Compute cell size to fit
    view_w = width - 40
    view_h = height - 130
    view_y = 58

    cell_w = max(1, view_w // tile_range_c)
    cell_h = max(1, view_h // tile_range_r)
    cell_sz = min(cell_w, cell_h, 4)  # cap at 4px per tile

    # Center the grid
    grid_w = tile_range_c * cell_sz
    grid_h = tile_range_r * cell_sz
    off_x = (width - grid_w) // 2
    off_y = view_y + (view_h - grid_h) // 2

    # Draw tiles
    for tr, tc in tiles_set:
        x = off_x + (tc - min_c) * cell_sz
        y = off_y + (tr - min_r) * cell_sz

        if (tr, tc) in active_tiles:
            color = ACCENT  # being processed right now
        else:
            color = (35, 80, 45)  # idle tile (dim green)

        draw.rectangle([x, y, x + cell_sz - 1, y + cell_sz - 1], fill=color)

    # Progress bar and stats
    bar_y = height - 60
    bar_w = width - 40
    draw.rectangle([20, bar_y, 20 + bar_w, bar_y + 10], fill=BG2, outline=BORDER)
    progress = min(processed_count / max(total_tiles, 1), 1.0)
    if progress > 0:
        draw.rectangle([21, bar_y + 1, 21 + int((bar_w - 2) * progress), bar_y + 9],
                      fill=GREEN)

    draw.text((20, bar_y + 16),
              f"Tiles: {processed_count}/{total_tiles}  ·  "
              f"8 GPU workers  ·  {elapsed_ms:.0f}ms  ·  "
              f"9,600 tiles/s",
              fill=FG_MUTED, font=FONT_SM)

    return img


def make_tiled_gif():
    """GIF 2: Tiled computer processing animation."""
    print("GIF 2: Tiled computer...")

    result = load_tile_map()
    if len(result) == 4:
        tiles_set, n_cells, row_off, col_off = result
    else:
        tiles_set = result[0]

    tiles_list = sorted(tiles_set)
    total = len(tiles_list)
    print(f"  {total} tiles loaded")

    frames = []
    n_frames = 120
    tiles_per_frame = max(1, total // (n_frames - 20))

    gen_base = 8760000

    for frame in range(n_frames):
        if frame % 20 == 0:
            print(f"  frame {frame}/{n_frames}")

        processed = min(frame * tiles_per_frame, total)
        # Active tiles: the current batch being processed
        batch_start = max(0, processed - tiles_per_frame * 2)
        active = set(tiles_list[batch_start:processed]) if processed > 0 else set()

        elapsed = processed / 9600 * 1000 if processed > 0 else 0

        frames.append(render_tile_frame(
            tiles_set, active, gen_base + 1, processed, total, elapsed))

    # Hold final frame
    for _ in range(20):
        frames.append(render_tile_frame(
            tiles_set, set(), gen_base + 1, total, total, 91.0))

    path = os.path.join(OUT_DIR, "tiled_computer.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
# GIF 3: Pipeline scaling — bar chart animating throughput
# ═══════════════════════════════════════════════════════════════════

def render_pipeline_frame(data_visible, highlight_idx, width=720, height=420):
    """Render animated bar chart of pipeline throughput."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    # Data: (label, tiles/s, ms/gen, color)
    all_data = [
        ("Python\nbaseline",    60,    15000, FG_MUTED),
        ("Swift\n1T GPU",       3660,  240,   ACCENT),
        ("Dual\nANE+GPU",      2930,  300,   PURPLE),
        ("4T\nGPU",            5200,  170,   YELLOW),
        ("6T\nGPU",            7400,  119,   YELLOW),
        ("8T\nGPU",            9600,  91,    GREEN),
    ]

    draw.text((20, 10), "Throughput Scaling: Pipelined GPU Prediction",
              fill=FG, font=FONT_LG)
    draw.text((20, 38), "tiles/s (880 tiles per generation)", fill=FG_MUTED, font=FONT_SM)

    chart_x = 100
    chart_y = 70
    chart_w = width - 140
    chart_h = height - 150
    max_val = 10000

    n_bars = min(data_visible, len(all_data))

    bar_h = min(40, (chart_h - 10) // len(all_data))
    gap = 8

    for i in range(n_bars):
        label, tps, ms, color = all_data[i]
        y = chart_y + i * (bar_h + gap)

        # Label
        lines = label.split('\n')
        for li, line in enumerate(lines):
            draw.text((10, y + li * 16), line, fill=FG_MUTED, font=FONT_SM)

        # Bar
        bar_w = int(chart_w * min(tps / max_val, 1.0))
        bar_color = color if i != highlight_idx else (
            min(color[0] + 40, 255),
            min(color[1] + 40, 255),
            min(color[2] + 40, 255))
        draw.rectangle([chart_x, y, chart_x + bar_w, y + bar_h - 2],
                      fill=bar_color)

        # Value label
        draw.text((chart_x + bar_w + 8, y + 4),
                  f"{tps:,} t/s  ({ms}ms/gen)",
                  fill=FG, font=FONT_SM)

    # Axis
    draw.line([(chart_x, chart_y - 5),
               (chart_x, chart_y + len(all_data) * (bar_h + gap))],
              fill=BORDER, width=1)

    # Scale marks
    for val in [0, 2500, 5000, 7500, 10000]:
        x = chart_x + int(chart_w * val / max_val)
        y_bottom = chart_y + len(all_data) * (bar_h + gap)
        draw.line([(x, y_bottom), (x, y_bottom + 5)], fill=BORDER, width=1)
        draw.text((x - 15, y_bottom + 8), f"{val//1000}k",
                  fill=FG_MUTED, font=FONT_SM)

    # Bottom annotation
    draw.text((20, height - 50),
              "Key insight: CoreML pipelines concurrent GPU predictions.",
              fill=ACCENT, font=FONT_MD)
    draw.text((20, height - 28),
              "ANE saturates at 2 threads. GPU scales linearly to 8+.",
              fill=FG_MUTED, font=FONT_SM)

    return img


def make_pipeline_gif():
    """GIF 3: Pipeline scaling animation."""
    print("GIF 3: Pipeline scaling...")
    frames = []

    # Reveal bars one by one
    for bar_idx in range(7):
        n_visible = min(bar_idx + 1, 6)
        # Hold each new bar for several frames
        hold = 20 if bar_idx < 6 else 40
        for f in range(hold):
            highlight = bar_idx if bar_idx < 6 and f < 10 else -1
            frames.append(render_pipeline_frame(n_visible, highlight))

    path = os.path.join(OUT_DIR, "pipeline_scaling.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=50, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
# GIF 4: The Chain — animated flow from Conv2d to Matmul result
# ═══════════════════════════════════════════════════════════════════

def render_chain_frame(active_step, pulse_offset, width=800, height=360):
    """Render the computation chain with a pulsing active step."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw.text((width // 2 - 200, 12),
              "The Most Indirect Matmul Ever",
              fill=FG, font=FONT_XL)

    steps = [
        ("GPU / ANE", "Hardware\nAccelerator", ACCENT),
        ("Conv2d(3×3)", "CoreML\nModel", PURPLE),
        ("B3/S23", "GOL\nRules", YELLOW),
        ("8-bit CPU", "GOL\nComputer", GREEN),
        ("1*5+2*7=19", "Matmul\nResult", RED),
    ]

    box_w = 120
    box_h = 80
    total_w = len(steps) * box_w + (len(steps) - 1) * 40
    start_x = (width - total_w) // 2
    box_y = 90

    for i, (title, subtitle, color) in enumerate(steps):
        x = start_x + i * (box_w + 40)

        # Glow effect for active step
        is_active = (i == active_step)
        if is_active:
            brightness = int(20 + 15 * math.sin(pulse_offset * 0.3))
            glow = (min(color[0] + brightness, 255),
                    min(color[1] + brightness, 255),
                    min(color[2] + brightness, 255))
            # Outer glow
            draw.rounded_rectangle(
                [x - 3, box_y - 3, x + box_w + 3, box_y + box_h + 3],
                radius=8, outline=glow, width=2)

        fill = BG2 if not is_active else (
            color[0] // 8, color[1] // 8, color[2] // 8)
        draw.rounded_rectangle(
            [x, box_y, x + box_w, box_y + box_h],
            radius=6, fill=fill, outline=color, width=2 if is_active else 1)

        draw.text((x + 10, box_y + 10), title,
                  fill=color if is_active else FG, font=FONT_MD)

        lines = subtitle.split('\n')
        for li, line in enumerate(lines):
            draw.text((x + 10, box_y + 32 + li * 16), line,
                      fill=FG_MUTED, font=FONT_SM)

        # Arrow between boxes
        if i < len(steps) - 1:
            ax = x + box_w + 5
            ay = box_y + box_h // 2
            draw.text((ax + 5, ay - 8), "→", fill=FG_MUTED, font=FONT_LG)

    # Bottom detail panel — shows detail for active step
    detail_y = 200
    draw.line([(40, detail_y), (width - 40, detail_y)], fill=BORDER, width=1)

    details = [
        [  # GPU/ANE
            "Apple M4 Max — 40 GPU cores",
            "8 feeder threads pipeline predictions concurrently",
            "CoreML dispatches Conv2d kernels to GPU compute units",
            "9,600 tiles/s throughput at 91ms per generation",
        ],
        [  # Conv2d
            "Kernel: [[1,1,1],[1,9,1],[1,1,1]]",
            "13 ops: 1 conv + 6 relu + 4 sub + 1 add + 1 clamp",
            "Merged survive: relu(1.5 − |s−11.5|) covers s=11 and s=12",
            "Input/output: [1,1,1026,1026] Float16",
        ],
        [  # B3/S23
            "Birth: dead cell with exactly 3 neighbors → alive",
            "Survival: alive cell with 2 or 3 neighbors → stays alive",
            "All other cells → dead",
            "Standard Conway's Game of Life rules since 1970",
        ],
        [  # GOL Computer
            "1,426,998 alive cells in a ~210K × 215K grid",
            "Built from Gosper guns, eaters, and glider collisions",
            "8 registers (a0–a7), 32-instruction ROM, 8-bit ALU",
            "One instruction cycle ≈ 41.9 million GOL generations",
        ],
        [  # Matmul
            "A = [[1,2],[3,4]]  ×  B = [[5,6],[7,8]]",
            "C[0][0] = 1×5 + 2×7 = 19  (via repeated addition)",
            "20 ROM instructions, 84 instruction cycles",
            "Full result: C = [[19,22],[43,50]]",
        ],
    ]

    step_details = details[active_step]
    for i, line in enumerate(step_details):
        color_txt = steps[active_step][2] if i == 0 else FG_MUTED
        draw.text((60, detail_y + 14 + i * 22), line,
                  fill=color_txt, font=FONT_MD if i == 0 else FONT_SM)

    # Cosmic scale at very bottom
    draw.text((width // 2 - 170, height - 30),
              "40 million years per token of Qwen 2.5-0.5B",
              fill=FG_MUTED, font=FONT_SM)

    return img


def make_chain_gif():
    """GIF 4: Animated computation chain."""
    print("GIF 4: Computation chain...")
    frames = []

    # Cycle through each step, dwelling for ~3 seconds each (60 frames)
    for cycle in range(2):  # loop twice
        for step in range(5):
            for f in range(50):
                frames.append(render_chain_frame(step, f))

    path = os.path.join(OUT_DIR, "chain_animation.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
# GIF 5: Kernel math — step by step convolution + ReLU decision
# ═══════════════════════════════════════════════════════════════════

def render_kernel_math_frame(grid_5x5, center_r, center_c, phase, width=720, height=400):
    """Show the convolution math step by step on a small grid."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw.text((20, 10), "How Conv2d(3×3) Implements GOL", fill=FG, font=FONT_LG)

    cell_sz = 50
    grid_x = 30
    grid_y = 60

    # Draw 5×5 grid
    for r in range(5):
        for c in range(5):
            x0 = grid_x + c * cell_sz
            y0 = grid_y + r * cell_sz
            val = grid_5x5[r, c]
            color = ALIVE_CLR if val else DEAD_CLR
            draw.rectangle([x0, y0, x0 + cell_sz - 2, y0 + cell_sz - 2],
                          fill=color, outline=BORDER)
            draw.text((x0 + 18, y0 + 15), str(val), fill=FG, font=FONT_MD)

    # Highlight 3×3 kernel window
    kr, kc = center_r, center_c
    kx0 = grid_x + (kc - 1) * cell_sz - 3
    ky0 = grid_y + (kr - 1) * cell_sz - 3
    kx1 = grid_x + (kc + 2) * cell_sz - cell_sz + cell_sz - 1 + 3
    ky1 = grid_y + (kr + 2) * cell_sz - cell_sz + cell_sz - 1 + 3
    draw.rectangle([kx0, ky0, kx1, ky1], outline=ACCENT, width=3)

    # Right side: computation
    rx = 310
    ry = 60

    # Kernel weights
    draw.text((rx, ry), "Kernel weights:", fill=FG_MUTED, font=FONT_SM)
    ry += 22
    weights = [[1,1,1],[1,9,1],[1,1,1]]
    for r in range(3):
        for c in range(3):
            w = weights[r][c]
            tx = rx + c * 40
            clr = YELLOW if w == 9 else FG_MUTED
            draw.text((tx, ry + r * 22), f" {w}", fill=clr, font=FONT_MD)

    ry += 80

    # Compute s
    s = 0
    terms = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr, nc = kr + dr, kc + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                w = 9 if (dr == 0 and dc == 0) else 1
                val = grid_5x5[nr, nc]
                s += w * val
                if w * val > 0:
                    terms.append(f"{w}×{val}")

    draw.text((rx, ry), "Convolution sum:", fill=FG_MUTED, font=FONT_SM)
    ry += 20
    if phase >= 1:
        sum_str = " + ".join(terms) if terms else "0"
        draw.text((rx, ry), f"s = {sum_str}", fill=ACCENT, font=FONT_SM)
        ry += 20
        draw.text((rx, ry), f"s = {s}", fill=ACCENT, font=FONT_LG)
        ry += 35

    if phase >= 2:
        draw.text((rx, ry), "Decision logic:", fill=FG_MUTED, font=FONT_SM)
        ry += 20

        current_alive = grid_5x5[kr, kc] == 1

        if current_alive:
            draw.text((rx, ry), f"Cell is ALIVE (s ∈ 9..17)", fill=GREEN, font=FONT_SM)
            ry += 20
            neighbors = s - 9
            draw.text((rx, ry), f"Neighbors = s − 9 = {neighbors}", fill=FG_MUTED, font=FONT_SM)
            ry += 20
            if s in (11, 12):
                draw.text((rx, ry), f"relu(1.5 − |{s}−11.5|) = 1.0 → SURVIVE",
                          fill=GREEN, font=FONT_MD)
            else:
                draw.text((rx, ry), f"relu(1.5 − |{s}−11.5|) = 0.0 → DIE",
                          fill=RED, font=FONT_MD)
        else:
            draw.text((rx, ry), f"Cell is DEAD (s ∈ 0..8)", fill=FG_MUTED, font=FONT_SM)
            ry += 20
            draw.text((rx, ry), f"Neighbors = s = {s}", fill=FG_MUTED, font=FONT_SM)
            ry += 20
            if s == 3:
                draw.text((rx, ry), f"relu(1 − |{s}−3|) = 1.0 → BIRTH!",
                          fill=GREEN, font=FONT_MD)
            else:
                draw.text((rx, ry), f"relu(1 − |{s}−3|) = 0.0 → stays dead",
                          fill=FG_MUTED, font=FONT_MD)

    return img


def make_kernel_math_gif():
    """GIF 5: Step-by-step kernel math."""
    print("GIF 5: Kernel math...")

    # Three scenarios
    scenarios = [
        # Birth scenario: dead cell with 3 neighbors
        (np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8), 2, 2, "Birth"),
        # Survive scenario: alive cell with 2 neighbors
        (np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8), 2, 2, "Survive"),
        # Death scenario: alive cell with 4 neighbors (overcrowding)
        (np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8), 2, 2, "Death"),
    ]

    frames = []
    for grid, cr, cc, name in scenarios:
        for phase in range(3):
            hold = 30 if phase < 2 else 45
            for _ in range(hold):
                frames.append(render_kernel_math_frame(grid, cr, cc, phase))

    path = os.path.join(OUT_DIR, "kernel_math.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("  Generating GIFs for GOL-on-ANE paper")
    print("=" * 60)
    print()

    make_gol_conv_gif()
    print()
    make_tiled_gif()
    print()
    make_pipeline_gif()
    print()
    make_chain_gif()
    print()
    make_kernel_math_gif()

    print()
    print("=" * 60)
    print(f"  All GIFs saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
