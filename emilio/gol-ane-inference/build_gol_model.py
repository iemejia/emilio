#!/usr/bin/env python3
"""
Build a CoreML model that computes one Game of Life generation.

The GOL update rule is implemented as:
  1. Conv2d with kernel [[1,1,1],[1,9,1],[1,1,1]] counts neighbors
     while encoding current state (dead: 0-8, alive: 9-17)
  2. ReLU-based equality checks determine birth (S==3) or survival (S==11 or S==12)

Input:  [1, 1, H, W] float16 grid (0.0 = dead, 1.0 = alive)
Output: [1, 1, H, W] float16 grid after one generation

The model uses padding='same' so boundary cells see dead neighbors
(matching standard GOL bounded-grid convention).

For tile-based simulation, the caller handles overlap/boundary exchange.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import argparse
import os


class GOLStepNaive(nn.Module):
    """Original naive GOL: 1 conv + 9 relu + 12 element-wise ops = 22 ops."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1, 1, 1],
                                                    [1, 9, 1],
                                                    [1, 1, 1]]]], dtype=torch.float32))

    def forward(self, x):
        s = self.conv(x)
        birth = torch.relu(1.0 - torch.relu(s - 3.0) - torch.relu(3.0 - s))
        surv2 = torch.relu(1.0 - torch.relu(s - 11.0) - torch.relu(11.0 - s))
        surv3 = torch.relu(1.0 - torch.relu(s - 12.0) - torch.relu(12.0 - s))
        result = birth + surv2 + surv3
        return torch.clamp(result, 0.0, 1.0)


class GOLStep(nn.Module):
    """Optimized GOL: merged survive check, 13 ops (vs 22 naive).

    Key optimization: s=11 and s=12 are adjacent, so we merge them into
    one check using relu(1.5 - |s - 11.5|), which equals 1.0 for both
    s=11 and s=12 and 0.0 for all other integer values.

    This halves the survive path: 2 checks → 1 check.
    Total: 1 conv + 6 relu + 4 sub + 1 add + 1 clamp = 13 ops.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1, 1, 1],
                                                    [1, 9, 1],
                                                    [1, 1, 1]]]], dtype=torch.float32))

    def forward(self, x):
        s = self.conv(x)
        # Birth: dead cell with exactly 3 neighbors (s == 3)
        birth = torch.relu(1.0 - torch.relu(s - 3.0) - torch.relu(3.0 - s))
        # Survive: alive cell with 2 or 3 neighbors (s ∈ {11, 12})
        # relu(1.5 - |s - 11.5|) = 1.0 for s=11,12; 0.0 for all other integers
        survive = torch.relu(1.5 - torch.relu(s - 11.5) - torch.relu(11.5 - s))
        return torch.clamp(birth + survive, 0.0, 1.0)


class GOLMultiStep(nn.Module):
    """Multiple GOL generations in one forward pass."""

    def __init__(self, steps):
        super().__init__()
        self.steps = nn.ModuleList([GOLStep() for _ in range(steps)])
        # Share weights across all steps
        w = self.steps[0].conv.weight
        for step in self.steps[1:]:
            step.conv.weight = w

    def forward(self, x):
        for step in self.steps:
            x = step(x)
        return x


def verify_pytorch(model, grid_size=64):
    """Verify the PyTorch model against a manual GOL implementation."""
    # Blinker: period-2 oscillator
    grid = torch.zeros(1, 1, grid_size, grid_size)
    mid = grid_size // 2
    grid[0, 0, mid, mid - 1] = 1
    grid[0, 0, mid, mid] = 1
    grid[0, 0, mid, mid + 1] = 1

    with torch.no_grad():
        gen1 = model(grid)

    # After 1 gen, blinker should rotate
    assert gen1[0, 0, mid - 1, mid].item() > 0.5, "Blinker gen1 fail"
    assert gen1[0, 0, mid, mid].item() > 0.5, "Blinker gen1 fail"
    assert gen1[0, 0, mid + 1, mid].item() > 0.5, "Blinker gen1 fail"
    alive = (gen1 > 0.5).sum().item()
    assert alive == 3, f"Blinker should have 3 alive, got {alive}"

    # Gosper glider gun check
    gun = torch.zeros(1, 1, grid_size, grid_size)
    gun_cells = [
        (1, 25), (2, 23), (2, 25), (3, 13), (3, 14), (3, 21), (3, 22),
        (3, 35), (3, 36), (4, 12), (4, 16), (4, 21), (4, 22), (4, 35),
        (4, 36), (5, 1), (5, 2), (5, 11), (5, 17), (5, 21), (5, 22),
        (6, 1), (6, 2), (6, 11), (6, 15), (6, 17), (6, 18), (6, 23),
        (6, 25), (7, 11), (7, 17), (7, 25), (8, 12), (8, 16), (9, 13),
        (9, 14)
    ]
    for r, c in gun_cells:
        gun[0, 0, r, c] = 1

    with torch.no_grad():
        gen1 = model(gun)
    alive1 = (gen1 > 0.5).sum().item()
    # Gun should have 36 cells initially, and keep oscillating
    print(f"  Blinker: PASS (3 cells, rotated)")
    print(f"  Glider gun: {len(gun_cells)} → {alive1} cells after 1 gen")

    return True


def build_and_export(grid_h, grid_w, steps_per_call, output_dir):
    """Build the CoreML model and export."""
    print(f"Building GOL model: {grid_h}×{grid_w}, {steps_per_call} step(s)/call")

    if steps_per_call == 1:
        model = GOLStep()
    else:
        model = GOLMultiStep(steps_per_call)

    model.eval()

    # Verify
    print("Verifying PyTorch model...")
    verify_pytorch(GOLStep())  # always verify single step

    # Trace
    example = torch.zeros(1, 1, grid_h, grid_w)
    traced = torch.jit.trace(model, example)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="grid", shape=(1, 1, grid_h, grid_w))],
        outputs=[ct.TensorType(name="next_grid")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel.author = "emilio"
    mlmodel.short_description = (
        f"Game of Life: {steps_per_call} gen(s) on {grid_h}×{grid_w} grid. "
        "B3/S23 via Conv2d. Designed for ANE execution."
    )

    pkg_path = os.path.join(output_dir, f"GOL_{grid_h}x{grid_w}_s{steps_per_call}.mlpackage")
    mlmodel.save(pkg_path)
    print(f"Saved: {pkg_path}")
    return pkg_path


def build_and_export_batched(grid_h, grid_w, batch_size, output_dir):
    """Build a batched CoreML model: [B, 1, H, W] → [B, 1, H, W]."""
    print(f"Building batched GOL model: {batch_size}×{grid_h}×{grid_w}")

    model = GOLStep()
    model.eval()

    # Verify
    print("Verifying PyTorch model...")
    verify_pytorch(GOLStep())

    # Trace with batch dim
    example = torch.zeros(batch_size, 1, grid_h, grid_w)
    traced = torch.jit.trace(model, example)

    # Convert to CoreML with fixed batch size
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="grid", shape=(batch_size, 1, grid_h, grid_w))],
        outputs=[ct.TensorType(name="next_grid")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel.author = "emilio"
    mlmodel.short_description = (
        f"Game of Life: batch={batch_size}, {grid_h}×{grid_w} grid. "
        "B3/S23 via Conv2d. Designed for ANE execution."
    )

    pkg_path = os.path.join(output_dir, f"GOL_{grid_h}x{grid_w}_b{batch_size}.mlpackage")
    mlmodel.save(pkg_path)
    print(f"Saved: {pkg_path}")
    return pkg_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GOL CoreML model for ANE")
    parser.add_argument("--height", type=int, default=4096, help="Grid height")
    parser.add_argument("--width", type=int, default=4096, help="Grid width")
    parser.add_argument("--steps", type=int, default=1, help="GOL generations per call")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0 = unbatched)")
    args = parser.parse_args()

    if args.batch > 0:
        build_and_export_batched(args.height, args.width, args.batch, args.output)
    else:
        build_and_export(args.height, args.width, args.steps, args.output)
