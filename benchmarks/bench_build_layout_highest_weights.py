"""Micro-benchmark for BuildLayoutWithHighestWeightsFirst (category 19)."""

from __future__ import annotations

import os

import numpy as np
import pyperf

from nornir_imageregistration.layout import BuildLayoutWithHighestWeightsFirst, Layout


def _synthetic_section_layout(num_tiles: int, rng_seed: int = 0) -> Layout:
    """Grid-like section with many cross-tile offsets (typical arrange-mosaic input)."""
    rng = np.random.default_rng(rng_seed)
    layout = Layout()
    side = int(np.ceil(np.sqrt(num_tiles)))
    pitch = 512.0
    for tile_id in range(num_tiles):
        row, col = divmod(tile_id, side)
        layout.CreateNode(tile_id, np.array([row * pitch, col * pitch], dtype=np.float64))
    for tile_id in range(num_tiles):
        row, col = divmod(tile_id, side)
        for dr, dc in ((0, 1), (1, 0), (1, 1)):
            nr, nc = row + dr, col + dc
            neighbor = nr * side + nc
            if neighbor >= num_tiles:
                continue
            offset = np.array([dr * pitch, dc * pitch], dtype=np.float64)
            weight = float(rng.uniform(0.3, 1.0))
            a_id, b_id = (tile_id, neighbor) if tile_id < neighbor else (neighbor, tile_id)
            if not layout.ContainsOffset((a_id, b_id)):
                layout.SetOffset(a_id, b_id, offset if tile_id < neighbor else -offset, weight)
    return layout


def main() -> None:
    num_tiles = int(os.environ.get("NORNIR_BENCH_NUM_TILES", "4096"))
    source = _synthetic_section_layout(num_tiles)

    def worker() -> None:
        BuildLayoutWithHighestWeightsFirst(source)

    runner = pyperf.Runner()
    runner.bench_func(f"build_layout_highest_weights_{num_tiles}", worker)


if __name__ == "__main__":
    main()
