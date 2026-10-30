#!/usr/bin/env python3
"""Benchmark tile pyramid builders: baseline NFS round-trip vs in-memory CPU/GPU paths.

Compares:

1. **Baseline** — per-level ``Shrink`` (same I/O pattern as ``BuildTilePyramids``)
2. **CPU-new** — :func:`BuildTilePyramidsMemoryCpu` (load once, chain downsamples)
3. **GPU-new** — :func:`BuildTilePyramidsMemoryGpu` (skipped when CuPy unavailable)

Usage
-----
  python bench_build_pyramids.py \\
      --tile-dir /volumes/RPC3/TEM/0601/TEM/Leveled/TilePyramid/001 \\
      --levels 1,2,4,8,16,32 --max-tiles 64

  python bench_build_pyramids.py --tile-dir ... --verify
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image

import nornir_imageregistration
from nornir_imageregistration.computational_lib import ComputationLib
from nornir_imageregistration.core._core import (
    BuildTilePyramidsMemoryCpu,
    BuildTilePyramidsMemoryGpu,
)


def _collect_tile_paths(tile_dir: Path, max_tiles: int | None) -> list[Path]:
    paths = sorted(tile_dir.glob("*.png"))
    if not paths:
        paths = sorted(tile_dir.glob("*.tif")) + sorted(tile_dir.glob("*.tiff"))
    if not paths:
        raise FileNotFoundError(f"No PNG/TIF tiles found in {tile_dir}")
    if max_tiles is not None:
        paths = paths[:max_tiles]
    return paths


def _parse_levels(levels_str: str) -> list[int]:
    return sorted({int(x.strip()) for x in levels_str.split(",") if x.strip()})


def _level_dir(root: Path, level: int) -> Path:
    return root / f"{level:03d}"


def _shrink_factors(levels: list[int]) -> list[float]:
    return [float(levels[i - 1]) / float(levels[i]) for i in range(1, len(levels))]


def _prepare_source_tree(tile_paths: list[Path], out_root: Path, finest_level: int) -> Path:
    src_dir = _level_dir(out_root, finest_level)
    src_dir.mkdir(parents=True, exist_ok=True)
    for p in tile_paths:
        shutil.copy2(p, src_dir / p.name)
    return src_dir


def _run_baseline_shrink(
    tile_paths: list[Path],
    out_root: Path,
    levels: list[int],
    n_workers: int,
) -> float:
    """Per-level-pair Shrink — mirrors BuildTilePyramids NFS pattern."""
    t0 = time.perf_counter()
    for i in range(1, len(levels)):
        up = levels[i - 1]
        down = levels[i]
        factor = float(up) / float(down)
        src_dir = _level_dir(out_root, up)
        dst_dir = _level_dir(out_root, down)
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_paths = [str(src_dir / p.name) for p in tile_paths]
        dst_paths = [str(dst_dir / p.name) for p in tile_paths]

        def _one(src: str, dst: str) -> None:
            nornir_imageregistration.Shrink(src, dst, factor)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_one, s, d) for s, d in zip(src_paths, dst_paths)]
            for f in concurrent.futures.as_completed(futs):
                f.result()
    return time.perf_counter() - t0


def _run_memory_cpu(
    tile_paths: list[Path],
    out_root: Path,
    levels: list[int],
    n_workers: int,
) -> float:
    finest = levels[0]
    src_dir = _level_dir(out_root, finest)
    factors = _shrink_factors(levels)
    tiles: dict[str, list[str | None]] = {}
    for p in tile_paths:
        input_path = str(src_dir / p.name)
        outputs: list[str | None] = []
        for lvl in levels[1:]:
            dst = _level_dir(out_root, lvl)
            dst.mkdir(parents=True, exist_ok=True)
            outputs.append(str(dst / p.name))
        tiles[input_path] = outputs

    t0 = time.perf_counter()
    BuildTilePyramidsMemoryCpu(tiles, factors, num_threads=n_workers)
    return time.perf_counter() - t0


def _run_memory_gpu(
    tile_paths: list[Path],
    out_root: Path,
    levels: list[int],
) -> float | None:
    if not nornir_imageregistration.HasCupy():
        return None
    nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
    nornir_imageregistration.TryInitCupyContext()

    finest = levels[0]
    src_dir = _level_dir(out_root, finest)
    factors = _shrink_factors(levels)
    tiles: dict[str, list[str | None]] = {}
    for p in tile_paths:
        input_path = str(src_dir / p.name)
        outputs: list[str | None] = []
        for lvl in levels[1:]:
            dst = _level_dir(out_root, lvl)
            dst.mkdir(parents=True, exist_ok=True)
            outputs.append(str(dst / p.name))
        tiles[input_path] = outputs

    t0 = time.perf_counter()
    BuildTilePyramidsMemoryGpu(tiles, factors)
    return time.perf_counter() - t0


def _verify_against_baseline(
    baseline_root: Path,
    candidate_root: Path,
    levels: list[int],
    tile_paths: list[Path],
) -> None:
    print("\n[VERIFY] mean absolute error vs baseline (per level):")
    for lvl in levels[1:]:
        errors = []
        for p in tile_paths:
            ref = np.array(Image.open(_level_dir(baseline_root, lvl) / p.name))
            cand = np.array(Image.open(_level_dir(candidate_root, lvl) / p.name))
            if ref.shape != cand.shape:
                print(f"  level {lvl:03d} {p.name}: shape mismatch {ref.shape} vs {cand.shape}")
                continue
            errors.append(float(np.mean(np.abs(ref.astype(np.float32) - cand.astype(np.float32)))))
        if errors:
            print(f"  level {lvl:03d}: mean MAE = {np.mean(errors):.4f} (max {np.max(errors):.4f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark tile pyramid builders")
    parser.add_argument("--tile-dir", required=True, type=Path,
                        help="Finest pyramid level directory (e.g. .../TilePyramid/001)")
    parser.add_argument("--levels", default="1,2,4,8,16,32",
                        help="Comma-separated downsample levels (default: 1,2,4,8,16,32)")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Limit number of tiles processed")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Parent directory for benchmark outputs (default: temp dir)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Repeat each backend (default: 1)")
    parser.add_argument("--verify", action="store_true",
                        help="Compare CPU/GPU outputs to baseline pixel-wise")
    args = parser.parse_args()

    tile_paths = _collect_tile_paths(args.tile_dir, args.max_tiles)
    levels = _parse_levels(args.levels)
    if levels[0] != 1:
        print(f"Note: finest level in --levels is {levels[0]}, not 1")

    n_workers = min(os.cpu_count() or 4, len(tile_paths)) * 2
    n_workers = max(n_workers, 1)

    out_parent = args.output_dir
    if out_parent is None:
        out_parent = Path(tempfile.mkdtemp(prefix="bench_pyramids_"))
        print(f"Using temp output dir: {out_parent}")
    else:
        out_parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[float]] = {}

    for backend in ("baseline", "cpu", "gpu"):
        times: list[float] = []
        for it in range(args.iterations):
            tag = f"{backend}_run{it}"
            run_root = out_parent / tag
            if run_root.exists():
                shutil.rmtree(run_root)
            run_root.mkdir(parents=True)
            _prepare_source_tree(tile_paths, run_root, levels[0])

            if backend == "baseline":
                elapsed = _run_baseline_shrink(tile_paths, run_root, levels, n_workers)
            elif backend == "cpu":
                elapsed = _run_memory_cpu(tile_paths, run_root, levels, n_workers)
            else:
                elapsed = _run_memory_gpu(tile_paths, run_root, levels)
                if elapsed is None:
                    print("[GPU-new] skipped (CuPy unavailable)")
                    break
            times.append(elapsed)
            print(f"[{backend}] iteration {it + 1}/{args.iterations}: {elapsed:.3f}s  "
                  f"({len(tile_paths) / elapsed:.2f} tiles/s)")

        if times:
            results[backend] = times
            print(f"[{backend}] mean: {np.mean(times):.3f}s\n")

    if args.verify and "baseline" in results:
        baseline_dir = out_parent / "baseline_run0"
        if "cpu" in results:
            _verify_against_baseline(
                baseline_dir, out_parent / "cpu_run0", levels, tile_paths)
        if "gpu" in results:
            _verify_against_baseline(
                baseline_dir, out_parent / "gpu_run0", levels, tile_paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
