#!/usr/bin/env python3
"""CPU vs GPU wall-time benchmark for contrast conversion and pyramid building.

Benchmarks two workflows:

1. **Contrast-only** (original hot path of AdjustContrast):
   - CPU:  ConvertImagesInDict
   - GPU:  ConvertImagesInDictGpu  (batch-size sweep)

2. **Contrast + full tile pyramid** (new merged path vs original two-stage):
   - Baseline CPU two-stage:   ConvertImagesInDict  → per-level Pillow LANCZOS shrink
   - Baseline GPU two-stage:   ConvertImagesInDictGpu → per-level Pillow LANCZOS shrink
   - Merged CPU:               ConvertImagesInDictPyramid
   - Merged GPU:               ConvertImagesInDictGpuPyramid  (batch-size sweep)

Pyramid benchmarks are enabled by ``--pyramid``.  The levels to build are set
with ``--levels`` (default ``1,2,4,8,16``).

Usage
-----
  # Contrast-only sweep (previous benchmark)
  python bench_adjust_contrast.py --tile-dir /volumes/RPC3/TEM/0001/TEM/Raw8/TilePyramid/001 \\
      --max-tiles 32 --batch-mb 64,128,256,512

  # Full pyramid benchmark at default levels 1,2,4,8,16
  python bench_adjust_contrast.py --tile-dir /volumes/RPC3/TEM/0001/TEM/Raw8/TilePyramid/001 \\
      --max-tiles 32 --pyramid --levels 1,2,4,8,16 --batch-mb 64

Flags
-----
--tile-dir      (required) Directory of input PNG tiles (e.g. TilePyramid/001).
--max-tiles     Limit number of tiles to process (default: all in directory).
--min-cutoff    Intensity cutoff as a fraction 0-1 (default: 0.0 = 0th percentile).
--max-cutoff    Intensity cutoff as a fraction 0-1 (default: 1.0 = 100th percentile).
--iterations    Number of benchmark repetitions per backend (default: 3).
--backends      cpu, gpu, or both (default: both).
--batch-mb      Comma-separated list of GPU batch sizes in MB to sweep
                (default: 64).  Each value is tested independently.
--pyramid       Also benchmark the merged contrast+pyramid path.
--levels        Comma-separated pyramid downsample levels when --pyramid is set
                (default: 1,2,4,8,16).  Level 1 is always the contrast output.
--profile       Dump a cProfile .profile file for each backend.
--output        Write StageTimings-compatible JSON to this file (default: stdout only).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import cProfile
import json
import multiprocessing
import os
import pstats
import shutil
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path

# Force 'fork' start method so multiprocessing workers inherit the parent's
# active computation library (ComputationLib.numpy / ComputationLib.cupy) set
# below.  The default 'forkserver' on Python 3.14 spawns a fresh server process
# that re-initialises the CUDA context; workers forked from that server inherit
# a broken context and crash with cudaErrorInitializationError.
# force=True allows re-setting when called after import of multiprocessing.
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # context already started; continue with existing method

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nornir_imageregistration
from nornir_imageregistration.computational_lib import ComputationLib
from nornir_imageregistration.core._core import (
    ConvertImagesInDict,
    ConvertImagesInDictGpu,
    ConvertImagesInDictPyramid,
    ConvertImagesInDictGpuPyramid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_tile_paths(tile_dir: Path, max_tiles: int | None) -> list[Path]:
    paths = sorted(tile_dir.glob('*.png'))
    if not paths:
        paths = sorted(tile_dir.glob('*.tif')) + sorted(tile_dir.glob('*.tiff'))
    if not paths:
        raise FileNotFoundError(f"No PNG/TIF tiles found in {tile_dir}")
    if max_tiles is not None:
        paths = paths[:max_tiles]
    return paths


def _build_io_dict(tile_paths: list[Path], out_dir: Path) -> dict[str, str]:
    return {str(p): str(out_dir / p.name) for p in tile_paths}


def _build_pyramid_dirs(tmp: Path, tag: str, levels: list[int]) -> list[Path]:
    """Create and return one output directory per pyramid level."""
    dirs = []
    for lvl in levels:
        d = tmp / f"{tag}_l{lvl:03d}"
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs


def _build_pyramid_io(tile_paths: list[Path],
                       level_dirs: list[Path]) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Return (io_l1, pyramid_output_dicts) keyed by original input path.

    ``io_l1``            maps input_path  → level-1 output path
    ``pyramid_output_dicts[i]`` maps input_path → level-(i+2) output path
    """
    io_l1 = {str(p): str(level_dirs[0] / p.name) for p in tile_paths}
    pyramid_dicts = [
        {str(p): str(d / p.name) for p in tile_paths}
        for d in level_dirs[1:]
    ]
    return io_l1, pyramid_dicts


def _pillow_shrink_level(src_paths: list[str], dst_paths: list[str],
                          n_workers: int) -> None:
    """Shrink a set of tiles by 0.5× using Pillow LANCZOS — simulates BuildTilePyramids."""
    def _shrink_one(src: str, dst: str) -> None:
        nornir_imageregistration.Shrink(src, dst, 0.5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_shrink_one, s, d) for s, d in zip(src_paths, dst_paths)]
        for f in concurrent.futures.as_completed(futures):
            f.result()


def _run_cpu(io_dict: dict[str, str], min_max: tuple[float, float], gamma: float) -> float:
    t = time.perf_counter()
    ConvertImagesInDict(io_dict, MinMax=min_max, Gamma=gamma)
    return time.perf_counter() - t


def _run_gpu(io_dict: dict[str, str], min_max: tuple[float, float], gamma: float,
             batch_bytes: int = 64 * 1024 * 1024) -> float:
    t = time.perf_counter()
    ConvertImagesInDictGpu(io_dict, MinMax=min_max, Gamma=gamma, batch_bytes=batch_bytes)
    return time.perf_counter() - t


def _run_baseline_two_stage(
    contrast_fn,
    io_l1: dict[str, str],
    level_dirs: list[Path],
    tile_paths: list[Path],
    min_max: tuple[float, float],
    gamma: float,
    n_workers: int,
    contrast_kwargs: dict | None = None,
) -> float:
    """Time the original two-stage path: contrast then per-level Pillow LANCZOS shrink.

    This mirrors what AdjustContrastGpu + BuildTilePyramids does in production:
    - Stage 1: write level-1 tiles (contrast adjust via ``contrast_fn``).
    - Stage 2: for each coarser level, read the previous level's files and
      write the next level via Pillow LANCZOS resize (same algorithm as Shrink).

    The per-level barrier (all N tiles at level k done before starting level k+1)
    is preserved, matching BuildTilePyramids' outer level loop.
    """
    t = time.perf_counter()

    # Stage 1: contrast
    kw = contrast_kwargs or {}
    contrast_fn(io_l1, MinMax=min_max, Gamma=gamma, **kw)

    # Stage 2: pyramid — level by level, each level reads from the previous level's output
    tile_names = [p.name for p in tile_paths]
    prev_dir = level_dirs[0]
    for next_dir in level_dirs[1:]:
        src_paths = [str(prev_dir / name) for name in tile_names]
        dst_paths = [str(next_dir / name) for name in tile_names]
        _pillow_shrink_level(src_paths, dst_paths, n_workers)
        prev_dir = next_dir

    return time.perf_counter() - t


def _run_merged_cpu(
    io_l1: dict[str, str],
    pyramid_output_dicts: list[dict[str, str]],
    min_max: tuple[float, float],
    gamma: float,
) -> float:
    t = time.perf_counter()
    ConvertImagesInDictPyramid(io_l1, pyramid_output_dicts, MinMax=min_max, Gamma=gamma)
    return time.perf_counter() - t


def _run_merged_gpu(
    io_l1: dict[str, str],
    pyramid_output_dicts: list[dict[str, str]],
    min_max: tuple[float, float],
    gamma: float,
    batch_bytes: int = 64 * 1024 * 1024,
) -> float:
    t = time.perf_counter()
    ConvertImagesInDictGpuPyramid(io_l1, pyramid_output_dicts,
                                   MinMax=min_max, Gamma=gamma, batch_bytes=batch_bytes)
    return time.perf_counter() - t


def _profile_run(fn, io_dict, min_max, gamma, profile_path: Path) -> float:
    profiler = cProfile.Profile()
    profiler.enable()
    elapsed = fn(io_dict, min_max, gamma)
    profiler.disable()
    profiler.dump_stats(str(profile_path))
    stream = StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    print(stream.getvalue())
    return elapsed


# ---------------------------------------------------------------------------
# Table + JSON
# ---------------------------------------------------------------------------

def _format_table(results: list[tuple[str, int | None, list[float]]], tile_count: int) -> str:
    """Format a results table.

    results: list of (label, batch_mb_or_None, times)
    """
    hdr = (f"\n{'Backend':<12} {'batch_MB':>8} {'chunk_tiles':>11} "
           f"{'N':>3} {'Min (s)':>8} {'Mean (s)':>9} {'tiles/s':>8} {'Speedup':>8}")
    lines = [hdr, "-" * 72]
    cpu_mean: float | None = None
    for label, batch_mb, times in results:
        if not times:
            continue
        t_min = min(times)
        t_mean = sum(times) / len(times)
        tps = tile_count / t_mean
        speedup_str = ""
        batch_str = f"{batch_mb:>8}" if batch_mb is not None else f"{'--':>8}"
        chunk_str = "        --"
        if label == 'cpu':
            cpu_mean = t_mean
        elif cpu_mean is not None:
            speedup_str = f"{cpu_mean / t_mean:.2f}x"
        lines.append(
            f"{label:<12} {batch_str} {chunk_str} "
            f"{len(times):>3} {t_min:>8.3f} {t_mean:>9.3f} {tps:>8.1f} {speedup_str:>8}"
        )
    return "\n".join(lines)


def _format_sweep_table(results: list[tuple[str, int | None, list[float]]],
                        tile_count: int, tile_float32_bytes: int) -> str:
    """Format sweep results with chunk_tiles column filled from batch_mb."""
    hdr = (f"\n{'Backend':<12} {'batch_MB':>8} {'chunk_tiles':>11} "
           f"{'N':>3} {'Min (s)':>8} {'Mean (s)':>9} {'tiles/s':>8} {'Speedup':>8}")
    lines = [hdr, "-" * 72]
    cpu_mean: float | None = None
    for label, batch_mb, times in results:
        if not times:
            continue
        t_min = min(times)
        t_mean = sum(times) / len(times)
        tps = tile_count / t_mean
        speedup_str = ""
        if label == 'cpu':
            cpu_mean = t_mean
            batch_str = f"{'--':>8}"
            chunk_str = f"{'--':>11}"
        else:
            batch_bytes_val = (batch_mb or 64) * 1024 * 1024
            chunk_tiles = max(1, batch_bytes_val // tile_float32_bytes)
            batch_str = f"{batch_mb:>8}"
            chunk_str = f"{chunk_tiles:>11}"
            if cpu_mean is not None:
                speedup_str = f"{cpu_mean / t_mean:.2f}x"
        lines.append(
            f"{label:<12} {batch_str} {chunk_str} "
            f"{len(times):>3} {t_min:>8.3f} {t_mean:>9.3f} {tps:>8.1f} {speedup_str:>8}"
        )
    return "\n".join(lines)


def _format_pyramid_table(results: list[tuple[str, int | None, list[float]]],
                           tile_count: int, n_levels: int) -> str:
    """Format pyramid benchmark results.  baseline_cpu is the reference speedup."""
    hdr = (f"\n{'Backend':<22} {'batch_MB':>8} {'N':>3} "
           f"{'Min (s)':>8} {'Mean (s)':>9} {'tiles/s':>8} {'Speedup':>8}")
    lines = [hdr, "-" * 72]
    baseline_mean: float | None = None
    for label, batch_mb, times in results:
        if not times:
            continue
        t_min = min(times)
        t_mean = sum(times) / len(times)
        # tiles/s = total tile saves across all pyramid levels per second
        tps = (tile_count * n_levels) / t_mean
        speedup_str = ""
        batch_str = f"{batch_mb:>8}" if batch_mb is not None else f"{'--':>8}"
        if label == 'baseline_cpu':
            baseline_mean = t_mean
        elif baseline_mean is not None:
            speedup_str = f"{baseline_mean / t_mean:.2f}x"
        lines.append(
            f"{label:<22} {batch_str} {len(times):>3} "
            f"{t_min:>8.3f} {t_mean:>9.3f} {tps:>8.1f} {speedup_str:>8}"
        )
    return "\n".join(lines)


def _to_json(results: list[tuple[str, int | None, list[float]]],
             tile_count: int, tile_dir: str) -> list[dict]:
    blocks = []
    for label, batch_mb, times in results:
        if not times:
            continue
        pipeline = 'AdjustContrast' if label == 'cpu' else 'AdjustContrastGpu'
        blocks.append({
            "pipeline": pipeline,
            "backend": label,
            "batch_mb": batch_mb,
            "tile_count": tile_count,
            "tile_dir": tile_dir,
            "mean_seconds": sum(times) / len(times),
            "stages": [{"stage": f"iter_{i+1}", "seconds": t} for i, t in enumerate(times)],
        })
    return blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tile-dir', required=True, type=Path,
                   help='Directory of input PNG tiles.')
    p.add_argument('--max-tiles', type=int, default=None,
                   help='Max number of tiles to include (default: all).')
    p.add_argument('--min-cutoff', type=float, default=0.0,
                   help='Intensity min cutoff fraction 0-1 (default: 0.0).')
    p.add_argument('--max-cutoff', type=float, default=1.0,
                   help='Intensity max cutoff fraction 0-1 (default: 1.0).')
    p.add_argument('--gamma', type=float, default=1.0,
                   help='Gamma correction (default: 1.0 = no correction).')
    p.add_argument('--iterations', type=int, default=3,
                   help='Repetitions per backend (default: 3).')
    p.add_argument('--backends', choices=['cpu', 'gpu', 'both'], default='both')
    p.add_argument('--batch-mb', type=str, default='64,128,256,512',
                   help='Comma-separated GPU batch sizes in MB to sweep (default: 64). '
                        'Each value is tested independently. Example: 64,128,256,512')
    p.add_argument('--pyramid', action='store_true',
                   help='Also benchmark the merged contrast+pyramid path vs the original '
                        'two-stage path (ConvertImagesInDict[Gpu] + BuildTilePyramids).')
    p.add_argument('--levels', type=str, default='1,2,4,8,16',
                   help='Comma-separated pyramid downsample levels for --pyramid '
                        '(default: 1,2,4,8,16).  Level 1 is always the contrast output.')
    p.add_argument('--profile', action='store_true',
                   help='Emit a cProfile .profile file per backend.')
    p.add_argument('--output', type=Path, default=None,
                   help='Write JSON results to this file (default: stdout).')
    return p.parse_args(argv)


def _probe_tile_float32_bytes(tile_paths: list[Path]) -> int:
    """Return float32 byte size of one tile by loading the first."""
    from nornir_imageregistration.core._core import _LoadImageByExtension
    arr = _LoadImageByExtension(str(tile_paths[0]), None)
    if arr is None:
        return 4 * 1024 * 1024  # fallback: 4 MB
    return int(arr.size) * 4


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    tile_dir: Path = args.tile_dir.resolve()
    tile_paths = _collect_tile_paths(tile_dir, args.max_tiles)
    tile_count = len(tile_paths)
    min_max = (args.min_cutoff, args.max_cutoff)
    gamma = args.gamma if args.gamma != 1.0 else None

    # Parse batch-mb sweep list
    batch_mb_list: list[int] = []
    for part in args.batch_mb.split(','):
        part = part.strip()
        if part:
            batch_mb_list.append(int(part))
    if not batch_mb_list:
        batch_mb_list = [64]

    # Parse pyramid levels
    pyramid_levels: list[int] = []
    for part in args.levels.split(','):
        part = part.strip()
        if part:
            pyramid_levels.append(int(part))
    pyramid_levels = sorted(set(pyramid_levels))
    if not pyramid_levels or pyramid_levels[0] != 1:
        pyramid_levels = [1] + [lvl for lvl in pyramid_levels if lvl > 1]
    n_pyramid_levels = len(pyramid_levels)  # includes level-1

    tile_float32_bytes = _probe_tile_float32_bytes(tile_paths)
    tile_mb = tile_float32_bytes / (1024 * 1024)
    n_workers = min(multiprocessing.cpu_count() * 2, tile_count + 1)

    print(f"Tile directory  : {tile_dir}")
    print(f"Tiles           : {tile_count}")
    print(f"Tile float32    : {tile_mb:.1f} MB  ({tile_float32_bytes // (1024*1024)} MiB)")
    print(f"MinMax          : {min_max}  Gamma: {gamma}")
    print(f"Batch sweep (MB): {batch_mb_list}")
    if args.pyramid:
        print(f"Pyramid levels  : {pyramid_levels}  ({n_pyramid_levels} total)")

    # results: list of (label, batch_mb_or_None, times)
    results: list[tuple[str, int | None, list[float]]] = []

    with tempfile.TemporaryDirectory(prefix='bench_ac_') as tmp_root:
        tmp = Path(tmp_root)

        run_cpu = args.backends in ('cpu', 'both')
        run_gpu = args.backends in ('gpu', 'both')

        # ---- CPU baseline (contrast only) ----
        if run_cpu:
            nornir_imageregistration.SetActiveComputationLib(ComputationLib.numpy)
            print(f"\n[CPU] ConvertImagesInDict  x{args.iterations} ...")
            cpu_times: list[float] = []
            for i in range(args.iterations):
                out_dir = tmp / f"cpu_{i}"
                out_dir.mkdir()
                io = _build_io_dict(tile_paths, out_dir)
                profile_path = (tmp / f"cpu_{i}.profile") if args.profile else None
                if profile_path:
                    elapsed = _profile_run(_run_cpu, io, min_max, gamma, profile_path)
                else:
                    elapsed = _run_cpu(io, min_max, gamma)
                cpu_times.append(elapsed)
                print(f"  iter {i+1}: {elapsed:.3f}s  ({tile_count/elapsed:.1f} tiles/s)")
                shutil.rmtree(out_dir)
            results.append(('cpu', None, cpu_times))

        # ---- GPU sweep over batch sizes (contrast only) ----
        if run_gpu:
            if not nornir_imageregistration.HasCupy():
                print("\n[GPU] CuPy not available — skipping.")
            else:
                nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
                for batch_mb in batch_mb_list:
                    batch_bytes = batch_mb * 1024 * 1024
                    chunk_tiles = max(1, batch_bytes // tile_float32_bytes)
                    label = f"gpu_{batch_mb}MB"
                    print(f"\n[GPU] batch={batch_mb} MB  chunk={chunk_tiles} tiles  "
                          f"x{args.iterations} ...")
                    gpu_times: list[float] = []
                    for i in range(args.iterations):
                        out_dir = tmp / f"gpu_{batch_mb}_{i}"
                        out_dir.mkdir()
                        io = _build_io_dict(tile_paths, out_dir)
                        elapsed = _run_gpu(io, min_max, gamma, batch_bytes=batch_bytes)
                        gpu_times.append(elapsed)
                        print(f"  iter {i+1}: {elapsed:.3f}s  ({tile_count/elapsed:.1f} tiles/s)")
                        shutil.rmtree(out_dir)
                    results.append((label, batch_mb, gpu_times))

    print(_format_sweep_table(results, tile_count, tile_float32_bytes))

    # ---- Pyramid benchmarks ----
    if args.pyramid:
        print(f"\n{'='*72}")
        print(f"PYRAMID BENCHMARK  levels={pyramid_levels}  tiles={tile_count}")
        print(f"{'='*72}")
        print("Baseline = original two-stage: contrast → per-level Pillow LANCZOS shrink")
        print("Merged   = new single-pass: contrast + GPU downsample chain\n")

        # (label, batch_mb, times) for pyramid results
        pyr_results: list[tuple[str, int | None, list[float]]] = []

        with tempfile.TemporaryDirectory(prefix='bench_pyr_') as pyr_root:
            pyr_tmp = Path(pyr_root)

            # ---- Baseline CPU two-stage (ConvertImagesInDict + Pillow pyramid) ----
            if run_cpu:
                nornir_imageregistration.SetActiveComputationLib(ComputationLib.numpy)
                print(f"[BASELINE CPU] two-stage  x{args.iterations} ...")
                baseline_cpu_times: list[float] = []
                for i in range(args.iterations):
                    level_dirs = _build_pyramid_dirs(pyr_tmp, f"base_cpu_{i}", pyramid_levels)
                    io_l1, _ = _build_pyramid_io(tile_paths, level_dirs)
                    elapsed = _run_baseline_two_stage(
                        ConvertImagesInDict, io_l1, level_dirs, tile_paths,
                        min_max, gamma, n_workers)
                    baseline_cpu_times.append(elapsed)
                    print(f"  iter {i+1}: {elapsed:.3f}s")
                    for d in level_dirs:
                        shutil.rmtree(d, ignore_errors=True)
                pyr_results.append(('baseline_cpu', None, baseline_cpu_times))

            # ---- Baseline GPU two-stage (ConvertImagesInDictGpu + Pillow pyramid) ----
            if run_gpu and nornir_imageregistration.HasCupy():
                nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
                for batch_mb in batch_mb_list:
                    batch_bytes = batch_mb * 1024 * 1024
                    label = f"baseline_gpu_{batch_mb}MB"
                    print(f"\n[BASELINE GPU] two-stage  batch={batch_mb} MB  x{args.iterations} ...")
                    baseline_gpu_times: list[float] = []
                    for i in range(args.iterations):
                        level_dirs = _build_pyramid_dirs(pyr_tmp, f"base_gpu{batch_mb}_{i}", pyramid_levels)
                        io_l1, _ = _build_pyramid_io(tile_paths, level_dirs)
                        elapsed = _run_baseline_two_stage(
                            ConvertImagesInDictGpu, io_l1, level_dirs, tile_paths,
                            min_max, gamma, n_workers,
                            contrast_kwargs={'batch_bytes': batch_bytes})
                        baseline_gpu_times.append(elapsed)
                        print(f"  iter {i+1}: {elapsed:.3f}s")
                        for d in level_dirs:
                            shutil.rmtree(d, ignore_errors=True)
                    pyr_results.append((label, batch_mb, baseline_gpu_times))

            # ---- Merged CPU (ConvertImagesInDictPyramid) ----
            if run_cpu:
                nornir_imageregistration.SetActiveComputationLib(ComputationLib.numpy)
                print(f"\n[MERGED CPU] ConvertImagesInDictPyramid  x{args.iterations} ...")
                merged_cpu_times: list[float] = []
                for i in range(args.iterations):
                    level_dirs = _build_pyramid_dirs(pyr_tmp, f"merge_cpu_{i}", pyramid_levels)
                    io_l1, pyramid_output_dicts = _build_pyramid_io(tile_paths, level_dirs)
                    elapsed = _run_merged_cpu(io_l1, pyramid_output_dicts, min_max, gamma)
                    merged_cpu_times.append(elapsed)
                    print(f"  iter {i+1}: {elapsed:.3f}s")
                    for d in level_dirs:
                        shutil.rmtree(d, ignore_errors=True)
                pyr_results.append(('merged_cpu', None, merged_cpu_times))

            # ---- Merged GPU (ConvertImagesInDictGpuPyramid) ----
            if run_gpu and nornir_imageregistration.HasCupy():
                nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
                for batch_mb in batch_mb_list:
                    batch_bytes = batch_mb * 1024 * 1024
                    label = f"merged_gpu_{batch_mb}MB"
                    print(f"\n[MERGED GPU] ConvertImagesInDictGpuPyramid  "
                          f"batch={batch_mb} MB  x{args.iterations} ...")
                    merged_gpu_times: list[float] = []
                    for i in range(args.iterations):
                        level_dirs = _build_pyramid_dirs(pyr_tmp, f"merge_gpu{batch_mb}_{i}", pyramid_levels)
                        io_l1, pyramid_output_dicts = _build_pyramid_io(tile_paths, level_dirs)
                        elapsed = _run_merged_gpu(io_l1, pyramid_output_dicts, min_max, gamma,
                                                   batch_bytes=batch_bytes)
                        merged_gpu_times.append(elapsed)
                        print(f"  iter {i+1}: {elapsed:.3f}s")
                        for d in level_dirs:
                            shutil.rmtree(d, ignore_errors=True)
                    pyr_results.append((label, batch_mb, merged_gpu_times))

        print(_format_pyramid_table(pyr_results, tile_count, n_pyramid_levels))

    # ---- JSON output ----
    json_data = _to_json(results, tile_count, str(tile_dir))
    json_str = json.dumps(json_data, indent=2)
    if args.output:
        args.output.write_text(json_str, encoding='utf-8')
        print(f"\nJSON written to: {args.output}")
    else:
        print("\nJSON:\n" + json_str)

    return 0


if __name__ == '__main__':
    sys.exit(main())
