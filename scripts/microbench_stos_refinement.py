#!/usr/bin/env python3
"""Micro-benchmark one Grid8 stos refine with cold vs clean CuPy memory pool."""

from __future__ import annotations

import argparse
import cProfile
import glob
import os
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nornir_imageregistration
from nornir_imageregistration.local_distortion_correction import (
    RefineStosFile,
    _release_refinement_worker_memory,
)


def _warm_gpu_memory() -> None:
    """Allocate and retain CuPy buffers to mimic post-mosaic pool pressure."""
    if not nornir_imageregistration.UsingCupy():
        return
    import cupy as cp

    retained: list = []
    for _ in range(8):
        retained.append(cp.zeros((4096, 4096), dtype=cp.float32))
    globals()['_MICROBENCH_GPU_RETAIN'] = retained


def _find_heavy_stos(volume_dir: Path) -> tuple[Path, Path]:
    """Locate input/output paths for the heaviest Grid8 automatic stos pair."""
    automatic = sorted(volume_dir.glob('TEM/Grid8/Automatic/*.stos'))
    if not automatic:
        raise FileNotFoundError(f'No Grid8 automatic stos files under {volume_dir}')
    preferred = [path for path in automatic if '693' in path.name and '691' in path.name]
    input_stos = preferred[0] if preferred else automatic[0]
    output_stos = input_stos.with_name(f'microbench_{input_stos.name}')
    return input_stos, output_stos


def _run_once(input_stos: Path, output_stos: Path, iterations: int) -> float:
    """Run RefineStosFile once and return wall seconds."""
    if output_stos.exists():
        output_stos.unlink()
    start = time.perf_counter()
    RefineStosFile(
        InputStos=str(input_stos),
        OutputStosPath=str(output_stos),
        num_iterations=iterations,
        cell_size=(128, 128),
        grid_spacing=(96, 96),
        min_unmasked_area=0.24,
    )
    return time.perf_counter() - start


def _profile_once(input_stos: Path, output_stos: Path, iterations: int, profile_path: Path) -> float:
    """Profile one RefineStosFile call."""
    profiler = cProfile.Profile()
    profiler.enable()
    elapsed = _run_once(input_stos, output_stos, iterations)
    profiler.disable()
    profiler.dump_stats(str(profile_path))
    return elapsed


def _print_profile_summary(profile_path: Path) -> None:
    """Print top cumulative stats and AttemptAlignPoint needle."""
    stream = StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(15)
    print(stream.getvalue())
    stream.truncate(0)
    stream.seek(0)
    stats.print_stats('AttemptAlignPoint')
    print(stream.getvalue())


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume-dir', type=Path,
                        default=Path(os.environ.get('TESTOUTPUTPATH', '/tmp/nornir-test-output')) / 'TestIDocBuild',
                        help='Built TestIDocBuild volume directory containing Grid8 stos inputs')
    parser.add_argument('--iterations', type=int, default=3, help='RefineStosFile num_iterations')
    parser.add_argument('--repeats', type=int, default=3, help='Repeats per memory mode')
    parser.add_argument('--profile', action='store_true', help='Write cProfile for the first clean run')
    args = parser.parse_args()

    input_stos, output_stos = _find_heavy_stos(args.volume_dir)
    print(f'Input stos: {input_stos}')
    print(f'UsingCupy: {nornir_imageregistration.UsingCupy()}')

    results: dict[str, list[float]] = {'clean': [], 'cold': []}
    for mode in ('clean', 'cold'):
        for repeat in range(args.repeats):
            _release_refinement_worker_memory()
            if mode == 'cold':
                _warm_gpu_memory()
            else:
                globals().pop('_MICROBENCH_GPU_RETAIN', None)
                _release_refinement_worker_memory()
            out_path = output_stos.with_name(f'{mode}_{repeat}_{output_stos.name}')
            if args.profile and mode == 'clean' and repeat == 0:
                profile_path = args.volume_dir / f'microbench_{mode}.profile'
                elapsed = _profile_once(input_stos, out_path, args.iterations, profile_path)
                _print_profile_summary(profile_path)
            else:
                elapsed = _run_once(input_stos, out_path, args.iterations)
            results[mode].append(elapsed)
            print(f'{mode} repeat {repeat + 1}: {elapsed:.2f}s')

    clean_mean = sum(results['clean']) / len(results['clean'])
    cold_mean = sum(results['cold']) / len(results['cold'])
    print(f'clean mean: {clean_mean:.2f}s')
    print(f'cold mean: {cold_mean:.2f}s')
    print(f'cold - clean: {cold_mean - clean_mean:+.2f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
