#!/usr/bin/env python3
"""Unified micro-benchmark for mosaic and STOS grid refinement.

Usage:
  python scripts/microbench_grid_refine.py --mode mosaic --backend both
  python scripts/microbench_grid_refine.py --mode stos --volume-dir /path/to/TestIDocBuild

This harness replaces the separate ``microbench_mosaic_refine.py`` and
``microbench_stos_refinement.py`` entry points while keeping their workflows.
Legacy scripts remain as thin wrappers that forward to this module.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _enable_flags_from_argv() -> None:
    """Pin env flags before nornir imports when requested on the CLI."""
    if '--phase-timing' in sys.argv or '--phase_timing' in sys.argv:
        os.environ['NORNIR_REFINE_PHASE_TIMING'] = '1'
    if '--batched' in sys.argv:
        os.environ['NORNIR_REFINE_BATCHED'] = '1'
    elif '--no-batched' in sys.argv:
        os.environ['NORNIR_REFINE_BATCHED'] = '0'
    if '--mosaic-cutoff' in sys.argv:
        os.environ['NORNIR_REFINE_MOSAIC_CUTOFF'] = '1'
    if '--stos-regularize' in sys.argv:
        os.environ['NORNIR_REFINE_STOS_REGULARIZE'] = '1'


_enable_flags_from_argv()

import nornir_imageregistration  # noqa: E402
from nornir_imageregistration.computational_lib import ComputationLib  # noqa: E402
from nornir_imageregistration.local_distortion_correction import (  # noqa: E402
    MosaicRefinementDiagnostics,
    RefineStosFile,
    _release_refinement_worker_memory,
)

_GRID690_DATASET = 'RC2_4Square_Assembled'
_GRID690_TRANSLATED = 'Translated_Prune_Max0.5.mosaic'
_GRID690_TILE_SUBDIR = ('Leveled', 'TilePyramid', '004')


def _artifact_dir(mode: str) -> Path:
    """Return a writable directory for profile/JSON artifacts."""
    for env_name in ('TEST_OUTPUT_DIR', 'TESTOUTPUTPATH'):
        value = os.environ.get(env_name, '').strip()
        if value:
            out = Path(value) / f'microbench_grid_refine_{mode}'
            try:
                out.mkdir(parents=True, exist_ok=True)
                return out
            except OSError:
                continue
    out = Path(tempfile.gettempdir()) / f'microbench_grid_refine_{mode}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _parse_pair(text: str, name: str) -> tuple[int, int]:
    """Parse 'N', 'NxM', or 'N,M' into an integer pair."""
    parts = text.replace('x', ',').replace('X', ',').split(',')
    values = [int(p.strip()) for p in parts if p.strip() != '']
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError(f'{name} must be N, NxM, or N,M')


def _set_backend(backend: str) -> bool:
    """Activate the requested backend; return False if cupy was requested but absent."""
    if backend == 'cupy':
        if not nornir_imageregistration.HasCupy():
            return False
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
        nornir_imageregistration.TryInitCupyContext()
        return True
    nornir_imageregistration.SetActiveComputationLib(ComputationLib.numpy)
    return True


def _discover_grid690() -> tuple[Path, Path] | None:
    """Return (translated mosaic, tile dir) for the Grid690 fixture if usable."""
    testinput = os.environ.get('TESTINPUTPATH', '').strip()
    candidates: list[Path] = []
    if testinput:
        candidates.append(
            Path(testinput) / 'PlatformRaw' / 'IDOC' / _GRID690_DATASET / 'TEM' / '0690' / 'TEM')
    tests_fixtures = _REPO_ROOT / 'tests' / 'fixtures'
    candidates.append(tests_fixtures / 'RC2_4Square_Assembled_Grid690' / 'TEM' / '0690' / 'TEM')
    for root in candidates:
        mosaic = root / _GRID690_TRANSLATED
        tile_dir = root.joinpath(*_GRID690_TILE_SUBDIR)
        if mosaic.is_file() and tile_dir.is_dir():
            return mosaic, tile_dir
    return None


def _find_heavy_stos(volume_dir: Path) -> tuple[Path, Path]:
    """Locate input/output paths for a Grid8 automatic stos pair."""
    automatic = sorted(volume_dir.glob('TEM/Grid8/Automatic/*.stos'))
    if not automatic:
        raise FileNotFoundError(f'No Grid8 automatic stos files under {volume_dir}')
    preferred = [path for path in automatic if '693' in path.name and '691' in path.name]
    input_stos = preferred[0] if preferred else automatic[0]
    output_stos = input_stos.with_name(f'microbench_{input_stos.name}')
    return input_stos, output_stos


def _run_mosaic_once(args: argparse.Namespace, mosaic: Path, tile_dir: Path
                     ) -> tuple[float, MosaicRefinementDiagnostics]:
    """Run one RefineGridMosaic and return (wall seconds, diagnostics)."""
    _release_refinement_worker_memory()
    start = time.perf_counter()
    _, diagnostics = nornir_imageregistration.RefineGridMosaic(
        str(mosaic),
        str(tile_dir),
        iterations=args.iterations,
        cell_size=args.cell_size,
        mesh_shape=args.mesh_shape,
        displacement_threshold=args.displacement_threshold,
        min_overlap=args.min_overlap,
        imageScale=args.image_scale,
        return_diagnostics=True)
    return time.perf_counter() - start, diagnostics


def _run_stos_once(input_stos: Path, output_stos: Path, iterations: int) -> float:
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


def _print_profile_summary(profile_path: Path, needles: tuple[str, ...]) -> None:
    """Print top cumulative stats and selected needles."""
    stream = StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    print(stream.getvalue())
    for needle in needles:
        stream.truncate(0)
        stream.seek(0)
        stats.print_stats(needle)
        print(stream.getvalue())


def _run_mosaic(args: argparse.Namespace) -> int:
    """Execute mosaic microbench mode."""
    discovered = _discover_grid690()
    mosaic = Path(args.mosaic) if args.mosaic else (discovered[0] if discovered else None)
    tile_dir = Path(args.tiles_dir) if args.tiles_dir else (discovered[1] if discovered else None)
    if mosaic is None or tile_dir is None:
        print('Could not locate mosaic + tiles-dir; pass --mosaic and --tiles-dir')
        return 2

    backends = ['numpy', 'cupy'] if args.backend == 'both' else [args.backend]
    print(f'Mosaic: {mosaic}')
    print(f'Tiles:  {tile_dir}')
    for backend in backends:
        if not _set_backend(backend):
            print(f'Skipping {backend}: CuPy unavailable')
            continue
        print(f'=== backend={backend} UsingCupy={nornir_imageregistration.UsingCupy()} ===')
        times: list[float] = []
        for repeat in range(args.repeats):
            if args.profile and repeat == 0:
                profile_path = _artifact_dir('mosaic') / f'mosaic_{backend}.profile'
                profiler = cProfile.Profile()
                profiler.enable()
                elapsed, diagnostics = _run_mosaic_once(args, mosaic, tile_dir)
                profiler.disable()
                profiler.dump_stats(str(profile_path))
                _print_profile_summary(
                    profile_path,
                    ('_phase_correlate_refinement_cell', 'find_offset', 'AttemptAlignPoint'))
            else:
                elapsed, diagnostics = _run_mosaic_once(args, mosaic, tile_dir)
            times.append(elapsed)
            measured = sum(
                int(tile_diag.get('measured', 0))
                for pass_diag in diagnostics.vertex_diagnostics_per_pass
                for tile_diag in pass_diag.values())
            rate = measured / elapsed if elapsed > 0 else 0.0
            print(f'  repeat {repeat + 1}: {elapsed:.2f}s  passes={diagnostics.iterations_completed}  '
                  f'cells={measured}  cells/s={rate:.1f}  converged={diagnostics.converged}')
        print(f'  best wall: {min(times):.2f}s')
    return 0


def _run_stos(args: argparse.Namespace) -> int:
    """Execute STOS microbench mode."""
    volume_dir = Path(args.volume_dir)
    input_stos, output_stos = _find_heavy_stos(volume_dir)
    print(f'Input stos: {input_stos}')
    print(f'UsingCupy: {nornir_imageregistration.UsingCupy()}')
    times: list[float] = []
    for repeat in range(args.repeats):
        _release_refinement_worker_memory()
        out_path = output_stos.with_name(f'repeat_{repeat}_{output_stos.name}')
        if args.profile and repeat == 0:
            profile_path = _artifact_dir('stos') / 'stos.profile'
            profiler = cProfile.Profile()
            profiler.enable()
            elapsed = _run_stos_once(input_stos, out_path, args.iterations)
            profiler.disable()
            profiler.dump_stats(str(profile_path))
            _print_profile_summary(profile_path, ('AttemptAlignPoint', 'measure_translation'))
        else:
            elapsed = _run_stos_once(input_stos, out_path, args.iterations)
        times.append(elapsed)
        print(f'repeat {repeat + 1}: {elapsed:.2f}s')
    print(f'mean: {sum(times) / len(times):.2f}s')
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for mosaic or STOS grid-refine microbenchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('mosaic', 'stos'), required=True)
    parser.add_argument('--backend', choices=('numpy', 'cupy', 'both'), default='numpy')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--phase-timing', action='store_true')
    parser.add_argument('--batched', action='store_true')
    parser.add_argument('--no-batched', action='store_true')
    parser.add_argument('--mosaic-cutoff', action='store_true')
    parser.add_argument('--stos-regularize', action='store_true')
    parser.add_argument('--mosaic', type=str, default='')
    parser.add_argument('--tiles-dir', type=str, default='')
    parser.add_argument('--cell-size', type=lambda s: _parse_pair(s, 'cell-size'), default=(96, 96))
    parser.add_argument('--mesh-shape', type=lambda s: _parse_pair(s, 'mesh-shape'), default=(8, 8))
    parser.add_argument('--displacement-threshold', type=float, default=0.5)
    parser.add_argument('--min-overlap', type=float, default=0.25)
    parser.add_argument('--image-scale', type=float, default=0.25)
    parser.add_argument(
        '--volume-dir', type=Path,
        default=Path(os.environ.get('TESTOUTPUTPATH', '/tmp/nornir-test-output')) / 'TestIDocBuild')
    args = parser.parse_args(argv)

    if args.mode == 'mosaic':
        if args.backend == 'both':
            # Isolate each backend in a subprocess so CUDA context does not leak.
            rc = 0
            for backend in ('numpy', 'cupy'):
                cmd = [sys.executable, __file__, '--mode', 'mosaic', '--backend', backend,
                       '--iterations', str(args.iterations), '--repeats', str(args.repeats)]
                if args.mosaic:
                    cmd.extend(['--mosaic', args.mosaic])
                if args.tiles_dir:
                    cmd.extend(['--tiles-dir', args.tiles_dir])
                if args.phase_timing:
                    cmd.append('--phase-timing')
                if args.batched:
                    cmd.append('--batched')
                if args.no_batched:
                    cmd.append('--no-batched')
                if args.mosaic_cutoff:
                    cmd.append('--mosaic-cutoff')
                env = os.environ.copy()
                if backend == 'numpy':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                result = subprocess.run(cmd, env=env, check=False)
                rc = rc or result.returncode
            return rc
        return _run_mosaic(args)
    return _run_stos(args)


if __name__ == '__main__':
    raise SystemExit(main())
