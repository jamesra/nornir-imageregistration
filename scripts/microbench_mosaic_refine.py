#!/usr/bin/env python3
"""Micro-benchmark mosaic grid refinement (RefineGridMosaic) on CPU vs GPU.

Measures wall time, refinement diagnostics, and (with --phase-timing) the
per-phase wall-time split (prewarp / cell_extract / fft / host_sync /
regularize / apply) emitted by ``_refine_tileset``. Optionally profiles one
run with cProfile and prints needles for the known hot functions.

Inputs:
  --mosaic + --tiles-dir : explicit translated .mosaic and tile-level image dir
  (default)              : auto-discover the RC2_4Square_Assembled Grid690
                           fixture under TESTINPUTPATH

Backend is selected with --backend {numpy,cupy,both}; cupy runs are skipped
cleanly when CuPy is unavailable.

To capture the per-phase breakdown, pass --phase-timing (sets
NORNIR_REFINE_PHASE_TIMING before importing nornir_imageregistration).
"""

from __future__ import annotations

import argparse
import cProfile
import json
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


def _enable_phase_timing_from_argv() -> None:
    """Set NORNIR_REFINE_PHASE_TIMING before nornir imports when requested."""
    if '--phase-timing' in sys.argv or '--phase_timing' in sys.argv:
        os.environ['NORNIR_REFINE_PHASE_TIMING'] = '1'


def _enable_batched_from_argv() -> None:
    """Pin NORNIR_REFINE_BATCHED_GPU before nornir imports for a clean A/B.

    The batched-GPU vertex path is the production default under CuPy, so this
    harness pins it explicitly: ``--batched`` forces it on, its absence forces
    the legacy serial path on, so the matrix always contrasts the two.
    """
    os.environ['NORNIR_REFINE_BATCHED_GPU'] = '1' if '--batched' in sys.argv else '0'


_enable_phase_timing_from_argv()
_enable_batched_from_argv()

import nornir_imageregistration  # noqa: E402
from nornir_imageregistration.computational_lib import ComputationLib  # noqa: E402
from nornir_imageregistration.local_distortion_correction import (  # noqa: E402
    MosaicRefinementDiagnostics,
    _release_refinement_worker_memory,
)

# Grid690 fixture parameters (match tests/grid_seam_metrics.py repro pipeline).
_GRID690_DATASET = 'RC2_4Square_Assembled'
_GRID690_TRANSLATED = 'Translated_Prune_Max0.5.mosaic'
_GRID690_TILE_SUBDIR = ('Leveled', 'TilePyramid', '004')
_DEFAULT_CELL_SIZE = (96, 96)
_DEFAULT_MESH_SHAPE = (8, 8)
_DEFAULT_DISPLACEMENT_THRESHOLD = 0.5
_DEFAULT_IMAGE_SCALE = 0.25


def _artifact_dir() -> Path:
    """Return a writable directory for profile/JSON artifacts (tile dirs may be read-only)."""
    for env_name in ('TEST_OUTPUT_DIR', 'TESTOUTPUTPATH'):
        value = os.environ.get(env_name, '').strip()
        if value:
            out = Path(value) / 'microbench_mosaic_refine'
            try:
                out.mkdir(parents=True, exist_ok=True)
                return out
            except OSError:
                continue
    out = Path(tempfile.gettempdir()) / 'microbench_mosaic_refine'
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


def _discover_grid690() -> tuple[Path, Path] | None:
    """Return (translated mosaic, tile dir) for the Grid690 fixture if usable."""
    testinput = os.environ.get('TESTINPUTPATH', '').strip()
    candidates: list[Path] = []
    if testinput:
        candidates.append(
            Path(testinput) / 'PlatformRaw' / 'IDOC' / _GRID690_DATASET / 'TEM' / '0690' / 'TEM')
    tests_fixtures = _REPO_ROOT / 'tests' / 'fixtures'
    candidates.append(tests_fixtures / 'RC2_4Square_Assembled_Grid690' / 'TEM' / '0690' / 'TEM')
    candidates.append(tests_fixtures / 'IDocBuildTest_Grid690' / 'TEM' / '0690' / 'TEM')
    for root in candidates:
        mosaic = root / _GRID690_TRANSLATED
        tile_dir = root.joinpath(*_GRID690_TILE_SUBDIR)
        if mosaic.is_file() and tile_dir.is_dir():
            return mosaic, tile_dir
    return None


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


def _measured_cells(diagnostics: MosaicRefinementDiagnostics) -> int:
    """Total FFT cells measured across all passes (a vertices/sec numerator)."""
    total = 0
    for pass_diag in diagnostics.vertex_diagnostics_per_pass:
        for tile_diag in pass_diag.values():
            total += int(tile_diag.get('measured', 0))
    return total


def _run_once(args, mosaic: Path, tile_dir: Path) -> tuple[float, MosaicRefinementDiagnostics]:
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
    elapsed = time.perf_counter() - start
    return elapsed, diagnostics


def _profile_once(args, mosaic: Path, tile_dir: Path, profile_path: Path
                  ) -> tuple[float, MosaicRefinementDiagnostics]:
    """Profile one RefineGridMosaic call and dump stats to *profile_path*."""
    profiler = cProfile.Profile()
    profiler.enable()
    elapsed, diagnostics = _run_once(args, mosaic, tile_dir)
    profiler.disable()
    profiler.dump_stats(str(profile_path))
    return elapsed, diagnostics


def _print_profile_summary(profile_path: Path) -> None:
    """Print top cumulative stats plus needles for the refine hot functions."""
    stream = StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    print(stream.getvalue())
    for needle in ('_phase_correlate_refinement_cell', 'find_offset',
                   '_prewarp_tile_for_grid_refine', '_regularize_displacements',
                   '_extract_refinement_cell'):
        stream.truncate(0)
        stream.seek(0)
        stats.print_stats(needle)
        print(stream.getvalue())


def _summarize(backend: str, repeat: int, elapsed: float,
               diagnostics: MosaicRefinementDiagnostics) -> dict[str, float | int | str | bool]:
    """Build a result record for one run."""
    cells = _measured_cells(diagnostics)
    return {
        'backend': backend,
        'repeat': repeat,
        'wall_s': elapsed,
        'passes': diagnostics.iterations_completed,
        'converged': diagnostics.converged,
        'cells': cells,
        'cells_per_s': (cells / elapsed) if elapsed > 0 else 0.0,
        'cell_size': tuple(diagnostics.resolved_cell_size),
        'mesh_shape': tuple(diagnostics.resolved_mesh_shape),
    }


def _print_results_table(results: list[dict]) -> None:
    """Print a compact comparison table over all runs."""
    print()
    header = (f'{"backend":<9} {"rep":>3} {"cell":>9} {"mesh":>7} {"passes":>6} '
              f'{"conv":>5} {"wall_s":>9} {"cells":>8} {"cells/s":>10}')
    print(header)
    print('-' * len(header))
    for r in results:
        cell = f'{r["cell_size"][0]}x{r["cell_size"][1]}'
        mesh = f'{r["mesh_shape"][0]}x{r["mesh_shape"][1]}'
        print(f'{r["backend"]:<9} {r["repeat"]:>3} {cell:>9} {mesh:>7} '
              f'{r["passes"]:>6} {str(r["converged"]):>5} {r["wall_s"]:>9.2f} '
              f'{r["cells"]:>8} {r["cells_per_s"]:>10.1f}')

    # Best (fastest) wall per backend for a quick speedup read.
    by_backend: dict[str, float] = {}
    for r in results:
        by_backend.setdefault(r['backend'], float('inf'))
        by_backend[r['backend']] = min(by_backend[r['backend']], r['wall_s'])
    if 'numpy' in by_backend and 'cupy' in by_backend and by_backend['cupy'] > 0:
        speedup = by_backend['numpy'] / by_backend['cupy']
        print(f'\nbest wall: numpy={by_backend["numpy"]:.2f}s  cupy={by_backend["cupy"]:.2f}s  '
              f'(numpy/cupy = {speedup:.2f}x)')


def _run_backend(args, backend: str, mosaic: Path, tile_dir: Path,
                 cell_sizes: list[tuple[int, int]]) -> list[dict]:
    """Run all cell sizes / repeats for one backend in-process; return result records."""
    results: list[dict] = []
    if not _set_backend(backend):
        print(f'\n[skip] backend {backend}: CuPy unavailable')
        return results
    # Batched is CuPy-only (gated by UsingCupy); reflect that in the record label.
    batched_active = getattr(args, 'batched', False) and backend == 'cupy'
    backend_label = f'{backend}-bat' if batched_active else backend
    for cell_size in cell_sizes:
        args.cell_size = cell_size
        for repeat in range(args.repeats):
            profile_this = args.profile and repeat == 0
            label = f'{backend_label} cell={cell_size[0]}x{cell_size[1]} rep={repeat + 1}'
            print(f'\n=== {label} ===')
            if profile_this:
                profile_path = _artifact_dir() / f'{backend_label}_{cell_size[0]}.profile'
                elapsed, diagnostics = _profile_once(args, mosaic, tile_dir, profile_path)
                _print_profile_summary(profile_path)
            else:
                elapsed, diagnostics = _run_once(args, mosaic, tile_dir)
            record = _summarize(backend_label, repeat + 1, elapsed, diagnostics)
            results.append(record)
            print(f'{label}: {elapsed:.2f}s  passes={record["passes"]} '
                  f'converged={record["converged"]}  cells={record["cells"]}  '
                  f'cells/s={record["cells_per_s"]:.1f}')
    return results


def _forward_args(args, backend: str, json_path: Path, mosaic: Path, tile_dir: Path,
                  cell_sizes: list[tuple[int, int]]) -> list[str]:
    """Build a subprocess argv to benchmark one backend in an isolated process."""
    argv = [sys.executable, os.path.abspath(__file__),
            '--backend', backend,
            '--mosaic', str(mosaic),
            '--tiles-dir', str(tile_dir),
            '--iterations', str(args.iterations),
            '--mesh-shape', f'{args.mesh_shape[0]}x{args.mesh_shape[1]}',
            '--displacement-threshold', str(args.displacement_threshold),
            '--image-scale', str(args.image_scale),
            '--min-overlap', str(args.min_overlap),
            '--repeats', str(args.repeats),
            '--emit-json', str(json_path)]
    argv.append('--cell-sweep')
    argv.extend(f'{c[0]}x{c[1]}' for c in cell_sizes)
    if args.profile:
        argv.append('--profile')
    if args.phase_timing:
        argv.append('--phase-timing')
    if args.batched:
        argv.append('--batched')
    return argv


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mosaic', type=Path, default=None,
                        help='Translated .mosaic transform path (defaults to Grid690 fixture)')
    parser.add_argument('--tiles-dir', type=Path, default=None,
                        help='Directory of tile images referenced by the mosaic')
    parser.add_argument('--iterations', type=int, default=10, help='Max refine passes')
    parser.add_argument('--cell-size', type=lambda s: _parse_pair(s, 'cell-size'),
                        default=_DEFAULT_CELL_SIZE, help='Cell size N, NxM, or N,M')
    parser.add_argument('--mesh-shape', type=lambda s: _parse_pair(s, 'mesh-shape'),
                        default=_DEFAULT_MESH_SHAPE, help='Mesh shape (rows, cols)')
    parser.add_argument('--displacement-threshold', type=float,
                        default=_DEFAULT_DISPLACEMENT_THRESHOLD, help='Early-stop threshold (px)')
    parser.add_argument('--image-scale', type=float, default=_DEFAULT_IMAGE_SCALE,
                        help='Target-space scale (1/downsample)')
    parser.add_argument('--min-overlap', type=float, default=0.25, help='Min cell valid fraction')
    parser.add_argument('--repeats', type=int, default=2, help='Repeats per backend/cell')
    parser.add_argument('--backend', choices=('numpy', 'cupy', 'both'), default='both',
                        help='Computation backend(s) to benchmark')
    parser.add_argument('--cell-sweep', type=lambda s: _parse_pair(s, 'cell-sweep'),
                        nargs='*', default=None,
                        help='Optional list of cell sizes to sweep (overrides --cell-size)')
    parser.add_argument('--profile', action='store_true',
                        help='cProfile the first run of each backend')
    parser.add_argument('--phase-timing', action='store_true',
                        help='Enable per-phase timing log in _refine_tileset')
    parser.add_argument('--batched', action='store_true',
                        help='Enable opt-in batched-GPU vertex path (CuPy only)')
    parser.add_argument('--emit-json', type=Path, default=None,
                        help='Internal: write this backend run results as JSON (subprocess use)')
    args = parser.parse_args()

    if args.mosaic is not None and args.tiles_dir is not None:
        mosaic, tile_dir = args.mosaic, args.tiles_dir
    else:
        discovered = _discover_grid690()
        if discovered is None:
            print('ERROR: no --mosaic/--tiles-dir given and Grid690 fixture not found.',
                  file=sys.stderr)
            print('Set TESTINPUTPATH or pass --mosaic and --tiles-dir.', file=sys.stderr)
            return 2
        mosaic, tile_dir = discovered

    if not mosaic.is_file():
        print(f'ERROR: mosaic not found: {mosaic}', file=sys.stderr)
        return 2
    if not tile_dir.is_dir():
        print(f'ERROR: tiles dir not found: {tile_dir}', file=sys.stderr)
        return 2

    backends = ('numpy', 'cupy') if args.backend == 'both' else (args.backend,)
    cell_sizes = args.cell_sweep if args.cell_sweep else [tuple(args.cell_size)]

    print(f'Mosaic:    {mosaic}')
    print(f'Tiles dir: {tile_dir}')
    print(f'HasCupy:   {nornir_imageregistration.HasCupy()}')
    print(f'Backends:  {backends}')
    print(f'Cells:     {cell_sizes}')
    print(f'iterations={args.iterations} mesh={tuple(args.mesh_shape)} '
          f'image_scale={args.image_scale} threshold={args.displacement_threshold}')

    results: list[dict] = []
    if len(backends) == 1:
        # Single backend: run in-process (also the subprocess case used by 'both').
        results = _run_backend(args, backends[0], mosaic, tile_dir, cell_sizes)
        if args.emit_json is not None:
            args.emit_json.write_text(json.dumps(results), encoding='utf-8')
            return 0 if results else 1
    else:
        # Multiple backends: isolate each in its own process so a numpy-run fork
        # pool cannot inherit a broken CUDA context from CuPy initialization.
        for backend in backends:
            json_path = _artifact_dir() / f'results_{backend}.json'
            argv = _forward_args(args, backend, json_path, mosaic, tile_dir, cell_sizes)
            sub_env = dict(os.environ)
            if backend == 'numpy':
                # Hide the GPU so the CPU path is fork-safe (no inherited CUDA context)
                # and the measurement reflects a true CPU-only machine.
                sub_env['CUDA_VISIBLE_DEVICES'] = ''
            print(f'\n##### isolated subprocess: backend {backend} #####')
            completed = subprocess.run(argv, env=sub_env)
            if completed.returncode != 0:
                print(f'[warn] backend {backend} subprocess exited {completed.returncode}')
            if json_path.is_file():
                try:
                    results.extend(json.loads(json_path.read_text(encoding='utf-8')))
                except (OSError, ValueError) as exc:
                    print(f'[warn] could not read {json_path}: {exc}')

    if not results:
        print('\nNo runs completed.')
        return 1

    # Records loaded from JSON have list cell_size/mesh_shape; normalize to tuples.
    for r in results:
        r['cell_size'] = tuple(r['cell_size'])
        r['mesh_shape'] = tuple(r['mesh_shape'])
    _print_results_table(results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
