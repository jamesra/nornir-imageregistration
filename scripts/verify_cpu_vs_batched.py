#!/usr/bin/env python3
"""Verify a batched refine output matches the serial CPU output within tolerance.

Runs RefineGridMosaic on the Grid690 fixture with two legs, each isolated in
its own subprocess so a forked CPU pool cannot inherit a CUDA context. The
reference leg is always ``numpy`` serial; the comparison leg is:

- ``cupy`` with batching on (default), or
- ``numpy`` with batching on (``--cpu-batched-parity``), to validate the CPU
  batched path.

Each subprocess saves its refined mosaic; the parent loads both (plus the C++
golden mosaic) and reports ref-vs-cmp and each-vs-golden mean/max target-point
deltas. Exit 0 iff the comparison matches the reference within tolerance.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (_REPO_ROOT, _REPO_ROOT / 'tests'):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

# CPU-vs-batched tolerance (working-resolution px). Both paths converge to the
# same registration; the batched argmax+centroid peak finder differs from the
# connected-component find_peak only at the sub-pixel level.
CPU_VS_BATCHED_MEAN_LIMIT = 1.0
CPU_VS_BATCHED_MAX_LIMIT = 3.0


def _run_subprocess_leg(backend: str, out_path: Path) -> None:
    """Run one refinement leg in-process and save the refined mosaic (subprocess entry)."""
    import nornir_imageregistration
    from nornir_imageregistration.computational_lib import ComputationLib
    from grid_seam_metrics import grid690_fixture_root, refine_grid690

    if backend == 'cupy':
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
        nornir_imageregistration.TryInitCupyContext()
    else:
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.numpy)

    fixture_root = grid690_fixture_root()
    refined, diagnostics = refine_grid690(fixture_root)
    refined.SaveToMosaicFile(str(out_path))
    print(f'[{backend}] passes={diagnostics.iterations_completed} '
          f'converged={diagnostics.converged} saved={out_path}')


def _spawn_leg(backend: str, batched: bool, out_path: Path) -> int:
    """Spawn an isolated subprocess to run one backend leg."""
    env = dict(os.environ)
    env.pop('NORNIR_REFINE_BATCHED_GPU', None)
    if batched:
        env['NORNIR_REFINE_BATCHED'] = '1'
    else:
        env['NORNIR_REFINE_BATCHED'] = '0'
    if backend == 'numpy':
        env['CUDA_VISIBLE_DEVICES'] = ''
    argv = [sys.executable, os.path.abspath(__file__),
            '--run-leg', backend, '--out', str(out_path)]
    label = f'{backend}{" +batched" if batched else ""}'
    print(f'\n##### isolated subprocess: {label} #####')
    return subprocess.run(argv, env=env).returncode


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-leg', choices=('numpy', 'cupy'), default=None,
                        help='Internal: run one backend leg and save the mosaic')
    parser.add_argument('--out', type=Path, default=None,
                        help='Internal: mosaic output path for --run-leg')
    parser.add_argument('--cpu-batched-parity', action='store_true',
                        help='Compare numpy-serial (ref) vs numpy-batched (cmp) instead of '
                             'numpy-serial vs cupy-batched. Validates the CPU batched path.')
    args = parser.parse_args()

    if args.run_leg is not None:
        if args.out is None:
            print('ERROR: --out required with --run-leg', file=sys.stderr)
            return 2
        _run_subprocess_leg(args.run_leg, args.out)
        return 0

    import nornir_imageregistration
    from grid_seam_metrics import (
        GOLDEN_GRID_MOSAIC_NAME,
        compare_mosaic_target_points_to_golden,
        grid690_fixture_is_usable,
        grid690_fixture_root,
    )

    fixture_root = grid690_fixture_root()
    if not grid690_fixture_is_usable(fixture_root):
        print(f'ERROR: Grid690 fixture not usable at {fixture_root}', file=sys.stderr)
        return 2

    # (backend, batched) for the reference and the comparison leg.
    cmp_backend = 'numpy' if args.cpu_batched_parity else 'cupy'
    ref_label = 'cpu-serial'
    cmp_label = 'cpu-batched' if args.cpu_batched_parity else 'gpu-batched'

    with tempfile.TemporaryDirectory(prefix='verify_cpu_vs_batched_') as tmp:
        cpu_path = Path(tmp) / 'cpu.mosaic'
        batched_path = Path(tmp) / 'batched.mosaic'

        if _spawn_leg('numpy', batched=False, out_path=cpu_path) != 0 or not cpu_path.is_file():
            print(f'ERROR: {ref_label} leg failed', file=sys.stderr)
            return 1
        if _spawn_leg(cmp_backend, batched=True, out_path=batched_path) != 0 or not batched_path.is_file():
            print(f'ERROR: {cmp_label} leg failed', file=sys.stderr)
            return 1

        cpu = nornir_imageregistration.Mosaic.LoadFromMosaicFile(str(cpu_path))
        batched = nornir_imageregistration.Mosaic.LoadFromMosaicFile(str(batched_path))

        cpu_vs_batched_mean, cpu_vs_batched_per_tile = compare_mosaic_target_points_to_golden(batched, cpu)
        cpu_vs_batched_max = max(cpu_vs_batched_per_tile.values())

        golden_path = os.path.join(fixture_root, GOLDEN_GRID_MOSAIC_NAME)
        golden = (nornir_imageregistration.Mosaic.LoadFromMosaicFile(golden_path)
                  if os.path.isfile(golden_path) else None)

    print(f'\n=== {ref_label} vs {cmp_label} target-point delta (working-res px) ===')
    print(f'mean: {cpu_vs_batched_mean:.4f}  (limit {CPU_VS_BATCHED_MEAN_LIMIT})')
    print(f'max : {cpu_vs_batched_max:.4f}  (limit {CPU_VS_BATCHED_MAX_LIMIT})')

    if golden is not None:
        cpu_g, _ = compare_mosaic_target_points_to_golden(cpu, golden)
        bat_g, _ = compare_mosaic_target_points_to_golden(batched, golden)
        print('\n=== Mean delta vs golden (working-res px) ===')
        print(f'{ref_label:<11}: {cpu_g:.4f}')
        print(f'{cmp_label:<11}: {bat_g:.4f}')

    mean_ok = cpu_vs_batched_mean <= CPU_VS_BATCHED_MEAN_LIMIT
    max_ok = cpu_vs_batched_max <= CPU_VS_BATCHED_MAX_LIMIT
    verdict = mean_ok and max_ok
    print('\n=== Verdict ===')
    print(f'mean <= {CPU_VS_BATCHED_MEAN_LIMIT}: {"PASS" if mean_ok else "FAIL"}')
    print(f'max  <= {CPU_VS_BATCHED_MAX_LIMIT}: {"PASS" if max_ok else "FAIL"}')
    print(f'\n{ref_label.upper()}-vs-{cmp_label.upper()} MATCH: {"PASS" if verdict else "FAIL"}')
    return 0 if verdict else 1


if __name__ == '__main__':
    raise SystemExit(main())
