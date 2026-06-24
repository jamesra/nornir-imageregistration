#!/usr/bin/env python3
"""Compare serial vs batched-GPU mosaic grid refinement against the golden mosaic.

Runs RefineGridMosaic on the RC2_4Square_Assembled Grid690 fixture twice in one
process - once with the serial connected-component find_peak, once with the
opt-in batched-GPU vertex path (NORNIR_REFINE_BATCHED_GPU=1) - and reports:

- mean/max target-point delta vs the C++ golden mosaic for each path,
- seam MAE for each path,
- batched-vs-serial target-point delta (the prototype's self-consistency).

Gate (per the plan): batched stays within golden 2.2 px and seam MAE < 35, and
batched-vs-serial mean target delta within ~1.0 working-res px.

Requires CuPy for the batched path to engage; on a CPU-only host the batched
flag is a no-op (gated by UsingCupy) and the two runs will be identical.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (_REPO_ROOT, _REPO_ROOT / 'tests'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import nornir_imageregistration
from nornir_imageregistration.computational_lib import ComputationLib
from nornir_imageregistration.local_distortion_correction import _release_refinement_worker_memory

from grid_seam_metrics import (  # noqa: E402  (tests/ helper)
    GOLDEN_GRID_MOSAIC_NAME,
    SEAM_MIN_OVERLAP,
    compare_mosaic_target_points_to_golden,
    grid690_fixture_is_usable,
    grid690_fixture_root,
    measure_mosaic_seam_scores,
    refine_grid690,
    _grid690_tile_dir,
)
from grid690_diagnostics import REGISTRATION_DOWNSAMPLE  # noqa: E402  (tests/ helper)

GOLDEN_TARGET_DELTA_LIMIT = 2.2
SEAM_MAE_LIMIT = 35.0
BATCHED_VS_SERIAL_LIMIT = 1.0


def _run(fixture_root: str, batched: bool):
    """Run one refinement (serial or batched) and return the refined mosaic + diagnostics."""
    # Batched is the production default under CuPy; force '0' to exercise the
    # legacy serial path for the A/B comparison.
    os.environ['NORNIR_REFINE_BATCHED_GPU'] = '1' if batched else '0'
    _release_refinement_worker_memory()
    return refine_grid690(fixture_root)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fixture-root', type=Path, default=None,
                        help='Grid690 fixture root (defaults to TESTINPUTPATH discovery)')
    args = parser.parse_args()

    fixture_root = str(args.fixture_root) if args.fixture_root else grid690_fixture_root()
    if not grid690_fixture_is_usable(fixture_root):
        print(f'ERROR: Grid690 fixture not usable at {fixture_root}', file=sys.stderr)
        return 2

    if not nornir_imageregistration.HasCupy():
        print('WARNING: CuPy unavailable; batched flag is a no-op and runs will match.')
    else:
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
        nornir_imageregistration.TryInitCupyContext()

    print(f'Fixture:  {fixture_root}')
    print(f'UsingCupy: {nornir_imageregistration.UsingCupy()}')

    tile_dir = _grid690_tile_dir(fixture_root)
    golden_path = os.path.join(fixture_root, GOLDEN_GRID_MOSAIC_NAME)
    golden = (nornir_imageregistration.Mosaic.LoadFromMosaicFile(golden_path)
              if os.path.isfile(golden_path) else None)

    print('\nRunning serial refinement...')
    serial, serial_diag = _run(fixture_root, batched=False)
    print('Running batched refinement...')
    batched, batched_diag = _run(fixture_root, batched=True)

    serial_seams = measure_mosaic_seam_scores(
        serial, tile_dir, REGISTRATION_DOWNSAMPLE, min_overlap=SEAM_MIN_OVERLAP)
    batched_seams = measure_mosaic_seam_scores(
        batched, tile_dir, REGISTRATION_DOWNSAMPLE, min_overlap=SEAM_MIN_OVERLAP)

    print('\n=== Convergence ===')
    print(f'serial : passes={serial_diag.iterations_completed} converged={serial_diag.converged}')
    print(f'batched: passes={batched_diag.iterations_completed} converged={batched_diag.converged}')

    print('\n=== Seam MAE (lower is better) ===')
    print(f'serial : mean={serial_seams.mean_mae:.3f} max={serial_seams.max_mae:.3f}')
    print(f'batched: mean={batched_seams.mean_mae:.3f} max={batched_seams.max_mae:.3f}')

    golden_ok = True
    if golden is not None:
        serial_delta, _ = compare_mosaic_target_points_to_golden(serial, golden)
        batched_delta, _ = compare_mosaic_target_points_to_golden(batched, golden)
        print('\n=== Mean target-point delta vs golden (px) ===')
        print(f'serial : {serial_delta:.3f}')
        print(f'batched: {batched_delta:.3f}  (limit {GOLDEN_TARGET_DELTA_LIMIT})')
        golden_ok = batched_delta <= GOLDEN_TARGET_DELTA_LIMIT
    else:
        print('\n(no golden mosaic present; skipping golden comparison)')

    bvs_delta, _ = compare_mosaic_target_points_to_golden(batched, serial)
    print('\n=== Batched vs serial mean target-point delta (px) ===')
    print(f'{bvs_delta:.3f}  (limit {BATCHED_VS_SERIAL_LIMIT})')

    seam_ok = batched_seams.max_mae < SEAM_MAE_LIMIT
    bvs_ok = bvs_delta <= BATCHED_VS_SERIAL_LIMIT

    print('\n=== Gate ===')
    print(f'golden delta <= {GOLDEN_TARGET_DELTA_LIMIT}: {"PASS" if golden_ok else "FAIL"}')
    print(f'seam max MAE < {SEAM_MAE_LIMIT}:        {"PASS" if seam_ok else "FAIL"}')
    print(f'batched vs serial <= {BATCHED_VS_SERIAL_LIMIT}:  {"PASS" if bvs_ok else "FAIL"}')
    verdict = golden_ok and seam_ok and bvs_ok
    print(f'\nPARITY VERDICT: {"PASS" if verdict else "FAIL"}')
    return 0 if verdict else 1


if __name__ == '__main__':
    raise SystemExit(main())
