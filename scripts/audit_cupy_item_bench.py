#!/usr/bin/env python3
"""Opt-in CuPy audit microbenches. Not imported by production code.

Re-measures host↔device items from the CuPy audit (P3, A5, A7, T1, T2, R2, P4, R7)
plus GPU ``cdist`` / nearest-neighbor. Requires a CuPy session.

  NORNIR_HEADLESS=1 NORNIR_COMPUTATIONAL_LIBRARY=cupy \\
    python scripts/audit_cupy_item_bench.py
  python scripts/audit_cupy_item_bench.py t2 cdist
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

os.environ.setdefault("NORNIR_HEADLESS", "1")
os.environ.setdefault("NORNIR_COMPUTATIONAL_LIBRARY", "cupy")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.computational_lib import (
    ComputationLib,
    HasCuVS,
    SetActiveComputationLib,
    UsingCupy,
)


def _sync() -> None:
    """Wait for outstanding CUDA work when CuPy is active."""
    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def _time(fn: Callable[[], object], n: int = 5, warmup: int = 1) -> float:
    """Return mean seconds for *n* synchronized calls after *warmup*."""
    for _ in range(warmup):
        fn()
        _sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
        _sync()
    return (time.perf_counter() - t0) / n


def bench_p3() -> None:
    """Log-polar upload/coerce vs host-only, plus a full LogPolar registration."""
    import cupy as cp
    from nornir_imageregistration import stos_brute
    from nornir_imageregistration.settings import SliceToSliceMethod, StosBruteSettings

    print("=== P3 LogPolar upload/coerce + full LogPolar registration ===")
    rng = np.random.default_rng(0)
    src = rng.random((1024, 1024), dtype=np.float32)
    tgt = np.roll(src, (12, -8), axis=(0, 1))
    src_stats = nornir_imageregistration.ImageStats.CalcStats(src)
    tgt_stats = nornir_imageregistration.ImageStats.CalcStats(tgt)

    def upload_then_coerce() -> tuple:
        s = cp.asarray(src)
        t = cp.asarray(tgt)
        _sync()
        return (
            stos_brute._coerce_to_source_module(s, np),
            stos_brute._coerce_to_source_module(t, np),
        )

    def coerce_only() -> tuple:
        return (
            stos_brute._coerce_to_source_module(src, np),
            stos_brute._coerce_to_source_module(tgt, np),
        )

    print(f"  upload+coerce 1024^2: {_time(upload_then_coerce, n=8) * 1e3:.2f} ms")
    print(f"  coerce-only 1024^2:   {_time(coerce_only, n=8) * 1e3:.2f} ms")

    def logpolar_only() -> object:
        return stos_brute._find_angle_and_scale_with_logpolar(
            source_image=src, target_image=tgt,
            source_stats=src_stats, target_stats=tgt_stats, min_overlap=0.5)

    def logpolar_after_upload() -> object:
        return stos_brute._find_angle_and_scale_with_logpolar(
            source_image=cp.asarray(src), target_image=cp.asarray(tgt),
            source_stats=src_stats, target_stats=tgt_stats, min_overlap=0.5)

    print(f"  logpolar host inputs:   {_time(logpolar_only, n=3) * 1e3:.1f} ms")
    print(f"  logpolar after upload:  {_time(logpolar_after_upload, n=3) * 1e3:.1f} ms")

    src_h = nornir_imageregistration.ImagePermutationHelper(src, None)
    tgt_h = nornir_imageregistration.ImagePermutationHelper(tgt, None)
    settings = StosBruteSettings(
        method=SliceToSliceMethod.LogPolar,
        min_overlap=0.5,
        try_flipped=False,
        larget_dimension=1024,
        angles={0.0},
    )

    def full_logpolar() -> object:
        return stos_brute.SliceToSliceRigidRegistrationWithPreprocessedImages(
            source_image_data=src_h,
            target_image_data=tgt_h,
            settings=settings,
            SingleThread=True,
        )

    t = _time(full_logpolar, n=2, warmup=1)
    rec = full_logpolar()
    print(f"  full LogPolar (no flip) 1024: {t * 1e3:.1f} ms  angle={rec.angle:.2f} weight={rec.weight:.4f}")


def bench_a7() -> None:
    """Empty-ROI and invalid-subroi assemble paths."""
    from nornir_imageregistration.assemble import _TransformImageUsingCoords

    print("=== A7 empty-ROI / invalid subroi ===")
    empty_src = np.zeros((0, 2), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)

    def empty_warp() -> object:
        return _TransformImageUsingCoords(
            empty_src, empty_src, img,
            output_origin=(0, 0), output_area=(32, 32), cval=0)

    coords = np.array([[-100.0, -100.0]], dtype=np.float32)
    img2 = np.ones((8, 8), dtype=np.float32)

    def invalid_subroi() -> object:
        return _TransformImageUsingCoords(
            coords, coords, img2,
            output_origin=(0, 0), output_area=(16, 16), cval=0)

    print(f"  empty coords: {_time(empty_warp, n=20) * 1e3:.3f} ms")
    print(f"  invalid subroi: {_time(invalid_subroi, n=20) * 1e3:.3f} ms")


def bench_t1() -> None:
    """Vectorized TPS Beta matrix on GPU vs CPU."""
    import cupy as cp
    from nornir_imageregistration.transforms.one_way_rbftransform import (
        OneWayRBFWithLinearCorrection,
        OneWayRBFWithLinearCorrection_GPUComponent,
    )

    print("=== T1 CreateBetaMatrix GPU ===")
    rng = np.random.default_rng(1)
    n = 256
    pts = cp.asarray(rng.random((n, 2), dtype=np.float32) * 1000.0)

    def beta() -> object:
        return OneWayRBFWithLinearCorrection_GPUComponent.CreateBetaMatrix(
            pts, OneWayRBFWithLinearCorrection_GPUComponent.DefaultBasisFunction)

    print(f"  GPU Beta N={n}: {_time(beta, n=3) * 1e3:.1f} ms")
    pts_np = rng.random((n, 2), dtype=np.float32) * 1000.0
    cpu = OneWayRBFWithLinearCorrection.CreateBetaMatrix(
        pts_np, OneWayRBFWithLinearCorrection.DefaultBasisFunction)
    gpu = OneWayRBFWithLinearCorrection_GPUComponent.CreateBetaMatrix(
        cp.asarray(pts_np), OneWayRBFWithLinearCorrection_GPUComponent.DefaultBasisFunction)
    max_err = float(np.max(np.abs(cpu - cp.asnumpy(gpu))))
    print(f"  CPU vs GPU max abs err: {max_err:.4g}")


def bench_t2() -> None:
    """Nearest-neighbor index: cKDTree below 4096, CuVS at or above."""
    from nornir_imageregistration.nearest_neighbor import build_nearest_neighbor_index

    print("=== T2 nearest-neighbor build+query (k=1) ===")
    rng = np.random.default_rng(0)
    for n in (256, 1024, 4096, 10000):
        pts = rng.random((n, 2), dtype=np.float32)

        def run() -> object:
            idx = build_nearest_neighbor_index(pts)
            return idx.query(pts[:8], k=1)

        kind = type(build_nearest_neighbor_index(pts)).__name__
        print(f"  N={n} build+query8: {_time(run, n=5) * 1e3:.2f} ms  {kind}")


def bench_cdist() -> None:
    """GPU pairwise cdist (CuVS when HasCuVS) vs host SciPy."""
    import cupy as cp
    from nornir_imageregistration.spatial_distance import cdist

    print(f"=== cdist (HasCuVS={HasCuVS()}) ===")
    rng = np.random.default_rng(0)
    for n in (256, 1024, 4096):
        host = rng.random((n, 2), dtype=np.float32)
        gpu = cp.asarray(host)

        def gpu_cdist() -> object:
            return cdist(gpu, gpu)

        def host_cdist() -> object:
            return cdist(host, host)

        print(
            f"  N={n} GPU: {_time(gpu_cdist, n=5) * 1e3:.2f} ms  "
            f"host: {_time(host_cdist, n=5) * 1e3:.2f} ms"
        )


def bench_r2() -> None:
    """Vectorized masked ZNCC over a cell stack."""
    import cupy as cp
    from nornir_imageregistration.local_distortion_correction import _masked_zncc_stack

    print("=== R2 masked_zncc_stack ===")
    rng = np.random.default_rng(2)
    n, h, w = 64, 96, 96
    a = cp.asarray(rng.random((n, h, w), dtype=np.float32))
    b = a + 0.05 * cp.asarray(rng.random((n, h, w), dtype=np.float32))

    def zncc() -> object:
        return _masked_zncc_stack(a, b)

    print(f"  stack {n}x{h}x{w}: {_time(zncc, n=5) * 1e3:.1f} ms")


def bench_p4() -> None:
    """Pad-for-PC with ImageStats vs min/max sync."""
    import cupy as cp
    from nornir_imageregistration.phasecorrelation import pad_image_for_phase_correlation

    print("=== P4 pad min/max sync ===")
    img = cp.asarray(np.random.default_rng(3).random((512, 512), dtype=np.float32))
    stats = nornir_imageregistration.ImageStats.CalcStats(img)

    def pad_no_stats() -> object:
        return pad_image_for_phase_correlation(img, min_overlap=0.5)

    def pad_with_stats() -> object:
        return pad_image_for_phase_correlation(
            img, min_overlap=0.5, image_median=stats.median, image_stddev=stats.std)

    print(f"  pad 512 no stats:   {_time(pad_no_stats, n=8) * 1e3:.2f} ms")
    print(f"  pad 512 with stats: {_time(pad_with_stats, n=8) * 1e3:.2f} ms")


def bench_r7() -> None:
    """Global FOV residual translation on a 1024 image downsampled to 512."""
    from nornir_imageregistration.refine_shared.coherent_residual import (
        estimate_global_fov_residual_translation,
    )
    from nornir_imageregistration.transforms.rigid import Rigid

    print("=== R7 global FOV residual ===")
    rng = np.random.default_rng(4)
    tgt = rng.random((1024, 1024), dtype=np.float32)
    src = np.roll(tgt, (20, -15), axis=(0, 1))
    transform = Rigid(target_offset=(0.0, 0.0), source_rotation_center=(512.0, 512.0), angle=0.0)

    def fov() -> object:
        return estimate_global_fov_residual_translation(transform, tgt, src, max_dim=512)

    t = _time(fov, n=2, warmup=1)
    peak = fov()
    print(f"  FOV 1024→512: {t * 1e3:.1f} ms  peak={peak}")


def bench_a5() -> None:
    """Serial vs thread-pool find_offset (arrange analog)."""
    import cupy as cp
    import nornir_pools
    from nornir_imageregistration.phasecorrelation import find_offset

    print("=== A5 serial vs thread-pool find_offset ===")
    rng = np.random.default_rng(5)
    pairs = []
    for i in range(12):
        a = rng.random((256, 256), dtype=np.float32)
        b = np.roll(a, (3 + i, -2), axis=(0, 1))
        pairs.append((cp.asarray(a), cp.asarray(b)))

    def serial() -> list:
        out = [find_offset(a, b) for a, b in pairs]
        _sync()
        return out

    def threaded() -> list:
        pool = nornir_pools.GetGlobalThreadPool()
        tasks = [pool.add_task(f"off{i}", find_offset, a, b) for i, (a, b) in enumerate(pairs)]
        out = [t.wait_return() for t in tasks]
        _sync()
        return out

    print(f"  serial 12x256:   {_time(serial, n=2, warmup=1) * 1e3:.1f} ms")
    print(f"  threads 12x256:  {_time(threaded, n=2, warmup=1) * 1e3:.1f} ms")


_BENCHES: dict[str, Callable[[], None]] = {
    "p3": bench_p3,
    "a7": bench_a7,
    "t1": bench_t1,
    "t2": bench_t2,
    "cdist": bench_cdist,
    "r2": bench_r2,
    "p4": bench_p4,
    "r7": bench_r7,
    "a5": bench_a5,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected audit microbenches; default is the full set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "items",
        nargs="*",
        choices=sorted(_BENCHES),
        help="Subset of items (default: all)",
    )
    args = parser.parse_args(argv)

    SetActiveComputationLib(ComputationLib.cupy)
    if not UsingCupy():
        print("Requires a CuPy session (NORNIR_COMPUTATIONAL_LIBRARY=cupy).", file=sys.stderr)
        return 1

    print(f"UsingCupy={UsingCupy()} HasCuVS={HasCuVS()}")
    for name in (args.items or list(_BENCHES)):
        _BENCHES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
