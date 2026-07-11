#!/usr/bin/env python3
"""Assess independent source-axis Y/X scale on RPC3 Manual StosBrute64 pairs.

Uses Manual ``CenteredSimilarity2DTransform`` angle for alignment only (no angle
search). Scale is applied in source image axes then rotated (scale-then-rotate).
Compares anisotropic ``(sy, sx)`` to Manual isotropic scalar and existing 1D
estimators (B1 radial FFT + ``_refine_scale_local``).

Example::

    python scripts/assess_stos_anisotropic_scale.py \\
        --manual-dir /storage4/RPC3/TEM/StosBrute64/Manual \\
        --write-csv /tmp/aniso_scale.csv --write-json /tmp/aniso_scale.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nornir_imageregistration  # noqa: E402
import nornir_imageregistration.stos_brute as stos_brute  # noqa: E402
from nornir_imageregistration.computational_lib import ComputationLib  # noqa: E402
from nornir_imageregistration.files.stosfile import StosFile  # noqa: E402
from nornir_imageregistration.hann_window_cache import HannWindowCache  # noqa: E402

_DEFAULT_MANUAL_DIRS = (
    Path('/storage4/RPC3/TEM/StosBrute64/Manual'),
    Path(os.environ.get('INPUT_NORNIR_DATA', '')) / 'RPC3' / 'TEM' / 'StosBrute64' / 'Manual',
)

_DEFAULT_PAIRS: tuple[str, ...] = (
    '68-69', '69-70', '71-72', '72-73', '73-74', '75-76', '76-77',
    '78-79', '80-81', '82-84', '84-85', '86-87', '89-90', '108-109',
)

_STOS_NAME = '{pair}_ctrl-TEM_Blob_map-TEM_Blob.stos'
_MIN_OVERLAP = 0.5
_DESCENT_PASSES = 2
# Peak weight below this is flagged low-confidence (unreliable anisotropy).
_LOW_WEIGHT_FLAG = 1.5
_ANISO_ABS_THRESHOLD = 0.02
_WEIGHT_LIFT_THRESHOLD = 1.05


@dataclass
class PairResult:
    pair: str
    stos_path: str
    manual_angle_deg: float
    manual_scalar: float
    scale_radial: float
    radial_peak_ratio: float
    scale_iso_refine: float
    sy: float
    sx: float
    abs_sy_sx: float
    geo_mean: float
    geo_mean_vs_manual: float
    iso_refine_vs_manual: float
    weight_manual_iso: float
    weight_iso_refine: float
    weight_aniso: float
    weight_lift_vs_manual: float
    weight_lift_vs_iso_refine: float
    low_weight: bool
    elapsed_s: float


def _resolve_manual_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f'--manual-dir not found: {explicit}')
        return explicit
    for candidate in _DEFAULT_MANUAL_DIRS:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        'No Manual directory found; pass --manual-dir '
        '(tried /storage4/... and $INPUT_NORNIR_DATA/...)')


def _set_backend(backend: str) -> None:
    if backend == 'cupy':
        if not nornir_imageregistration.HasCupy():
            raise RuntimeError('CuPy requested but not available')
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
        nornir_imageregistration.TryInitCupyContext()
    else:
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.numpy)


def _maybe_downscale(
        source: np.ndarray,
        target: np.ndarray,
        source_mask: np.ndarray | None,
        target_mask: np.ndarray | None,
        max_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    shapes = [source.shape, target.shape]
    scalar = float(nornir_imageregistration.ScalarForMaxDimension(max_dim, shapes))
    if scalar >= 1.0:
        return source, target, source_mask, target_mask
    source = nornir_imageregistration.ScaleImage(source, scalar)
    target = nornir_imageregistration.ScaleImage(target, scalar)
    if source_mask is not None:
        source_mask = nornir_imageregistration.ScaleImage(source_mask, scalar)
    if target_mask is not None:
        target_mask = nornir_imageregistration.ScaleImage(target_mask, scalar)
    return source, target, source_mask, target_mask


def _manual_angle_and_scalar(stos: StosFile) -> tuple[float, float]:
    """Return (angle_degrees, scalar) from Manual CenteredSimilarity / Rigid."""
    transform = nornir_imageregistration.transforms.LoadTransform(
        stos.Transform, pixelSpacing=1.0)
    angle_rad = float(getattr(transform, 'angle', 0.0) or 0.0)
    scalar = float(getattr(transform, 'scalar', 1.0) or 1.0)
    # Transform stores radians; spatial rotate/score use degrees (AngleSearchRange).
    angle_deg = stos_brute._normalize_angle_degrees(float(np.degrees(angle_rad)))
    return angle_deg, scalar


def _estimate_b1_scale(
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_stats: nornir_imageregistration.ImageStats,
        target_stats: nornir_imageregistration.ImageStats,
) -> tuple[float, float]:
    """Isotropic B1 radial-FFT scale seed (no angle)."""
    desired_height = int(nornir_imageregistration.NearestPowerOfTwo(
        max(source_image.shape[0], target_image.shape[0])))
    desired_width = int(nornir_imageregistration.NearestPowerOfTwo(
        max(source_image.shape[1], target_image.shape[1])))
    desired_shape = (desired_height, desired_width)
    max_dimension = max(desired_height, desired_width)
    radius_radial = stos_brute._radial_fft_max_radius(max_dimension)

    padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        target_image,
        min_overlap=_MIN_OVERLAP,
        image_median=target_stats.median,
        image_stddev=target_stats.std,
        new_height=desired_height,
        new_width=desired_width)
    padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        source_image,
        min_overlap=_MIN_OVERLAP,
        image_median=source_stats.median,
        image_stddev=source_stats.std,
        new_height=desired_height,
        new_width=desired_width)

    target_window = HannWindowCache.GetOrCreate(padded_target.shape)
    source_window = HannWindowCache.GetOrCreate(padded_source.shape)
    target_mag = stos_brute._logpolar_fft_magnitude(padded_target, target_window, use_dog=True)
    source_mag = stos_brute._logpolar_fft_magnitude(padded_source, source_window, use_dog=True)
    return stos_brute._estimate_scale_radial_fft(
        target_mag, source_mag, radius_radial, desired_shape)


def _make_aniso_scorer(
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_stats: nornir_imageregistration.ImageStats,
        target_stats: nornir_imageregistration.ImageStats,
        angle_deg: float,
):
    """Return score(sy, sx) using scale-then-rotate at Manual angle."""
    try:
        import cupy as _cp
        xp = _cp.get_array_module(source_image)
    except Exception:
        xp = np

    padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        target_image,
        image_median=target_stats.median,
        image_stddev=target_stats.std,
        min_overlap=_MIN_OVERLAP,
        original_shape=target_image.shape)
    target_mean = float(target_stats.mean)
    fft_target_cache: dict[tuple[int, int], np.ndarray] = {}

    def score(sy: float, sx: float) -> float:
        sy = float(np.clip(sy, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
        sx = float(np.clip(sx, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
        working = source_image
        working_stats = source_stats
        working_shape = source_image.shape
        if not (np.isclose(sy, 1.0) and np.isclose(sx, 1.0)):
            working = stos_brute._scale_registration_image(source_image, sy, sx)
            working_stats = nornir_imageregistration.ImageStats.CalcStats(working)
            working_shape = working.shape

        rotated = stos_brute.pad_and_rotate_image(
            image=working,
            angle=angle_deg,
            image_stats=working_stats,
            min_overlap=_MIN_OVERLAP)

        corr_h = max(padded_target.shape[0], rotated.shape[0])
        corr_w = max(padded_target.shape[1], rotated.shape[1])
        corr_shape = (int(corr_h), int(corr_w))

        if padded_target.shape == corr_shape:
            target_for_corr = padded_target
        else:
            target_for_corr = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                padded_target,
                new_width=corr_shape[1],
                new_height=corr_shape[0],
                image_median=target_stats.median,
                image_stddev=target_stats.std,
                min_overlap=1.0,
                power_of_two=False)

        if rotated.shape == corr_shape:
            source_for_corr = rotated
        else:
            source_for_corr = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                rotated,
                new_width=corr_shape[1],
                new_height=corr_shape[0],
                image_median=working_stats.median,
                image_stddev=working_stats.std,
                min_overlap=1.0,
                power_of_two=False)

        if corr_shape not in fft_target_cache:
            fft_target_cache[corr_shape] = xp.fft.fft2(target_for_corr - target_mean)
        source_fft = xp.fft.fft2(source_for_corr - float(working_stats.mean))
        correlation_image = nornir_imageregistration.phasecorrelation.fft_phase_correlation(
            fft_target_cache[corr_shape], source_fft, delete_input=False, correlation_coefficient=.66)
        del source_fft
        record = stos_brute._peak_from_correlation_image(
            correlation_image, target_image.shape, working_shape, angle_deg, _MIN_OVERLAP, xp)
        return float(record.weight)

    return score


def _optimize_1d_axis(
        score,
        fixed_other: float,
        seed: float,
        *,
        vary_y: bool,
) -> tuple[float, float]:
    """Coarse grid + ternary along one axis; returns (best_value, best_weight)."""
    lo_bound = stos_brute._SCALE_REFINE_MIN
    hi_bound = stos_brute._SCALE_REFINE_MAX
    seed = float(np.clip(seed, lo_bound, hi_bound))
    candidates = sorted({
        float(np.clip(seed + d, lo_bound, hi_bound))
        for d in stos_brute._SCALE_REFINE_NARROW_DELTAS
    })

    def _eval(v: float) -> float:
        if vary_y:
            return score(v, fixed_other)
        return score(fixed_other, v)

    best_v = seed
    best_w = _eval(seed)
    for v in candidates:
        w = _eval(v)
        if w > best_w:
            best_w = w
            best_v = v

    lo = max(lo_bound, best_v - stos_brute._SCALE_REFINE_LOCAL_HALF_WIDTH)
    hi = min(hi_bound, best_v + stos_brute._SCALE_REFINE_LOCAL_HALF_WIDTH)
    for _ in range(stos_brute._SCALE_REFINE_TERNARY_ITERATIONS):
        third = (hi - lo) / 3.0
        if third < 1e-6:
            break
        m1 = lo + third
        m2 = hi - third
        if _eval(m1) < _eval(m2):
            lo = m1
        else:
            hi = m2
    refined = float((lo + hi) * 0.5)
    refined_w = _eval(refined)
    if refined_w > best_w:
        return refined, refined_w
    return best_v, best_w


def _refine_anisotropic(
        score,
        seed: float,
) -> tuple[float, float, float]:
    """Coordinate descent on (sy, sx); returns (sy, sx, weight)."""
    sy = float(seed)
    sx = float(seed)
    best_w = score(sy, sx)
    for _ in range(_DESCENT_PASSES):
        sy, best_w = _optimize_1d_axis(score, sx, sy, vary_y=True)
        sx, best_w = _optimize_1d_axis(score, sy, sx, vary_y=False)
    return sy, sx, best_w


def _assess_pair(
        pair: str,
        stos_path: Path,
        max_dim: int,
) -> PairResult:
    t0 = time.perf_counter()
    stos = StosFile.Load(str(stos_path), resolve_paths=True)
    angle_deg, manual_scalar = _manual_angle_and_scalar(stos)

    source_path = stos.MappedImageFullPath
    target_path = stos.ControlImageFullPath
    source_mask_path = stos.MappedMaskFullPath
    target_mask_path = stos.ControlMaskFullPath
    if not source_path or not target_path:
        raise FileNotFoundError(f'{pair}: missing image paths in {stos_path}')
    if not os.path.isfile(source_path) or not os.path.isfile(target_path):
        raise FileNotFoundError(
            f'{pair}: images not found: {source_path!r} / {target_path!r}')

    source_h = nornir_imageregistration.ImagePermutationHelper(
        source_path, source_mask_path if source_mask_path and os.path.isfile(source_mask_path) else None)
    target_h = nornir_imageregistration.ImagePermutationHelper(
        target_path, target_mask_path if target_mask_path and os.path.isfile(target_mask_path) else None)

    source_raw = source_h.ImageWithMaskAsNoise
    target_raw = target_h.ImageWithMaskAsNoise
    if hasattr(source_raw, 'get'):
        source_raw = source_raw.get()
    if hasattr(target_raw, 'get'):
        target_raw = target_raw.get()
    source = np.asarray(source_raw, dtype=np.float32)
    target = np.asarray(target_raw, dtype=np.float32)
    source, target, _, _ = _maybe_downscale(source, target, None, None, max_dim)
    source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
    target_stats = nornir_imageregistration.ImageStats.CalcStats(target)

    scale_radial, radial_peak_ratio = _estimate_b1_scale(
        source, target, source_stats, target_stats)
    seed = float(np.clip(manual_scalar, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
    scale_iso_refine = stos_brute._refine_scale_local(
        source, target, source_stats, target_stats,
        angle=angle_deg,
        initial_scale=seed,
        min_overlap=_MIN_OVERLAP,
        wide_search=False,
    )

    score = _make_aniso_scorer(source, target, source_stats, target_stats, angle_deg)
    weight_manual_iso = score(manual_scalar, manual_scalar)
    weight_iso_refine = score(scale_iso_refine, scale_iso_refine)
    aniso_seed = float(scale_iso_refine)
    sy, sx, weight_aniso = _refine_anisotropic(score, aniso_seed)

    geo = float(np.sqrt(sy * sx))
    lift_manual = weight_aniso / weight_manual_iso if weight_manual_iso > 1e-12 else float('nan')
    lift_iso = weight_aniso / weight_iso_refine if weight_iso_refine > 1e-12 else float('nan')
    abs_diff = abs(sy - sx)
    low_weight = max(weight_aniso, weight_iso_refine, weight_manual_iso) < _LOW_WEIGHT_FLAG

    return PairResult(
        pair=pair,
        stos_path=str(stos_path),
        manual_angle_deg=angle_deg,
        manual_scalar=manual_scalar,
        scale_radial=scale_radial,
        radial_peak_ratio=radial_peak_ratio,
        scale_iso_refine=scale_iso_refine,
        sy=sy,
        sx=sx,
        abs_sy_sx=abs_diff,
        geo_mean=geo,
        geo_mean_vs_manual=abs(geo - manual_scalar),
        iso_refine_vs_manual=abs(scale_iso_refine - manual_scalar),
        weight_manual_iso=weight_manual_iso,
        weight_iso_refine=weight_iso_refine,
        weight_aniso=weight_aniso,
        weight_lift_vs_manual=lift_manual,
        weight_lift_vs_iso_refine=lift_iso,
        low_weight=low_weight,
        elapsed_s=time.perf_counter() - t0,
    )


def _print_table(results: list[PairResult]) -> None:
    header = (
        f'{"pair":>8} {"man_s":>7} {"iso":>7} {"sy":>7} {"sx":>7} '
        f'{"|d|":>6} {"lift_iso":>8} {"w_aniso":>8} {"flag":>4}'
    )
    print(header)
    print('-' * len(header))
    for r in results:
        flag = 'LOW' if r.low_weight else ''
        print(
            f'{r.pair:>8} {r.manual_scalar:7.4f} {r.scale_iso_refine:7.4f} '
            f'{r.sy:7.4f} {r.sx:7.4f} {r.abs_sy_sx:6.4f} '
            f'{r.weight_lift_vs_iso_refine:8.3f} {r.weight_aniso:8.3f} {flag:>4}'
        )


def _go_nogo_summary(results: list[PairResult]) -> dict:
    """Apply plan decision rule; return summary dict and print recommendation."""
    usable = [r for r in results if not r.low_weight]
    n = len(results)
    n_usable = len(usable)
    if n_usable == 0:
        usable = results

    abs_diffs = np.array([r.abs_sy_sx for r in usable], dtype=float)
    lifts = np.array([r.weight_lift_vs_iso_refine for r in usable], dtype=float)
    lifts = lifts[np.isfinite(lifts)]

    frac_ge_002 = float(np.mean(abs_diffs >= _ANISO_ABS_THRESHOLD)) if len(abs_diffs) else 0.0
    frac_ge_005 = float(np.mean(abs_diffs >= 0.05)) if len(abs_diffs) else 0.0
    mean_abs = float(np.mean(abs_diffs)) if len(abs_diffs) else float('nan')
    median_abs = float(np.median(abs_diffs)) if len(abs_diffs) else float('nan')
    mean_lift = float(np.mean(lifts)) if len(lifts) else float('nan')
    frac_lift = float(np.mean(lifts >= _WEIGHT_LIFT_THRESHOLD)) if len(lifts) else 0.0

    # Several pairs with |sy-sx|>=0.02 AND consistent weight lift.
    several = max(2, int(np.ceil(0.3 * max(n_usable, 1))))
    n_aniso_and_lift = sum(
        1 for r in usable
        if r.abs_sy_sx >= _ANISO_ABS_THRESHOLD
        and np.isfinite(r.weight_lift_vs_iso_refine)
        and r.weight_lift_vs_iso_refine >= _WEIGHT_LIFT_THRESHOLD
    )
    recommend_affine = n_aniso_and_lift >= several

    summary = {
        'n_pairs': n,
        'n_usable_weight': n_usable,
        'n_low_weight': n - len([r for r in results if not r.low_weight]),
        'mean_abs_sy_sx': mean_abs,
        'median_abs_sy_sx': median_abs,
        'frac_abs_ge_0.02': frac_ge_002,
        'frac_abs_ge_0.05': frac_ge_005,
        'mean_weight_lift_vs_iso_refine': mean_lift,
        'frac_lift_ge_1.05': frac_lift,
        'n_aniso_and_lift': n_aniso_and_lift,
        'several_threshold': several,
        'recommend_affine_per_axis': recommend_affine,
        'decision_rule': (
            f'recommend Affine if >= {several} usable pairs have '
            f'|sy-sx|>={_ANISO_ABS_THRESHOLD} and weight_lift>={_WEIGHT_LIFT_THRESHOLD}'
        ),
    }

    print()
    print('=== Summary ===')
    print(f'pairs={n} usable(non-low-weight)={n_usable} low_weight={summary["n_low_weight"]}')
    print(f'|sy-sx| mean={mean_abs:.4f} median={median_abs:.4f} '
          f'frac>=0.02={frac_ge_002:.2f} frac>=0.05={frac_ge_005:.2f}')
    print(f'weight_lift vs iso_refine: mean={mean_lift:.3f} frac>={_WEIGHT_LIFT_THRESHOLD}={frac_lift:.2f}')
    print(f'pairs with aniso+lift: {n_aniso_and_lift} (need >={several})')
    if recommend_affine:
        print('GO: evidence supports independent sy/sx (Affine / per-axis export worth pursuing).')
    else:
        print('NO-GO: keep isotropic CenteredSimilarity; anisotropy not strongly supported '
              '(or peak weights too weak). Prefer mesh refine for residual mismatch.')
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual-dir', type=Path, default=None)
    parser.add_argument(
        '--pairs', default=','.join(_DEFAULT_PAIRS),
        help='Comma-separated section pairs (default: 14 Manual calibration pairs)')
    parser.add_argument('--max-dim', type=int, default=1024)
    parser.add_argument('--backend', choices=('numpy', 'cupy'), default='numpy')
    parser.add_argument('--write-csv', type=Path, default=None)
    parser.add_argument('--write-json', type=Path, default=None)
    args = parser.parse_args()

    _set_backend(args.backend)
    manual_dir = _resolve_manual_dir(args.manual_dir)
    pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]

    results: list[PairResult] = []
    skipped: list[str] = []
    for pair in pairs:
        stos_path = manual_dir / _STOS_NAME.format(pair=pair)
        if not stos_path.is_file():
            print(f'SKIP {pair}: missing {stos_path}', file=sys.stderr)
            skipped.append(pair)
            continue
        try:
            print(f'Assessing {pair} ...', flush=True)
            result = _assess_pair(pair, stos_path, args.max_dim)
            results.append(result)
            print(
                f'  {pair}: sy={result.sy:.4f} sx={result.sx:.4f} '
                f'|d|={result.abs_sy_sx:.4f} lift={result.weight_lift_vs_iso_refine:.3f} '
                f'({result.elapsed_s:.1f}s)',
                flush=True,
            )
        except Exception as exc:
            print(f'SKIP {pair}: {exc}', file=sys.stderr)
            skipped.append(pair)

    if not results:
        print('No pairs assessed.', file=sys.stderr)
        return 1

    _print_table(results)
    summary = _go_nogo_summary(results)
    summary['skipped'] = skipped
    summary['manual_dir'] = str(manual_dir)

    rows = [asdict(r) for r in results]
    if args.write_csv:
        args.write_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.write_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote CSV {args.write_csv}')
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {'summary': summary, 'pairs': rows}
        args.write_json.write_text(json.dumps(payload, indent=2))
        print(f'Wrote JSON {args.write_json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
