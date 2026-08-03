"""
Slice-to-slice rigid registration (angle, isotropic scale, translation).

Scale search bounds are calibrated from RPC3 TEM manual registrations (StosBrute64
Manual, Blob @ downsample 64). Thirteen ``CenteredSimilarity2DTransform`` pairs
(68-69, 69-70, 71-72, 72-73, 73-74, 75-76, 76-77, 78-79, 80-81, 82-84, 84-85,
86-87, 89-90) under ``RPC3/TEM/StosBrute64/Manual/*_ctrl-TEM_Blob_map-TEM_Blob.stos``.

Measured scale factor (scalar):
  mean 1.0054, sample std 0.0717, min 0.9246, max 1.1016, median 1.0416
  mode ~0.92 (4 pairs) and expand cluster ~1.04-1.10

Measured |percent change| from unity (|scale - 1| x 100):
  mean 6.6%, sample std 2.0%, min 3.9%, max 10.2%, median 7.4%, mode 7.5%

Blind search grids use min/max scalar +/- ~2% margin (0.90-1.12). Metadata or
``initial_scale_hint`` narrows refinement around the hinted residual.

Scale is estimated with a decoupled log-polar warp of the DoG |FFT| magnitude spectrum
(Reddy & Chatterji 1996, Phase B1); the angle half-plane estimates rotation only.

Created on Oct 4, 2012

@author: u0490822
"""
import multiprocessing
import multiprocessing.sharedctypes
from time import sleep
import math
import numpy as np
from numpy.typing import NDArray
from typing import Sequence, AbstractSet, Optional
import logging
import skimage
import skimage.registration
import skimage.transform
import skimage.filters
from dataclasses import dataclass

from nornir_imageregistration import AlignmentRecord, IgnoreUnderflow
import nornir_imageregistration.phasecorrelation
from nornir_imageregistration.settings import StosBruteSettings, AngleSearchRange, SliceToSliceMethod
from nornir_imageregistration.nornir_image_types import ImageLike

# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp
    import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

import nornir_imageregistration
import nornir_shared.mathhelper
import nornir_pools
from nornir_imageregistration.hann_window_cache import HannWindowCache


def _normalize_angle_degrees(angle: float) -> float:
    a = float(angle)
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def _rotated_aabb_shape(height: int, width: int, angle_deg: float) -> tuple[int, int]:
    """Axis-aligned bounding box after a reshape=True rotation (SciPy/CuPyX ndimage).

    For a rectangle of size ``(height, width)`` rotated by *angle_deg* degrees,
    returns ``(new_height, new_width)`` of the minimal axis-aligned box that
    contains the rotated rectangle. Matches the geometric model used by
    ``scipy.ndimage.rotate(..., reshape=True)``.
    """
    rad = math.radians(float(angle_deg))
    c = abs(math.cos(rad))
    s = abs(math.sin(rad))
    new_h = height * c + width * s
    new_w = height * s + width * c
    # SciPy ceils the projected extents; add a 1-pixel safety margin so
    # pad_and_rotate_image(desired_shape=...) never has to grow past the plan.
    return (int(math.ceil(new_h)) + 1, int(math.ceil(new_w)) + 1)


def _fixed_correlation_shape(
        target_shape: tuple[int, int] | NDArray | Sequence[int],
        source_shape: tuple[int, int] | NDArray | Sequence[int],
        angles: Sequence[float] | AbstractSet[float] | NDArray,
        min_overlap: float) -> tuple[int, int]:
    """Power-of-two correlation frame covering target, source, and all rotated AABBs.

    Takes the element-wise max of the unrotated target/source shapes and the
    rotated source AABB for every angle in *angles*, then rounds each dimension
    up with ``NearestPowerOfTwoWithOverlap``.
    """
    th, tw = int(target_shape[0]), int(target_shape[1])
    sh, sw = int(source_shape[0]), int(source_shape[1])
    max_h = max(th, sh)
    max_w = max(tw, sw)
    for angle in angles:
        rh, rw = _rotated_aabb_shape(sh, sw, float(angle))
        if rh > max_h:
            max_h = rh
        if rw > max_w:
            max_w = rw
    out_h = int(nornir_imageregistration.NearestPowerOfTwoWithOverlap(max_h, min_overlap))
    out_w = int(nornir_imageregistration.NearestPowerOfTwoWithOverlap(max_w, min_overlap))
    return (out_h, out_w)


# Hybrid ambiguous-fallback tuning (calibrated against fixed ±20° at median ambiguous pairs).
_FALLBACK_GAMMA = 2.0
_FALLBACK_MAX_HALF_WIDTH = 45.0
_FALLBACK_WIDEN_UNCERTAINTY_SCALE = 2.0
_FALLBACK_WEIGHT_RATIO_FLOOR = 0.85

# Log-polar warp tuning (Phase A2–A4). Angle-only log-polar; scale uses radial FFT (B1).
_LOGPOLAR_WARP_ORDER = 0
_LOGPOLAR_RADIUS_DIVISOR = 4
# B1 scale: log-polar warp of DoG |FFT| magnitude (radius divisor 2 vs angle 4).
_RADIAL_FFT_RADIUS_DIVISOR = 2

# Match Pyre Operations→Log Polar (Fast): StosBruteSettings.min_overlap + LimitImageSize=818.
# AlignSections / buildmanager LogPolar should use these so GPU container matches desktop Pyre.
LOGPOLAR_PIPELINE_MIN_OVERLAP = 0.75
LOGPOLAR_PIPELINE_LARGEST_DIMENSION = 818

# Accept flipud only when final ScoreOneAngle weight beats upright by this factor.
# Early LogPolar peak_strength can prefer a wrong flip (e.g. RPC3 770–769).
_FLIP_FINAL_WEIGHT_MARGIN = 1.05

# RPC3 manual corpus (see module docstring). Reference-only; search bounds derived below.
_RPC3_MANUAL_SCALE_MEAN = 1.0054
_RPC3_MANUAL_SCALE_STD = 0.0717
_RPC3_MANUAL_SCALE_MIN = 0.9246
_RPC3_MANUAL_SCALE_MAX = 1.1016
_RPC3_MANUAL_SCALE_MEDIAN = 1.0416
_RPC3_MANUAL_ABS_PCT_CHANGE_MEAN = 6.6
_RPC3_MANUAL_ABS_PCT_CHANGE_STD = 2.0
_RPC3_MANUAL_ABS_PCT_CHANGE_MIN = 3.9
_RPC3_MANUAL_ABS_PCT_CHANGE_MAX = 10.2
_RPC3_MANUAL_ABS_PCT_CHANGE_MEDIAN = 7.4
_RPC3_MANUAL_ABS_PCT_CHANGE_MODE = 7.5
# Margin beyond observed min/max scalar for blind search (~2%).
_RPC3_SCALE_SEARCH_MARGIN = 0.02

# 1D scale search at fixed angle (ScoreOneAngle weight), envelope from RPC3 manual stats.
_SCALE_REFINE_MIN = _RPC3_MANUAL_SCALE_MIN - _RPC3_SCALE_SEARCH_MARGIN  # 0.9046 -> 0.90
_SCALE_REFINE_MAX = _RPC3_MANUAL_SCALE_MAX + _RPC3_SCALE_SEARCH_MARGIN  # 1.1216 -> 1.12
_SCALE_REFINE_MIN = round(_SCALE_REFINE_MIN, 2)
_SCALE_REFINE_MAX = round(_SCALE_REFINE_MAX, 2)
_SCALE_REFINE_COARSE_DELTAS: tuple[float, ...] = (
    -0.08, -0.06, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.06, 0.08,
)
_SCALE_REFINE_NARROW_DELTAS: tuple[float, ...] = (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04)
_SCALE_REFINE_TISSUE_GRID: tuple[float, ...] = tuple(
    float(s) for s in np.geomspace(_SCALE_REFINE_MIN, _SCALE_REFINE_MAX, 11)
)
_SCALE_REFINE_LOCAL_HALF_WIDTH = 0.02
_SCALE_REFINE_TERNARY_ITERATIONS = 14
# Minimum ScoreOneAngle weight gain to accept a wide-grid scale far from the seed.
_SCALE_REFINE_WIDE_MIN_WEIGHT_RATIO = 1.12
_SCALE_REFINE_WIDE_MAX_SEED_DELTA = 0.03


def _logpolar_warp_radius(max_dimension: int) -> int:
    """FFT magnitude radius for angle-only ``warp_polar``."""
    return max(8, max_dimension // _LOGPOLAR_RADIUS_DIVISOR)


def _radial_fft_max_radius(max_dimension: int) -> int:
    """Outer radius for B1 radial log-magnitude annuli."""
    return max(8, max_dimension // _RADIAL_FFT_RADIUS_DIVISOR)


def _logpolar_fft_magnitude(
        padded_image: NDArray[np.floating],
        window: NDArray[np.floating],
        *,
        use_dog: bool) -> NDArray[np.floating]:
    """Magnitude spectrum for log-polar warping (A8: DoG for angle, raw for scale)."""
    if use_dog:
        filtered = skimage.filters.difference_of_gaussians(padded_image, low_sigma=4, high_sigma=20)
    else:
        filtered = padded_image.astype(np.float32, copy=False)
    freq = np.fft.fft2(filtered * window)
    return np.abs(np.fft.fftshift(freq))


def _parabolic_peak_index(values: NDArray[np.floating], peak_index: int) -> float:
    """Refine an integer peak index with three-point parabolic interpolation (A5)."""
    if peak_index <= 0 or peak_index >= len(values) - 1:
        return float(peak_index)
    ym = float(values[peak_index - 1])
    y0 = float(values[peak_index])
    yp = float(values[peak_index + 1])
    denom = ym - 2.0 * y0 + yp
    if abs(denom) < 1e-12:
        return float(peak_index)
    delta = 0.5 * (ym - yp) / denom
    return float(peak_index) + float(np.clip(delta, -0.5, 0.5))


def _estimate_scale_radial_fft(
        target_magnitude: NDArray[np.floating],
        source_magnitude: NDArray[np.floating],
        max_radius: int,
        output_shape: tuple[int, int]) -> tuple[float, float]:
    """Estimate isotropic scale from decoupled log-polar |FFT| correlation (B1).

    Full-plane ``warp_polar`` of the DoG magnitude spectrum (rotation decoupled from
    the angle half-plane). Column shift along log-radius gives scale
    (Reddy & Chatterji 1996). Returns ``(scale, peak_ratio)``.
    """
    target_log_polar = skimage.transform.warp_polar(
        target_magnitude,
        radius=max_radius,
        output_shape=output_shape,
        scaling='log',
        order=_LOGPOLAR_WARP_ORDER,
    )
    source_log_polar = skimage.transform.warp_polar(
        source_magnitude,
        radius=max_radius,
        output_shape=output_shape,
        scaling='log',
        order=_LOGPOLAR_WARP_ORDER,
    )
    phase_correlation = nornir_imageregistration.phasecorrelation.image_phase_correlation(
        target_log_polar, source_log_polar)
    phase_correlation_shifted = np.fft.fftshift(phase_correlation)
    peak_ratio = float(_correlation_peak_ratio(phase_correlation_shifted))
    try:
        peak_search = phase_correlation_shifted.astype(np.float32, copy=True)
        peak_search -= peak_search.min()
        peak_search /= peak_search.max()
    except FloatingPointError:
        return 1.0, 0.0
    peak = nornir_imageregistration.phasecorrelation.find_peak(peak_search)
    klog = output_shape[1] / float(np.log(max(max_radius, 2)))
    refined_col_offset = float(peak.scaled_offset[1])
    scale = float(np.exp(refined_col_offset / klog))
    return float(np.clip(scale, _SCALE_REFINE_MIN, _SCALE_REFINE_MAX)), peak_ratio


def _scale_at_final_angle(
        source_image: NDArray[np.floating],
        target_image: NDArray[np.floating],
        source_stats: nornir_imageregistration.ImageStats,
        target_stats: nornir_imageregistration.ImageStats,
        final_angle: float,
        scale_seed: float,
        min_overlap: float,
        *,
        wide_search: bool = False) -> float:
    """Re-estimate scale at the finalized angle (A1/A7); log-polar scale is only a seed."""
    return _refine_scale_local(
        source_image, target_image, source_stats, target_stats,
        final_angle, scale_seed, min_overlap, wide_search=wide_search)


def _smooth01(value: float, low: float, high: float) -> float:
    """Linear ramp low→0, high→1, clamped."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


@dataclass(frozen=True)
class LogPolarDiagnostics:
    angle_peak_ratio: float
    translation_peak_ratio: float
    strength_delta_ratio: float
    degrees_per_pixel: float
    peak_strength: float


def _logpolar_narrow_angle_range(
        center_angle: float,
        diagnostics: LogPolarDiagnostics,
        *,
        half_width_mult: float = 4.0,
) -> list[float]:
    """Narrow angle grid from log-polar row spacing (degrees per pixel in warp)."""
    half_width = max(half_width_mult * diagnostics.degrees_per_pixel, 1.0)
    step = max(0.2, float(diagnostics.degrees_per_pixel))
    center = _normalize_angle_degrees(center_angle)
    return sorted({
        _normalize_angle_degrees(center + offset)
        for offset in np.arange(-half_width, half_width + 1e-6, step)
    })


def _logpolar_confidence(d: LogPolarDiagnostics) -> float:
    return min(
        _smooth01(d.angle_peak_ratio, 1.0, 1.5),
        _smooth01(d.translation_peak_ratio, 1.0, 1.3),
        _smooth01(d.strength_delta_ratio, 0.0, 0.25),
    )


def _fallback_search_geometry(
        confidence: float,
        degrees_per_pixel: float,
        *,
        uncertainty_scale: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return half_width_deg, coarse_step_deg, fine_half_width_deg, fine_step_deg."""
    uncertainty = min(1.0, ((1.0 - confidence) ** _FALLBACK_GAMMA) * uncertainty_scale)
    min_hw = max(3.0 * degrees_per_pixel, 2.0)
    half_width = min(_FALLBACK_MAX_HALF_WIDTH, min_hw + (_FALLBACK_MAX_HALF_WIDTH - min_hw) * uncertainty)
    coarse_step = 0.2 + 1.8 * uncertainty
    fine_half_width = min(1.8, half_width * (0.1 + 0.2 * confidence))
    return half_width, coarse_step, fine_half_width, 0.2


def _angles_from_geometry(
        center: float,
        half_width: float,
        coarse_step: float,
        fine_half_width: float,
        fine_step: float,
) -> list[float]:
    center = _normalize_angle_degrees(center)
    coarse = {
        _normalize_angle_degrees(center + offset)
        for offset in np.arange(-half_width, half_width + 1e-6, coarse_step)
    }
    fine = {
        _normalize_angle_degrees(center + offset)
        for offset in np.arange(-fine_half_width, fine_half_width + 1e-6, fine_step)
    }
    return sorted(coarse | fine)


def _adaptive_fallback_angle_range(
        recovered_angle: float,
        diagnostics: LogPolarDiagnostics,
        *,
        uncertainty_scale: float = 1.0,
) -> list[float]:
    confidence = _logpolar_confidence(diagnostics)
    geometry = _fallback_search_geometry(
        confidence, diagnostics.degrees_per_pixel, uncertainty_scale=uncertainty_scale,
    )
    return _angles_from_geometry(recovered_angle, *geometry)


def _brute_fallback_needs_widen(
        brute: AlignmentRecord,
        logpolar: 'AngleScaleResult',
) -> bool:
    d = logpolar.diagnostics
    if d is None:
        return False
    weight_ratio = brute.weight / max(d.peak_strength, 1e-6)
    return weight_ratio < _FALLBACK_WEIGHT_RATIO_FLOOR


@dataclass
class HybridFallbackStats:
    confidence: float
    angle_count: int
    pass_used: int


_last_hybrid_fallback_stats: HybridFallbackStats | None = None


def get_last_hybrid_fallback_stats() -> HybridFallbackStats | None:
    """Stats from the most recent ambiguous log-polar fallback (for benchmarks)."""
    return _last_hybrid_fallback_stats


def _correlation_peak_ratio(arr: NDArray) -> float:
    arr_np = np.asarray(arr)
    if np.iscomplexobj(arr_np):
        arr_np = np.abs(arr_np)
    arr_np = np.asarray(arr_np, dtype=np.float32).ravel()
    if arr_np.size < 2:
        return 0.0
    finite = np.isfinite(arr_np)
    if not np.any(finite):
        return 0.0
    finite_values = arr_np[finite]
    if finite_values.size < 2:
        return 0.0
    top2 = np.partition(finite_values, -2)[-2:]
    second = float(max(min(top2), 1e-6))
    first = float(max(top2))
    return first / second


def _normalize_scale_xy(scale_factors: float | Sequence[float] | NDArray[np.floating] | None) -> tuple[float, float]:
    """Return (scale_y, scale_x) for warped→fixed pixel-size correction."""
    if scale_factors is None:
        return 1.0, 1.0
    arr = np.asarray(scale_factors, dtype=np.float64).ravel()
    if arr.size == 0:
        return 1.0, 1.0
    if arr.size == 1:
        s = float(arr[0])
        return s, s
    return float(arr[0]), float(arr[1])


def _isotropic_scale_from_xy(scale_y: float, scale_x: float) -> float:
    if np.isclose(scale_y, scale_x):
        return scale_y
    return float(np.sqrt(scale_y * scale_x))


def _scale_registration_image(image: NDArray[np.floating],
                              scale_y: float,
                              scale_x: float | None = None) -> NDArray[np.floating]:
    """Resample *image* for registration (warped/source → fixed scale)."""
    if scale_x is None:
        scale_x = scale_y
    if np.isclose(scale_y, 1.0) and np.isclose(scale_x, 1.0):
        return image
    if np.isclose(scale_y, scale_x):
        return nornir_imageregistration.ScaleImage(image, scale_y)
    sp = cupyx.scipy.get_array_module(image)
    order = 1 if min(scale_y, scale_x) < 1.0 else 3
    if sp is np:
        return sp.ndimage.zoom(image.astype(np.float32, copy=False), zoom=(scale_y, scale_x), order=order)
    return sp.ndimage.zoom(image, zoom=(scale_y, scale_x), order=order)


def _scaled_shape(shape: tuple[int, int], scale_y: float, scale_x: float) -> tuple[int, int]:
    return int(round(shape[0] * scale_y)), int(round(shape[1] * scale_x))


def _resolve_scale_search_params(
        settings: StosBruteSettings,
        metadata_scale_iso: float,
        logpolar_residual_scale: float | None = None,
) -> tuple[float | None, bool]:
    """Return (residual scale center for search/refine, force_scale_search)."""
    metadata_known = abs(metadata_scale_iso - 1.0) > 0.001
    meta = metadata_scale_iso if abs(metadata_scale_iso) > 1e-12 else 1.0

    if settings.initial_scale_hint is not None:
        return float(settings.initial_scale_hint) / meta, True

    if settings.estimated_scale_hint is not None:
        return float(settings.estimated_scale_hint) / meta, True

    if logpolar_residual_scale is not None:
        residual = float(logpolar_residual_scale)
        force = metadata_known or abs(residual - 1.0) > 0.002
        return residual, force

    if metadata_known:
        return 1.0, True

    return None, False


def _scale_search_candidates(metadata_applied: float,
                             hint: float | None,
                             *,
                             force_search: bool = False) -> list[float]:
    """Isotropic scale factors to try on source (after metadata pre-scale)."""
    center = 1.0 if hint is None else float(hint)
    metadata_known = abs(metadata_applied - 1.0) > 0.001
    if not force_search and abs(center - 1.0) < 0.002 and not metadata_known:
        return list(_SCALE_REFINE_TISSUE_GRID)
    deltas = _SCALE_REFINE_COARSE_DELTAS
    candidates = {
        float(np.clip(scale, _SCALE_REFINE_MIN, _SCALE_REFINE_MAX))
        for scale in (*_SCALE_REFINE_TISSUE_GRID, *(center + d for d in deltas))
    }
    if metadata_known:
        candidates.add(1.0)
    return sorted(candidates)


def _refine_scale_initial_center(settings: StosBruteSettings,
                                 metadata_scale_iso: float,
                                 logpolar_detected_scale: float,
                                 resolved_residual_hint: float | None) -> float:
    """Scale center for local refinement: user hint beats log-polar estimate."""
    if settings.initial_scale_hint is not None:
        meta = metadata_scale_iso if abs(metadata_scale_iso) > 1e-12 else 1.0
        return float(settings.initial_scale_hint) / meta
    if resolved_residual_hint is not None:
        return float(resolved_residual_hint)
    return float(logpolar_detected_scale)


def _refine_scale_local(source_image: NDArray[np.floating],
                        target_image: NDArray[np.floating],
                        source_stats: nornir_imageregistration.ImageStats,
                        target_stats: nornir_imageregistration.ImageStats,
                        angle: float,
                        initial_scale: float,
                        min_overlap: float,
                        *,
                        wide_search: bool = False) -> float:
    """Refine isotropic scale with coarse grid + local ternary search (A6)."""
    seed = float(np.clip(initial_scale, _SCALE_REFINE_MIN, _SCALE_REFINE_MAX))

    def _score(scale: float) -> float:
        record = ScoreOneAngle(
            target_original=target_image,
            source_original=source_image,
            target_image_shape=target_image.shape,
            source_image_shape=source_image.shape,
            angle=angle,
            target_stats=target_stats,
            source_stats=source_stats,
            min_overlap=min_overlap,
            source_scale=float(scale),
        )
        return float(record.weight)

    seed_weight = _score(seed)
    narrow_deltas = _SCALE_REFINE_NARROW_DELTAS
    inverse_seed = (1.0 / seed) if wide_search and seed > 1e-6 else None
    scale_sources_list: list[float] = [
        *(seed + delta for delta in narrow_deltas),
    ]
    if wide_search:
        scale_sources_list.extend(_SCALE_REFINE_TISSUE_GRID)
        if inverse_seed is not None:
            scale_sources_list.extend(inverse_seed + delta for delta in narrow_deltas)
    coarse_candidates = sorted({
        float(np.clip(scale, _SCALE_REFINE_MIN, _SCALE_REFINE_MAX))
        for scale in scale_sources_list
    })
    best_scale = seed
    best_weight = seed_weight
    for scale in coarse_candidates:
        weight = _score(scale)
        if weight > best_weight:
            best_weight = weight
            best_scale = float(scale)

    lo = max(_SCALE_REFINE_MIN, best_scale - _SCALE_REFINE_LOCAL_HALF_WIDTH)
    hi = min(_SCALE_REFINE_MAX, best_scale + _SCALE_REFINE_LOCAL_HALF_WIDTH)
    for _ in range(_SCALE_REFINE_TERNARY_ITERATIONS):
        third = (hi - lo) / 3.0
        if third < 1e-6:
            break
        m1 = lo + third
        m2 = hi - third
        if _score(m1) < _score(m2):
            lo = m1
        else:
            hi = m2
    refined = float((lo + hi) * 0.5)
    refined_weight = _score(refined)
    if refined_weight > best_weight:
        best_weight = refined_weight
        best_scale = refined
    if (
        wide_search
        and abs(best_scale - seed) > _SCALE_REFINE_WIDE_MAX_SEED_DELTA
        and best_weight < seed_weight * _SCALE_REFINE_WIDE_MIN_WEIGHT_RATIO
    ):
        return seed
    if (
        not wide_search
        and abs(best_scale - seed) > 0.02
        and best_weight < seed_weight * 1.35
    ):
        return seed
    return best_scale


@dataclass
class AngleScaleResult:
    angle: float
    scale: float
    weight: float
    translation: tuple[float, float]
    flippedud: bool = False
    ambiguous: bool = False
    diagnostics: LogPolarDiagnostics | None = None


def _coerce_to_source_module(x: NDArray, xp) -> NDArray:
    """Place *x* on array module *xp* (numpy or cupy), matching ``cupy.get_array_module``."""
    if cp.get_array_module(x) is xp:
        return xp.asarray(x)
    if xp is np:
        if hasattr(x, "get"):
            return x.get()  # type: ignore[attr-defined, union-attr]
        return np.asarray(x)
    return xp.asarray(np.asarray(x))


def _coerce_registration_image_pair(source_image: NDArray, target_image: NDArray) -> tuple[NDArray, NDArray]:
    """Align *target_image* to the same array module as *source_image* (no global CuPy upgrade)."""
    xp = cp.get_array_module(source_image)
    return _coerce_to_source_module(source_image, xp), _coerce_to_source_module(target_image, xp)


def rotate_image(image: NDArray,
                 angle: float,
                 image_stats: nornir_imageregistration.ImageStats) -> NDArray:
    """Rotates an image, filling empty space with noise that matches the image stats
    :return: The rotated image and the image stats, the original objects if rotation is 0 / image_stats was passed"""

    if angle == 0:
        return image

    xp = cp.get_array_module(image)
    xp_scipy = cupyx.scipy.get_array_module(image)
    rotate = xp_scipy.ndimage.rotate

    # im_target = cp.asarray(im_target) if use_cp and not isinstance(im_target, cp.ndarray) else im_target
    # im_source = cp.asarray(im_source) if use_cp  and not isinstance(im_source, cp.ndarray)  else im_source

    # gc.set_debug(gc.DEBUG_LEAK)
    if image_stats is None:
        image_stats = nornir_imageregistration.ImageStats.CalcStats(image_stats)

    # This confused me for years, but the implementation of rotate calls affine_transform with
    # the rotation matrix.  However the docs for affine_transform state it needs to be called
    # with the inverse transform.  Hence negating the angle here.
    with IgnoreUnderflow():
        if xp is not np:
            im_rotated = rotate(image, axes=(1, 0), angle=-angle, cval=np.nan)
        else:
            im_rotated = rotate(image.astype(np.float32, copy=False), axes=(1, 0), angle=-angle,
                                cval=np.nan).astype(image.dtype, copy=False)  # Numpy cannot rotate float16 images

    xp_out = cp.get_array_module(im_rotated)
    im_result_empty_entries = xp_out.isnan(im_rotated)
    n_bad = int(xp_out.sum(im_result_empty_entries))
    if n_bad:
        noise = image_stats.GenerateNoise(n_bad, dtype=image.dtype, xp=xp_out)  # type: ignore[arg-type]
        if cp.get_array_module(noise) is not xp_out:
            if xp_out is np:
                noise = noise.get() if hasattr(noise, "get") else np.asarray(noise).ravel()
            else:
                noise = xp_out.asarray(np.asarray(noise))
        im_rotated[im_result_empty_entries] = noise

    return im_rotated


def pad_and_rotate_image(image: NDArray,
                         angle: float,
                         image_stats: nornir_imageregistration.ImageStats,
                         desired_shape: tuple[int, int] | None = None,
                         min_overlap: float = 0.75,
                         original_shape: NDArray | tuple[int, int] | None = None,
                         power_of_two: bool = False,
                         ) -> NDArray:
    """
    Rotates and image and pads it to ensure it has the requested dimensions, filling empty space with noise that matches the image stats.
    :param image:
    :param desired_shape: The desired shape of the image after rotation
    :param image_stats:
    :param min_overlap:
    :param original_shape: If the input image has been previously padded, this is the original shape of the image
    :param power_of_two: If True, the image will be padded to the nearest power of two.  This may be largest than the desired_shape
    :return: The rotated image and the image stats, the original objects if rotation is 0 / image_stats was passed
    """

    if original_shape is None:
        orginal_shape = image.shape

    if desired_shape is None:
        desired_shape = (None, None)  # type: ignore[assignment]

    rotated_image = rotate_image(image, angle=angle, image_stats=image_stats) if angle != 0 else image

    # if desired_shape is not None and rotated_image.shape[0] > desired_shape[0] or rotated_image.shape[1] > desired_shape[1]:
    #    raise ValueError("Need to add support to pad_and_rotate_image for expanding the desired image size")

    if power_of_two:
        desired_shape = nornir_imageregistration.NearestPowerOfTwo(rotated_image.shape)  # type: ignore[assignment]

    padded_rotated_image = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(rotated_image,
                                                                                                     image_median=image_stats.median,
                                                                                                     image_stddev=image_stats.std,
                                                                                                     min_overlap=min_overlap,
                                                                                                     original_shape=original_shape,
                                                                                                     new_height=
                                                                                                     desired_shape[0],  # type: ignore[index]
                                                                                                     new_width=
                                                                                                     desired_shape[  # type: ignore[index]
                                                                                                         1])
    return padded_rotated_image


# from memory_profiler import profile
def SliceToSliceRigidRegistration(target_image: ImageLike,
                                  source_image: ImageLike,
                                  target_mask: ImageLike | None = None,
                                  source_mask: ImageLike | None = None,
                                  LargestDimension: int | None = None,
                                  AngleSearchRange: Sequence[float] | AbstractSet[float] | None = None,
                                  MinOverlap: float = 0.5,
                                  WarpedImageScaleFactors=None,
                                  SingleThread: bool = False,
                                  Cluster: bool = False,
                                  TestFlip: bool = True,
                                  estimate_angle: bool = True,
                                  method: SliceToSliceMethod = SliceToSliceMethod.LogPolar,
                                  initial_scale_hint: float | None = None) -> nornir_imageregistration.AlignmentRecord:
    """Given two images this function returns the rotation angle which best aligns them
       Largest dimension determines how large the images used for alignment should be.

       :param target_image: Control/reference image (target space)
       :param source_image: Mapped/moving image (source space)
       :param target_mask:
       :param source_mask:
       :param SingleThread:
       :param Cluster:
       :param TestFlip:
       :param estimate_angle: If true, run a log_polar registration and append the result to the angles to search
       :param int LargestDimension: The input images should be scaled so the largest image dimension is equal to this value, default is None
       :param float MinOverlap: The minimum amount of overlap we require in the images.  Higher values reduce false positives but may not register offset images
       :param float AngleSearchRange: A list of rotation angles to test.  Pass None for the default which is every two degrees
       :param float WarpedImageScaleFactors: Scale the source image input by this amount before attempting registration
       :param float initial_scale_hint: Total scale on source image (e.g. current transform) to seed scale search
       """
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    if AngleSearchRange is not None:
        if not isinstance(AngleSearchRange, set):
            AngleSearchRange = set(AngleSearchRange)  # type: ignore[assignment]
        # if isinstance(AngleSearchRange, np.ndarray):

        if 0 not in set(AngleSearchRange):  # type: ignore[arg-type]
            logger = logging.getLogger(__name__ + '.SliceToSliceRigidRegistration')
            logger.warning("AngleSearchRange should contain 0 degrees to ensure the best match is found")
    else: 
        AngleSearchRange = set(map(float, range(0, 358, 2)))

    SingleThread = True if use_cp else SingleThread

    source_image_data = nornir_imageregistration.ImagePermutationHelper(source_image, source_mask)
    target_image_data = nornir_imageregistration.ImagePermutationHelper(target_image, target_mask)

    if estimate_angle and method == SliceToSliceMethod.LogPolar:
        estimate_angle = False
        # raise ValueError("LogPolar method is redundant with setting estimate_angle to true")

    estimated_scale_hint = None
    if estimate_angle:
        # Estimate the angle and scale
        estimated_angle_best_match = _find_angle_and_scale_with_logpolar(
            source_image=source_image_data.ImageWithMaskAsNoise,
            target_image=target_image_data.ImageWithMaskAsNoise,
            source_stats=source_image_data.Stats,
            target_stats=target_image_data.Stats,
            min_overlap=MinOverlap)
        if abs(estimated_angle_best_match.angle) > 0.25:
            AngleSearchRange.add(estimated_angle_best_match.angle)  # type: ignore[union-attr]
        estimated_scale_hint = float(estimated_angle_best_match.scale)

    settings = StosBruteSettings(method=method,
                                 angles=AngleSearchRange,
                                 min_overlap=MinOverlap,
                                 source_image_scale_factors=WarpedImageScaleFactors,
                                 larget_dimension=LargestDimension,
                                 try_flipped=TestFlip,
                                 estimated_scale_hint=estimated_scale_hint,
                                 initial_scale_hint=initial_scale_hint)

    return SliceToSliceRigidRegistrationWithPreprocessedImages(source_image_data=source_image_data,
                                                               target_image_data=target_image_data,
                                                               settings=settings,
                                                               SingleThread=SingleThread,
                                                               Cluster=Cluster)


def NarrowAngleSearchRangeWithResult(angle_range: NDArray[np.floating],
                                     min_step_size: float,
                                     target_angle: float) -> set[float]:
    """
    Given a range of angles, returns a smaller search range around an estimated correct angle
    :param angle_range: The original search range of angles we want to narrow down
    :param min_step_size: Minimum difference between angles in the results
    :param target_angle: The angle previously estimated to be the best match
    :return: A narrower search range to refine the angle search in a future iteration
    """
    if len(angle_range) < 2:
        raise ValueError("Angle search range must contain at least two angles to be refined")

    sorted_angles = sorted(angle_range)
    iMatch = sorted_angles.index(target_angle)
    iBelow = iMatch - 1 if iMatch - 1 >= 0 else len(sorted_angles) - 1
    iAbove = iMatch + 1 if iMatch + 1 < len(sorted_angles) else 0
    below = sorted_angles[iMatch - 1] if iMatch - 1 >= 0 else sorted_angles[0] - np.abs(
        sorted_angles[1] - sorted_angles[0])
    above = sorted_angles[iMatch + 1] if iMatch + 1 < len(sorted_angles) else sorted_angles[
                                                                                  iMatch] + np.abs(
        sorted_angles[iMatch] - sorted_angles[iMatch - 1])
    refine_search_range = above - below
    nSteps = 20
    stepsize = refine_search_range / nSteps

    if stepsize < min_step_size:
        nSteps = int(refine_search_range / min_step_size)
        stepsize = refine_search_range / nSteps

    refined_angle_search_range = {(x * stepsize) + below for x in range(1, nSteps)}

    # Ensure we include the best match angle
    refined_angle_search_range.add(target_angle)
    return refined_angle_search_range


def SliceToSliceRigidRegistrationWithPreprocessedImages(
        source_image_data: nornir_imageregistration.ImagePermutationHelper,
        target_image_data: nornir_imageregistration.ImagePermutationHelper,
        settings: StosBruteSettings,
        SingleThread: bool = False,
        Cluster: bool = False) -> nornir_imageregistration.AlignmentRecord:
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    target_image = target_image_data.ImageWithMaskAsNoise
    source_image = source_image_data.ImageWithMaskAsNoise

    target_stats = target_image_data.Stats
    source_stats = source_image_data.Stats

    del target_image_data
    del source_image_data

    target_image = cp.asarray(target_image) if use_cp and not isinstance(target_image, cp.ndarray) else target_image
    source_image = cp.asarray(source_image) if use_cp and not isinstance(source_image, cp.ndarray) else source_image

    metadata_scale_y, metadata_scale_x = _normalize_scale_xy(settings.source_image_scale_factors)
    metadata_scale_iso = _isotropic_scale_from_xy(metadata_scale_y, metadata_scale_x)
    if settings.source_image_scaling_required:
        source_image = _scale_registration_image(source_image, metadata_scale_y, metadata_scale_x)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source_image)

    detected_scale = 1.0

    scalar = 1.0
    if settings.larget_dimension is not None:
        scalar = nornir_imageregistration.ScalarForMaxDimension(settings.larget_dimension,
                                                                [target_image.shape, source_image.shape])
        if scalar > 1.0:
            scalar = 1.0

    if scalar != 1.0:
        target_image = nornir_imageregistration.ScaleImage(target_image, scalar)
        source_image = nornir_imageregistration.ScaleImage(source_image, scalar)

    def _find_logpolar_with_fallback(candidate_source_image: NDArray[np.floating]) -> AngleScaleResult:
        global _last_hybrid_fallback_stats

        logpolar_result = _find_angle_and_scale_with_logpolar(source_image=candidate_source_image,
                                                              target_image=target_image,
                                                              source_stats=source_stats,
                                                              target_stats=target_stats,
                                                              min_overlap=settings.min_overlap)
        if not logpolar_result.ambiguous:
            _last_hybrid_fallback_stats = None
            return logpolar_result

        diagnostics = logpolar_result.diagnostics
        assert diagnostics is not None

        # Hybrid fallback stays on the active backend (CuPy ScoreManyAnglesGpu when enabled).
        # Weak-pair NumPy≠CuPy is underdetermination, not a host-scoring requirement.
        confidence = _logpolar_confidence(diagnostics)
        pass_used = 1
        fallback_angles = _adaptive_fallback_angle_range(logpolar_result.angle, diagnostics)
        angle_count = len(fallback_angles)

        brute_force_result = _find_best_angle(source_image=candidate_source_image,
                                              target_image=target_image,
                                              source_stats=source_stats,
                                              target_stats=target_stats,
                                              angle_range=fallback_angles,
                                              min_overlap=settings.min_overlap,
                                              SingleThread=SingleThread,
                                              use_cluster=Cluster)

        if _brute_fallback_needs_widen(brute_force_result, logpolar_result):
            widened_angles = _adaptive_fallback_angle_range(
                logpolar_result.angle,
                diagnostics,
                uncertainty_scale=_FALLBACK_WIDEN_UNCERTAINTY_SCALE,
            )
            pass_used = 2
            angle_count = len(widened_angles)
            widened_result = _find_best_angle(source_image=candidate_source_image,
                                              target_image=target_image,
                                              source_stats=source_stats,
                                              target_stats=target_stats,
                                              angle_range=widened_angles,
                                              min_overlap=settings.min_overlap,
                                              SingleThread=SingleThread,
                                              use_cluster=Cluster)
            if widened_result.weight > brute_force_result.weight:
                brute_force_result = widened_result

        if _brute_fallback_needs_widen(brute_force_result, logpolar_result):
            pass_used = 3
            angle_count = len(settings.angle_range)
            full_sweep_result = _find_best_angle(source_image=candidate_source_image,
                                                 target_image=target_image,
                                                 source_stats=source_stats,
                                                 target_stats=target_stats,
                                                 angle_range=settings.angle_range,
                                                 min_overlap=settings.min_overlap,
                                                 SingleThread=SingleThread,
                                                 use_cluster=Cluster)
            if full_sweep_result.weight > brute_force_result.weight:
                brute_force_result = full_sweep_result

        _last_hybrid_fallback_stats = HybridFallbackStats(
            confidence=confidence,
            angle_count=angle_count,
            pass_used=pass_used,
        )

        if nornir_imageregistration.in_debug_mode():
            logging.getLogger(__name__).debug(
                'hybrid fallback: confidence=%.3f angles=%d pass=%d weight=%.4f',
                confidence, angle_count, pass_used, brute_force_result.weight,
            )

        final_angle = float(brute_force_result.angle)
        scale_seed = float(logpolar_result.scale)
        scale_at_angle = _scale_at_final_angle(
            candidate_source_image, target_image, source_stats, target_stats,
            final_angle, scale_seed, settings.min_overlap, wide_search=True)

        if nornir_imageregistration.in_debug_mode():
            logging.getLogger(__name__).debug(
                'hybrid fallback scale: logpolar_seed=%.6f refined=%.6f angle=%.4f',
                scale_seed, scale_at_angle, final_angle,
            )

        return AngleScaleResult(angle=final_angle,
                                scale=scale_at_angle,
                                weight=float(brute_force_result.weight),
                                translation=(float(brute_force_result.peak[0]), float(brute_force_result.peak[1])),
                                ambiguous=False,
                                diagnostics=diagnostics)

    resolved_scale_hint, force_scale_search = _resolve_scale_search_params(
        settings, metadata_scale_iso)
    if (settings.method == nornir_imageregistration.settings.SliceToSliceMethod.BruteForce
            and resolved_scale_hint is None and not force_scale_search):
        logpolar_probe = _find_angle_and_scale_with_logpolar(
            source_image=source_image,
            target_image=target_image,
            source_stats=source_stats,
            target_stats=target_stats,
            min_overlap=settings.min_overlap)
        resolved_scale_hint, force_scale_search = _resolve_scale_search_params(
            settings, metadata_scale_iso, float(logpolar_probe.scale))

    # Replace extrema with noise
    if settings.method == nornir_imageregistration.settings.SliceToSliceMethod.LogPolar:
        best_match = _find_logpolar_with_fallback(source_image)
        detected_scale = float(best_match.scale)
    else:
        best_match, detected_scale = _find_best_angle_with_scale_search(
            source_image=source_image,
            target_image=target_image,
            source_stats=source_stats,
            target_stats=target_stats,
            angle_range=settings.angle_range,
            min_overlap=settings.min_overlap,
            metadata_applied=metadata_scale_iso,
            scale_hint=resolved_scale_hint,
            SingleThread=SingleThread,
            use_cluster=Cluster,
            force_search=force_scale_search,
        )

    def _finalize_logpolar_candidate(
            candidate_source: NDArray[np.floating],
            seed: AngleScaleResult,
            flipped_ud: bool) -> nornir_imageregistration.AlignmentRecord:
        """Narrow-angle/scale refine + ScoreOneAngle for one LogPolar orientation."""
        detected = float(seed.scale)
        final_angle = float(seed.angle)
        if (
            get_last_hybrid_fallback_stats() is None
            and seed.diagnostics is not None
            and seed.diagnostics.angle_peak_ratio < 1.35
        ):
            angle_range = _logpolar_narrow_angle_range(final_angle, seed.diagnostics)
            angle_refined = _find_best_angle(
                source_image=candidate_source,
                target_image=target_image,
                source_stats=source_stats,
                target_stats=target_stats,
                angle_range=angle_range,
                min_overlap=settings.min_overlap,
                SingleThread=SingleThread,
                source_scale=1.0,
            )
            if float(angle_refined.weight) > float(seed.weight):
                final_angle = float(angle_refined.angle)
        refine_initial = _refine_scale_initial_center(
            settings, metadata_scale_iso, detected, resolved_scale_hint)
        detected = _refine_scale_local(
            candidate_source, target_image, source_stats, target_stats,
            final_angle, refine_initial, settings.min_overlap,
            wide_search=False)
        translation_results = ScoreOneAngle(
            source_original=candidate_source,
            target_original=target_image,
            target_image_shape=target_image.shape,
            source_image_shape=candidate_source.shape,
            angle=final_angle,
            target_stats=target_stats,
            source_stats=source_stats,
            target_image_prepadded=False,
            min_overlap=settings.min_overlap,
            source_scale=detected)
        return nornir_imageregistration.AlignmentRecord(
            peak=translation_results.peak,
            weight=translation_results.weight,
            angle=final_angle,
            flipped_ud=flipped_ud,
            scale=metadata_scale_iso * detected)

    def _finalize_bruteforce_candidate(
            candidate_source: NDArray[np.floating],
            seed: nornir_imageregistration.AlignmentRecord,
            seed_scale: float,
            flipped_ud: bool) -> nornir_imageregistration.AlignmentRecord:
        """Angle/scale refine + ScoreOneAngle for one BruteForce orientation."""
        if not settings.angle_range_defined():
            refined = _find_best_angle(
                source_image=candidate_source, target_image=target_image,
                source_stats=source_stats, target_stats=target_stats,
                angle_range=[(x * 0.2 + seed.angle) for x in range(-9, 10)],
                min_overlap=settings.min_overlap, SingleThread=SingleThread)
        else:
            min_step_size = 0.25
            if len(settings.angle_range) > 2:
                refined_angle_search_range = NarrowAngleSearchRangeWithResult(
                    settings.angle_range, min_step_size, seed.angle)
                refined = _find_best_angle(
                    source_image=candidate_source, target_image=target_image,
                    source_stats=source_stats, target_stats=target_stats,
                    angle_range=np.array(list(refined_angle_search_range), float),
                    min_overlap=settings.min_overlap, SingleThread=SingleThread)
            else:
                refined = seed

        refine_initial = _refine_scale_initial_center(
            settings, metadata_scale_iso, seed_scale, resolved_scale_hint)
        use_wide_scale_search = (
            (force_scale_search or resolved_scale_hint is None)
            and settings.initial_scale_hint is None)
        detected = _refine_scale_local(
            candidate_source, target_image, source_stats, target_stats,
            float(refined.angle), refine_initial, settings.min_overlap,
            wide_search=use_wide_scale_search)
        translation_results = ScoreOneAngle(
            source_original=candidate_source,
            target_original=target_image,
            target_image_shape=target_image.shape,
            source_image_shape=candidate_source.shape,
            angle=float(refined.angle),
            target_stats=target_stats,
            source_stats=source_stats,
            target_image_prepadded=False,
            min_overlap=settings.min_overlap,
            source_scale=detected)
        return nornir_imageregistration.AlignmentRecord(
            peak=translation_results.peak,
            weight=translation_results.weight,
            angle=float(refined.angle),
            flipped_ud=flipped_ud,
            scale=metadata_scale_iso * detected)

    upright_source = source_image
    upright_seed = best_match
    upright_scale = float(detected_scale)

    # Finalize upright before any flipped LogPolar call so hybrid-fallback
    # globals still describe the upright seed.
    if settings.method == nornir_imageregistration.settings.SliceToSliceMethod.LogPolar:
        upright_final = _finalize_logpolar_candidate(upright_source, upright_seed, False)
    else:
        upright_final = _finalize_bruteforce_candidate(
            upright_source, upright_seed, upright_scale, False)  # type: ignore[arg-type]

    best_refined_match = upright_final
    if settings.try_flipped:
        _xp_img = cp.get_array_module(source_image)
        flipped_source = _xp_img.flipud(source_image)
        if settings.method == nornir_imageregistration.settings.SliceToSliceMethod.LogPolar:
            flipped_seed = _find_logpolar_with_fallback(flipped_source)
            flipped_final = _finalize_logpolar_candidate(flipped_source, flipped_seed, True)
        else:
            flipped_seed, flipped_scale = _find_best_angle_with_scale_search(
                source_image=flipped_source,
                target_image=target_image,
                source_stats=source_stats,
                target_stats=target_stats,
                angle_range=settings.angle_range,
                min_overlap=settings.min_overlap,
                metadata_applied=metadata_scale_iso,
                scale_hint=resolved_scale_hint,
                SingleThread=SingleThread,
                use_cluster=Cluster,
                force_search=force_scale_search,
            )
            flipped_final = _finalize_bruteforce_candidate(
                flipped_source, flipped_seed, flipped_scale, True)
        if float(flipped_final.weight) > float(upright_final.weight) * _FLIP_FINAL_WEIGHT_MARGIN:
            best_refined_match = flipped_final

    is_flipped = bool(best_refined_match.flippedud)

    if scalar != 1.0:
        AdjustedPeak = (best_refined_match.peak[0] * (1 / scalar), best_refined_match.peak[1] * (1 / scalar))  # type: ignore[union-attr]
        best_refined_match = nornir_imageregistration.AlignmentRecord(
            AdjustedPeak,
            best_refined_match.weight,
            best_refined_match.angle,
            is_flipped,
            scale=best_refined_match.scale)

    return best_refined_match  # type: ignore[return-value]


def _peak_from_correlation_image(
        correlation_image: NDArray,
        target_image_shape: tuple[int, int],
        source_image_shape: tuple[int, int],
        angle: float,
        min_overlap: float,
        xp,
        *,
        already_shifted: bool = False) -> nornir_imageregistration.AlignmentRecord:
    """Overlap mask + find_peak → AlignmentRecord.

    When *already_shifted* is False (default), applies fftshift and subtracts the
    minimum so callers that still hold a pre-shift buffer (assessment scripts)
    need not change. Prefer shifting in the caller and passing
    ``already_shifted=True`` so the pre-shift array can be deleted first.
    """
    if not already_shifted:
        xp_scipy = cupyx.scipy.get_array_module(correlation_image)
        correlation_image = xp_scipy.fft.fftshift(correlation_image)
        try:
            correlation_image -= correlation_image.min()
        except FloatingPointError as e:
            print(f"Floating point error: {e} for {correlation_image.min()} or {correlation_image.max()}")
            return nornir_imageregistration.AlignmentRecord((0, 0), 0, angle)

    overlap_mask = nornir_imageregistration.overlapmasking.GetOverlapMaskOnDevice(
        target_image_shape, source_image_shape, correlation_image.shape, min_overlap,
        MaxOverlap=1.0, xp=xp)

    peak_result = nornir_imageregistration.phasecorrelation.find_peak(
        correlation_image, overlap_mask, allow_in_place=True)
    del overlap_mask
    del correlation_image
    return nornir_imageregistration.AlignmentRecord(
        peak_result.scaled_offset,
        peak_result.peak_strength,
        angle,
        peak_ratio=float(peak_result.peak_ratio))


def _score_one_angle_core(
        im_target: NDArray,
        im_source: NDArray,
        target_image_shape: tuple[int, int],
        source_image_shape: tuple[int, int],
        angle: float,
        target_stats: nornir_imageregistration.ImageStats,
        source_stats: nornir_imageregistration.ImageStats,
        target_image_prepadded: bool,
        min_overlap: float,
        source_scale: float = 1.0,
        fixed_shape: tuple[int, int] | None = None,
        fft_target: NDArray | None = None) -> nornir_imageregistration.AlignmentRecord:
    """Score one angle using arrays already on the active computation backend.

    When *fixed_shape* is set (multi-angle sweeps), the rotated source is padded to
    that frame and the target is assumed already prepadded to the same size.
    Optional *fft_target* reuses a precomputed FFT of the padded target.
    """

    xp = cp.get_array_module(im_target)

    working_source = im_source
    working_source_shape = source_image_shape
    working_source_stats = source_stats
    if not np.isclose(source_scale, 1.0):
        working_source = _scale_registration_image(im_source, source_scale)
        working_source_shape = _scaled_shape(source_image_shape, source_scale, source_scale)
        working_source_stats = nornir_imageregistration.ImageStats.CalcStats(working_source)

    if fixed_shape is not None:
        rotated_padded_source = pad_and_rotate_image(
            image=working_source,
            angle=angle,
            image_stats=working_source_stats,
            desired_shape=fixed_shape,
            min_overlap=min_overlap)
        if not target_image_prepadded:
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                im_target,
                image_median=target_stats.median,
                image_stddev=target_stats.std,
                min_overlap=1.0,
                new_height=fixed_shape[0],
                new_width=fixed_shape[1],
                original_shape=target_image_shape)
        else:
            padded_target = im_target
        if tuple(int(s) for s in padded_target.shape) != tuple(fixed_shape):
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                padded_target,
                image_median=target_stats.median,
                image_stddev=target_stats.std,
                min_overlap=1.0,
                new_height=fixed_shape[0],
                new_width=fixed_shape[1])
        # If rotate under-estimated, pad_image may have kept a larger frame — grow to match.
        if rotated_padded_source.shape != padded_target.shape:
            target_height = max(padded_target.shape[0], rotated_padded_source.shape[0])
            target_width = max(padded_target.shape[1], rotated_padded_source.shape[1])
            if padded_target.shape != (target_height, target_width):
                padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                    padded_target,
                    new_width=target_width,
                    new_height=target_height,
                    image_median=target_stats.median,
                    image_stddev=target_stats.std,
                    min_overlap=1.0)
                fft_target = None  # cached FFT no longer matches
            if rotated_padded_source.shape != padded_target.shape:
                rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                    rotated_padded_source,
                    new_width=padded_target.shape[1],
                    new_height=padded_target.shape[0],
                    image_median=working_source_stats.median,
                    image_stddev=working_source_stats.std,
                    min_overlap=1.0)
        rotated_source = rotated_padded_source
    else:
        rotated_source = pad_and_rotate_image(image=working_source,
                                              angle=angle,
                                              image_stats=working_source_stats,
                                              min_overlap=min_overlap)

        assert rotated_source.shape[0] > 0
        assert rotated_source.shape[1] > 0

        if not target_image_prepadded:
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                im_target,
                image_median=target_stats.median,
                image_stddev=target_stats.std,
                min_overlap=min_overlap,
                original_shape=target_image_shape)
        else:
            padded_target = im_target

        target_height = max(padded_target.shape[0], rotated_source.shape[0])
        target_width = max(padded_target.shape[1], rotated_source.shape[1])

        if not np.array_equal(im_target.shape, np.array((target_height, target_width))):
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                im_target,
                new_width=target_width,
                new_height=target_height,
                image_median=target_stats.median,
                image_stddev=target_stats.std,
                min_overlap=1.0)
            fft_target = None

        if np.array_equal(rotated_source.shape, np.array((target_height, target_width))):
            rotated_padded_source = rotated_source
        else:
            rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                rotated_source,
                new_width=target_width,
                new_height=target_height,
                image_median=working_source_stats.median,
                image_stddev=working_source_stats.std,
                min_overlap=1.0)

    assert np.array_equal(padded_target.shape, rotated_padded_source.shape)

    use_cached_fft = (
        fft_target is not None
        and tuple(int(s) for s in fft_target.shape) == tuple(int(s) for s in padded_target.shape)
    )
    correlation_image = nornir_imageregistration.phasecorrelation.image_phase_correlation(
        target_image=padded_target,
        source_image=rotated_padded_source,
        target_mean=target_stats.mean,
        source_mean=source_stats.mean,
        correlation_coefficient=.66,
        fft_target=fft_target if use_cached_fft else None)

    del rotated_source
    if padded_target is not im_target:
        del padded_target
    del rotated_padded_source
    if working_source is not im_source:
        del working_source

    # fftshift in this frame so the pre-shift correlation buffer can be dropped
    # before find_peak allocates threshold/label images.
    xp_scipy = cupyx.scipy.get_array_module(correlation_image)
    shifted = xp_scipy.fft.fftshift(correlation_image)
    del correlation_image
    try:
        shifted -= shifted.min()
    except FloatingPointError as e:
        print(f"Floating point error: {e} for {shifted.min()} or {shifted.max()}")
        return nornir_imageregistration.AlignmentRecord((0, 0), 0, angle)

    return _peak_from_correlation_image(
        shifted, target_image_shape, working_source_shape, angle, min_overlap, xp,
        already_shifted=True)


def _find_best_angle_at_scale(source_image: NDArray[np.floating],
                              target_image: NDArray[np.floating],
                              source_stats: nornir_imageregistration.ImageStats,
                              target_stats: nornir_imageregistration.ImageStats,
                              angle_range: NDArray[np.floating] | Sequence[float],
                              min_overlap: float,
                              source_scale: float,
                              SingleThread: bool,
                              use_cluster: bool) -> nornir_imageregistration.AlignmentRecord:
    if np.isclose(source_scale, 1.0):
        return _find_best_angle(source_image=source_image,
                                target_image=target_image,
                                source_stats=source_stats,
                                target_stats=target_stats,
                                angle_range=angle_range,
                                min_overlap=min_overlap,
                                SingleThread=SingleThread,
                                use_cluster=use_cluster,
                                source_scale=1.0)

    scaled_source = _scale_registration_image(source_image, source_scale)
    scaled_stats = nornir_imageregistration.ImageStats.CalcStats(scaled_source)
    return _find_best_angle(source_image=scaled_source,
                            target_image=target_image,
                            source_stats=scaled_stats,
                            target_stats=target_stats,
                            angle_range=angle_range,
                            min_overlap=min_overlap,
                            SingleThread=SingleThread,
                            use_cluster=use_cluster,
                            source_scale=1.0)


def _find_best_angle_with_scale_search(source_image: NDArray[np.floating],
                                       target_image: NDArray[np.floating],
                                       source_stats: nornir_imageregistration.ImageStats,
                                       target_stats: nornir_imageregistration.ImageStats,
                                       angle_range: NDArray[np.floating] | Sequence[float],
                                       min_overlap: float,
                                       metadata_applied: float,
                                       scale_hint: float | None,
                                       SingleThread: bool,
                                       use_cluster: bool,
                                       *,
                                       force_search: bool = False) -> tuple[nornir_imageregistration.AlignmentRecord, float]:
    candidates = _scale_search_candidates(metadata_applied, scale_hint, force_search=force_search)
    best_match: nornir_imageregistration.AlignmentRecord | None = None
    best_scale = 1.0
    for candidate in candidates:
        match = _find_best_angle_at_scale(source_image, target_image, source_stats, target_stats,
                                          angle_range, min_overlap, candidate,
                                          SingleThread, use_cluster)
        if best_match is None or match.weight > best_match.weight:
            best_match = match
            best_scale = candidate
    assert best_match is not None
    return best_match, best_scale


def ScoreManyAnglesGpu(target_original: NDArray,
                       source_original: NDArray,
                       target_image_shape: tuple[int, int],
                       source_image_shape: tuple[int, int],
                       angles: Sequence[float],
                       target_stats: nornir_imageregistration.ImageStats | None = None,
                       source_stats: nornir_imageregistration.ImageStats | None = None,
                       min_overlap: float = 0.75,
                       fixed_shape: tuple[int, int] | None = None) -> list[nornir_imageregistration.AlignmentRecord]:
    """Score multiple rotation angles with one source/target upload on GPU.

    When *fixed_shape* is provided, every angle pads to that frame and a single
    target FFT is reused across the sweep.
    """

    im_target = nornir_imageregistration.ImageParamToImageArray(
        target_original, dtype=nornir_imageregistration.default_image_dtype())
    im_source = nornir_imageregistration.ImageParamToImageArray(
        source_original, dtype=nornir_imageregistration.default_image_dtype())

    if source_stats is None:
        source_stats = nornir_imageregistration.ImageStats.CalcStats(im_source)

    if target_stats is None:
        target_stats = nornir_imageregistration.ImageStats.CalcStats(im_target)

    xp = cp.get_array_module(im_target)
    fft_target = None
    if fixed_shape is not None and tuple(int(s) for s in im_target.shape) == tuple(fixed_shape):
        fft_target = xp.fft.fft2(im_target - target_stats.mean)

    try:
        return [
            _score_one_angle_core(
                im_target,
                im_source,
                target_image_shape,
                source_image_shape,
                float(angle),
                target_stats,
                source_stats,
                target_image_prepadded=True,
                min_overlap=min_overlap,
                fixed_shape=fixed_shape,
                fft_target=fft_target,
            )
            for angle in angles
        ]
    finally:
        if fft_target is not None:
            del fft_target


def ScoreOneAngle(target_original: NDArray, source_original: NDArray,
                  target_image_shape: tuple[int, int], source_image_shape: tuple[int, int],
                  angle: float,
                  target_stats: nornir_imageregistration.ImageStats | None = None,
                  source_stats: nornir_imageregistration.ImageStats | None = None,
                  target_image_prepadded: bool = True,
                  min_overlap: float = 0.75,
                  source_scale: float = 1.0,
                  fixed_shape: tuple[int, int] | None = None,
                  fft_target: NDArray | None = None) -> nornir_imageregistration.AlignmentRecord:
    """Returns an alignment score for a fixed image and an image rotated at a specified angle"""

    # print(f'Scoring {angle} degrees')
    try:
        im_target = nornir_imageregistration.ImageParamToImageArray(target_original,
                                                                    dtype=nornir_imageregistration.default_image_dtype())
        im_source = nornir_imageregistration.ImageParamToImageArray(source_original,
                                                                    dtype=nornir_imageregistration.default_image_dtype())

        if source_stats is None:
            source_stats = nornir_imageregistration.ImageStats.CalcStats(im_source)

        if target_stats is None:
            target_stats = nornir_imageregistration.ImageStats.CalcStats(im_target)

        return _score_one_angle_core(
            im_target,
            im_source,
            target_image_shape,
            source_image_shape,
            angle,
            target_stats,
            source_stats,
            target_image_prepadded,
            min_overlap,
            source_scale=source_scale,
            fixed_shape=fixed_shape,
            fft_target=fft_target)
    finally:
        nornir_imageregistration.close_shared_memory(target_original)  # type: ignore[arg-type]
        nornir_imageregistration.close_shared_memory(source_original)  # type: ignore[arg-type]


def GetFixedAndWarpedImageStats(imFixed: NDArray[np.floating], imWarped: NDArray[np.floating]) -> tuple[
    nornir_imageregistration.ImageStats, nornir_imageregistration.ImageStats]:
    tpool = nornir_pools.GetGlobalThreadPool()

    fixedStatsTask = tpool.add_task('FixedStats', nornir_imageregistration.ImageStats.CalcStats, imFixed)
    warpedStats = nornir_imageregistration.ImageStats.CalcStats(imWarped)

    fixedStats = fixedStatsTask.wait_return()

    return fixedStats, warpedStats


def _find_angle_and_scale_with_logpolar(source_image: NDArray[np.floating],
                                        target_image: NDArray[np.floating],
                                        source_stats: nornir_imageregistration.ImageStats,
                                        target_stats: nornir_imageregistration.ImageStats,
                                        min_overlap: float = 0.5) -> AngleScaleResult:
    """This function uses the log polar technique to determine the scale and angle of the best alignment between two images"""
    # Intentional host boundary: skimage log-polar path is CPU-only. Coerce each image
    # independently (ROIs may be mixed CuPy/NumPy); then ScoreOneAngle can stay on the
    # process backend for phase correlation when CuPy is enabled.
    source_image = _coerce_to_source_module(source_image, np)
    target_image = _coerce_to_source_module(target_image, np)

    desired_height = int(nornir_imageregistration.NearestPowerOfTwo(max([source_image.shape[0], target_image.shape[0]])))
    desired_width = int(nornir_imageregistration.NearestPowerOfTwo(max([source_image.shape[1], target_image.shape[1]])))
    desired_shape = np.array([desired_height, desired_width], dtype=int)

    max_dimension = max([desired_height, desired_width])
    radius_angle = _logpolar_warp_radius(max_dimension)
    radius_radial = _radial_fft_max_radius(max_dimension)

    """Angle from log-polar half-plane; scale from decoupled radial log-FFT (B1)."""
    padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(target_image,
                                                                                              min_overlap=min_overlap,
                                                                                              image_median=target_stats.median,
                                                                                              image_stddev=target_stats.std,
                                                                                              new_height=desired_height,
                                                                                              new_width=desired_width)

    padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(source_image,
                                                                                              min_overlap=min_overlap,
                                                                                              image_median=source_stats.median,
                                                                                              image_stddev=source_stats.std,
                                                                                              new_height=desired_height,
                                                                                              new_width=desired_width)

    target_window = HannWindowCache.GetOrCreate(padded_target.shape)
    source_window = HannWindowCache.GetOrCreate(padded_source.shape)

    target_freq_shift_angle = _logpolar_fft_magnitude(padded_target, target_window, use_dog=True)
    source_freq_shift_angle = _logpolar_fft_magnitude(padded_source, source_window, use_dog=True)
    target_freq_shift_radial = _logpolar_fft_magnitude(padded_target, target_window, use_dog=True)
    source_freq_shift_radial = _logpolar_fft_magnitude(padded_source, source_window, use_dog=True)

    target_image_log_polar = skimage.transform.warp_polar(target_freq_shift_angle,
                                                          radius=radius_angle,
                                                          output_shape=desired_shape,
                                                          scaling='log',
                                                          order=_LOGPOLAR_WARP_ORDER)
    source_image_log_polar = skimage.transform.warp_polar(source_freq_shift_angle,
                                                          radius=radius_angle,
                                                          output_shape=desired_shape,
                                                          scaling='log',
                                                          order=_LOGPOLAR_WARP_ORDER)

    target_image_log_polar_left_half = target_image_log_polar[:target_image_log_polar.shape[0] // 2, :]
    source_image_log_polar_left_half = source_image_log_polar[:source_image_log_polar.shape[0] // 2, :]

    phase_correlation = nornir_imageregistration.phasecorrelation.image_phase_correlation(
        target_image_log_polar_left_half, source_image_log_polar_left_half)

    # shifts, error, phasediff = skimage.registration.phase_cross_correlation(
    #     source_image_log_polar_left_half, target_image_log_polar_left_half, upsample_factor=2, normalization=None
    # )

    # phase_correlation_shifted = phase_correlation
    phase_correlation_shifted = np.fft.fftshift(phase_correlation)
    angle_scale_peak_ratio = float(_correlation_peak_ratio(phase_correlation_shifted))
    try:
        peak_search = phase_correlation_shifted.astype(np.float32, copy=True)
        peak_search -= peak_search.min()
        peak_search /= peak_search.max()
    except FloatingPointError as e:
        print(f"Floating point error: {e} for {phase_correlation.min()} or {phase_correlation.max()}")
        record = AngleScaleResult(angle=0, scale=1.0, weight=0, translation=(0, 0))
        return record

    angle_scale_peak = nornir_imageregistration.phasecorrelation.find_peak(peak_search)
    refined_row_offset = float(angle_scale_peak.scaled_offset[0])

    degrees_per_pixel = 360 / desired_shape[0]
    recovered_angle = float(degrees_per_pixel * refined_row_offset)
    scale_seed, radial_peak_ratio = _estimate_scale_radial_fft(
        target_freq_shift_radial,
        source_freq_shift_radial,
        radius_radial,
        (int(desired_shape[0]), int(desired_shape[1])),
    )

    if nornir_imageregistration.in_debug_mode():
        print(
            f'radial_fft scale_seed={scale_seed} peak_ratio={radial_peak_ratio} '
            f'radius_radial={radius_radial} angle={recovered_angle:.4f}')

    # B1 scale is a seed only; do not pre-scale before angle / 180° disambiguation.
    registration_source = source_image.astype(np.float32)

    # Check whether the angle is correct or needs to be adjusted by 180 degrees, also collect the translation vector
    rotated_padded_source = pad_and_rotate_image(image=registration_source,
                                                 angle=recovered_angle,
                                                 image_stats=source_stats,
                                                 min_overlap=min_overlap,
                                                 desired_shape=[desired_height, desired_width],  # type: ignore[arg-type]
                                                 power_of_two=True)

    if not np.array_equal(rotated_padded_source.shape, padded_target.shape):
        # If the target image does not match the dimensions of the rotated source image, make the size equal
        rotated_desired_shape = nornir_shared.mathhelper.max_shape([rotated_padded_source.shape, padded_target.shape])  # type: ignore[arg-type]
        rotated_desired_height, rotated_desired_width = rotated_desired_shape
        padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(target_image,
                                                                                                  min_overlap=min_overlap,
                                                                                                  image_median=target_stats.median,
                                                                                                  image_stddev=target_stats.std,
                                                                                                  new_height=rotated_desired_height,
                                                                                                  new_width=rotated_desired_width)

        if not np.array_equal(rotated_padded_source.shape, rotated_desired_shape):
            # If the rotated source image does not match the dimensions of the target image, make the size equal
            # rotated_desired_shape = nornir_shared.mathhelper.max_shape([rotated_padded_source.shape, padded_target.shape])
            # rotated_desired_height, rotated_desired_width = rotated_desired_shape
            rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                rotated_padded_source,
                min_overlap=min_overlap,
                image_median=source_stats.median,
                image_stddev=source_stats.std,
                new_height=rotated_desired_height,
                new_width=rotated_desired_width)
        target_window = HannWindowCache.GetOrCreate(rotated_desired_shape)
        source_window = HannWindowCache.GetOrCreate(rotated_desired_shape)
    else:
        rotated_desired_height, rotated_desired_width = desired_shape

    fft_target_ref = np.fft.fft2(padded_target * target_window)
    fft_source_ref = np.fft.fft2(rotated_padded_source * source_window)

    original_correlation = nornir_imageregistration.fft_phase_correlation(fft_target_ref, fft_source_ref)  # type: ignore[arg-type]

    # rotated_source = sp.ndimage.rotate(source_image.astype(np.float32), -recovered_angle + 180, reshape=True)
    # rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(rotated_source,
    #                                                                                               MinOverlap=min_overlap,
    #                                                                                               ImageMedian=source_stats.median,
    #                                                                                               ImageStdDev=source_stats.std,
    #                                                                                               NewHeight=desired_height,
    #                                                                                               NewWidth=desired_width)

    rotated_padded_source = pad_and_rotate_image(image=registration_source,
                                                 angle=recovered_angle + 180,
                                                 image_stats=source_stats,
                                                 min_overlap=min_overlap,
                                                 desired_shape=[rotated_desired_height, rotated_desired_width])  # type: ignore[arg-type]

    rotated_source_freq = np.fft.fft2(rotated_padded_source * source_window)
    rotated_correlation = nornir_imageregistration.fft_phase_correlation(fft_target_ref, rotated_source_freq)  # type: ignore[arg-type]

    original_peak = nornir_imageregistration.phasecorrelation.find_peak(original_correlation)
    rotated_peak = nornir_imageregistration.phasecorrelation.find_peak(rotated_correlation)
    original_peak_ratio = float(_correlation_peak_ratio(original_correlation))
    rotated_peak_ratio = float(_correlation_peak_ratio(rotated_correlation))
    translation_peak_ratio = max(original_peak_ratio, rotated_peak_ratio)

    rotated_180 = False
    if original_peak.peak_strength >= rotated_peak.peak_strength:
        selected_peak = original_peak
    else:
        strength_ratio = rotated_peak.peak_strength / max(original_peak.peak_strength, 1e-6)
        # When log-polar finds a small angle, a marginally stronger +180° FFT peak is often spurious.
        if abs(recovered_angle) <= 45.0 and strength_ratio < 1.15:
            selected_peak = original_peak
        else:
            selected_peak = rotated_peak
            recovered_angle -= 180
            if recovered_angle < -180:
                recovered_angle += 360
            rotated_180 = True

    if nornir_imageregistration.in_debug_mode():
        print(
            f'{original_peak.peak_strength} vs {rotated_peak.peak_strength} @ recovered angle {recovered_angle} '
            f'scale_seed {scale_seed} {'rotated_180' if rotated_180 else ""}')

    strength_delta_ratio = abs(float(original_peak.peak_strength) - float(rotated_peak.peak_strength)) / max(
        max(float(original_peak.peak_strength), float(rotated_peak.peak_strength)), 1e-6
    )
    ambiguous = bool(
        angle_scale_peak_ratio < 1.2
        or (translation_peak_ratio < 1.12 and angle_scale_peak_ratio < 1.35)
        or (strength_delta_ratio < 0.12 and angle_scale_peak_ratio < 1.35)
    )

    diagnostics = LogPolarDiagnostics(
        angle_peak_ratio=angle_scale_peak_ratio,
        translation_peak_ratio=translation_peak_ratio,
        strength_delta_ratio=strength_delta_ratio,
        degrees_per_pixel=float(degrees_per_pixel),
        peak_strength=float(angle_scale_peak.peak_strength),
    )

    shift_scale = float(scale_seed)
    refine_threshold = _RPC3_MANUAL_ABS_PCT_CHANGE_MIN / 100.0
    if rotated_180 or abs(scale_seed - 1.0) > refine_threshold:
        shift_scale = _scale_at_final_angle(
            source_image, target_image, source_stats, target_stats,
            recovered_angle, scale_seed, min_overlap, wide_search=False)
        if nornir_imageregistration.in_debug_mode():
            print(
                f'B1 fixed-angle scale: seed={scale_seed:.6f} refined={shift_scale:.6f} '
                f'angle={recovered_angle:.4f}')
    elif nornir_imageregistration.in_debug_mode():
        print(
            f'B1 scale seed={scale_seed:.6f} angle={recovered_angle:.4f} '
            f'peak_ratio={radial_peak_ratio:.3f}')

    return AngleScaleResult(
        angle=recovered_angle,
        scale=shift_scale,
        weight=angle_scale_peak.peak_strength,
        translation=selected_peak.scaled_offset,
        ambiguous=ambiguous,
        diagnostics=diagnostics,
    )


def _find_best_angle(source_image: NDArray[np.floating],
                     target_image: NDArray[np.floating],
                     source_stats: nornir_imageregistration.ImageStats,
                     target_stats: nornir_imageregistration.ImageStats,
                     angle_range: NDArray[np.floating] | Sequence[float],
                     min_overlap: float = 0.5,
                     SingleThread: bool = False,
                     use_cluster: bool = False,
                     source_scale: float = 1.0) -> nornir_imageregistration.AlignmentRecord:
    """Find the best angle to align two images.  This function can be very memory intensive.
       Setting SingleThread=True makes debugging easier"""

    try:
        Debug = False
        pool = None
        use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

        # Temporarily disable until we have  cluster pool working again.  Leaving this on eliminates shared memory which is a big optimization
        use_cluster = False

        if len(angle_range) <= 1:
            SingleThread = True

        if not SingleThread:
            if nornir_imageregistration.in_debug_mode():
                pool = nornir_pools.GetGlobalSerialPool()
            elif use_cluster:
                pool = nornir_pools.GetGlobalClusterPool()
            else:
                pool = nornir_pools.GetGlobalMultithreadingPool()

        # Preallocate lists to store results of each angle
        AngleMatchValues = list()  # type:  list[AlignmentRecord | None]
        taskList = list()  # type:  list[nornir_pools.Task | None]

        #    MaxRotatedDimension = max([max(imFixed), max(imWarped)]) * 1.4143
        #    MinRotatedDimension = max(min(imFixed), min(imWarped))
        #
        #    SmallPaddedFixed = pad_image_for_phase_correlation(imFixed, MaxOffset=0.1)
        #    LargePaddedFixed = pad_image_for_phase_correlation(imFixed, MaxOffset=0.1)

        source_shape = source_image.shape
        target_shape = target_image.shape

        # Multi-angle sweeps use one fixed Po2 frame covering the max rotated AABB so
        # the target FFT can be reused and per-angle target re-pads are avoided.
        fixed_shape: tuple[int, int] | None = None
        if len(angle_range) > 1:
            fixed_shape = _fixed_correlation_shape(
                target_shape, source_shape, angle_range, min_overlap)
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                target_image,
                min_overlap=1.0,
                image_median=target_stats.median,
                image_stddev=target_stats.std,
                new_height=fixed_shape[0],
                new_width=fixed_shape[1],
                original_shape=target_shape)
        else:
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                target_image,
                min_overlap=min_overlap,
                image_median=target_stats.median,
                image_stddev=target_stats.std)

        # Create a shared read-only memory map for the Padded fixed image

        if not (use_cluster or SingleThread):
            # temp_padded_fixed_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(padded_target)
            # temp_shared_warp_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(imWarped)

            # temp_padded_fixed_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data
            # temp_shared_warp_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data

            shared_target_metadata, shared_padded_target = nornir_imageregistration.npArrayToSharedArray(padded_target)
            shared_source_metadata, shared_source = nornir_imageregistration.npArrayToSharedArray(source_image
                                                                                                  )
            # shared_padded_target = np.save(padded_target, )
        else:
            shared_target_metadata = None
            shared_source_metadata = None
            shared_padded_target = padded_target.astype(nornir_imageregistration.default_image_dtype(),
                                                        copy=False) if not use_cp else cp.array(padded_target,
                                                                                                nornir_imageregistration.default_image_dtype())
            shared_source = source_image.astype(nornir_imageregistration.default_image_dtype(),
                                                copy=False) if not use_cp else cp.array(source_image,
                                                                                        nornir_imageregistration.default_image_dtype())

        CheckTaskInterval = 16

        max_task_count = multiprocessing.cpu_count() * 1.5

        if use_cp and len(angle_range) > 1:
            AngleMatchValues = ScoreManyAnglesGpu(
                target_original=shared_padded_target,
                source_original=shared_source,
                target_image_shape=target_shape,
                source_image_shape=source_shape,
                angles=angle_range,
                target_stats=target_stats,
                source_stats=source_stats,
                min_overlap=min_overlap,
                fixed_shape=fixed_shape,
            )
        else:
            # Single-thread multi-angle: FFT the padded target once and reuse.
            shared_fft_target = None
            if SingleThread and fixed_shape is not None:
                xp_fft = cp.get_array_module(shared_padded_target)
                shared_fft_target = xp_fft.fft.fft2(shared_padded_target - target_stats.mean)

            for i, theta in enumerate(angle_range):
                if SingleThread:
                    record = ScoreOneAngle(target_original=shared_padded_target, source_original=shared_source,
                                           target_image_shape=target_shape, source_image_shape=source_shape,
                                           angle=theta,
                                           target_stats=target_stats, source_stats=source_stats,
                                           min_overlap=min_overlap,
                                           fixed_shape=fixed_shape,
                                           fft_target=shared_fft_target)
                    AngleMatchValues.append(record)
                elif use_cluster:
                    task = pool.add_task(str(theta), ScoreOneAngle,  # type: ignore[union-attr]
                                         target_original=shared_padded_target, source_original=shared_source,
                                         target_image_shape=target_shape, source_image_shape=source_shape,
                                         angle=theta,
                                         target_stats=target_stats, source_stats=source_stats,
                                         min_overlap=min_overlap,
                                         fixed_shape=fixed_shape)
                    taskList.append(task)
                else:
                    task = pool.add_task(str(theta), ScoreOneAngle,  # type: ignore[union-attr]
                                         target_original=shared_target_metadata, source_original=shared_source_metadata,
                                         target_image_shape=target_shape, source_image_shape=source_shape,
                                         angle=theta,
                                         target_stats=target_stats, source_stats=source_stats,
                                         min_overlap=min_overlap,
                                         fixed_shape=fixed_shape)
                    taskList.append(task)

                if not i % CheckTaskInterval == 0:
                    continue

                # I don't like this, but it lets me delete tasks before filling the queue which may save some memory.
                # No sense checking unless we've already filled the queue though
                if len(taskList) > max_task_count:
                    for iTask in range(len(taskList) - 1, -1, -1):
                        if taskList[iTask].iscompleted:  # type: ignore[union-attr]
                            record = taskList[iTask].wait_return()  # type: ignore[union-attr]
                            AngleMatchValues.append(record)
                            del taskList[iTask]

                # TestOneAngle(shared_padded_target, shared_source, angle, None, MinOverlap)

            # taskList.sort(key=tpool.Task.name)

            while len(taskList) > 0:
                for iTask in range(len(taskList) - 1, -1, -1):
                    if taskList[iTask].iscompleted:  # type: ignore[union-attr]
                        record = taskList[iTask].wait_return()  # type: ignore[union-attr]
                        AngleMatchValues.append(record)
                        del taskList[iTask]

                if len(taskList) > 0:
                    # Wait a bit before checking the task list
                    sleep(0.5)

            # print(str(record.angle) + ' = ' + str(record.peak) + ' weight: ' + str(record.weight) + '\n')

            # ShowGrayscale(NormCorrelationImage)

        # print(str(AngleMatchValues))

        # Delete the pool to ensure extra python threads do not stick around
        # if pool is not None:
        #    pool.shutdown()

        del padded_target

        BestMatch = max(AngleMatchValues, key=nornir_imageregistration.AlignmentRecord.WeightKey)  # type: ignore[arg-type]
        return BestMatch
    finally:

        if shared_target_metadata is not None:
            nornir_imageregistration.unlink_shared_memory(shared_target_metadata)
        if shared_source_metadata is not None:
            nornir_imageregistration.unlink_shared_memory(shared_source_metadata)

            # os.remove(temp_shared_warp_memmap.path)
            # os.remove(temp_padded_fixed_memmap.path)


def __ExecuteProfiler():
    SliceToSliceRigidRegistration('C:/Src/Git/nornir-testdata/Images/0162_ds32.png',
                                  'C:/Src/Git/nornir-testdata/Images/0164_ds32.png',
                                  AngleSearchRange=list(range(-175, -174, 1)),
                                  SingleThread=True)


if __name__ == '__main__':
    from nornir_shared import NearestPowerOfTwo, misc

    misc.RunWithProfiler("__ExecuteProfiler()", r"C:\Temp\StosBrute")
    # __ExecuteProfiler()
    pass

