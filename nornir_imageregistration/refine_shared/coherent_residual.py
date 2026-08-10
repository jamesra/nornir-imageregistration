"""Coherent residual and pathological global-FOV pose recovery for STOS refine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.failure_mode_stats import peak_direction_coherence
from nornir_imageregistration.refine_shared.peak_ratio_gates import PEAK_RATIO_MIN, finite_peak_ratio

# Always-on constants (no feature-flag env). Healthy pairs (~33% locks) never trigger.
LOCK_FRAC_TRIGGER = 0.05
COHERENCE_MIN = 0.85
MIN_UNIQUE_PEAKS = 50
MIN_UNIQUE_TRAVEL = 1.0
# Keep peaks within ~60° of the preliminary median direction (robust consensus).
INLIER_COS_MIN = 0.5
# Downsampled whole-FOV phase-correlation recovery when unique peaks are useless.
GLOBAL_FOV_MAX_DIM = 512
# After Track A / global FOV TranslateFixed, refuse sparse mesh rebuilds that
# would discard the dense translated control set (e.g. Grid16 240-241 ~14 pts).
MIN_MESH_FRAC_AFTER_RESIDUAL = 0.05
MIN_MESH_ABS_AFTER_RESIDUAL = 100


class _AlignmentRecordLike(Protocol):
    peak: NDArray[np.floating]
    peak_ratio: float | None


@dataclass(frozen=True)
class CoherentResidualTranslation:
    """Median unique-peak translation to compose into the STOS control transform."""

    translation: NDArray[np.float64]
    coherence: float
    n_unique: int
    n_inliers: int = 0


@dataclass(frozen=True)
class CoherentResidualDiagnosis:
    """Track A attempt result plus skip telemetry (always filled)."""

    result: CoherentResidualTranslation | None
    skip_reason: str | None
    n_unique: int
    n_inliers: int
    coherence: float


def _median_direction_inliers(
        peak_arr: NDArray[np.float64],
        *,
        min_travel: float,
        inlier_cos_min: float,
) -> NDArray[np.bool_]:
    """Return mask of peaks whose unit direction agrees with the median direction."""
    norms = np.linalg.norm(peak_arr, axis=1)
    keep_travel = norms >= float(min_travel)
    if not np.any(keep_travel):
        return np.zeros(peak_arr.shape[0], dtype=bool)

    unit = np.zeros_like(peak_arr)
    unit[keep_travel] = peak_arr[keep_travel] / norms[keep_travel, None]
    # Preliminary median peak among traveling unique cells, then its unit vector.
    med = np.median(peak_arr[keep_travel], axis=0)
    med_norm = float(np.linalg.norm(med))
    if med_norm < float(min_travel):
        # Fall back to mean unit direction when median is near zero.
        mean_u = unit[keep_travel].mean(axis=0)
        mean_n = float(np.linalg.norm(mean_u))
        if mean_n <= 0.0:
            return keep_travel
        ref = mean_u / mean_n
    else:
        ref = med / med_norm

    dots = unit @ ref
    return keep_travel & (dots >= float(inlier_cos_min))


def _collect_unique_traveling_peaks(
        records: Sequence[_AlignmentRecordLike],
        *,
        min_travel: float,
        peak_ratio_min: float,
) -> NDArray[np.float64]:
    """Stack unique traveling peaks as ``(N, 2)`` (possibly empty)."""
    peaks: list[NDArray[np.float64]] = []
    for record in records:
        ratio = finite_peak_ratio(record)
        if ratio is None or float(ratio) < float(peak_ratio_min):
            continue
        peak = np.asarray(record.peak, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(peak)) < float(min_travel):
            continue
        peaks.append(peak)
    if not peaks:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(peaks, dtype=np.float64)


def diagnose_coherent_residual_translation(
        records: Sequence[_AlignmentRecordLike],
        lock_fraction: float,
        *,
        lock_frac_trigger: float = LOCK_FRAC_TRIGGER,
        coherence_min: float = COHERENCE_MIN,
        min_unique: int = MIN_UNIQUE_PEAKS,
        min_travel: float = MIN_UNIQUE_TRAVEL,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        inlier_cos_min: float = INLIER_COS_MIN,
) -> CoherentResidualDiagnosis:
    """Diagnose Track A; always reports n_unique / n_inliers / coherence / skip_reason."""
    if float(lock_fraction) >= float(lock_frac_trigger):
        return CoherentResidualDiagnosis(
            result=None,
            skip_reason=f'lock_frac>={lock_frac_trigger}',
            n_unique=0,
            n_inliers=0,
            coherence=0.0,
        )
    if len(records) == 0:
        return CoherentResidualDiagnosis(
            result=None,
            skip_reason='no_records',
            n_unique=0,
            n_inliers=0,
            coherence=0.0,
        )

    peak_arr = _collect_unique_traveling_peaks(
        records, min_travel=min_travel, peak_ratio_min=peak_ratio_min)
    n_unique = int(peak_arr.shape[0])
    if n_unique < int(min_unique):
        return CoherentResidualDiagnosis(
            result=None,
            skip_reason=f'n_unique={n_unique}<{min_unique}',
            n_unique=n_unique,
            n_inliers=0,
            coherence=0.0,
        )

    inlier_mask = _median_direction_inliers(
        peak_arr, min_travel=min_travel, inlier_cos_min=inlier_cos_min)
    inliers = peak_arr[inlier_mask]
    n_inliers = int(inliers.shape[0])
    if n_inliers < int(min_unique):
        coh_partial = peak_direction_coherence(
            peak_arr[:, 0], peak_arr[:, 1], min_travel=min_travel) if n_unique else 0.0
        return CoherentResidualDiagnosis(
            result=None,
            skip_reason=f'n_inliers={n_inliers}<{min_unique}',
            n_unique=n_unique,
            n_inliers=n_inliers,
            coherence=float(coh_partial),
        )

    coherence = peak_direction_coherence(
        inliers[:, 0], inliers[:, 1], min_travel=min_travel)
    if coherence < float(coherence_min):
        return CoherentResidualDiagnosis(
            result=None,
            skip_reason=f'coherence={coherence:.3f}<{coherence_min}',
            n_unique=n_unique,
            n_inliers=n_inliers,
            coherence=float(coherence),
        )

    translation = np.median(inliers, axis=0).astype(np.float64, copy=False)
    result = CoherentResidualTranslation(
        translation=translation,
        coherence=float(coherence),
        n_unique=n_unique,
        n_inliers=n_inliers,
    )
    return CoherentResidualDiagnosis(
        result=result,
        skip_reason=None,
        n_unique=n_unique,
        n_inliers=n_inliers,
        coherence=float(coherence),
    )


def estimate_coherent_residual_translation(
        records: Sequence[_AlignmentRecordLike],
        lock_fraction: float,
        *,
        lock_frac_trigger: float = LOCK_FRAC_TRIGGER,
        coherence_min: float = COHERENCE_MIN,
        min_unique: int = MIN_UNIQUE_PEAKS,
        min_travel: float = MIN_UNIQUE_TRAVEL,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        inlier_cos_min: float = INLIER_COS_MIN,
) -> CoherentResidualTranslation | None:
    """Return a coherent residual translation when locks are scarce and unique peaks agree.

    Uses cells with known ``peak_ratio >= peak_ratio_min`` and travel ``>= min_travel``.
    Peaks opposing the preliminary median direction (unit-dot ``< inlier_cos_min``)
    are dropped before the coherence check so a dominant cluster is not vetoed by
    a minority of outliers (e.g. Grid16 240-241).

    Returns ``None`` when the lock fraction is already healthy, too few inlier
    unique peaks exist, or inlier peak directions are not coherent.
    """
    return diagnose_coherent_residual_translation(
        records,
        lock_fraction,
        lock_frac_trigger=lock_frac_trigger,
        coherence_min=coherence_min,
        min_unique=min_unique,
        min_travel=min_travel,
        peak_ratio_min=peak_ratio_min,
        inlier_cos_min=inlier_cos_min,
    ).result


def should_attempt_global_fov_recovery(
        diagnosis: CoherentResidualDiagnosis,
        lock_fraction: float,
        *,
        lock_frac_trigger: float = LOCK_FRAC_TRIGGER,
        min_unique: int = MIN_UNIQUE_PEAKS,
        coherence_min: float = COHERENCE_MIN,
) -> bool:
    """True when locks are scarce and Track A unique evidence is insufficient/incoherent.

    Requires at least one unique cell peak. When every local peak is rejected
    (``n_unique == 0``), whole-FOV phase correlation must not invent a rigid
    translation — there is no cell-level evidence for a residual.
    """
    if float(lock_fraction) >= float(lock_frac_trigger):
        return False
    if diagnosis.result is not None:
        return False
    # No unique cell peaks ⇒ do not TranslateFixed via Track B.
    if int(diagnosis.n_unique) <= 0:
        return False
    # Healthy lock fraction already excluded; attempt when unique soup is too small
    # or directions do not form a coherent inlier cluster (wrap-like opposites).
    if diagnosis.n_unique < int(min_unique):
        return True
    if diagnosis.n_inliers < int(min_unique):
        return True
    if diagnosis.coherence < float(coherence_min):
        return True
    return False


def should_preserve_post_residual_transform(
        *,
        residual_applied: bool,
        n_mesh: int,
        n_grid: int,
        n_locks: int,
        lock_frac_trigger: float = LOCK_FRAC_TRIGGER,
        min_mesh_frac: float = MIN_MESH_FRAC_AFTER_RESIDUAL,
        min_mesh_abs: int = MIN_MESH_ABS_AFTER_RESIDUAL,
) -> bool:
    """True when a post-residual sparse mesh would discard TranslateFixed.

    After Track A / global FOV recovery, wrap/ambiguous soup can travel-drop
    almost every cell so the mesh collapses to a handful of points. Rebuilding
    from that set undoes the dense translated pose. Healthy lock fractions
    (~33%) never enter; dense meshes after residual also keep the normal path.
    """
    if not residual_applied:
        return False
    grid_n = max(1, int(n_grid))
    if float(n_locks) / float(grid_n) >= float(lock_frac_trigger):
        return False
    min_keep = max(int(min_mesh_abs), int(float(min_mesh_frac) * float(grid_n)))
    return int(n_mesh) < int(min_keep)


def estimate_global_fov_residual_translation(
        transform,
        target_image: NDArray,
        source_image: NDArray,
        *,
        max_dim: int = GLOBAL_FOV_MAX_DIM,
) -> NDArray[np.float64] | None:
    """Estimate one residual translation via downsampled whole-FOV phase correlation.

    Renders the source into target space under *transform* at reduced resolution,
    runs ``find_offset``, and returns the residual in full-resolution pixels for
    ``TranslateFixed``. Returns ``None`` on failure or degenerate images.

    Mutates *transform* only transiently via ``Scale`` (restored before return)
    when the transform implements ``ITransformScaling``.
    """
    import nornir_imageregistration
    from nornir_imageregistration.phasecorrelation import find_offset

    try:
        target = nornir_imageregistration.ImageParamToImageArray(
            target_image, dtype=nornir_imageregistration.default_image_dtype())
        source = nornir_imageregistration.ImageParamToImageArray(
            source_image, dtype=nornir_imageregistration.default_image_dtype())
    except Exception:
        return None

    target = np.asarray(getattr(target, 'get', lambda: target)(), dtype=np.float64)
    source = np.asarray(getattr(source, 'get', lambda: source)(), dtype=np.float64)
    if target.size == 0 or source.size == 0:
        return None

    max_side = float(max(int(target.shape[0]), int(target.shape[1]), 1))
    scalar = float(max_dim) / max_side
    if scalar <= 0.0 or not np.isfinite(scalar):
        return None
    if scalar > 1.0:
        scalar = 1.0

    try:
        target_ds = nornir_imageregistration.ResizeImage(target, scalar)
        source_ds = nornir_imageregistration.ResizeImage(source, scalar)
    except Exception:
        return None

    scale_fn = getattr(transform, 'Scale', None)
    if not callable(scale_fn):
        return None

    scale_fn(scalar)
    try:
        from nornir_imageregistration.assemble import SourceImageToTargetSpace
        area = np.asarray(target_ds.shape[:2], dtype=np.float64)
        warped = SourceImageToTargetSpace(
            transform,
            DataToTransform=source_ds,
            output_botleft=np.asarray((0.0, 0.0), dtype=np.float64),
            output_area=area,
            extrapolate=True,
            cval=0.0,
        )
        warped = np.asarray(getattr(warped, 'get', lambda: warped)(), dtype=np.float64)
        # Replace NaNs from OOB with median so PC stays defined.
        if not np.isfinite(warped).all():
            finite = warped[np.isfinite(warped)]
            fill = float(np.median(finite)) if finite.size else 0.0
            warped = np.where(np.isfinite(warped), warped, fill)
        record = find_offset(target_ds, warped)
        peak = np.asarray(record.peak, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(peak)):
            return None
        # Peak is in downsampled pixels; convert to full-resolution offset.
        return (peak / float(scalar)).astype(np.float64, copy=False)
    except Exception:
        return None
    finally:
        # Undo the temporary Scale.
        try:
            scale_fn(1.0 / float(scalar))
        except Exception:
            pass
