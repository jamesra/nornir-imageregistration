"""Strain / warp pickers for refine-fixture crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.peak_ratio_gates import PEAK_RATIO_MIN
from nornir_imageregistration.transforms import IControlPoints, ITriangulatedTargetSpace
from nornir_imageregistration.transforms import converters, metrics


@dataclass(frozen=True)
class StrainCrop:
    """Bounding box and cell IDs for a high-strain region plus halo."""

    bbox_yxhw: tuple[float, float, float, float]  # y0, x0, height, width (source space)
    strain_ids: tuple[tuple[int, int], ...]
    halo_ids: tuple[tuple[int, int], ...]
    certified_ids: tuple[tuple[int, int], ...]
    angle_delta_max: float
    linear_residual_p90: float
    unique_frac_in_crop: float
    localized: bool


def linear_residual_distances(transform: IControlPoints) -> NDArray[np.float64]:
    """Per-control-point Euclidean distance from a rigid fit of *transform*.

    Same residual used inside ``BlendWithLinear``; does not blend the mesh.
    """
    rigid = converters.ConvertControlPointsToRigidTransformForBlend(transform)
    source = np.asarray(transform.SourcePoints, dtype=np.float64)
    target = np.asarray(transform.TargetPoints, dtype=np.float64)
    linear = np.asarray(rigid.Transform(source), dtype=np.float64)
    return np.linalg.norm(target - linear, axis=1)


def vertex_angle_delta_max(transform: ITriangulatedTargetSpace) -> NDArray[np.float64]:
    """Per-vertex max triangle angle delta (radians)."""
    values = metrics.TriangleVertexAngleDelta(transform)
    out = np.zeros(transform.NumControlPoints, dtype=np.float64)
    for i, vals in enumerate(values):
        arr = np.asarray(vals, dtype=np.float64)
        out[i] = float(np.max(arr)) if arr.size else 0.0
    return out


def _grid_ids_from_points(
        source_points: NDArray[np.float64],
        grid_spacing: tuple[float, float] | None,
) -> list[tuple[int, int]]:
    """Infer integer grid (row, col) from source points when spacing is known."""
    if grid_spacing is None:
        return [(i, 0) for i in range(source_points.shape[0])]
    sy, sx = float(grid_spacing[0]), float(grid_spacing[1])
    if sy <= 0 or sx <= 0:
        return [(i, 0) for i in range(source_points.shape[0])]
    origin = source_points.min(axis=0)
    rows = np.rint((source_points[:, 0] - origin[0]) / sy).astype(np.int64)
    cols = np.rint((source_points[:, 1] - origin[1]) / sx).astype(np.int64)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _four_connected_halo(
        ids: set[tuple[int, int]],
        hops: int,
) -> set[tuple[int, int]]:
    """Expand *ids* by *hops* of 4-connected neighbours (same component only)."""
    frontier = set(ids)
    grown = set(ids)
    for _ in range(max(0, hops)):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb not in grown:
                    nxt.add(nb)
        grown |= nxt
        frontier = nxt
    return grown


def is_localized_high_strain(
        residual: NDArray[np.floating],
        *,
        high_frac: float = 0.25,
        p90_vs_median: float = 3.0,
) -> bool:
    """True when high residual is an island, not uniform soup."""
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if residual.size < 8:
        return False
    med = float(np.median(residual))
    p90 = float(np.percentile(residual, 90))
    p75 = float(np.percentile(residual, 75))
    if med <= 1e-6:
        return p90 > 1.0
    # Prefer p90/median when the tail is heavy; also accept a clear p75 spike.
    spike = max(p90, p75)
    high = residual >= float(np.percentile(residual, 100.0 * (1.0 - high_frac)))
    high_mean = float(np.mean(high))
    return bool(spike >= p90_vs_median * max(med, 1e-6) and 0.02 <= high_mean <= 0.55)


def pick_strain_crop_from_transform(
        transform: IControlPoints,
        *,
        grid_spacing: tuple[float, float] | None = None,
        halo_hops: int = 2,
        residual_quantile: float = 0.85,
        unique_mask: NDArray[np.bool_] | None = None,
) -> StrainCrop | None:
    """Pick a crop bbox from linear residual (+ optional unique mask).

    Returns None when residuals are flat or no unique tissue exists in the
    high-strain region (likely damage).
    """
    if not isinstance(transform, IControlPoints):
        raise TypeError('transform must implement IControlPoints')

    source = np.asarray(transform.SourcePoints, dtype=np.float64)
    residual = linear_residual_distances(transform)
    angle_max = 0.0
    if isinstance(transform, ITriangulatedTargetSpace):
        try:
            angle_max = float(np.max(vertex_angle_delta_max(transform)))
        except Exception:
            angle_max = 0.0

    thr = float(np.quantile(residual, residual_quantile))
    hot = residual >= thr
    if unique_mask is not None:
        unique_mask = np.asarray(unique_mask, dtype=bool).reshape(-1)
        if unique_mask.shape[0] == hot.shape[0]:
            hot_unique = hot & unique_mask
            if not np.any(hot_unique) and np.any(unique_mask):
                # No unique tissue in the warp island → damage / soup.
                return None
            if np.any(hot_unique):
                hot = hot_unique

    if not np.any(hot):
        return None

    ids = _grid_ids_from_points(source, grid_spacing)
    strain_ids = [ids[i] for i in range(len(ids)) if bool(hot[i])]
    strain_set = set(strain_ids)
    # Halo among known grid ids only.
    known = set(ids)
    halo = _four_connected_halo(strain_set, halo_hops) & known
    halo_only = halo - strain_set

    pts = source[hot]
    y0, x0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    y1, x1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    # Expand bbox by roughly one grid cell for halo padding.
    pad_y = float(grid_spacing[0]) if grid_spacing else 64.0
    pad_x = float(grid_spacing[1]) if grid_spacing else 64.0
    y0 = max(0.0, y0 - pad_y * halo_hops)
    x0 = max(0.0, x0 - pad_x * halo_hops)
    y1 = y1 + pad_y * halo_hops
    x1 = x1 + pad_x * halo_hops

    unique_frac = float(np.mean(unique_mask)) if unique_mask is not None else float('nan')
    return StrainCrop(
        bbox_yxhw=(y0, x0, y1 - y0, x1 - x0),
        strain_ids=tuple(sorted(strain_set)),
        halo_ids=tuple(sorted(halo_only)),
        certified_ids=tuple(sorted(strain_set)),
        angle_delta_max=angle_max,
        linear_residual_p90=float(np.percentile(residual, 90)),
        unique_frac_in_crop=unique_frac,
        localized=is_localized_high_strain(residual),
    )


def unique_mask_from_diagnostics(
        npz_path: str | Path,
        *,
        peak_ratio_min: float = PEAK_RATIO_MIN,
) -> tuple[NDArray[np.bool_], list[tuple[int, int]]] | None:
    """Load unique-cell mask from a ``refine_pass*_diagnostics.npz`` file."""
    path = Path(npz_path)
    if not path.is_file():
        return None
    data = np.load(path)
    if 'peak_ratio' not in data.files or 'grid_row' not in data.files:
        return None
    ratios = np.asarray(data['peak_ratio'], dtype=np.float64)
    rows = np.asarray(data['grid_row'], dtype=np.int64)
    cols = np.asarray(data['grid_col'], dtype=np.int64)
    unique = np.isfinite(ratios) & (ratios >= float(peak_ratio_min))
    ids = [(int(r), int(c)) for r, c in zip(rows, cols)]
    return unique, ids


def suggest_tags_from_strain(
        crop: StrainCrop,
        *,
        unique_frac_series: Sequence[float] | None = None,
        lock_frac: float | None = None,
) -> list[str]:
    """Return suggestable tag slugs from strain / lock statistics."""
    tags: list[str] = []
    if crop.localized and crop.linear_residual_p90 > 5.0:
        tags.append('high-relative-distortion')
    if unique_frac_series is not None and len(unique_frac_series) >= 2:
        if unique_frac_series[-1] < unique_frac_series[0] * 0.7 and unique_frac_series[0] > 0.05:
            tags.append('unique-collapse')
    if lock_frac is not None and 0.29 <= lock_frac <= 0.40 and crop.linear_residual_p90 < 20.0:
        tags.append('healthy')
    if not crop.localized and crop.unique_frac_in_crop == crop.unique_frac_in_crop:
        if crop.unique_frac_in_crop < 0.02:
            tags.append('damage')
    return tags


def classify_struggle_vs_damage(
        *,
        manual_zncc: float | None,
        automatic_zncc: float | None,
        unique_frac: float | None,
        localized: bool,
) -> str:
    """Return ``struggle``, ``damage``, or ``unknown`` for Manual candidates."""
    if unique_frac is not None and unique_frac < 0.02 and not localized:
        return 'damage'
    if (
            manual_zncc is not None
            and automatic_zncc is not None
            and manual_zncc > automatic_zncc + 0.05
            and (unique_frac is None or unique_frac >= 0.02)
    ):
        return 'struggle'
    if manual_zncc is not None and automatic_zncc is not None:
        if manual_zncc < 0.2 and automatic_zncc < 0.2:
            return 'damage'
    if localized and (unique_frac is None or unique_frac >= 0.02):
        return 'struggle'
    return 'unknown'
