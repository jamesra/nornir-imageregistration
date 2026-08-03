"""Locked-anchor grid smoothing for STOS mesh rebuild."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.displacement_regularize import regularize_displacements


class _AlignmentRecordLike(Protocol):
    ID: tuple[int, int]
    peak: NDArray[np.floating]
    weight: float
    angle: float
    SourcePoint: NDArray[np.floating]
    TargetPoint: NDArray[np.floating]
    flippedud: bool


class _TransformLike(Protocol):
    def Transform(self, points: NDArray[np.floating], **kwargs) -> NDArray[np.floating]:
        ...


@dataclass(frozen=True)
class AnchorSmoothSettings:
    """Knobs for locked-anchor mesh smoothing."""

    min_anchor_count: int = 3
    median_radius: int = 1

    @classmethod
    def from_grid_refinement(cls, settings: object) -> AnchorSmoothSettings:
        """Build from a GridRefinement-like object."""
        return cls(
            min_anchor_count=int(getattr(settings, 'anchor_smooth_min_locks', 3)),
            median_radius=int(getattr(settings, 'anchor_smooth_median_radius', 1)),
        )


def mesh_dims_from_records(
        *record_groups: Mapping[tuple[int, int], _AlignmentRecordLike] | Sequence[_AlignmentRecordLike],
) -> tuple[int, int]:
    """Return ``(rows, cols)`` covering all grid IDs in the given record groups."""
    max_row = -1
    max_col = -1
    for group in record_groups:
        if isinstance(group, Mapping):
            records = group.values()
        else:
            records = group
        for rec in records:
            row, col = int(rec.ID[0]), int(rec.ID[1])
            max_row = max(max_row, row)
            max_col = max(col, max_col)
    if max_row < 0 or max_col < 0:
        return 0, 0
    return max_row + 1, max_col + 1


def _as_numpy_points(points: NDArray[np.floating]) -> NDArray[np.float64]:
    if isinstance(points, np.ndarray):
        return np.asarray(points, dtype=np.float64)
    try:
        return np.asarray(points.get(), dtype=np.float64)  # type: ignore[union-attr]
    except AttributeError:
        return np.asarray(points, dtype=np.float64)


def compute_locked_displacement_field(
        finalized: Mapping[tuple[int, int], _AlignmentRecordLike],
        transform: _TransformLike,
        mesh_dims: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Build a dense shift field seeded only from locked anchors.

    For each locked cell at grid ID ``(r, c)``:

    - ``predicted = transform.Transform(SourcePoint)``
    - ``shift = TargetPoint - predicted`` (valid for baked locks with ``peak=0``)
    - ``measured[r, c] = True``

    Unlocked cells remain unmeasured with zero shift.
    """
    mesh_rows, mesh_cols = int(mesh_dims[0]), int(mesh_dims[1])
    if mesh_rows <= 0 or mesh_cols <= 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

    shifts = np.zeros((mesh_rows * mesh_cols, 2), dtype=np.float64)
    measured = np.zeros(mesh_rows * mesh_cols, dtype=bool)
    if not finalized:
        return shifts, measured

    source_points = np.asarray(
        [np.asarray(rec.SourcePoint, dtype=np.float64).reshape(2) for rec in finalized.values()],
        dtype=np.float64)
    target_points = np.asarray(
        [np.asarray(rec.TargetPoint, dtype=np.float64).reshape(2) for rec in finalized.values()],
        dtype=np.float64)
    predicted = _as_numpy_points(transform.Transform(source_points)).reshape(-1, 2)
    locked_shifts = target_points - predicted

    for (row, col), shift in zip(finalized.keys(), locked_shifts):
        idx = int(row) * mesh_cols + int(col)
        shifts[idx, :] = shift
        measured[idx] = True

    return shifts, measured


def smooth_peaks_from_locked_anchors(
        finalized: Mapping[tuple[int, int], _AlignmentRecordLike],
        alignment_points: Sequence[_AlignmentRecordLike],
        transform: _TransformLike,
        settings: AnchorSmoothSettings | object,
        discontinuity_ids: set[tuple[int, int]] | None = None,
) -> list:
    """Gap-fill a peak field from locked anchors and emit mesh alignment records.

    Raw measurements in ``alignment_points`` supply grid topology and metadata
    (weight, angle, source/target anchors). Peaks come from
    ``regularize_displacements`` seeded by locked cells only.

    When ``discontinuity_ids`` is set, those cells keep their **raw** peaks
    instead of the median/Gaussian-smoothed field so unique fold/tear
    discontinuities are not blurred into neighbors. Callers should pass
    ``soft_discontinuity_ids`` ∪ unique large-travel ids ∪ coherent disc-front
    ids; ambiguous dirt/false peaks that are not a coherent front must be
    smoothed from locked anchors (raw-preserving all disc tags caused a
    disc-count feedback loop).
    """
    if isinstance(settings, AnchorSmoothSettings):
        smooth_settings = settings
    else:
        smooth_settings = AnchorSmoothSettings.from_grid_refinement(settings)

    mesh_dims = mesh_dims_from_records(finalized, alignment_points)
    mesh_rows, mesh_cols = mesh_dims
    if mesh_rows <= 0 or mesh_cols <= 0:
        return list(alignment_points)

    discontinuity_ids = discontinuity_ids or set()
    shifts, measured = compute_locked_displacement_field(finalized, transform, mesh_dims)
    # Seed discontinuity cells with their raw peaks before regularization so
    # gap-fill neighbors see the fold, then restore raw peaks after blur.
    raw_peaks_by_id: dict[tuple[int, int], NDArray[np.float64]] = {}
    for rec in alignment_points:
        key = (int(rec.ID[0]), int(rec.ID[1]))
        raw_peaks_by_id[key] = np.asarray(rec.peak, dtype=np.float64).reshape(2)
        if key in discontinuity_ids:
            idx = key[0] * mesh_cols + key[1]
            if 0 <= idx < shifts.shape[0]:
                shifts[idx, :] = raw_peaks_by_id[key]
                measured[idx] = True

    smoothed_shifts, _ = regularize_displacements(
        shifts,
        measured,
        mesh_dims,
        median_radius=int(smooth_settings.median_radius),
    )
    for key in discontinuity_ids:
        if key in raw_peaks_by_id:
            idx = key[0] * mesh_cols + key[1]
            if 0 <= idx < smoothed_shifts.shape[0]:
                smoothed_shifts[idx, :] = raw_peaks_by_id[key]

    from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord

    record_by_id: dict[tuple[int, int], _AlignmentRecordLike] = {}
    for rec in alignment_points:
        record_by_id[tuple(int(v) for v in rec.ID)] = rec
    for key, rec in finalized.items():
        record_by_id[(int(key[0]), int(key[1]))] = rec

    smoothed_records: list = []
    for key in sorted(record_by_id.keys()):
        rec = record_by_id[key]
        row, col = int(key[0]), int(key[1])
        idx = row * mesh_cols + col
        source_point = np.asarray(rec.SourcePoint, dtype=np.float64).reshape(2)
        predicted_target = _as_numpy_points(transform.Transform(source_point.reshape(1, 2))).reshape(2)
        peak = np.asarray(smoothed_shifts[idx, :], dtype=np.float32)
        smoothed_records.append(EnhancedAlignmentRecord(
            ID=key,
            TargetPoint=predicted_target,
            SourcePoint=source_point,
            peak=peak,
            weight=float(rec.weight),
            angle=float(getattr(rec, 'angle', 0.0) or 0.0),
            flipped_ud=bool(getattr(rec, 'flippedud', False)),
            peak_ratio=getattr(rec, 'peak_ratio', None),
        ))
    return smoothed_records


def should_use_anchor_smooth_mesh(
        finalized: Mapping[tuple[int, int], _AlignmentRecordLike],
        settings: AnchorSmoothSettings | object,
) -> bool:
    """Return True when locked-anchor mesh smoothing should drive mesh rebuild."""
    if isinstance(settings, AnchorSmoothSettings):
        smooth_settings = settings
    else:
        smooth_settings = AnchorSmoothSettings.from_grid_refinement(settings)
    return len(finalized) >= int(smooth_settings.min_anchor_count)
