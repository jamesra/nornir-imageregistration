"""Scheduling state for finalized-cell registration rechecks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration

CellId = tuple[int, int]
LOCAL_RECHECK_EPS_FRACTION: float = 0.2


def local_recheck_threshold(stability_epsilon: float) -> float:
    """Use a conservative fraction of the movement stability tolerance."""
    return max(0.0, float(stability_epsilon)) * LOCAL_RECHECK_EPS_FRACTION


@dataclass(frozen=True)
class ArraySnapshot:
    """Exact compact snapshot of an immutable scoring input array."""

    shape: tuple[int, ...]
    dtype: str
    packed: bytes
    sha256: str


@dataclass
class TransformSnapshot:
    """Canonical transform type and ordered control-point values."""

    type_name: str
    control_points: NDArray[np.float64]


@dataclass
class RecheckContextSnapshot:
    """Inputs whose exact equality makes a finalized recheck reusable."""

    transform: TransformSnapshot
    cell_size: tuple[int, int]
    angles_to_search: tuple[float, ...]
    final_pass_angles: tuple[float, ...]
    max_travel_for_improvement: float | None
    ring_scale_fraction_max: float
    ring_angle_max_degrees: float
    ring_allow_flip_change: bool
    peak_ratio_exclusion_radius: int
    target_mask: ArraySnapshot | None
    source_mask: ArraySnapshot | None


@dataclass
class FinalizedRecheckState:
    """Prior exact context and local geometry retained across refine passes."""

    context: RecheckContextSnapshot | None = None
    local_geometry_by_id: dict[CellId, NDArray[np.float64]] = field(default_factory=dict)
    successfully_rechecked_ids: set[CellId] = field(default_factory=set)

    def prune(self, retained_ids: set[CellId]) -> None:
        """Discard state for cells that are no longer finalized."""
        stale = set(self.local_geometry_by_id) - retained_ids
        for key in stale:
            del self.local_geometry_by_id[key]
        self.successfully_rechecked_ids.intersection_update(retained_ids)


@dataclass
class FinalizedRecheckPlan:
    """Finalized IDs to recheck or skip plus shadow-rule evidence."""

    recheck_ids: list[CellId]
    skip_ids: list[CellId]
    proposed_local_skip_ids: list[CellId]
    current_geometry_by_id: dict[CellId, NDArray[np.float64]]
    local_movement_by_id: dict[CellId, float]
    exact_context_match: bool


def _host_array(value: Any) -> NDArray:
    """Return an arbitrary NumPy/CuPy value as a host array."""
    return np.asarray(getattr(value, 'get', lambda: value)())


def snapshot_array(value: Any | None) -> ArraySnapshot | None:
    """Pack an array losslessly for exact later comparison."""
    if value is None:
        return None
    array = np.ascontiguousarray(_host_array(value))
    if array.dtype == np.bool_:
        payload = np.packbits(array.reshape(-1), bitorder='little').tobytes()
    else:
        payload = array.tobytes()
    return ArraySnapshot(
        shape=tuple(int(v) for v in array.shape),
        dtype=array.dtype.str,
        packed=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def array_snapshots_equal(left: ArraySnapshot | None, right: ArraySnapshot | None) -> bool:
    """Return exact equality after using the digest as a fast rejection."""
    if left is None or right is None:
        return left is right
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.sha256 == right.sha256
        and left.packed == right.packed
    )


def canonical_transform_control_points(transform: Any) -> NDArray[np.float64]:
    """Return target/source control points in deterministic lexicographic order."""
    source = _host_array(transform.SourcePoints).astype(np.float64, copy=False).reshape(-1, 2)
    target = _host_array(transform.TargetPoints).astype(np.float64, copy=False).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError('transform SourcePoints and TargetPoints must have matching shapes')
    return canonical_control_point_array(np.hstack((target, source)))


def canonical_control_point_array(points: NDArray) -> NDArray[np.float64]:
    """Return target/source control-point rows in deterministic exact order."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 4)
    if points.shape[0] <= 1:
        return np.ascontiguousarray(points)
    order = np.lexsort((points[:, 3], points[:, 2], points[:, 1], points[:, 0]))
    return np.ascontiguousarray(points[order])


def transform_snapshots_equal(left: TransformSnapshot, right: TransformSnapshot) -> bool:
    """Compare transform type and canonical points exactly."""
    if left.type_name != right.type_name:
        return False
    if left.control_points.shape != right.control_points.shape:
        return False
    if not np.array_equal(left.control_points, right.control_points, equal_nan=True):
        return False
    return True


def context_snapshots_equal(
        left: RecheckContextSnapshot | None,
        right: RecheckContextSnapshot | None,
) -> bool:
    """Return True only when every recheck input is exactly unchanged."""
    if left is None or right is None:
        return False
    return (
        transform_snapshots_equal(left.transform, right.transform)
        and left.cell_size == right.cell_size
        and left.angles_to_search == right.angles_to_search
        and left.final_pass_angles == right.final_pass_angles
        and left.max_travel_for_improvement == right.max_travel_for_improvement
        and left.ring_scale_fraction_max == right.ring_scale_fraction_max
        and left.ring_angle_max_degrees == right.ring_angle_max_degrees
        and left.ring_allow_flip_change == right.ring_allow_flip_change
        and left.peak_ratio_exclusion_radius == right.peak_ratio_exclusion_radius
        and array_snapshots_equal(left.target_mask, right.target_mask)
        and array_snapshots_equal(left.source_mask, right.source_mask)
    )


def build_recheck_context(
        transform: Any,
        settings: Any,
        *,
        target_mask_snapshot: ArraySnapshot | None = None,
        source_mask_snapshot: ArraySnapshot | None = None,
) -> RecheckContextSnapshot:
    """Capture all transform and settings inputs relevant to a finalized recheck."""
    cell_size = np.asarray(settings.cell_size, dtype=np.int64).reshape(2)
    target_mask = (
        snapshot_array(getattr(settings, 'target_mask', None))
        if target_mask_snapshot is None
        else target_mask_snapshot
    )
    source_mask = (
        snapshot_array(getattr(settings, 'source_mask', None))
        if source_mask_snapshot is None
        else source_mask_snapshot
    )
    return RecheckContextSnapshot(
        transform=TransformSnapshot(
            type_name=f'{type(transform).__module__}.{type(transform).__qualname__}',
            control_points=canonical_transform_control_points(transform),
        ),
        cell_size=(int(cell_size.item(0)), int(cell_size.item(1))),
        angles_to_search=tuple(float(v) for v in settings.angles_to_search),
        final_pass_angles=tuple(float(v) for v in settings.final_pass_angles),
        max_travel_for_improvement=(
            None
            if settings.max_travel_for_finalization_improvement is None
            else float(settings.max_travel_for_finalization_improvement)
        ),
        ring_scale_fraction_max=float(settings.ring_scale_fraction_max),
        ring_angle_max_degrees=float(settings.ring_angle_max_degrees),
        ring_allow_flip_change=bool(settings.ring_allow_flip_change),
        peak_ratio_exclusion_radius=int(settings.peak_ratio_exclusion_radius),
        target_mask=target_mask,
        source_mask=source_mask,
    )


def _sample_local_geometry(
        transform: Any,
        records: Sequence[Any],
        cell_size: NDArray[np.integer],
) -> dict[CellId, NDArray[np.float64]]:
    """Sample batched forward rings and inverse target-ROI perimeters."""
    if not records:
        return {}
    centers = np.asarray(
        [np.asarray(record.SourcePoint, dtype=np.float64).reshape(2) for record in records],
        dtype=np.float64,
    )
    target_centers = np.asarray(
        [np.asarray(record.TargetPoint, dtype=np.float64).reshape(2) for record in records],
        dtype=np.float64,
    )
    half_y = float(cell_size[0]) * 0.5
    half_x = float(cell_size[1]) * 0.5
    radius = half_x
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring_offsets = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    forward_source = np.concatenate(
        (centers[:, None, :], centers[:, None, :] + ring_offsets[None, :, :]),
        axis=1,
    )

    roi_offsets = np.asarray([
        (-half_y, -half_x),
        (-half_y, 0.0),
        (-half_y, half_x),
        (0.0, half_x),
        (half_y, half_x),
        (half_y, 0.0),
        (half_y, -half_x),
        (0.0, -half_x),
    ], dtype=np.float64)
    inverse_target = target_centers[:, None, :] + roi_offsets[None, :, :]

    forward_values = _host_array(
        transform.Transform(forward_source.reshape(-1, 2))
    ).astype(np.float64, copy=False).reshape(len(records), 9, 2)
    inverse_values = _host_array(
        transform.InverseTransform(inverse_target.reshape(-1, 2))
    ).astype(np.float64, copy=False).reshape(len(records), 8, 2)
    combined = np.concatenate((forward_values, inverse_values), axis=1)
    return {
        (int(record.ID[0]), int(record.ID[1])): np.ascontiguousarray(combined[index])
        for index, record in enumerate(records)
    }


def local_geometry_unchanged(
        previous: NDArray[np.float64],
        current: NDArray[np.float64],
        *,
        threshold: float,
) -> bool:
    """Return True when every sampled local point moved at most *threshold*."""
    if previous.shape != current.shape or not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return False
    movement = np.linalg.norm(current - previous, axis=1)
    return bool(np.all(movement <= float(threshold)))


def local_geometry_max_movement(
        previous: NDArray[np.float64],
        current: NDArray[np.float64],
) -> float:
    """Return the largest sampled local movement, or infinity if incomparable."""
    if previous.shape != current.shape or not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return float('inf')
    movement = np.linalg.norm(current - previous, axis=1)
    return float(np.max(movement)) if movement.size else 0.0


def plan_finalized_rechecks(
        transform: Any,
        alignment_records: Mapping[CellId, Any],
        settings: Any,
        state: FinalizedRecheckState,
        *,
        mode: str,
        local_threshold: float,
) -> tuple[FinalizedRecheckPlan, RecheckContextSnapshot]:
    """Plan exact-safe and local finalized-cell rechecks for one pass."""
    if mode not in ('all', 'shadow', 'local'):
        mode = 'all'
    context = build_recheck_context(
        transform,
        settings,
        target_mask_snapshot=snapshot_array(getattr(settings, 'target_mask', None)),
        source_mask_snapshot=snapshot_array(getattr(settings, 'source_mask', None)),
    )
    records = list(alignment_records.values())
    ids = [(int(record.ID[0]), int(record.ID[1])) for record in records]
    exact_match = context_snapshots_equal(state.context, context)
    if exact_match:
        skip_ids = [key for key in ids if key in state.successfully_rechecked_ids]
        skip_set = set(skip_ids)
        missing_records = [
            record
            for record in records
            if (int(record.ID[0]), int(record.ID[1])) not in skip_set
        ]
        current_geometry = dict(state.local_geometry_by_id)
        if mode != 'all':
            current_geometry.update(_sample_local_geometry(
                transform,
                missing_records,
                np.asarray(settings.cell_size, dtype=np.int64).reshape(2),
            ))
        return FinalizedRecheckPlan(
            recheck_ids=[key for key in ids if key not in skip_set],
            skip_ids=skip_ids,
            proposed_local_skip_ids=skip_ids,
            current_geometry_by_id=current_geometry,
            local_movement_by_id={key: 0.0 for key in skip_ids},
            exact_context_match=True,
        ), context

    if mode == 'all':
        return FinalizedRecheckPlan(
            recheck_ids=ids,
            skip_ids=[],
            proposed_local_skip_ids=[],
            current_geometry_by_id={},
            local_movement_by_id={},
            exact_context_match=False,
        ), context

    geometry = _sample_local_geometry(
        transform,
        records,
        np.asarray(settings.cell_size, dtype=np.int64).reshape(2),
    )
    movement_by_id = {
        key: local_geometry_max_movement(state.local_geometry_by_id[key], geometry[key])
        for key in ids
        if key in state.local_geometry_by_id
    }
    proposed_skips = [
        key for key in ids
        if movement_by_id.get(key, float('inf')) <= float(local_threshold)
    ]
    skip_ids = proposed_skips if mode == 'local' else []
    skip_set = set(skip_ids)
    recheck_ids = [key for key in ids if key not in skip_set]
    return FinalizedRecheckPlan(
        recheck_ids=recheck_ids,
        skip_ids=skip_ids,
        proposed_local_skip_ids=proposed_skips,
        current_geometry_by_id=geometry,
        local_movement_by_id=movement_by_id,
        exact_context_match=False,
    ), context


def update_recheck_state(
        state: FinalizedRecheckState,
        context: RecheckContextSnapshot,
        plan: FinalizedRecheckPlan,
        *,
        successfully_rechecked_ids: Sequence[CellId],
) -> None:
    """Commit context and local samples only for cells actually rechecked."""
    state.context = context
    state.successfully_rechecked_ids.update(successfully_rechecked_ids)
    for key in successfully_rechecked_ids:
        geometry = plan.current_geometry_by_id.get(key)
        if geometry is not None:
            state.local_geometry_by_id[key] = geometry.copy()


def merge_recheck_results(
        existing: Mapping[CellId, Any],
        rechecked: Mapping[CellId, Any],
        plan: FinalizedRecheckPlan,
) -> dict[CellId, Any]:
    """Merge rechecked records with unchanged records retained by the plan."""
    output = {
        key: existing[key]
        for key in plan.skip_ids
        if key in existing
    }
    output.update(rechecked)
    return output
