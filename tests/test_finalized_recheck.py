"""Tests for exact-safe and local finalized-cell recheck scheduling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nornir_imageregistration.refine_shared.finalized_recheck import (
    FinalizedRecheckPlan,
    FinalizedRecheckState,
    local_recheck_threshold,
    merge_recheck_results,
    plan_finalized_rechecks,
    update_recheck_state,
)


class _AffineTransform:
    """Minimal affine transform exposing the grid-transform point interface."""

    def __init__(self, matrix: np.ndarray, offset: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.SourcePoints = np.asarray(((0, 0), (0, 10), (10, 0)), dtype=np.float64)
        self.TargetPoints = self.Transform(self.SourcePoints)

    def Transform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.matrix.T + self.offset

    def InverseTransform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return (points - self.offset) @ np.linalg.inv(self.matrix).T


def _settings() -> SimpleNamespace:
    """Return the recheck-relevant subset of GridRefinement settings."""
    return SimpleNamespace(
        cell_size=np.asarray((8, 8), dtype=np.int64),
        angles_to_search=[-1, 0, 1],
        final_pass_angles=[0],
        max_travel_for_finalization_improvement=3.0,
        ring_scale_fraction_max=0.2,
        ring_angle_max_degrees=10.0,
        ring_allow_flip_change=False,
        peak_ratio_exclusion_radius=3,
        target_mask=np.ones((8, 8), dtype=bool),
        source_mask=np.ones((8, 8), dtype=bool),
    )


def _records() -> dict[tuple[int, int], SimpleNamespace]:
    """Return two finalized-record stand-ins."""
    return {
        (0, 0): SimpleNamespace(
            ID=(0, 0),
            SourcePoint=np.asarray((0.0, 0.0)),
            TargetPoint=np.asarray((0.0, 0.0)),
        ),
        (0, 1): SimpleNamespace(
            ID=(0, 1),
            SourcePoint=np.asarray((0.0, 10.0)),
            TargetPoint=np.asarray((0.0, 10.0)),
        ),
    }


def test_missing_signatures_recheck_every_cell() -> None:
    """A fresh state cannot skip finalized cells."""
    transform = _AffineTransform(np.eye(2), np.zeros(2))
    plan, _ = plan_finalized_rechecks(
        transform,
        _records(),
        _settings(),
        FinalizedRecheckState(),
        mode='local',
        local_threshold=0.5,
    )
    assert plan.recheck_ids == [(0, 0), (0, 1)]
    assert plan.skip_ids == []


def test_local_threshold_is_conservative_fraction() -> None:
    """Local skipping stays narrower than the finalize stability tolerance."""
    assert local_recheck_threshold(0.5) == 0.1


def test_exact_transform_signature_reuses_prior_results() -> None:
    """An exactly unchanged context skips all expensive rechecks."""
    transform = _AffineTransform(np.eye(2), np.zeros(2))
    records = _records()
    settings = _settings()
    state = FinalizedRecheckState()
    first, context = plan_finalized_rechecks(
        transform,
        records,
        settings,
        state,
        mode='all',
        local_threshold=0.5,
    )
    update_recheck_state(
        state,
        context,
        first,
        successfully_rechecked_ids=first.recheck_ids,
    )
    second, _ = plan_finalized_rechecks(
        transform,
        records,
        settings,
        state,
        mode='all',
        local_threshold=0.5,
    )
    assert second.exact_context_match is True
    assert second.recheck_ids == []
    assert second.skip_ids == [(0, 0), (0, 1)]


def test_mask_change_invalidates_exact_reuse() -> None:
    """Exact reuse detects mask content changes, not only array identity."""
    transform = _AffineTransform(np.eye(2), np.zeros(2))
    records = _records()
    settings = _settings()
    state = FinalizedRecheckState()
    first, context = plan_finalized_rechecks(
        transform,
        records,
        settings,
        state,
        mode='all',
        local_threshold=0.5,
    )
    update_recheck_state(state, context, first, successfully_rechecked_ids=first.recheck_ids)
    settings.target_mask[0, 0] = False
    second, _ = plan_finalized_rechecks(
        transform,
        records,
        settings,
        state,
        mode='all',
        local_threshold=0.5,
    )
    assert second.exact_context_match is False
    assert second.recheck_ids == [(0, 0), (0, 1)]


def test_center_unchanged_ring_changed_is_not_locally_skipped() -> None:
    """A fixed center cannot hide scale changes on the ring and ROI perimeter."""
    records = {(0, 0): _records()[(0, 0)]}
    settings = _settings()
    state = FinalizedRecheckState()
    identity = _AffineTransform(np.eye(2), np.zeros(2))
    first, context = plan_finalized_rechecks(
        identity,
        records,
        settings,
        state,
        mode='shadow',
        local_threshold=0.1,
    )
    update_recheck_state(state, context, first, successfully_rechecked_ids=first.recheck_ids)

    scaled = _AffineTransform(np.eye(2) * 1.1, np.zeros(2))
    second, _ = plan_finalized_rechecks(
        scaled,
        records,
        settings,
        state,
        mode='shadow',
        local_threshold=0.1,
    )
    assert second.proposed_local_skip_ids == []
    assert second.recheck_ids == [(0, 0)]


def test_shadow_records_skips_but_rechecks_all() -> None:
    """Shadow mode reports locally stable cells without changing execution."""
    records = _records()
    settings = _settings()
    state = FinalizedRecheckState()
    identity = _AffineTransform(np.eye(2), np.zeros(2))
    first, context = plan_finalized_rechecks(
        identity,
        records,
        settings,
        state,
        mode='shadow',
        local_threshold=0.5,
    )
    update_recheck_state(state, context, first, successfully_rechecked_ids=first.recheck_ids)
    shifted = _AffineTransform(np.eye(2), np.asarray((0.1, 0.1)))
    second, _ = plan_finalized_rechecks(
        shifted,
        records,
        settings,
        state,
        mode='shadow',
        local_threshold=0.5,
    )
    assert second.proposed_local_skip_ids == [(0, 0), (0, 1)]
    assert second.skip_ids == []
    assert second.recheck_ids == [(0, 0), (0, 1)]


def test_local_mode_skips_only_stable_signed_geometry() -> None:
    """Local mode omits stable cells after they have one successful signature."""
    records = _records()
    settings = _settings()
    state = FinalizedRecheckState()
    identity = _AffineTransform(np.eye(2), np.zeros(2))
    first, context = plan_finalized_rechecks(
        identity,
        records,
        settings,
        state,
        mode='local',
        local_threshold=0.5,
    )
    update_recheck_state(state, context, first, successfully_rechecked_ids=first.recheck_ids)
    shifted = _AffineTransform(np.eye(2), np.asarray((0.1, 0.1)))
    second, _ = plan_finalized_rechecks(
        shifted,
        records,
        settings,
        state,
        mode='local',
        local_threshold=0.5,
    )
    assert second.recheck_ids == []
    assert second.skip_ids == [(0, 0), (0, 1)]


def test_local_merge_retains_skipped_records() -> None:
    """Local orchestration cannot drop records omitted from registration."""
    old_a = object()
    old_b = object()
    new_b = object()
    plan = FinalizedRecheckPlan(
        recheck_ids=[(0, 1)],
        skip_ids=[(0, 0)],
        proposed_local_skip_ids=[(0, 0)],
        current_geometry_by_id={},
        local_movement_by_id={(0, 0): 0.1},
        exact_context_match=False,
    )
    merged = merge_recheck_results(
        {(0, 0): old_a, (0, 1): old_b},
        {(0, 1): new_b},
        plan,
    )
    assert merged == {(0, 0): old_a, (0, 1): new_b}
