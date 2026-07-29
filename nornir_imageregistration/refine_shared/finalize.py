"""Multi-criteria finalize / unlock helpers for STOS grid refinement.

Locks require weight at the transform-inclusion bar, small travel, pass delay,
and optional peak stability. Stale locks that disagree with the evolving mesh
can be unlocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config


class _AlignmentRecordLike(Protocol):
    ID: tuple[int, int]
    peak: NDArray[np.floating]
    weight: float
    SourcePoint: NDArray[np.floating]
    TargetPoint: NDArray[np.floating]
    AdjustedTargetPoint: NDArray[np.floating]


class _TransformLike(Protocol):
    def Transform(self, points: NDArray[np.floating], **kwargs) -> NDArray[np.floating]:
        ...


@dataclass
class FinalizeCandidateState:
    """Per-cell state tracked across passes before a point is locked."""

    peak: NDArray[np.floating]
    weight: float
    pass_index: int
    consecutive_stable: int = 1


@dataclass
class FinalizeEvaluationResult:
    """Outcome of one finalize evaluation pass."""

    lock_mask: NDArray[np.bool_]
    candidates: dict[tuple[int, int], FinalizeCandidateState]
    deferred_stability_count: int
    rejected_weight_count: int
    rejected_travel_count: int
    rejected_pass_count: int


@dataclass(frozen=True)
class FinalizeSettings:
    """Finalize / unlock knobs used by evaluate_finalize_candidates."""

    max_travel_for_finalization: float
    min_finalize_pass: int = 2
    finalize_stability_passes: int = 2
    finalize_stability_epsilon_px: float = 0.5
    finalize_unlock_travel_multiplier: float = 1.5
    weight_drop_fraction: float = 0.15

    @classmethod
    def from_grid_refinement(cls, settings: object) -> FinalizeSettings:
        """Build from a GridRefinement-like object with optional new attributes."""
        return cls(
            max_travel_for_finalization=float(getattr(settings, 'max_travel_for_finalization')),
            min_finalize_pass=int(getattr(settings, 'min_finalize_pass', 2)),
            finalize_stability_passes=int(getattr(settings, 'finalize_stability_passes', 2)),
            finalize_stability_epsilon_px=float(
                getattr(settings, 'finalize_stability_epsilon_px', 0.5)),
            finalize_unlock_travel_multiplier=float(
                getattr(settings, 'finalize_unlock_travel_multiplier', 1.5)),
        )


def use_legacy_finalize_gate() -> bool:
    """True when ``NORNIR_REFINE_FINALIZE_LEGACY=1`` restores distance+2% floor locking."""
    return get_runtime_config(refresh=True).finalize_legacy


def evaluate_finalize_candidates(
        records: Sequence[_AlignmentRecordLike],
        transform_cutoff: float,
        settings: FinalizeSettings,
        pass_index: int,
        prior_candidates: Mapping[tuple[int, int], FinalizeCandidateState] | None = None,
) -> FinalizeEvaluationResult:
    """Evaluate which alignment records may lock this pass.

    All of the following must hold for a lock (unless legacy mode is active
    elsewhere):

    - ``pass_index >= min_finalize_pass``
    - ``‖peak‖ <= max_travel_for_finalization``
    - ``weight >= transform_cutoff`` (caller should pass the inflection-derived
      mesh-inclusion cutoff from ``estimate_registration_weight_cutoff``)
    - peak is stable for ``finalize_stability_passes`` consecutive passes
    """
    n = len(records)
    lock_mask = np.zeros(n, dtype=bool)
    if n == 0:
        return FinalizeEvaluationResult(
            lock_mask=lock_mask,
            candidates={},
            deferred_stability_count=0,
            rejected_weight_count=0,
            rejected_travel_count=0,
            rejected_pass_count=0,
        )

    prior = dict(prior_candidates or {})
    weights = np.asarray([float(r.weight) for r in records], dtype=np.float64)
    peaks = np.asarray([np.asarray(r.peak, dtype=np.float64).reshape(2) for r in records])
    travels = np.linalg.norm(peaks, axis=1)

    # transform_cutoff is the STOS inflection / mesh-inclusion bar computed on the
    # full pass weight set in RefineTransform; do not recompute estimate_cutoff on
    # the unfinalized subset alone (that subset can be flat and rejects good locks).
    weight_ok = weights >= float(transform_cutoff)
    travel_ok = travels <= float(settings.max_travel_for_finalization)
    pass_ok = np.full(n, pass_index >= int(settings.min_finalize_pass), dtype=bool)

    rejected_weight = int(np.count_nonzero(~weight_ok))
    rejected_travel = int(np.count_nonzero(weight_ok & ~travel_ok))
    rejected_pass = int(np.count_nonzero(weight_ok & travel_ok & ~pass_ok))

    updated: dict[tuple[int, int], FinalizeCandidateState] = {}
    deferred_stability = 0
    epsilon = float(settings.finalize_stability_epsilon_px)
    need_stable = max(1, int(settings.finalize_stability_passes))
    drop_frac = float(settings.weight_drop_fraction)

    for i, record in enumerate(records):
        key = tuple(record.ID)  # type: ignore[arg-type]
        if not (bool(weight_ok[i]) and bool(travel_ok[i]) and bool(pass_ok[i])):
            # Drop prior candidate state when the cell fails hard gates this pass.
            continue

        peak = peaks[i]
        weight = float(weights[i])
        prev = prior.get(key)
        if prev is None:
            updated[key] = FinalizeCandidateState(
                peak=peak.copy(), weight=weight, pass_index=pass_index, consecutive_stable=1)
            if need_stable > 1:
                deferred_stability += 1
            else:
                lock_mask[i] = True
            continue

        peak_delta = float(np.linalg.norm(peak - np.asarray(prev.peak, dtype=np.float64)))
        weight_ok_vs_prev = weight >= float(prev.weight) * (1.0 - drop_frac)
        if peak_delta <= epsilon and weight_ok_vs_prev:
            consecutive = int(prev.consecutive_stable) + 1
        else:
            consecutive = 1

        updated[key] = FinalizeCandidateState(
            peak=peak.copy(),
            weight=weight,
            pass_index=pass_index,
            consecutive_stable=consecutive,
        )
        if consecutive >= need_stable:
            lock_mask[i] = True
        else:
            deferred_stability += 1

    return FinalizeEvaluationResult(
        lock_mask=lock_mask,
        candidates=updated,
        deferred_stability_count=deferred_stability,
        rejected_weight_count=rejected_weight,
        rejected_travel_count=rejected_travel,
        rejected_pass_count=rejected_pass,
    )


def unlock_stale_finalized(
        finalized: Mapping[tuple[int, int], _AlignmentRecordLike],
        transform: _TransformLike,
        settings: FinalizeSettings,
) -> tuple[dict[tuple[int, int], _AlignmentRecordLike], list[tuple[int, int]]]:
    """Remove finalized points that disagree with the current transform prediction.

    Compares each locked point's ``AdjustedTargetPoint`` (same control target the
    mesh uses) to ``transform.Transform(source)``. When the distance exceeds
    ``max_travel * unlock_multiplier``, the lock is dropped.

    Returns ``(kept_finalized, unlocked_keys)``. Unlock is disabled when
    ``finalize_unlock_travel_multiplier <= 0``.
    """
    if not finalized:
        return dict(finalized), []

    multiplier = float(settings.finalize_unlock_travel_multiplier)
    if multiplier <= 0.0:
        return dict(finalized), []

    unlock_travel = float(settings.max_travel_for_finalization) * multiplier
    source_points = np.asarray(
        [np.asarray(rec.SourcePoint, dtype=np.float64).reshape(2) for rec in finalized.values()],
        dtype=np.float64)
    # Mesh fixed points use AdjustedTargetPoint; unlock must compare the same.
    adjusted_targets = np.asarray(
        [np.asarray(rec.AdjustedTargetPoint, dtype=np.float64).reshape(2) for rec in finalized.values()],
        dtype=np.float64)
    predicted = np.asarray(transform.Transform(source_points), dtype=np.float64).reshape(-1, 2)
    if hasattr(predicted, 'get'):
        predicted = np.asarray(predicted.get(), dtype=np.float64).reshape(-1, 2)

    deltas = np.linalg.norm(adjusted_targets - predicted, axis=1)
    keys = list(finalized.keys())
    kept: dict[tuple[int, int], _AlignmentRecordLike] = {}
    unlocked: list[tuple[int, int]] = []
    for key, delta in zip(keys, deltas):
        if float(delta) > unlock_travel:
            unlocked.append(key)
        else:
            kept[key] = finalized[key]
    # #region agent log
    try:
        import json as _json
        import time as _time
        _payload = {
            'sessionId': '6f4e34',
            'runId': 'lean-fix',
            'hypothesisId': 'A,C',
            'location': 'finalize.py:unlock_stale_finalized',
            'message': 'unlock deltas',
            'data': {
                'n_finalized': len(keys),
                'n_unlocked': len(unlocked),
                'unlock_travel': unlock_travel,
                'delta_max': float(np.max(deltas)) if len(deltas) else 0.0,
                'delta_mean': float(np.mean(deltas)) if len(deltas) else 0.0,
                'delta_p90': float(np.percentile(deltas, 90)) if len(deltas) else 0.0,
            },
            'timestamp': int(_time.time() * 1000),
        }
        with open('/workspace/.cursor/debug-6f4e34.log', 'a', encoding='utf-8') as _f:
            _f.write(_json.dumps(_payload) + '\n')
    except Exception:
        pass
    # #endregion
    return kept, unlocked


def filter_records_for_mesh_inclusion(
        records: Sequence[_AlignmentRecordLike],
        max_travel: float,
        min_keep: int = 3,
) -> tuple[list, int]:
    """Drop free alignment records whose peak travel exceeds ``max_travel``.

    Large residual peaks that are not yet lockable must not reshape the mesh —
    they fold edges when weight alone includes them. If filtering would leave
    fewer than ``min_keep`` points, keep the smallest-travel subset instead.
    """
    records_list = list(records)
    n = len(records_list)
    if n == 0:
        return records_list, 0

    travels = np.asarray(
        [float(np.linalg.norm(np.asarray(r.peak, dtype=np.float64).reshape(2)))
         for r in records_list],
        dtype=np.float64)
    keep_mask = travels <= float(max_travel)
    kept = [r for r, ok in zip(records_list, keep_mask) if bool(ok)]
    if len(kept) >= int(min_keep):
        return kept, int(n - len(kept))

    order = np.argsort(travels)
    n_fallback = max(int(min_keep), min(n, max(int(min_keep), n // 20)))
    kept = [records_list[int(i)] for i in order[:n_fallback]]
    return kept, int(n - len(kept))


def legacy_finalize_mask(
        records: Sequence[_AlignmentRecordLike],
        max_travel_distance: float,
        polyfit_weights: NDArray[np.floating] | None = None,
        floor_percentile: float = 2.0,
) -> NDArray[np.bool_]:
    """Distance-primary lock with a low weight-percentile floor (pre-fix behavior)."""
    n = len(records)
    if n == 0:
        return np.zeros(0, dtype=bool)

    weights = np.asarray([float(r.weight) for r in records], dtype=np.float64)
    peaks = np.asarray([np.asarray(r.peak, dtype=np.float64).reshape(2) for r in records])
    travels = np.linalg.norm(peaks, axis=1)
    if polyfit_weights is not None and len(polyfit_weights) > 0:
        weight_cutoff = float(np.percentile(polyfit_weights, floor_percentile))
    else:
        weight_cutoff = float(np.percentile(weights, floor_percentile))
    return (weights >= weight_cutoff) & (travels <= float(max_travel_distance))
