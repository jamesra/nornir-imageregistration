"""Pair-adaptive best-effort mode for sub-par STOS grid refine passes.

Uses only per-pass relative statistics (fractions / ranks). Does not compute
FOV warps or pre-registration contrast deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.peak_ratio_gates import (
    PEAK_RATIO_MIN,
    finite_peak_ratio,
    is_ambiguous_peak,
)

# Fraction of lock-candidates within the near-settled travel bar that indicates
# an identity-heavy pass (plurality, not necessarily a strict majority).
IDENTITY_LOCK_CAND_FRAC: float = 1.0 / 3.0

# Ambiguous residual movers as a fraction of measured cells. Slightly below
# classic lock-fraction trigger (0.05) because peak-ambiguous residuals are
# scarcer than unique FREE peaks that FOV-hot already counts.
HIGH_TRAVEL_FRAC_MIN: float = 0.04

# Near-settled travel bar as a fraction of max_travel (finalize travel limit).
IDENTITY_TRAVEL_FRAC_OF_MAX: float = 0.25

# Lock-candidates that clear ZNCC but sit at travel≈0 must also clear this
# prominence quantile within the lock-cand set (best-effort only).
IDENTITY_PROMINENCE_QUANTILE: float = 0.75

# Among PEAK_AMBIGUOUS rejects, promote the upper half by peak_ratio to FREE for mesh.
AMBIGUOUS_MESH_PROMOTE_QUANTILE: float = 0.5

# Cell min-dimension that STOS_PEAK_RATIO_EXCLUSION_RADIUS (10) was tuned against
# (GridRefinement default cell_size 128).
ZNCC_DECOY_REFERENCE_MIN_DIM: float = 128.0

# Size-independent floor on decoy MAD-sigma in ZNCC units (replaces 1/sqrt(N)).
ZNCC_DECOY_SIGMA_FLOOR: float = 0.02


class _AlignmentRecordLike(Protocol):
    ID: tuple[int, int]
    peak: NDArray[np.floating]
    peak_ratio: float | None


@dataclass(frozen=True)
class BestEffortAssessment:
    """Pass-level best-effort decision from relative cell statistics."""

    active: bool
    identity_lock_cand_frac: float
    high_travel_frac: float
    n_lock_cand: int
    n_high_travel: int
    n_measured: int


def zncc_decoy_radius_px(
        cell_h: int,
        cell_w: int,
        *,
        base_radius: float,
        reference_min_dim: float = ZNCC_DECOY_REFERENCE_MIN_DIM,
) -> float:
    """Scale the ZNCC decoy ring with cell size so geometry matches at 128/256/512.

    At ``min(cell_h, cell_w) == reference_min_dim``, returns ``base_radius``
    (legacy STOS exclusion radius at the default 128 px cell).
    """
    min_dim = float(min(int(cell_h), int(cell_w)))
    if min_dim <= 0.0 or float(reference_min_dim) <= 0.0:
        return max(1.0, float(base_radius))
    scaled = float(base_radius) * (min_dim / float(reference_min_dim))
    return max(1.0, scaled)


def assess_best_effort_mode(
        records: Sequence[_AlignmentRecordLike],
        *,
        lock_candidate: NDArray[np.bool_] | Sequence[bool],
        max_travel: float,
        travel_eps: float = 0.5,
        identity_frac_min: float = IDENTITY_LOCK_CAND_FRAC,
        high_travel_frac_min: float = HIGH_TRAVEL_FRAC_MIN,
        peak_ratio_min: float = PEAK_RATIO_MIN,
) -> BestEffortAssessment:
    """Return whether this pass should run best-effort identity / mesh policy.

    Active when lock-candidates are mostly near-settled **and** a non-trivial
    fraction of measured cells are residual **peak-ambiguous** movers.

    Near-settled uses ``max(travel_eps, 0.1 * max_travel)`` so remasure residuals
    of a few pixels still count as identity-like relative to the finalize travel
    bar (cell-sized ``max_travel``), without treating mid-range repair peaks as
    settled. Ambiguous movers are ``peak_ratio < peak_ratio_min`` with travel
    above that same bar.
    """
    n = len(records)
    if n == 0:
        return BestEffortAssessment(
            active=False,
            identity_lock_cand_frac=0.0,
            high_travel_frac=0.0,
            n_lock_cand=0,
            n_high_travel=0,
            n_measured=0,
        )

    lc = np.asarray(lock_candidate, dtype=bool).reshape(-1)
    if lc.shape[0] != n:
        raise ValueError('lock_candidate must match records length')

    travels = np.asarray(
        [float(np.linalg.norm(np.asarray(r.peak, dtype=np.float64).reshape(2)))
         for r in records],
        dtype=np.float64)
    identity_bar = max(
        float(travel_eps), float(IDENTITY_TRAVEL_FRAC_OF_MAX) * float(max_travel))

    # Ambiguous movers: any residual above stability eps with a weak peak_ratio.
    # Do not require identity_bar here — mid-range residuals (e.g. 5–40 px) are
    # exactly the repair signal on sub-par pairs.
    ambiguous_mover = np.zeros(n, dtype=bool)
    for i, record in enumerate(records):
        if travels[i] <= float(travel_eps):
            continue
        ratio = finite_peak_ratio(record)
        if ratio is not None and is_ambiguous_peak(ratio, min_ratio=peak_ratio_min):
            ambiguous_mover[i] = True
    n_high = int(np.count_nonzero(ambiguous_mover))
    high_frac = float(n_high) / float(n)

    n_lc = int(np.count_nonzero(lc))
    if n_lc == 0:
        id_frac = 0.0
    else:
        id_frac = float(np.count_nonzero(lc & (travels <= identity_bar))) / float(n_lc)

    active = (
        n_lc > 0
        and id_frac >= float(identity_frac_min)
        and high_frac >= float(high_travel_frac_min)
    )
    return BestEffortAssessment(
        active=active,
        identity_lock_cand_frac=id_frac,
        high_travel_frac=high_frac,
        n_lock_cand=n_lc,
        n_high_travel=n_high,
        n_measured=n,
    )


def identity_prominence_floor(
        prominences: NDArray[np.floating],
        lock_candidate: NDArray[np.bool_],
        *,
        quantile: float = IDENTITY_PROMINENCE_QUANTILE,
) -> float | None:
    """Prominence quantile among lock-candidates with finite scores, or None."""
    lc = np.asarray(lock_candidate, dtype=bool).reshape(-1)
    prom = np.asarray(prominences, dtype=np.float64).reshape(-1)
    if lc.shape[0] != prom.shape[0]:
        raise ValueError('prominences must match lock_candidate length')
    sample = prom[lc & np.isfinite(prom)]
    if sample.size == 0:
        return None
    return float(np.quantile(sample, float(quantile)))


def ranked_ambiguous_mesh_ids(
        records: Sequence[_AlignmentRecordLike],
        reject_reasons: Sequence[object],
        *,
        max_travel: float,
        travel_eps: float = 0.5,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        promote_quantile: float = AMBIGUOUS_MESH_PROMOTE_QUANTILE,
) -> set[tuple[int, int]]:
    """PEAK_AMBIGUOUS IDs in the upper peak_ratio quantile with usable travel.

    Candidates must have finite ``peak_ratio < peak_ratio_min``, travel in
    ``(travel_eps, max_travel]``, and rank at/above ``promote_quantile`` within
    that ambiguous set. Low-content rejects are never included.
    """
    from nornir_imageregistration.refine_shared.cell_roles import RejectReason

    ambiguous: list[tuple[tuple[int, int], float, float]] = []
    for record, reason in zip(records, reject_reasons):
        if reason != RejectReason.PEAK_AMBIGUOUS:
            continue
        ratio = finite_peak_ratio(record)
        if ratio is None or not is_ambiguous_peak(ratio, min_ratio=peak_ratio_min):
            continue
        peak = np.asarray(record.peak, dtype=np.float64).reshape(2)
        travel = float(np.linalg.norm(peak))
        if travel <= float(travel_eps) or travel > float(max_travel):
            continue
        key = (int(record.ID[0]), int(record.ID[1]))
        ambiguous.append((key, float(ratio), travel))

    if not ambiguous:
        return set()

    ratios = np.asarray([a[1] for a in ambiguous], dtype=np.float64)
    floor = float(np.quantile(ratios, float(promote_quantile)))
    return {key for key, ratio, _travel in ambiguous if ratio >= floor}


def apply_best_effort_ambiguous_mesh_promotion(
        roles: list,
        reject_reasons: list,
        records: Sequence[_AlignmentRecordLike],
        promote_ids: set[tuple[int, int]],
) -> tuple[list, list, dict[tuple[int, int], object]]:
    """Promote selected PEAK_AMBIGUOUS rejects to FREE for mesh (not lockable).

    Clears reject reason to NONE for promoted IDs. Does not touch LOW_CONTENT.
    """
    from nornir_imageregistration.refine_shared.cell_roles import RejectReason, Role

    if not promote_ids:
        role_by_id = {
            (int(r.ID[0]), int(r.ID[1])): roles[i]
            for i, r in enumerate(records)
        }
        return roles, reject_reasons, role_by_id

    new_roles = list(roles)
    new_reasons = list(reject_reasons)
    for i, record in enumerate(records):
        key = (int(record.ID[0]), int(record.ID[1]))
        if key not in promote_ids:
            continue
        if new_reasons[i] != RejectReason.PEAK_AMBIGUOUS:
            continue
        new_roles[i] = Role.FREE
        new_reasons[i] = RejectReason.NONE
    role_by_id = {
        (int(r.ID[0]), int(r.ID[1])): new_roles[i]
        for i, r in enumerate(records)
    }
    return new_roles, new_reasons, role_by_id


__all__ = [
    'AMBIGUOUS_MESH_PROMOTE_QUANTILE',
    'BestEffortAssessment',
    'HIGH_TRAVEL_FRAC_MIN',
    'IDENTITY_LOCK_CAND_FRAC',
    'IDENTITY_PROMINENCE_QUANTILE',
    'IDENTITY_TRAVEL_FRAC_OF_MAX',
    'ZNCC_DECOY_REFERENCE_MIN_DIM',
    'ZNCC_DECOY_SIGMA_FLOOR',
    'apply_best_effort_ambiguous_mesh_promotion',
    'assess_best_effort_mode',
    'identity_prominence_floor',
    'ranked_ambiguous_mesh_ids',
    'zncc_decoy_radius_px',
]
