"""Trust tiers for the trusted-mesh refine path: LOCKED / PROVISIONAL / UNTRUSTED."""

from __future__ import annotations

from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.coherent_residual import INLIER_COS_MIN
from nornir_imageregistration.refine_shared.peak_ratio_gates import PEAK_RATIO_MIN, finite_peak_ratio

# Minimum 4-connected cluster of unique peaks when no locks exist yet.
CLUSTER_MIN_SIZE: int = 3
# Travel within this factor of the cluster median is mutually consistent.
CLUSTER_TRAVEL_FACTOR: float = 2.0


class TrustTier(IntEnum):
    """Per-cell trust for mesh membership."""

    UNTRUSTED = 0
    PROVISIONAL = 1
    LOCKED = 2


def _record_id(record: object) -> tuple[int, int]:
    key = getattr(record, 'ID')
    return (int(key[0]), int(key[1]))


def _peak_arr(record: object) -> NDArray[np.float64]:
    return np.asarray(getattr(record, 'peak'), dtype=np.float64).reshape(2)


def _travel(record: object) -> float:
    return float(np.linalg.norm(_peak_arr(record)))


def is_unique_peak(record: object, *, peak_ratio_min: float = PEAK_RATIO_MIN) -> bool:
    """True when a finite peak_ratio clears the uniqueness bar."""
    ratio = finite_peak_ratio(record)
    return ratio is not None and float(ratio) >= float(peak_ratio_min)


def agreement_tolerance(
        *,
        n_locks_within_two_hops: int,
        max_travel: float,
        cell_half_size: float,
) -> float:
    """Support-scaled agreement radius in pixels."""
    if n_locks_within_two_hops >= 3:
        return float(max_travel)
    if n_locks_within_two_hops >= 1:
        return float(0.5 * (max_travel + cell_half_size))
    return float(cell_half_size)


def _four_neighbors(key: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = key
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _hops_within(
        origin: tuple[int, int],
        targets: set[tuple[int, int]],
        *,
        max_hops: int = 2,
) -> int:
    """Count how many *targets* lie within *max_hops* of *origin* (4-connected)."""
    if not targets:
        return 0
    seen = {origin}
    frontier = {origin}
    found = 0
    for _ in range(max_hops):
        nxt: set[tuple[int, int]] = set()
        for node in frontier:
            for nb in _four_neighbors(node):
                if nb in seen:
                    continue
                seen.add(nb)
                nxt.add(nb)
                if nb in targets:
                    found += 1
        frontier = nxt
    return found


def find_unique_clusters(
        records: Sequence[object],
        *,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        inlier_cos_min: float = INLIER_COS_MIN,
        travel_factor: float = CLUSTER_TRAVEL_FACTOR,
        min_size: int = CLUSTER_MIN_SIZE,
) -> list[set[tuple[int, int]]]:
    """4-connected clusters of unique peaks that agree in direction and travel."""
    unique = [rec for rec in records if is_unique_peak(rec, peak_ratio_min=peak_ratio_min)]
    if not unique:
        return []
    by_id = {_record_id(rec): rec for rec in unique}
    remaining = set(by_id)
    clusters: list[set[tuple[int, int]]] = []
    while remaining:
        seed = next(iter(remaining))
        component: set[tuple[int, int]] = set()
        stack = [seed]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            component.add(node)
            for nb in _four_neighbors(node):
                if nb in remaining:
                    stack.append(nb)
        if len(component) < int(min_size):
            continue
        peaks = np.vstack([_peak_arr(by_id[k]) for k in component])
        travels = np.linalg.norm(peaks, axis=1)
        med_travel = float(np.median(travels))
        if med_travel <= 1e-6:
            continue
        unit = peaks / np.maximum(travels[:, None], 1e-12)
        med_peak = np.median(peaks, axis=0)
        med_n = float(np.linalg.norm(med_peak))
        if med_n <= 1e-6:
            continue
        ref = med_peak / med_n
        dots = unit @ ref
        travel_ok = (travels >= med_travel / float(travel_factor)) & (
            travels <= med_travel * float(travel_factor))
        keep = {k for k, d, t_ok in zip(component, dots, travel_ok)
                if float(d) >= float(inlier_cos_min) and bool(t_ok)}
        if len(keep) >= int(min_size):
            clusters.append(keep)
    return clusters


def assign_trust_tiers(
        records: Sequence[object],
        *,
        locked_ids: set[tuple[int, int]] | None = None,
        zncc_pass_ids: set[tuple[int, int]] | None = None,
        max_travel: float = 2.0,
        cell_half_size: float = 64.0,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        converged_ids: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], TrustTier]:
    """Assign LOCKED / PROVISIONAL / UNTRUSTED for each measured record.

    LOCKED requires unique peak, ZNCC pass (when provided), and convergence.
    PROVISIONAL requires unique peak plus neighbour agreement or a seeding cluster.
    """
    locked_ids = locked_ids or set()
    zncc_pass_ids = zncc_pass_ids  # None → do not require ZNCC for provisional
    converged_ids = converged_ids or set()
    by_id = {_record_id(rec): rec for rec in records}
    tiers: dict[tuple[int, int], TrustTier] = {
        key: TrustTier.UNTRUSTED for key in by_id
    }

    unique_ids = {
        key for key, rec in by_id.items()
        if is_unique_peak(rec, peak_ratio_min=peak_ratio_min)
    }

    # Promote locked+converged unique ZNCC-pass cells.
    for key in unique_ids:
        if key not in locked_ids and key not in converged_ids:
            continue
        if zncc_pass_ids is not None and key not in zncc_pass_ids:
            continue
        tiers[key] = TrustTier.LOCKED

    locked_now = {k for k, t in tiers.items() if t == TrustTier.LOCKED} | locked_ids

    clusters: list[set[tuple[int, int]]] = []
    if not locked_now:
        clusters = find_unique_clusters(records, peak_ratio_min=peak_ratio_min)
        cluster_members = set().union(*clusters) if clusters else set()
    else:
        cluster_members = set()

    for key in unique_ids:
        if tiers[key] == TrustTier.LOCKED:
            continue
        if zncc_pass_ids is not None and key not in zncc_pass_ids and key not in cluster_members:
            # Without ZNCC, cluster seeding can still provisional-promote uniqueness.
            if key not in cluster_members:
                continue
        n_support = _hops_within(key, locked_now, max_hops=2)
        if n_support == 0 and key not in cluster_members:
            continue
        tol = agreement_tolerance(
            n_locks_within_two_hops=n_support,
            max_travel=max_travel,
            cell_half_size=cell_half_size,
        )
        if n_support > 0:
            # Agree with nearest locked neighbour peak direction/travel.
            rec = by_id[key]
            peak = _peak_arr(rec)
            ok = False
            for nb in _four_neighbors(key):
                if nb not in locked_now or nb not in by_id:
                    # Also accept locked-only ids without a fresh record as support.
                    if nb in locked_now:
                        ok = True
                        break
                    continue
                nb_peak = _peak_arr(by_id[nb])
                if float(np.linalg.norm(peak - nb_peak)) <= tol:
                    ok = True
                    break
            if ok:
                tiers[key] = TrustTier.PROVISIONAL
        elif key in cluster_members:
            tiers[key] = TrustTier.PROVISIONAL

    return tiers


def trusted_set_snapshot(
        tiers: Mapping[tuple[int, int], TrustTier],
) -> frozenset[tuple[tuple[int, int], int]]:
    """Immutable (id, tier) set for change detection."""
    return frozenset(
        (key, int(tier))
        for key, tier in tiers.items()
        if int(tier) != int(TrustTier.UNTRUSTED)
    )


def demote_disagreeing(
        tiers: dict[tuple[int, int], TrustTier],
        records: Sequence[object],
        *,
        max_travel: float,
        cell_half_size: float,
) -> dict[tuple[int, int], TrustTier]:
    """Demote provisional cells that no longer agree with locked neighbours."""
    by_id = {_record_id(rec): rec for rec in records}
    locked = {k for k, t in tiers.items() if t == TrustTier.LOCKED}
    out = dict(tiers)
    for key, tier in list(tiers.items()):
        if tier != TrustTier.PROVISIONAL or key not in by_id:
            continue
        n_support = _hops_within(key, locked, max_hops=2)
        if n_support == 0:
            out[key] = TrustTier.UNTRUSTED
            continue
        tol = agreement_tolerance(
            n_locks_within_two_hops=n_support,
            max_travel=max_travel,
            cell_half_size=cell_half_size,
        )
        peak = _peak_arr(by_id[key])
        agreed = False
        for nb in _four_neighbors(key):
            if nb not in locked:
                continue
            if nb not in by_id:
                agreed = True
                break
            if float(np.linalg.norm(peak - _peak_arr(by_id[nb]))) <= tol:
                agreed = True
                break
        if not agreed:
            out[key] = TrustTier.UNTRUSTED
    return out


def mesh_records_from_tiers(
        records: Sequence[object],
        tiers: Mapping[tuple[int, int], TrustTier],
) -> tuple[list[object], list[object]]:
    """Split records into (locked_fixed, provisional_movable) for mesh build."""
    locked: list[object] = []
    provisional: list[object] = []
    for rec in records:
        key = _record_id(rec)
        tier = tiers.get(key, TrustTier.UNTRUSTED)
        if tier == TrustTier.LOCKED:
            locked.append(rec)
        elif tier == TrustTier.PROVISIONAL:
            provisional.append(rec)
    return locked, provisional
