"""Host-side peak_ratio gates for STOS false-peak finalize / disc soft floors.

Operates only on already-materialized ``AlignmentRecord.peak_ratio`` Python
floats (post batched ``host_sync``). Do not call into CuPy or recompute
``masked_peak_ratio`` here.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Provisional thresholds — tune after Grid16 PASS_DIAGNOSTICS NPZ calibration.
PEAK_RATIO_MIN: float = 1.20
PEAK_RATIO_EARLY: float = 1.50


def finite_peak_ratio(record: object) -> float | None:
    """Return a finite stored peak_ratio, or None if missing / non-finite."""
    raw = getattr(record, 'peak_ratio', None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def is_ambiguous_peak(ratio: float | None, *, min_ratio: float = PEAK_RATIO_MIN) -> bool:
    """True when ratio is known and below the hard-reject floor."""
    if ratio is None:
        return False
    return float(ratio) < float(min_ratio)


def is_early_lock_ratio(ratio: float | None, *, early_ratio: float = PEAK_RATIO_EARLY) -> bool:
    """True when ratio is known and at/above the early-lock bar."""
    if ratio is None:
        return False
    return float(ratio) >= float(early_ratio)


def soft_discontinuity_ids(
        records: Sequence[object],
        discontinuity_ids: set[tuple[int, int]],
        *,
        min_ratio: float = PEAK_RATIO_MIN,
) -> set[tuple[int, int]]:
    """Discontinuity cells eligible for relaxed travel / soft weight.

    Only cells with a finite ``peak_ratio >= min_ratio`` receive soft floors.
    Ambiguous or missing ratios keep strict travel/weight.
    """
    if not discontinuity_ids:
        return set()
    eligible: set[tuple[int, int]] = set()
    for record in records:
        key = (int(record.ID[0]), int(record.ID[1]))  # type: ignore[attr-defined]
        if key not in discontinuity_ids:
            continue
        ratio = finite_peak_ratio(record)
        if ratio is None or float(ratio) < float(min_ratio):
            continue
        eligible.add(key)
    return eligible


def ambiguous_record_ids(
        records: Sequence[object],
        *,
        min_ratio: float = PEAK_RATIO_MIN,
) -> set[tuple[int, int]]:
    """Grid IDs whose stored peak_ratio is known and below ``min_ratio``."""
    ids: set[tuple[int, int]] = set()
    for record in records:
        if is_ambiguous_peak(finite_peak_ratio(record), min_ratio=min_ratio):
            ids.add((int(record.ID[0]), int(record.ID[1])))  # type: ignore[attr-defined]
    return ids


def exclude_ambiguous_mesh_records(
        records: Sequence[object],
        *,
        min_keep: int = 3,
        min_ratio: float = PEAK_RATIO_MIN,
) -> tuple[list, int]:
    """Drop ambiguous free records so raw false peaks do not reshape the mesh.

    When filtering would leave fewer than ``min_keep`` points, keep all
    non-ambiguous records and fill with the lowest-travel ambiguous cells
    (emergency only — avoids an empty triangulation before any locks exist).

    Returns ``(kept_records, n_ambiguous_dropped)``.
    """
    records_list = list(records)
    if not records_list:
        return records_list, 0

    clear: list = []
    ambiguous: list = []
    for record in records_list:
        if is_ambiguous_peak(finite_peak_ratio(record), min_ratio=min_ratio):
            ambiguous.append(record)
        else:
            clear.append(record)

    if len(clear) >= int(min_keep):
        return clear, len(ambiguous)

    if not ambiguous:
        return clear, 0

    def _travel(rec: object) -> float:
        peak = getattr(rec, 'peak', None)
        if peak is None:
            return float('inf')
        return float(np.linalg.norm(np.asarray(peak, dtype=np.float64).reshape(2)))

    ambiguous_sorted = sorted(ambiguous, key=_travel)
    need = max(0, int(min_keep) - len(clear))
    kept = clear + ambiguous_sorted[:need]
    dropped = len(ambiguous) - len(ambiguous_sorted[:need])
    return kept, dropped
