"""Shared estimate_cutoff wrappers for mosaic and STOS refinement."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.mathfuncs import estimate_cutoff


def filter_weights_by_estimate_cutoff(weights: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Return a boolean mask of weights that pass the ``estimate_cutoff`` inflection.

    Weights below the inflection cutoff are treated as failed registrations.
    When fewer than three positive weights are present, all positive weights are kept.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    keep = np.zeros(weights.shape[0], dtype=bool)
    positive = weights > 0
    if not np.any(positive):
        return keep

    positive_weights = weights[positive]
    if positive_weights.shape[0] < 3:
        keep[positive] = True
        return keep

    try:
        _, inflection_percentile, _, polyfit_weights = estimate_cutoff(positive_weights)
        cutoff_value = float(polyfit_weights[inflection_percentile])  # type: ignore[index]
    except ValueError:
        keep[positive] = True
        return keep

    keep[positive] = positive_weights >= cutoff_value
    return keep


def filter_records_by_registration_weight(
        point_pair_updates: np.ndarray) -> np.ndarray:
    """Drop low-confidence structured updates using ``estimate_cutoff``.

    Expects a structured ndarray with a ``Weight`` field (overlap-pair path).
    """
    if point_pair_updates.size == 0:
        return point_pair_updates

    weights = np.asarray(point_pair_updates['Weight'], dtype=np.float64)
    keep_mask = filter_weights_by_estimate_cutoff(weights)
    return point_pair_updates[keep_mask]


def filter_alignment_records_by_weight(
        alignment_records: Sequence,
        weight_attr: str = 'weight') -> list:
    """Filter a sequence of alignment records by registration weight cutoff."""
    if len(alignment_records) == 0:
        return list(alignment_records)

    weights = np.asarray(
        [float(getattr(record, weight_attr)) for record in alignment_records],
        dtype=np.float64)
    keep = filter_weights_by_estimate_cutoff(weights)
    return [record for record, ok in zip(alignment_records, keep) if ok]
