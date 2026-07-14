"""Shared estimate_cutoff wrappers for mosaic and STOS refinement."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.mathfuncs import estimate_cutoff


class RegistrationWeightCutoff(NamedTuple):
    """Cutoff indices/values for registration-weight gating in STOS refine."""

    cutoff_percentile_index: int
    inflection_percentile_index: int
    cutoff_value: float
    percentile_curve: NDArray[np.floating]
    used_fallback: bool


def estimate_registration_weight_cutoff(
        weights: NDArray[np.floating]) -> RegistrationWeightCutoff:
    """Run ``estimate_cutoff`` on registration weights with a keep-all fallback.

    When the percentile curve has no inflection (flat or strictly mono scores),
    return index ``0`` and the minimum percentile sample so callers that keep
    ``weight >= cutoff`` retain essentially all measurements instead of aborting.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("Cannot estimate cutoff from empty registration weights")

    try:
        result = estimate_cutoff(weights)
        curve = result.y_fit
        if curve is None:
            percentile_points = np.linspace(0, 100, 101)
            curve = np.percentile(weights, percentile_points)
        return RegistrationWeightCutoff(
            int(result.cutoff_percentile_index),
            int(result.highest_inflection_point),
            float(result.cutoff_value),
            np.asarray(curve, dtype=np.float64),
            False,
        )
    except ValueError:
        percentile_points = np.linspace(0, 100, 101)
        curve = np.asarray(np.percentile(weights, percentile_points), dtype=np.float64)
        return RegistrationWeightCutoff(0, 0, float(curve[0]), curve, True)


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

    cutoff = estimate_registration_weight_cutoff(positive_weights)
    # Match RefineTransform / prior mosaic behavior: gate at the inflection
    # sample on the fitted percentile curve, not the Average-method elbow value
    # (which can sit above the max observed weight).
    gate = float(cutoff.percentile_curve[cutoff.inflection_percentile_index])
    keep[positive] = positive_weights >= gate
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
