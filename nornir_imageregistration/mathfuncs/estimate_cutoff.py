import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

from .calculate_deviation import calculate_deviation
from .cutoff_types import CutoffMethod, EstimateCutoffResult
from .find_inflection_points import find_inflection_points


def linear_percentile_curve(
    records: NDArray[np.floating],
    percentiles: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return linear-interpolated percentiles from a single sort (Hyndman-Fan type 7).

    Prefer this over ``xp.percentile(records, q)`` when *q* has many points: CuPy's
    percentile evaluates each quantile separately, which dominates serial ``find_peak``.
    """
    xp = cp.get_array_module(records)
    flat = xp.ravel(records)
    n = int(flat.size)
    if n == 0:
        raise ValueError("No values to compute percentiles")
    sorted_vals = xp.sort(flat)
    q = xp.asarray(percentiles, dtype=xp.float64)
    pos = (q / 100.0) * xp.float64(max(n - 1, 0))
    lo = xp.floor(pos).astype(xp.int64)
    hi = xp.minimum(lo + 1, n - 1)
    weight = pos - lo.astype(xp.float64)
    lo_vals = sorted_vals[lo].astype(xp.float64, copy=False)
    hi_vals = sorted_vals[hi].astype(xp.float64, copy=False)
    return lo_vals * (1.0 - weight) + hi_vals * weight


def estimate_cutoff(
    records: NDArray[np.floating],
    percentiles: NDArray[np.floating] | None = None,
    method: CutoffMethod = CutoffMethod.Average,
    polyfit_degree: int | None = None,
    precomputed_percentile_values: NDArray[np.floating] | None = None,
) -> EstimateCutoffResult:
    """Estimate a cutoff percentile where values begin to increase rapidly.

    :param precomputed_percentile_values: Optional curve already sampled at
        *percentiles* (or at ``linspace(0, 100, n)`` when *percentiles* is None).
        Skips ``percentile`` over *records* so callers can transfer a compact curve.
    """
    xp = cp.get_array_module(
        precomputed_percentile_values if precomputed_percentile_values is not None else records)
    if percentiles is None:
        if precomputed_percentile_values is not None:
            n_curve = int(xp.asarray(precomputed_percentile_values).shape[0])
            percentile_points = xp.linspace(0, 100, n_curve)
        else:
            percentile_points = xp.linspace(0, 100, 101)
    else:
        percentile_points = xp.sort(xp.asarray(percentiles, dtype=xp.float64))
    if precomputed_percentile_values is not None:
        percentile_values = xp.asarray(precomputed_percentile_values, dtype=xp.float64)
    else:
        try:
            percentile_values = xp.percentile(records, percentile_points, method="linear")
        except TypeError:
            percentile_values = xp.percentile(records, percentile_points)

    if method != CutoffMethod.Raw:
        degree = 5 if polyfit_degree is None else polyfit_degree
        if degree < 1:
            raise ValueError("Polyfit degree must be at least 1")

        if xp is np:
            coefficients = np.polyfit(percentile_points, percentile_values, degree)
            polynomial = np.poly1d(coefficients)
            y_fit = polynomial(percentile_points)
        else:
            # polyfit/poly1d parity can vary by CuPy version; use host for this compact vector.
            percentiles_host = nornir_imageregistration.EnsureNumpyArray(percentile_points)
            percentile_values_host = nornir_imageregistration.EnsureNumpyArray(percentile_values)
            coefficients = np.polyfit(percentiles_host, percentile_values_host, degree)
            polynomial = np.poly1d(coefficients)
            y_fit = xp.asarray(polynomial(percentiles_host))
        inflection_results = find_inflection_points(percentile_points, y_fit)
    else:
        inflection_results = find_inflection_points(percentile_points, percentile_values)
        y_fit = None

    if len(inflection_results.indices) == 0:
        raise ValueError("No inflection points found in the data")

    highest_inflection_point = inflection_results.values[-1]
    highest_inflection_point_index = int(inflection_results.indices[-1])

    if method == CutoffMethod.Raw:
        cross_products = calculate_deviation(
            values=percentile_values, above_index=highest_inflection_point_index
        )
        cutoff_percentile_index = int(xp.argmin(cross_products[:, 1])) + highest_inflection_point_index
        cutoff_value = float(percentile_values[cutoff_percentile_index])
    elif method == CutoffMethod.Polyfit:
        assert y_fit is not None
        cross_products = calculate_deviation(
            values=y_fit, above_index=highest_inflection_point_index
        )
        cutoff_percentile_index = int(xp.argmin(cross_products[:, 1])) + highest_inflection_point_index
        cutoff_value = float(y_fit[cutoff_percentile_index])
    elif method == CutoffMethod.Average:
        raw_cross_products = calculate_deviation(
            values=percentile_values, above_index=highest_inflection_point_index
        )
        raw_cutoff_percentile_index = (
            int(xp.argmin(raw_cross_products[:, 1])) + highest_inflection_point_index
        )
        raw_cutoff_value = float(percentile_values[raw_cutoff_percentile_index])

        assert y_fit is not None
        poly_cross_products = calculate_deviation(values=y_fit, above_index=highest_inflection_point_index)
        poly_cutoff_percentile_index = (
            int(xp.argmin(poly_cross_products[:, 1])) + highest_inflection_point_index
        )
        poly_cutoff_value = float(y_fit[poly_cutoff_percentile_index])

        cutoff_percentile_index = (raw_cutoff_percentile_index + poly_cutoff_percentile_index) // 2
        cutoff_value = (raw_cutoff_value + poly_cutoff_value) / 2
    else:
        raise ValueError(f"Unknown method: {method}")

    return EstimateCutoffResult(
        cutoff_percentile_index,
        highest_inflection_point_index,
        cutoff_value,
        y_fit,
    )
