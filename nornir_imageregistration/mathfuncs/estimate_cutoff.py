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


def estimate_cutoff(
    records: NDArray[np.floating],
    percentiles: NDArray[np.floating] | None = None,
    method: CutoffMethod = CutoffMethod.Average,
    polyfit_degree: int | None = None,
) -> EstimateCutoffResult:
    """Estimate a cutoff percentile where values begin to increase rapidly."""
    xp = cp.get_array_module(records)
    percentile_points = xp.linspace(0, 100, 101) if percentiles is None else xp.sort(percentiles)
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
