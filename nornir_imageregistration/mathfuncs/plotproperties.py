"""Contains routines for determining the properties of a data plot, such as inflection points of a 2D graph. """
import enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple

import nornir_imageregistration


class InflectionPointsResult(NamedTuple):
    indicies: NDArray[int]
    values: NDArray[np.floating]


def find_inflection_points(x: NDArray, y: NDArray) -> InflectionPointsResult:
    """
    Find the inflection points of an XY plot.
    :param x: The x values of the plot
    :param y: The y values of the plot
    :return: The x values of the inflection points
    """

    # Calculate the first derivative
    dy = np.gradient(y, x)

    # Calculate the second derivative
    d2y = np.gradient(dy, x)

    # Find where the second derivative changes sign
    inflection_point_candidates = np.where(np.diff(np.sign(d2y)))[0]

    dNy = np.gradient(d2y, x)
    is_odd_derivative = True
    verified_inflection_points = np.array([], dtype=int)
    while len(inflection_point_candidates) > 0:
        # Check if the next derivative is non-zero.  If it is an odd derivative and non-zero it is an inflection point.#
        # Otherwise it is an undulation point
        if is_odd_derivative:
            new_inflection_points = np.where(np.logical_not(
                np.isclose(dNy[inflection_point_candidates], 0)))[0]
            verified_inflection_points = np.union1d(inflection_point_candidates,
                                                    inflection_point_candidates[new_inflection_points])
            inflection_point_candidates = np.setdiff1d(inflection_point_candidates, verified_inflection_points,
                                                       assume_unique=True)
        else:
            undulation_points = np.where(np.isclose(dNy[inflection_point_candidates], 0))[0]
            undulation_indicies = inflection_point_candidates[undulation_points]
            inflection_point_candidates = np.setdiff1d(inflection_point_candidates,
                                                       undulation_indicies,
                                                       assume_unique=True)

        dNy = np.gradient(dNy, x)
        is_odd_derivative = not is_odd_derivative

    return InflectionPointsResult(verified_inflection_points, x[verified_inflection_points])


def calculate_deviation(values: NDArray[float],
                        above_index: int | None = None) -> NDArray[np.floating]:
    """
    Project the y values onto a line that runs from min(a) to max(a).
    The point furthest from the line will have the largest absolute cross product.
    :param y_values: The y values to project onto the line.  The x values are generated as the percentile values from 0 to 100.
    :param above_index: If specified, only indicies above this value will be considered.
    The line's origin used for the cross product will be at values[above].
    IF ABOVE IS SPECIFIED, THE RETURNED CROSS PRODUCTS WILL BE OFFSET BY ABOVE INDICES. The first index will be the above value.
    :return: The point with the lowest cross product indicating it is furthest from the line
    used as a cutoff value for registration quality.

    :return: The cross product values in an array representing each percentile.  Index 42 is the 42nd percentile
    """
    a = values
    p = np.linspace(0, 100, 101)

    start_index = 0 if above_index is None else above_index

    percentile_values = np.percentile(a, p, method='linear')

    # Create a vector for each record from the record to the min and max points
    percentile_value_subset = percentile_values[start_index:]
    percentile_subset = p[start_index:]
    min_value = percentile_value_subset[0]  # Sorted data, so indicies are fine
    max_value = percentile_value_subset[-1]

    min_vectors = np.vstack(
        np.array(((percentile_value_subset - min_value),
                  percentile_subset - percentile_subset[0],
                  np.zeros(len(percentile_subset))
                  ))
    ).T
    max_vectors = np.vstack(
        np.array(((max_value - percentile_value_subset),
                  percentile_subset - percentile_subset[-1],
                  np.zeros(len(percentile_subset))
                  ))
    ).T

    cross_products = np.cross(min_vectors, max_vectors)

    # Project the points onto the line
    return np.vstack((percentile_subset, cross_products[:, 2])).T


class CutoffMethod(enum.Enum):
    """
    The method used to determine a cutoff value

    Attributes:
        Raw: Use the raw data to find the cutoff
        Polyfit: Use a polyfit to determine the cutoff
        Average: Average the raw cutoff and polyfit cutoff together
    """
    Raw = enum.auto(),  # Use the raw data to find the cutoff
    Polyfit = enum.auto(),  # Use a polyfit to determine the cutoff
    Average = enum.auto(),  # Average the raw cutoff and polyfit cutoff together


class EstimateCutoffResult(NamedTuple):
    cutoff_percentile_index: int  # The index of the percentile value in the input array that is the cutoff
    highest_inflection_point: int  # The index of the highest inflection point found in the input
    cutoff_value: float  # The cutoff value, equivalent to records[cutoff_percentile_index]
    y_fit: NDArray[
        float]  # The polyfit line calculated to identify the inflection point, and if the polyfit or average method was chosen this result contributed to the cutoff value


def estimate_cutoff(records: NDArray[float],
                    percentiles: NDArray[float] | None = None,
                    method: CutoffMethod = CutoffMethod.Average,
                    polyfit_degree: int | None = None) -> EstimateCutoffResult:
    """
    Given a set of numbers, estimate a cutoff that seperates the point where the values begin to increase rapidly.
    :param records: A list of records to estimate the cutoff for.  Finds the inflection point with the highest x
    value.  Then finds the percentile higher than that with the largest cross product, which is the furthest point
    from a line drawn from the min to max values.  This is where the values tend to begin increasing rapidly
    indicating that registrations are successful.
    :param percentiles: The percentiles to evaluate when searching for a cutoff.  If None, the percentiles are 0 to 100 in 1% increments.
    :param method: The method to use to determine the cutoff
    :param polyfit_degree: The degree of the polynomial to fit to the data.  If None, 5 is the default.
    :return: The index of the percentile and  and the value at the cutoff
    """
    percentiles = np.linspace(0, 100, 101) if percentiles is None else np.sort(percentiles)
    percentile_values = np.percentile(records, percentiles)

    if method != CutoffMethod.Raw:
        # Sort the percentiles and values
        # Add a polyfit to the linear line
        degree = 5 if polyfit_degree is None else polyfit_degree

        if degree < 1:
            raise ValueError("Polyfit degree must be at least 1")

        coefficients = np.polyfit(percentiles, percentile_values, degree)
        # Generate the polynomial function from the coefficients
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(percentiles)

        inflection_results = find_inflection_points(percentiles, y_fit)
    else:
        inflection_results = find_inflection_points(percentiles, percentile_values)
        y_fit = None

    if len(inflection_results.indicies) == 0:
        raise ValueError("No inflection points found in the data")

    # The points after the highest inflection point are the ones considered for maximum deviation
    highest_inflection_point = inflection_results.values[-1]
    highest_inflection_point_index = inflection_results.indicies[-1]

    if method == CutoffMethod.Raw:
        cross_products = nornir_imageregistration.mathfuncs.plotproperties.calculate_deviation(values=percentile_values,
                                                                                               above_index=highest_inflection_point_index)
        cutoff_percentile_index = np.argmin(cross_products[:, 1]) + highest_inflection_point_index
        cutoff_value = percentile_values[cutoff_percentile_index]
    elif method == CutoffMethod.Polyfit:
        cross_products = nornir_imageregistration.mathfuncs.plotproperties.calculate_deviation(values=y_fit,
                                                                                               above_index=highest_inflection_point_index)
        cutoff_percentile_index = np.argmin(cross_products[:, 1]) + highest_inflection_point_index
        cutoff_value = y_fit[cutoff_percentile_index]
    elif method == CutoffMethod.Average:
        raw_cross_products = nornir_imageregistration.mathfuncs.plotproperties.calculate_deviation(
            values=percentile_values,
            above_index=highest_inflection_point_index)
        raw_cutoff_percentile_index = np.argmin(raw_cross_products[:, 1]) + highest_inflection_point_index
        raw_cutoff_value = percentile_values[raw_cutoff_percentile_index]

        poly_cross_products = nornir_imageregistration.mathfuncs.plotproperties.calculate_deviation(values=y_fit,
                                                                                                    above_index=highest_inflection_point_index)
        poly_cutoff_percentile_index = np.argmin(poly_cross_products[:, 1]) + highest_inflection_point_index
        poly_cutoff_value = y_fit[poly_cutoff_percentile_index]

        cutoff_percentile_index = (raw_cutoff_percentile_index + poly_cutoff_percentile_index) // 2
        cutoff_value = (raw_cutoff_value + poly_cutoff_value) / 2
    else:
        raise ValueError(f"Unknown method: {method}")

    return EstimateCutoffResult(cutoff_percentile_index, highest_inflection_point_index, cutoff_value, y_fit)
