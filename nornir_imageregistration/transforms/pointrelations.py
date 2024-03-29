import enum

import numpy as np
from numpy._typing import NDArray

from nornir_imageregistration import cupy_thunk as cp


class ControlPointRelation(enum.IntEnum):
    """Describes the relationship between two sets of control points.  Linear, flipped, or colinear."""
    LINEAR = 0  # Points are not flipped or colinear
    FLIPPED = 1
    COLINEAR = 2


def _get_pointset_crossproducts(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Returns the cross products of the vectors return by the differences (np.diff) between each point in the array, p[i] - p[i -1].
    """
    global _cross_product_abs_tolerance
    xp = cp.get_array_module(points)

    if points.shape[0] < 3:
        raise ValueError("Need at least 3 control points to determine if flipped")

    # Calculate vectors
    vectors = xp.diff(points, axis=0)  # Only need 2 vectors for speed

    # Grid transforms in particular have may colinear points.  So we start our search for a non-zero cross product
    # at the end of the list, and continue until we have two non-zero cross products
    cross_products = xp.cross(vectors[0], vectors[1:])

    return cross_products


def _get_pointset_crossproduct(points: NDArray[np.floating]) -> float:
    """
    Returns the largest absolute valued cross product of the vectors from np.diff
    """
    xp = cp.get_array_module(points)

    cross_products = _get_pointset_crossproducts(points)

    # Grid transforms in particular have may colinear points.  So we start our search for a non-zero cross product
    # at the end of the list, and continue until we have two non-zero cross products
    i_max_cross = xp.argmax(abs(cross_products))
    return cross_products[i_max_cross]


def calculate_point_relation(points: NDArray[np.floating]) -> ControlPointRelation:
    """
    Returns true if the points are colinear. False otherwise.
    Assume source_points and target_points are numpy arrays of shape (n, 2)
    where n is the number of points and 2 corresponds to the x and y coordinates of each point
    """
    global _cross_product_abs_tolerance
    cross = _get_pointset_crossproduct(points)
    xp = cp.get_array_module(points)

    # Check if flippednp.isclose(cross, 0)
    # If either cross product is close to zero the points are colinear.
    cross = 0 if xp.isclose(cross, 0) else cross

    # Both point sets are colinear
    if cross == 0:
        return ControlPointRelation.COLINEAR

    return ControlPointRelation.LINEAR if xp.sign(cross) >= 0 else ControlPointRelation.FLIPPED


def are_points_colinear(points: NDArray[np.floating]) -> bool:
    """
    Returns true if the points are colinear. False otherwise.
    Assume source_points and target_points are numpy arrays of shape (n, 2)
    where n is the number of points and 2 corresponds to the x and y coordinates of each point
    """
    relation = calculate_point_relation(points)
    return relation == ControlPointRelation.COLINEAR


def calculate_control_points_relationship(source_points: NDArray[np.floating],
                                          target_points: NDArray[np.floating]) -> ControlPointRelation:
    """
    Returns true if the control points are flipped on an axis. False otherwise.
    Assume source_points and target_points are numpy arrays of shape (n, 2)
    where n is the number of points and 2 corresponds to the x and y coordinates of each point
    """

    xp = cp.get_array_module(source_points)

    source_crosses = _get_pointset_crossproducts(source_points)
    target_crosses = _get_pointset_crossproducts(target_points)

    source_crosses[xp.isclose(source_crosses, 0)] = 0
    target_crosses[xp.isclose(target_crosses, 0)] = 0

    non_zero_crosses = xp.logical_and(source_crosses != 0, target_crosses != 0)

    source_cross_signs = xp.sign(source_crosses[non_zero_crosses])
    target_cross_signs = xp.sign(target_crosses[non_zero_crosses])

    # Check how many cross products have matching signs
    sign_comparisons = source_cross_signs == target_cross_signs
    num_matching_signs = xp.sum(sign_comparisons)
    if num_matching_signs < sign_comparisons.shape[0] / 2:
        return ControlPointRelation.FLIPPED

    return ControlPointRelation.LINEAR
