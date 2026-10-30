import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration

from .cutoff_types import InflectionPointsResult


def find_inflection_points(x: NDArray, y: NDArray) -> InflectionPointsResult:
    """Find inflection points of an XY plot (where second derivative changes sign)."""

    # Use NumPy for this solver; set operations have stable parity here and
    # input vectors are compact percentile-scale arrays.
    x_local = nornir_imageregistration.EnsureNumpyArray(x)
    y_local = nornir_imageregistration.EnsureNumpyArray(y)

    dy = np.gradient(y_local, x_local)
    d2y = np.gradient(dy, x_local)
    inflection_point_candidates = np.where(np.diff(np.sign(d2y)))[0]

    dNy = np.gradient(d2y, x_local)
    is_odd_derivative = True
    verified_inflection_points = np.array([], dtype=int)
    while len(inflection_point_candidates) > 0:
        if is_odd_derivative:
            new_inflection_points = np.where(
                np.logical_not(np.isclose(dNy[inflection_point_candidates], 0))
            )[0]
            verified_inflection_points = np.union1d(
                inflection_point_candidates,
                inflection_point_candidates[new_inflection_points],
            )
            inflection_point_candidates = np.setdiff1d(
                inflection_point_candidates,
                verified_inflection_points,
                assume_unique=True,
            )
        else:
            undulation_points = np.where(np.isclose(dNy[inflection_point_candidates], 0))[0]
            undulation_indices = inflection_point_candidates[undulation_points]
            inflection_point_candidates = np.setdiff1d(
                inflection_point_candidates,
                undulation_indices,
                assume_unique=True,
            )

        dNy = np.gradient(dNy, x_local)
        is_odd_derivative = not is_odd_derivative

    return InflectionPointsResult(verified_inflection_points, x_local[verified_inflection_points])
