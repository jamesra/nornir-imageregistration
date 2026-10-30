import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

def calculate_deviation(
    values: NDArray[np.floating], above_index: int | None = None
) -> NDArray[np.floating]:
    """Compute signed distance (cross product) of percentile points from the min-to-max line."""
    xp = cp.get_array_module(values)
    p = xp.linspace(0, 100, 101)
    start_index = 0 if above_index is None else above_index
    try:
        percentile_values = xp.percentile(values, p, method="linear")
    except TypeError:
        # Some CuPy versions don't support the NumPy `method` kwarg.
        percentile_values = xp.percentile(values, p)

    percentile_value_subset = percentile_values[start_index:]
    percentile_subset = p[start_index:]
    min_value = percentile_value_subset[0]
    max_value = percentile_value_subset[-1]

    min_vectors = xp.vstack(
        xp.array(
            (
                (percentile_value_subset - min_value),  # type: ignore[arg-type]
                percentile_subset - percentile_subset[0],
                xp.zeros(len(percentile_subset)),
            )
        )
    ).T
    max_vectors = xp.vstack(
        xp.array(
            (
                (max_value - percentile_value_subset),  # type: ignore[arg-type]
                percentile_subset - percentile_subset[-1],
                xp.zeros(len(percentile_subset)),
            )
        )
    ).T

    cross_products = xp.cross(min_vectors, max_vectors)
    return xp.vstack((percentile_subset, cross_products[:, 2])).T
