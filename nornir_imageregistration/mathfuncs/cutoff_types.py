import enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class InflectionPointsResult(NamedTuple):
    indices: NDArray[np.integer]
    values: NDArray[np.floating]


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
    cutoff_percentile_index: int
    highest_inflection_point: int
    cutoff_value: float
    y_fit: NDArray[np.floating] | None
