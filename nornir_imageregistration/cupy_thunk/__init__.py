import numpy as np
from numpy.typing import NDArray


def get_array_module(array: NDArray) -> np.ndarray:
    return np


ndarray = np.ndarray
