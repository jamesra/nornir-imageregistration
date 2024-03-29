import numpy as np
import types
from numpy.typing import NDArray


def get_array_module(array: NDArray) -> types.ModuleType:
    """Returns the module that created the array.  If we have a cupy thunk,
     that means it was already created by numpy"""
    return np


ndarray = np.ndarray
