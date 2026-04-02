"""
cupy_thunk: numpy-backed fallback for code written against the CuPy API.

When CuPy is not available (or not active), code can import this module as
their computation namespace and get numpy behaviour transparently.  All
public numpy symbols are re-exported so that attribute lookups like
``xp.zeros``, ``xp.linalg.inv``, ``xp.random.standard_normal`` all resolve.
"""
import numpy as np
import numpy.linalg as linalg
import numpy.random as random
import types
from numpy import *  # noqa: F401, F403  # type: ignore[reportWildcardImportFromLibrary]
from numpy.typing import NDArray


def get_array_module(array: NDArray) -> types.ModuleType:
    """Return the array module for *array*.

    For the numpy thunk, any array was created by numpy, so we always return
    the numpy module itself.
    """
    return np


ndarray = np.ndarray
