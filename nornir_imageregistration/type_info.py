import numpy as np
from numpy.typing import NDArray
from typing import Sequence

"""Describes a point using floats"""
PointLike = Sequence[float] | tuple[float, float] | NDArray[np.floating]

"""Describes a vector or offset using floats"""
VectorLike = Sequence[float] | tuple[float, float] | NDArray[np.floating]

"""Describes an area using floats"""
AreaLike = Sequence[float] | tuple[float, float] | NDArray[np.floating]

"""Describes an array shape or any 2D area using integers"""
ShapeLike = Sequence[int] | tuple[int, int] | NDArray[np.integer]
