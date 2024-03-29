from typing import Sequence

import numpy as np
from numpy.typing import NDArray

"""A type for points used for typing"""
PointLike = Sequence[float] | tuple[float, float] | NDArray[np.floating]
AreaLike = Sequence[float] | tuple[float, float] | NDArray[np.floating]
ShapeLike = Sequence[int] | tuple[int, int] | NDArray[np.integer]
RectLike = NDArray | Sequence[float] | tuple[float, float, float, float]
