from typing import Sequence
import numpy as np
from numpy.typing import NDArray

"""A type for points used for typing"""
RectLike = NDArray[np.floating | np.integer] | Sequence[float] | tuple[float, float, float, float]
