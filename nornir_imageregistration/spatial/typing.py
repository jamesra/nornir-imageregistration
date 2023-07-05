from typing import Sequence

from numpy.typing import NDArray

"""A type for points used for typing"""
PointLike = Sequence[float] | tuple[float, float] | NDArray[float]
RectLike = NDArray | Sequence[float] | tuple[float, float, float, float]
