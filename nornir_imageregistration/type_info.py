from __future__ import annotations

from dataclasses import dataclass

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
RectLike = NDArray[np.floating | np.integer] | ShapeLike | Sequence[float] | tuple[float, float, float, float]


@dataclass(frozen=True)
class Shape:
    """Integer image shape in NumPy order: ``y`` (height) then ``x`` (width).

    Index 0 is ``y`` and index 1 is ``x``, so this is a :data:`ShapeLike`.
    ``from_xy`` is the edge where XML ``TileXDim`` / ``TileYDim`` are swapped
    into that order.
    """

    y: int
    x: int

    @classmethod
    def from_xy(cls, x: int, y: int) -> Shape:
        """Build a shape from X-then-Y sizes, such as tileset XML attributes."""
        return cls(y=int(y), x=int(x))

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        yield self.y
        yield self.x

    def __getitem__(self, index: int) -> int:
        if index in (0, -2):
            return self.y
        if index in (1, -1):
            return self.x
        raise IndexError(index)

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> NDArray[np.integer]:
        del copy
        return np.array((self.y, self.x), dtype=np.int64 if dtype is None else dtype)
