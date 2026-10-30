from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class IGrid(ABC):

    @property
    @abstractmethod
    def cell_size(self) -> NDArray[np.integer]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def grid_dims(self) -> NDArray[np.integer]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def grid_spacing(self) -> NDArray[np.integer]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def coords(self) -> NDArray[np.integer]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def TargetPoints(self) -> NDArray[np.floating]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def SourcePoints(self) -> NDArray[np.floating]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def source_shape(self) -> NDArray[np.integer]:
        raise NotImplementedError()

    @property
    def axis_points(self) -> list[NDArray[np.floating]]:
        """The points along the axis, in source space, where the grid lines intersect the axis"""
        raise NotImplementedError()
