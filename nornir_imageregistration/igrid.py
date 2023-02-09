from abc import ABC, abstractmethod

from numpy.typing import NDArray


class IGrid(ABC):

    @property
    @abstractmethod
    def cell_size(self) -> NDArray[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def grid_dims(self) -> NDArray[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def grid_spacing(self) -> NDArray[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def coords(self) -> NDArray[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def TargetPoints(self) -> NDArray[float]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def SourcePoints(self) -> NDArray[float]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def source_shape(self) -> NDArray[int]:
        raise NotImplementedError()

    @property
    def axis_points(self) -> list[NDArray[float]]:
        """The points along the axis, in source space, where the grid lines intersect the axis"""
        raise NotImplementedError()
