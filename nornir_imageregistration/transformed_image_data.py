
from __future__ import annotations
import abc
from numpy.typing import NDArray
from typing import Tuple

class ITransformedImageData(abc.ABC):

    @abc.abstractmethod
    def image(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def centerDistanceImage(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def source_space_scale(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def target_space_scale(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def rendered_target_space_origin(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def Create(cls, image: NDArray, centerDistanceImage: NDArray,
               transform,
               source_space_scale: float, target_space_scale: float,
               rendered_target_space_origin: Tuple[float, float], SingleThreadedInvoke: bool) -> ITransformedImageData:
        raise NotImplementedError()

    @abc.abstractmethod
    def Clear(self):
        raise NotImplementedError()
