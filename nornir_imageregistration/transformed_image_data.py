from __future__ import annotations

import abc
from typing import Tuple

from numpy.typing import NDArray

import nornir_imageregistration


class ITransformedImageData(abc.ABC):

    @abc.abstractmethod
    def image(self) -> NDArray | nornir_imageregistration.Shared_Mem_Metadata:
        raise NotImplementedError()

    @abc.abstractmethod
    def centerDistanceImage(self) -> NDArray | nornir_imageregistration.Shared_Mem_Metadata:
        raise NotImplementedError()

    @abc.abstractmethod
    def source_space_scale(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def target_space_scale(self) -> float:
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

    def errormsg(self) -> str | None:
        raise NotImplementedError()


class TransformedImageDataError(ITransformedImageData):
    _errmsg: str

    @property
    def errormsg(self) -> str | None:
        return self._errmsg

    def __init__(self, error_msg: str):
        self._errmsg = error_msg
