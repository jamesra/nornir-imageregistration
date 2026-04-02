from __future__ import annotations

import abc
from enum import Enum, auto
from typing import Any, Tuple

from numpy.typing import NDArray

import nornir_imageregistration


class TransformedImageDataState(Enum):
    IN_MEMORY = auto()
    SHARED_MEMORY = auto()
    TEMP_FILE = auto()
    CLEARED = auto()


class ITransformedImageData(abc.ABC):

    @property
    @abc.abstractmethod
    def image(self) -> NDArray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def centerDistanceImage(self) -> NDArray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def source_space_scale(self) -> float:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def target_space_scale(self) -> float:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def rendered_target_space_origin(self) -> NDArray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def state(self) -> TransformedImageDataState:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def Create(cls, image: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
               centerDistanceImage: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
               transform,
               source_space_scale: float, target_space_scale: float,
               rendered_target_space_origin: Tuple[float, float], SingleThreadedInvoke: bool) -> ITransformedImageData:
        raise NotImplementedError()

    @abc.abstractmethod
    def Clear(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def errormsg(self) -> str | None:
        raise NotImplementedError()


class TransformedImageDataError(ITransformedImageData):
    _errmsg: str

    @property
    def image(self) -> NDArray:
        raise ValueError(self._errmsg)

    @property
    def centerDistanceImage(self) -> NDArray:
        raise ValueError(self._errmsg)

    @property
    def source_space_scale(self) -> float:
        raise ValueError(self._errmsg)

    @property
    def target_space_scale(self) -> float:
        raise ValueError(self._errmsg)

    @property
    def rendered_target_space_origin(self) -> NDArray:
        raise ValueError(self._errmsg)

    @property
    def state(self) -> TransformedImageDataState:
        return TransformedImageDataState.CLEARED

    @classmethod
    def Create(cls, image: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
               centerDistanceImage: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
               transform: Any,
               source_space_scale: float,
               target_space_scale: float,
               rendered_target_space_origin: Tuple[float, float],
               SingleThreadedInvoke: bool) -> ITransformedImageData:
        raise NotImplementedError("TransformedImageDataError cannot be created from image data")

    def Clear(self):
        return

    @property
    def errormsg(self) -> str | None:
        return self._errmsg

    def __init__(self, error_msg: str):
        self._errmsg = error_msg
