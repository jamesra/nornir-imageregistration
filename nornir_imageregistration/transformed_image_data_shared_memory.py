'''
Created on Jul 18, 2019

@author: u0490822

A helper class to marshal large images using the file system instead of in-memory. 
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.shared_mem_metadata import Shared_Mem_Metadata
from nornir_imageregistration.transformed_image_data import ITransformedImageData, TransformedImageDataState


@dataclass
class _InMemoryImageState:
    image: NDArray


@dataclass
class _SharedMemoryImageState:
    metadata: Shared_Mem_Metadata
    image: NDArray | None = None


class _ClearedImageState:
    pass


_CLEARED_IMAGE_STATE = _ClearedImageState()
_ImageState = _InMemoryImageState | _SharedMemoryImageState | _ClearedImageState


class TransformedImageDataViaSharedMemory(ITransformedImageData):
    _image_state: _ImageState
    _center_distance_image_state: _ImageState
    _source_space_scale: float
    _target_space_scale: float
    _rendered_target_space_origin: NDArray[np.float32]
    _transform: Any | None
    _errmsg: str | None

    @staticmethod
    def _state_from_input(value: NDArray | Shared_Mem_Metadata) -> _ImageState:
        if isinstance(value, Shared_Mem_Metadata):
            return _SharedMemoryImageState(metadata=value)
        return _InMemoryImageState(image=value)

    @staticmethod
    def _metadata_only_state(value: _ImageState) -> _ImageState:
        if isinstance(value, _SharedMemoryImageState):
            return _SharedMemoryImageState(metadata=value.metadata)
        return value

    @property
    def errormsg(self) -> str | None:
        return self._errmsg

    @property
    def state(self) -> TransformedImageDataState:
        if isinstance(self._image_state, _ClearedImageState) or isinstance(self._center_distance_image_state, _ClearedImageState):
            return TransformedImageDataState.CLEARED
        if isinstance(self._image_state, _SharedMemoryImageState) or isinstance(self._center_distance_image_state,
                                                                                _SharedMemoryImageState):
            return TransformedImageDataState.SHARED_MEMORY
        return TransformedImageDataState.IN_MEMORY

    '''
    Returns data from multiprocessing thread processes.  Uses memory mapped files when there is too much data for pickle to be efficient.
    '''

    memmap_threshold: int = 64 * 64

    @property
    def image_shared_mem_meta(self) -> Shared_Mem_Metadata | None:
        if isinstance(self._image_state, _SharedMemoryImageState):
            return self._image_state.metadata
        return None

    @property
    def image(self) -> NDArray:
        if isinstance(self._image_state, _InMemoryImageState):
            return self._image_state.image

        if isinstance(self._image_state, _SharedMemoryImageState):
            shared_state = self._image_state
            image = shared_state.image
            if image is None:
                image = nornir_imageregistration.ImageParamToImageArray(shared_state.metadata)
                self._image_state = _SharedMemoryImageState(metadata=shared_state.metadata, image=image)
            return image

        raise ValueError("No image associated with TransformedImageData")

    @property
    def center_distance_image_mem_meta(self) -> Shared_Mem_Metadata | None:
        if isinstance(self._center_distance_image_state, _SharedMemoryImageState):
            return self._center_distance_image_state.metadata
        return None

    @property
    def centerDistanceImage(self) -> NDArray:
        if isinstance(self._center_distance_image_state, _InMemoryImageState):
            return self._center_distance_image_state.image

        if isinstance(self._center_distance_image_state, _SharedMemoryImageState):
            shared_state = self._center_distance_image_state
            image = shared_state.image
            if image is None:
                image = nornir_imageregistration.ImageParamToImageArray(shared_state.metadata)
                self._center_distance_image_state = _SharedMemoryImageState(metadata=shared_state.metadata, image=image)
            return image

        raise ValueError("No distance image associated with TransformedImageData")

    @property
    def source_space_scale(self) -> float:
        return self._source_space_scale

    @property
    def target_space_scale(self) -> float:
        return self._target_space_scale

    @property
    def rendered_target_space_origin(self) -> NDArray[np.float32]:
        """
        The bottom left origin of the transformed data.  When requesting an assembled image for a target region
        rounding sometimes can occur this property contains the actual bottom left coordinate of the image data
        in target space.
        :return:
        """
        return self._rendered_target_space_origin

    # @property
    # def transform(self):
    #    return self._transform

    @classmethod
    def Create(cls, image: NDArray | Shared_Mem_Metadata, centerDistanceImage: NDArray | Shared_Mem_Metadata,
               transform,
               source_space_scale: float, target_space_scale: float,
               rendered_target_space_origin: Tuple[float, float],
               SingleThreadedInvoke: bool) -> TransformedImageDataViaSharedMemory:
        o = TransformedImageDataViaSharedMemory(source_space_scale=source_space_scale,
                                                target_space_scale=target_space_scale,
                                                rendered_target_space_origin=rendered_target_space_origin)
        o._image_state = cls._state_from_input(image)
        o._center_distance_image_state = cls._state_from_input(centerDistanceImage)
        o._transform = transform

        # if not SingleThreadedInvoke:
        #    o.ConvertToMemmapIfLarge()

        return o

    def ConvertToSharedMemory(self):
        if isinstance(self._image_state, _InMemoryImageState):
            metadata, image = nornir_imageregistration.npArrayToSharedArray(self._image_state.image)
            self._image_state = _SharedMemoryImageState(metadata=metadata, image=image)

        if isinstance(self._center_distance_image_state, _InMemoryImageState):
            metadata, image = nornir_imageregistration.npArrayToSharedArray(self._center_distance_image_state.image)
            self._center_distance_image_state = _SharedMemoryImageState(metadata=metadata, image=image)

    def Clear(self):
        if isinstance(self._image_state, _SharedMemoryImageState):
            nornir_imageregistration.unlink_shared_memory(self._image_state.metadata)
        self._image_state = _CLEARED_IMAGE_STATE

        if isinstance(self._center_distance_image_state, _SharedMemoryImageState):
            nornir_imageregistration.unlink_shared_memory(self._center_distance_image_state.metadata)
        self._center_distance_image_state = _CLEARED_IMAGE_STATE

    # def ConvertToMemmapIfLarge(self):
    #     if np.prod(self._image.shape) > TransformedImageData.memmap_threshold:
    #         self._image_path = self.CreateMemoryMappedFilesForImage("Image", self._image)
    #         # self._image_shape = self._image.shape
    #         # self._image_dtype = self._image.dtype
    #         self._image = None
    #
    #     if np.prod(self._centerDistanceImage.shape) > TransformedImageData.memmap_threshold:
    #         self._centerDistanceImage_path = self.CreateMemoryMappedFilesForImage("Distance", self._centerDistanceImage)
    #         # self._centerDistanceImage_shape = self._centerDistanceImage.shape
    #         # self._centerDistance_dtype = self._centerDistanceImage.dtype
    #         self._centerDistanceImage = None
    #
    #     return

    def __init__(self,
                 source_space_scale: float = 0.0,
                 target_space_scale: float = 0.0,
                 rendered_target_space_origin: Tuple[float, float] = (0.0, 0.0),
                 errorMsg: str | None = None):
        self._image_state = _CLEARED_IMAGE_STATE
        self._center_distance_image_state = _CLEARED_IMAGE_STATE
        self._source_space_scale = source_space_scale
        self._target_space_scale = target_space_scale
        self._rendered_target_space_origin = np.asarray(rendered_target_space_origin, dtype=np.float32)
        self._transform = None
        self._errmsg = errorMsg
        # self._image_path = None
        # self._centerDistanceImage_path = None
        # self._tempdir = None
        # self._image_shape = None
        # self._centerDistanceImage_shape = None
        # self._image_dtype = None
        # self._centerDistance_dtype = None

    def __getstate__(self):
        self.ConvertToSharedMemory()
        return {
            "_image_state": self._metadata_only_state(self._image_state),
            "_center_distance_image_state": self._metadata_only_state(self._center_distance_image_state),
            "_source_space_scale": self._source_space_scale,
            "_target_space_scale": self._target_space_scale,
            "_rendered_target_space_origin": self._rendered_target_space_origin,
            "_transform": self._transform,
            "_errmsg": self._errmsg,
        }

    def __setstate__(self, state):
        if not isinstance(state, Mapping):
            raise TypeError(f"Invalid state type: {type(state)!r}")

        self._image_state = state.get("_image_state", _CLEARED_IMAGE_STATE)
        if "_image_state" not in state and isinstance(state.get("_image_shared_mem_meta"), Shared_Mem_Metadata):
            self._image_state = _SharedMemoryImageState(state["_image_shared_mem_meta"])

        self._center_distance_image_state = state.get("_center_distance_image_state", _CLEARED_IMAGE_STATE)
        if "_center_distance_image_state" not in state and isinstance(state.get("_center_distance_image_shared_mem_meta"),
                                                                        Shared_Mem_Metadata):
            self._center_distance_image_state = _SharedMemoryImageState(state["_center_distance_image_shared_mem_meta"])

        self._source_space_scale = float(state.get("_source_space_scale", 0.0))
        self._target_space_scale = float(state.get("_target_space_scale", 0.0))
        self._rendered_target_space_origin = np.asarray(state.get("_rendered_target_space_origin", (0.0, 0.0)),
                                                        dtype=np.float32)
        self._transform = state.get("_transform")
        self._errmsg = state.get("_errmsg")
