'''
Created on Jul 18, 2019

@author: u0490822

A helper class to marshal large images using the file system instead of in-memory.
'''
from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration.shared_mem_metadata import Shared_Mem_Metadata
from nornir_imageregistration.transformed_image_data import ITransformedImageData, TransformedImageDataState


# When porting to Python 3.10 there was a regression where
# concurrent.futures.ThreadPoolExecutor system could not function in atext calls
# So I reverted to shutil until it is fixed
# atexit.register(nornir_shared.files.rmtree, _sharedTempRoot)
# atexit.register(shutil.rmtree, _sharedTempRoot, ignore_errors=True)


class TransformedImageDataViaTempFile(ITransformedImageData):
    """
    Returns data from multiprocessing thread processes.  Uses memory mapped files when there is too much data for pickle to be efficient
    """
    _image_path: str | None
    _centerDistanceImage_path: str | None
    _image: NDArray[np.floating] | None
    _centerDistanceImage: NDArray[np.floating] | None
    _source_space_scale: float
    _target_space_scale: float
    _image_state: TransformedImageDataState
    _center_distance_image_state: TransformedImageDataState
    _transform: Any | None
    _errmsg: str | None
    _rendered_target_space_origin: NDArray[np.float32]

    _temp_folder_created = False
    sharedTempRoot = None

    tempfile_threshold = 64 * 64

    @property
    def errormsg(self) -> str | None:
        return self._errmsg

    @property
    def state(self) -> TransformedImageDataState:
        if self._image_state == TransformedImageDataState.CLEARED or \
                self._center_distance_image_state == TransformedImageDataState.CLEARED:
            return TransformedImageDataState.CLEARED
        if self._image_state == TransformedImageDataState.TEMP_FILE or \
                self._center_distance_image_state == TransformedImageDataState.TEMP_FILE:
            return TransformedImageDataState.TEMP_FILE
        return TransformedImageDataState.IN_MEMORY

    #
    #     def __getstate__(self):
    #         odict = {}
    #         odict["_image"] = self._image
    #         odict["_centerDistanceImage"] = self._centerDistanceImage
    #         odict["_source_space_scale"] = self._source_space_scale
    #         odict["_transform"] = self._transform
    #         odict["_errmsg"] = self._errmsg
    #         odict["_image_path"] = self._image_path
    #         odict["_centerDistanceImage_path"] = self._centerDistanceImage_path
    #         odict["_tempdir"] = self._tempdir
    #         odict["_image_shape"] = self._image_shape
    #         odict["_centerDistanceImage_shape"] = self._centerDistanceImage_shape
    #         odict["_image_dtype"] = self._image_dtype
    #         odict["_centerDistance_dtype"] = self._centerDistance_dtype
    #         return odict
    #
    #     def __setstate__(self, dictionary):
    #         self.__dict__.update(dictionary)
    #

    @property
    def image(self) -> NDArray:
        if self._image_state == TransformedImageDataState.CLEARED:
            raise ValueError("No image associated with TransformedImageData")

        image = self._image
        if image is None:
            if self._image_path is None:
                raise ValueError("No image associated with TransformedImageData")

            image = np.load(self._image_path,
                            mmap_mode='r')  # np.memmap(self._image_path, mode='c', shape=self._image_shape, dtype=self._image_dtype)
            self._image = image

        return image

    @property
    def centerDistanceImage(self) -> NDArray:
        if self._center_distance_image_state == TransformedImageDataState.CLEARED:
            raise ValueError("No distance image associated with TransformedImageData")

        distance_image = self._centerDistanceImage
        if distance_image is None:
            if self._centerDistanceImage_path is None:
                raise ValueError("No distance image associated with TransformedImageData")

            distance_image = np.load(self._centerDistanceImage_path,
                                     mmap_mode='r')  # np.memmap(self._centerDistanceImage_path, mode='c', shape=self._centerDistanceImage_shape, dtype=self._centerDistance_dtype)
            self._centerDistanceImage = distance_image

        return distance_image

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
               rendered_target_space_origin: Tuple[float, float], SingleThreadedInvoke: bool):
        o = TransformedImageDataViaTempFile(source_space_scale=source_space_scale,
                                            target_space_scale=target_space_scale,
                                            rendered_target_space_origin=rendered_target_space_origin)
        o._image = nornir_imageregistration.ImageParamToImageArray(image)
        o._centerDistanceImage = nornir_imageregistration.ImageParamToImageArray(centerDistanceImage)
        o._image_state = TransformedImageDataState.IN_MEMORY
        o._center_distance_image_state = TransformedImageDataState.IN_MEMORY
        o._transform = transform

        o._image_path = None
        o._centerDistanceImage_path = None
        # o._transform = transform

        if not SingleThreadedInvoke:
            o.ConvertToTempFileIfLarge()

        return o

    @staticmethod
    def SaveArrayToTemporaryFile(name: str, image: NDArray) -> str:
        """
        Save the image to a temporary file and return the name of the temporary file
        :param name: Suffix to prepend to the filename
        :param image: NDArray to save
        :return: name of temporary file
        """
        if image is None:
            raise ValueError("image cannot be None")

        with tempfile.NamedTemporaryFile(suffix=name + '.npy', dir=TransformedImageDataViaTempFile._sharedTempRoot,
                                         delete=False) as tfile:
            np.save(tfile, image)
            return tfile.name

    def ConvertToTempFileIfLarge(self):
        '''
        Save our image data into files.  This gets it out of memory, lowering our footprint.  When we return
        to the calling process we do not need to marshal the images across a pipe.  This was replaced by
        use of SharedMemory, but that implementation seems to destroy the sharedmemory before it can be
        returned to the caller.
        :return:
        '''
        image = self.image
        center_distance_image = self.centerDistanceImage
        if np.prod(image.shape) > TransformedImageDataViaTempFile.tempfile_threshold:
            _image_path_task = None
            _centerDistanceImage_path_task = None

            # Create the temporary directory if it doesn't exist
            if not TransformedImageDataViaTempFile._temp_folder_created:
                temp_dir = nornir_imageregistration.gettempdir()
                TransformedImageDataViaTempFile._sharedTempRoot = tempfile.mkdtemp(
                    prefix="nornir-imageregistration.transformed_image_data.", dir=temp_dir)
                TransformedImageDataViaTempFile._temp_folder_created = True
                atexit.register(shutil.rmtree, TransformedImageDataViaTempFile._sharedTempRoot, ignore_errors=True)

            # TODO: Replace with a task group once we are on Python 3.11
            pool = nornir_pools.GetGlobalThreadPool()

            _image_path_task = pool.add_task("Image", self.SaveArrayToTemporaryFile, "Image", image)
            self._image = None
            self._image_state = TransformedImageDataState.TEMP_FILE

            _centerDistanceImage_path_task = pool.add_task("Distance",
                                                           self.SaveArrayToTemporaryFile, "Distance",
                                                           center_distance_image)
            self._centerDistanceImage = None
            self._center_distance_image_state = TransformedImageDataState.TEMP_FILE

            self._image_path = _image_path_task.wait_return()
            self._centerDistanceImage_path = _centerDistanceImage_path_task.wait_return()

        return

    def Clear(self):
        """Release loaded arrays and any temporary files."""
        self._image = None
        self._centerDistanceImage = None
        self._image_state = TransformedImageDataState.CLEARED
        self._center_distance_image_state = TransformedImageDataState.CLEARED
        self._transform = None

        # It is hard to delete these temporary files because it is ambiguous on when
        # numpy releases the underlying file
        if self._centerDistanceImage_path is not None or self._image_path is not None:
            pool = nornir_pools.GetGlobalThreadPool()
            pool.add_task(str(self._image_path), TransformedImageDataViaTempFile._RemoveTempFiles,
                          self._centerDistanceImage_path,
                          self._image_path)
            self._centerDistanceImage_path = None
            self._image_path = None

    @staticmethod
    def _RemoveTempFiles(_centerDistanceImage_path, _image_path):
        try:
            if _centerDistanceImage_path is not None:
                os.remove(_centerDistanceImage_path)
        except FileNotFoundError:
            pass
        except IOError as E:
            logging.warning("Could not delete temporary file {0}".format(_centerDistanceImage_path))
            pass

        try:
            if _image_path is not None:
                os.remove(_image_path)
        except FileNotFoundError:
            pass
        except IOError as E:
            logging.warning("Could not delete temporary file {0}".format(_image_path))
            pass

    def __init__(self,
                 source_space_scale: float = 0.0,
                 target_space_scale: float = 0.0,
                 rendered_target_space_origin: Tuple[float, float] = (0.0, 0.0),
                 errorMsg: str | None = None):
        self._image = None
        self._centerDistanceImage = None
        self._source_space_scale = source_space_scale
        self._target_space_scale = target_space_scale
        self._rendered_target_space_origin = np.asarray(rendered_target_space_origin, dtype=np.float32)
        self._image_state = TransformedImageDataState.CLEARED
        self._center_distance_image_state = TransformedImageDataState.CLEARED
        self._transform = None
        self._errmsg = errorMsg
        self._image_path = None
        self._centerDistanceImage_path = None
        self._tempdir = None
        # self._image_shape = None
        # self._centerDistanceImage_shape = None
        # self._image_dtype = None
        # self._centerDistance_dtype = None
