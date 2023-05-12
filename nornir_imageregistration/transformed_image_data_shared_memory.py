'''
Created on Jul 18, 2019

@author: u0490822

A helper class to marshal large images using the file system instead of in-memory. 
'''
from __future__ import annotations
import os
import weakref
import multiprocessing
import multiprocessing.shared_memory as shared_memory
from typing import Tuple
import tempfile
import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_pools
import logging
import shutil
import atexit

from nornir_imageregistration.shared_mem_metadata import Shared_Mem_Metadata
from numpy.distutils.fcompiler import none
from transformed_image_data import ITransformedImageData

class TransformedImageDataViaSharedMemory(ITransformedImageData):
    _image_shared_mem_meta : Shared_Mem_Metadata
    _center_distance_image_shared_mem_meta : Shared_Mem_Metadata
    _image: NDArray[float]
    _centerDistanceImage: NDArray[float]
    _source_space_scale: float
    _target_space_scale: float
    #_transform: nornir_imageregistration.ITransform
    _errmsg: str

    '''
    Returns data from multiprocessing thread processes.  Uses memory mapped files when there is too much data for pickle to be efficient.
    '''

    memmap_threshold: int = 64 * 64

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

    @property
    def image_shared_mem_meta(self) -> Shared_Mem_Metadata:
        return self._image_shared_mem_meta

    @property
    def image(self):
        if self._image is None:
            if self._image_shared_mem_meta is not None:
                self._image = nornir_imageregistration.ImageParamToImageArray(self._image_shared_mem_meta)
        
        if self._image is None:
            raise ValueError("No image associated with TransformedImageData")

        return self._image

    @property
    def center_distance_image_mem_meta(self) -> Shared_Mem_Metadata:
        return self._center_distance_image_shared_mem_meta

    @property
    def centerDistanceImage(self):
        if self._centerDistanceImage is None:
            if self._center_distance_image_shared_mem_meta is not None:
                self._centerDistanceImage = nornir_imageregistration.ImageParamToImageArray(self._center_distance_image_shared_mem_meta)
        
        if self._centerDistanceImage is None:
            raise ValueError("No distance image associated with TransformedImageData")

        return self._centerDistanceImage

    @property
    def source_space_scale(self):
        return self._source_space_scale

    @property
    def target_space_scale(self):
        return self._target_space_scale

    @property
    def rendered_target_space_origin(self):
        """
        The bottom left origin of the transformed data.  When requesting an assembled image for a target region
        rounding sometimes can occur this property contains the actual bottom left coordinate of the image data
        in target space.
        :return:
        """
        return self._rendered_target_space_origin

    #@property
    #def transform(self):
    #    return self._transform

    @property
    def errormsg(self):
        return self._errmsg

    @classmethod
    def Create(cls, image: NDArray, centerDistanceImage: NDArray,
               transform,
               source_space_scale: float, target_space_scale: float,
               rendered_target_space_origin: Tuple[float, float], SingleThreadedInvoke: bool) -> TransformedImageData:
        o = TransformedImageData()

        if isinstance(image, nornir_imageregistration.Shared_Mem_Metadata):
            o._image_shared_mem_meta = image
            o._image = None
        else:
            o._image = image
            o._image_shared_mem_meta = None

        if isinstance(centerDistanceImage, nornir_imageregistration.Shared_Mem_Metadata):
            o._centerDistanceImage = None
            o._center_distance_image_shared_mem_meta = centerDistanceImage
        else:
            o._centerDistanceImage = centerDistanceImage
            o._center_distance_image_shared_mem_meta = None

        #o._image_path = None
        #o._centerDistanceImage_path = None
        o._source_space_scale = source_space_scale
        o._target_space_scale = target_space_scale
        o._rendered_target_space_origin = np.array(rendered_target_space_origin, dtype=np.float32)
        # o._transform = transform

        #if not SingleThreadedInvoke:
        #    o.ConvertToMemmapIfLarge()

        return o

    def ConvertToSharedMemory(self):
        if self._image_shared_mem_meta is None and self._image is not None:
            self._image_shared_mem_meta, self._image = nornir_imageregistration.npArrayToReadOnlySharedArray(self._image)

        if self._center_distance_image_shared_mem_meta is None and self._centerDistanceImage is not None:
            self._center_distance_image_shared_mem_meta, self._centerDistanceImage = nornir_imageregistration.npArrayToReadOnlySharedArray(self._centerDistanceImage)

    def Clear(self):
        if self._image_shared_mem_meta is not None:
            nornir_imageregistration.unlink_shared_memory(self._image_shared_mem_meta)
            self._image_shared_mem_meta = None

        if self._center_distance_image_shared_mem_meta is not None:
            nornir_imageregistration.unlink_shared_memory(self._center_distance_image_shared_mem_meta)
            self._center_distance_image_shared_mem_meta = None

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

    @staticmethod
    def CreateMemoryMappedFilesForImage(name: str, image: NDArray) -> str:
        """
        Save the image to a temporary file and return the name of the temporary file
        :param name: Suffix to prepend to the filename
        :param image: NDArray to save
        :return: name of temporary file
        """

        if image is None:
            return None

        with tempfile.NamedTemporaryFile(suffix=name, dir=_sharedTempRoot, delete=False) as tfile:
            tempfilename = tfile.name
            np.save(tfile, image)
            # memmapped_image = np.memmap(tempfilename, dtype=image.dtype, mode='w+', shape=image.shape)
            # np.copyto(memmapped_image, image)
            # print("Write %s" % tempfilename)

            # del memmapped_image
            return tempfilename

    def Clear(self):
        """Sets attributes to None to encourage garbage collection"""
        #self._image = None
        #self._centerDistanceImage = None
        #self._source_space_scale = None
        #self._target_space_scale = None
        #self._transform = None

        #if self._centerDistanceImage_path is not None or self._image_path is not None:
        #    pool = nornir_pools.GetGlobalMultithreadingPool()
        #    pool.add_task(self._image_path, TransformedImageData._RemoveTempFiles, self._centerDistanceImage_path,
        #               self._image_path, self._tempdir)
        return

    @staticmethod
    def _RemoveTempFiles(_centerDistanceImage_path, _image_path, _tempdir):
        try:
            if _centerDistanceImage_path is not None:
                os.remove(_centerDistanceImage_path)
        except IOError as E:
            logging.warning("Could not delete temporary file {0}".format(_centerDistanceImage_path))
            pass

        try:
            if _image_path is not None:
                os.remove(_image_path)
        except IOError as E:
            logging.warning("Could not delete temporary file {0}".format(_image_path))
            pass

    def __init__(self, errorMsg: str | None = None):
        #self._image_shared_mem_finalizer = None
        #self._distance_image_shared_mem_finalizer = None
        self._image_shared_mem_meta = None
        self._center_distance_image_shared_mem_meta = None
        self._image = None
        self._centerDistanceImage = None
        self._source_space_scale = None
        self._target_space_scale = None
        self._transform = None
        self._errmsg = errorMsg
        #self._image_path = None
        #self._centerDistanceImage_path = None
        #self._tempdir = None
        # self._image_shape = None
        # self._centerDistanceImage_shape = None
        # self._image_dtype = None
        # self._centerDistance_dtype = None

    def __getstate__(self):
        self.ConvertToSharedMemory() 
        odict = {}
        odict.update(self.__dict__)
        del odict['_image']
        del odict['_centerDistanceImage']
    
        return odict
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._image = None
        self._centerDistanceImage = None
