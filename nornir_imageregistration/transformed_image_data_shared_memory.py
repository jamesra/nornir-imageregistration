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

from nornir_imageregistration.shared_mem_metadata import Shared_Mem_Metadata 
from nornir_imageregistration.transformed_image_data import ITransformedImageData

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
               rendered_target_space_origin: Tuple[float, float], SingleThreadedInvoke: bool) -> TransformedImageDataViaSharedMemory:
        o = TransformedImageDataViaSharedMemory()

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
            self._image_shared_mem_meta, self._image = nornir_imageregistration.npArrayToSharedArray(self._image)

        if self._center_distance_image_shared_mem_meta is None and self._centerDistanceImage is not None:
            self._center_distance_image_shared_mem_meta, self._centerDistanceImage = nornir_imageregistration.npArrayToSharedArray(
                self._centerDistanceImage)

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
 

    def __init__(self, errorMsg: str | None = None):
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
