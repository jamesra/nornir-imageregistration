'''
Created on Jul 18, 2019

@author: u0490822

A helper class to marshal large images using the file system instead of in-memory. 
'''
import os
from typing import Tuple
import tempfile
import numpy as np
from numpy.typing import NDArray
import nornir_pools
import logging
import shutil

#import atexit

_sharedTempRoot = tempfile.mkdtemp(prefix="nornir-imageregistration.transformed_image_data.", dir=tempfile.gettempdir())

# When porting to Python 3.10 there was a regression where
# concurrent.futures.ThreadPoolExecutor system could not function in atext calls
# So I reverted to shutil until it is fixed
# atexit.register(nornir_shared.files.rmtree, _sharedTempRoot)
#atexit.register(shutil.rmtree, _sharedTempRoot)

def SaveArrayToTemporaryFile(name: str, image: NDArray) -> str:
    """
    Save the image to a temporary file and return the name of the temporary file
    :param name: Suffix to prepend to the filename
    :param image: NDArray to save
    :return: name of temporary file
    """
    if image is None:
        raise ValueError("image cannot be None")

    with tempfile.NamedTemporaryFile(suffix=name, dir=_sharedTempRoot, delete=False) as tfile:
        np.save(tfile, image)
        return tfile.name


class TransformedImageData(object):
    '''
    Returns data from multiprocessing thread processes.  Uses memory mapped files when there is too much data for pickle to be efficient
    '''

    memmap_threshold = 64 * 64

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
    def image(self):
        if self._image is None:
            if self._image_path is None:
                return None

            self._image = np.load(self._image_path,
                                  mmap_mode='r')  # np.memmap(self._image_path, mode='c', shape=self._image_shape, dtype=self._image_dtype)

        return self._image

    @property
    def centerDistanceImage(self):
        if self._centerDistanceImage is None:
            if self._centerDistanceImage_path is None:
                return None

            self._centerDistanceImage = np.load(self._centerDistanceImage_path,
                                                mmap_mode='r')  # np.memmap(self._centerDistanceImage_path, mode='c', shape=self._centerDistanceImage_shape, dtype=self._centerDistance_dtype)

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

    @property
    def transform(self):
        return self._transform

    @property
    def errormsg(self):
        return self._errmsg

    @classmethod
    def Create(cls, image: NDArray, centerDistanceImage: NDArray,
               transform,
               source_space_scale: float, target_space_scale: float,
               rendered_target_space_origin: Tuple[float, float], SingleThreadedInvoke: bool):
        o = TransformedImageData()
        o._image = image
        o._centerDistanceImage = centerDistanceImage

        o._image_path = None
        o._centerDistanceImage_path = None
        o._source_space_scale = source_space_scale
        o._target_space_scale = target_space_scale
        o._rendered_target_space_origin = np.array(rendered_target_space_origin, dtype=np.float32)
        # o._transform = transform

        if not SingleThreadedInvoke:
            o.ConvertToTempFileIfLarge()

        return o

    def ConvertToMemmapIfLarge(self):
        if np.prod(self._image.shape) > TransformedImageData.memmap_threshold:
            self._image_path = self.SaveArrayToTemporaryFile("Image", self._image)
            self._image = None

        if np.prod(self._centerDistanceImage.shape) > TransformedImageData.memmap_threshold:
            self._centerDistanceImage_path = self.SaveArrayToTemporaryFile("Distance", self._centerDistanceImage)
            self._centerDistanceImage = None

        return

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

    def ConvertToTempFileIfLarge(self):
        '''
        Save our image data into files.  This gets it out of memory, lowering our footprint.  When we return
        to the calling process we do not need to marshal the images across a pipe.  This was replaced by
        use of SharedMemory, but that implementation seems to destroy the sharedmemory before it can be
        returned to the caller.
        :return:
        '''
        if np.prod(self._image.shape) > TransformedImageData.memmap_threshold:
            self._image_path = self.CreateMemoryMappedFilesForImage("Image", self._image)
            # self._image_shape = self._image.shape
            # self._image_dtype = self._image.dtype
            self._image = None

        if np.prod(self._centerDistanceImage.shape) > TransformedImageData.memmap_threshold:
            self._centerDistanceImage_path = self.CreateMemoryMappedFilesForImage("Distance", self._centerDistanceImage)
            # self._centerDistanceImage_shape = self._centerDistanceImage.shape
            # self._centerDistance_dtype = self._centerDistanceImage.dtype
            self._centerDistanceImage = None

        return

    def Clear(self):
        """Sets attributes to None to encourage garbage collection"""
        self._image = None
        self._centerDistanceImage = None
        self._source_space_scale = None
        self._target_space_scale = None
        self._transform = None

        if self._centerDistanceImage_path is not None or self._image_path is not None:
            pool = nornir_pools.GetGlobalMultithreadingPool()
            pool.add_task(self._image_path, TransformedImageData._RemoveTempFiles, self._centerDistanceImage_path,
                          self._image_path, self._tempdir)

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

    def __init__(self, errorMsg=None):
        self._image = None
        self._centerDistanceImage = None
        self._source_space_scale = None
        self._target_space_scale = None
        self._transform = None
        self._errmsg = errorMsg
        self._image_path = None
        self._centerDistanceImage_path = None
        self._tempdir = None
        # self._image_shape = None
        # self._centerDistanceImage_shape = None
        # self._image_dtype = None
        # self._centerDistance_dtype = None
