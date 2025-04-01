"""
Implements a class that caches filter windows for use in image processing.  Examples are hamming and distance filters.

"""

from skimage.filters import window
import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing import Callable
import tempfile
import os

import skimage.filters

import nornir_shared.files
import nornir_shared.prettyoutput as prettyoutput
from nornir_imageregistration.type_info import ShapeLike
import nornir_imageregistration

FilterWindowCreationFunction = Callable[[ShapeLike, DTypeLike | None], NDArray[np.floating]]


class WindowFilterCache:
    """
    A generic class that creates and caches filter windows for use in image processing based on size.
    Cached images are saved to disk in a temporary directory for easy access in multiprocessing.  T
    The images in the cache are read-only.
    """

    _creation_function: FilterWindowCreationFunction
    cache_dir: str
    _name: str
    _loaded_images: dict[ShapeLike, NDArray[np.floating]]

    def __init__(self, name: str, creation_function: FilterWindowCreationFunction, dtype: DTypeLike | None = None):
        """
        :param name: Name of the filter cache, used as a subdirectory in under the temp directory
        :param creation_function:  Function to call if a filter needs to be created for a given size
        """

        self._name = name
        self.cache_dir = os.path.join(tempfile.gettempdir(), name)
        self._creation_function = creation_function
        self._dtype = dtype if dtype is not None else nornir_imageregistration.default_depth_image_dtype()
        self._loaded_images = dict()

        os.makedirs(self.cache_dir, exist_ok=True)

    def __del__(self):
        try:
            nornir_shared.files.rmtree(self.cache_dir)
        except IOError:
            prettyoutput.LogErr("Unable to delete filter cache directory: %s" % self.cache_dir)
            pass

    def GetOrCreate(self, image_shape: ShapeLike, **kwargs) -> NDArray[np.floating]:
        """Get or create a cached image filter of the expected shape"""
        return self.__GetOrCreateCachedImage(image_shape)

    def KeepGetOrCreate(self, image: NDArray | None, image_shape: ShapeLike):
        """
        If image is the expected shape, returns image.  Otherwise returns or generates a cached image of the expected shape
        :param image: Existing image to return if it is the correct shape
        :param image_shape: Shape of the image to return
        :return:
        """

        if len(image_shape) != 2:
            raise ValueError("image_shape must be a 2 element tuple")

        if image and np.array_equal(image.shape, image_shape):
            return image

        return self.__GetOrCreateCachedImage(image_shape)

    def __GetOrCreateCachedImage(self, image_shape: ShapeLike, creation_kwargs: dict | None = None) -> NDArray[
        np.floating]:

        if image_shape in self._loaded_images:
            return self._loaded_images[image_shape]

        image_path = os.path.join(self.cache_dir, f'{image_shape[0]}x{image_shape[1]}.npy')
        output = None

        # output = nornir_imageregistration.LoadImage(distance_image_path)
        try:
            #             if use_memmap:
            #                 output = np.load(distance_array_path, mmap_mode='r')
            #             else:
            output = np.load(image_path, mmap_mode='r')
            if output.dtype != self._dtype:
                output = None
                prettyoutput.Log(f"Removed outdated image from {self._name} cache: {image_path}")
                os.remove(image_path)
            else:
                output.flags.writeable = False
                self._loaded_images[image_shape] = output
                return output
        except FileNotFoundError:
            # print("Distance_image %s does not exist" % distance_array_path)
            pass
        except Exception as e:
            print(f"{self._name}: Invalid image {image_path}\n{str(e)}")
            try:
                os.remove(image_path)
            except IOError as e:
                prettyoutput.LogErr(f"Unable to delete invalid image: {image_path}\n{str(e)}")
                pass
            pass

        if output is None:
            output = self._creation_function(image_shape, self._dtype)
            self._loaded_images[image_shape] = output
            try:
                np.save(image_path, output)
            except:
                prettyoutput.LogErr("Unable to save invalid image: %s" % image_path)

            output.flags.writeable = False

        return output


def CreateWindowFilterCache(window_type: str, dtype: DTypeLike = None) -> WindowFilterCache:
    """Create a window of the specified shape and type"""
    dtype = dtype if dtype is not None else nornir_imageregistration.default_image_dtype()
    return WindowFilterCache(window_type,
                             lambda shape, dtype: skimage.filters.window(window_type, shape=shape).astype(dtype,
                                                                                                          copy=False))
