"""

"""
import numpy as np
import numpy.typing
from numpy.typing import NDArray

import nornir_imageregistration


class ImagePermutationHelper(object):
    """
    A helper class that takes an image and optional mask.  It exposes the image, mask, a version
     of the image with random noise where the mask is over the image.  It also exposes a version
     of the mask with image extrema values added to the mask
    :param object:
    :return:
    """
    @property
    def Extrema_Area_Cutoff_In_Pixels(self) -> int:
        """
        :return:  In pixels, the minimum area of extreme pixel values for them to be masked
        """
        return self._extrema_size_cutoff_in_pixels

    @property
    def Image(self) -> NDArray:
        """
        :return: The image passed to the constructor
        """
        return self._image

    @property
    def Mask(self) -> NDArray:
        """
        :return:  The mask passed to the constructor, may be None
        """
        return self._mask

    @property
    def BlendedMask(self) -> NDArray:
        """
        :return: The mask combined with the extrema mask.  Is only the extrema mask if there was no Mask passed
        """
        return self._blended_mask

    @property
    def Stats(self) -> nornir_imageregistration.ImageStats:
        """
        :return: Statistics for unmasked portion of the image
        """
        return self._stats

    @property
    def ImageWithMaskAsNoise(self) -> NDArray:
        """
        :return:  The image with random noise over the masked regions
        """
        return self._image_with_mask_as_noise

    def __init__(self,
                 img: nornir_imageregistration.ImageLike,
                 mask: nornir_imageregistration.ImageLike | None = None,
                 extrema_mask_size_cuttoff: float | int | NDArray | None = None,
                 dtype: numpy.typing.DTypeLike | None = None):

        if dtype is None:
            dtype = img.dtype if np.issubdtype(img.dtype, np.floating) else np.float32

        img = nornir_imageregistration.ImageParamToImageArray(img, dtype=dtype)
        mask = nornir_imageregistration.ImageParamToImageArray(img, dtype=bool)

        self._extrema_size_cutoff_in_pixels = None
        if extrema_mask_size_cuttoff is None:
            extrema_mask_size_cuttoff = np.array((128, 128))

        if isinstance(extrema_mask_size_cuttoff, np.ndarray):
            self.extrema_size_cutoff_in_pixels = int(np.prod(extrema_mask_size_cuttoff))
        elif isinstance(extrema_mask_size_cuttoff, float):
            self.extrema_size_cutoff_in_pixels = int(np.prod(img.shape) * extrema_mask_size_cuttoff)
        elif not isinstance(extrema_mask_size_cuttoff, int):
            raise ValueError(f"extrema_mask_size_cutoff")
        else:
            self._extrema_size_cutoff_in_pixels = extrema_mask_size_cuttoff

        self._image = img.astype(dtype, copy=False)
        self._mask = mask
        self._extrema_mask = nornir_imageregistration.CreateExtremaMask(self._image, self._mask, size_cutoff=self.extrema_size_cutoff_in_pixels)
        self._blended_mask = np.logical_and(self._mask, self._extrema_mask) if self._mask is not None else self._extrema_mask
        self._stats = nornir_imageregistration.ImageStats.Create(self._image[self._blended_mask])
        self._image_with_mask_as_noise = nornir_imageregistration.RandomNoiseMask(self._image, self._blended_mask, Copy=True)