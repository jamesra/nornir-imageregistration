"""
Functions related to blurring images and low-pass filtering
"""

import logging
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy as sp
import scipy.ndimage as ndimage
import skimage.filters

import nornir_imageregistration
from skimage import feature


class SmartBlurConfig:
    kernel_size: int  # Must be odd
    sigma: float
    low_threshold: float
    high_threshold: float

    def __init__(self, kernel_size: int, sigma: float, low_threshold: float, high_threshold: float | None = None):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold if high_threshold is not None else 2 * low_threshold


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a Gaussian kernel.

    Parameters:
    - size: Size of the kernel (should be odd).
    - sigma: Standard deviation for Gaussian kernel.

    Returns:
    - Gaussian kernel as a 2D numpy array.
    """
    # Create a 1D range of values
    if size % 2 == 0:
        raise ValueError("Size of the kernel should be odd")

    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the Gaussian function
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    return kernel


def smart_blur(image: nornir_imageregistration.ImageLike,
               config: SmartBlurConfig) -> nornir_imageregistration.ImageLike:
    """
    Apply a smart blur to the image.  This filter only includes pixels that are within a threshold range of the center point in the gaussian kernel
    """

    gaussian_kernel = create_gaussian_kernel(config.kernel_size, config.sigma)

    def smart_kernel(subset_image: NDArray) -> float:
        """
        Calculate the kernel size for a smart blur
        """

        shape = config.kernel_size, config.kernel_size
        subset_image = np.reshape(subset_image, shape)
        # Find the difference from the center pixel to each other pixel in the image
        center_index = (shape[0] // 2, shape[1] // 2)
        center_value = subset_image[center_index]
        absdiff = abs(subset_image - center_value)
        threshold_mask = absdiff < config.low_threshold
        threshold_mask[shape[0] // 2, shape[1] // 2] = True  # Ensure the pixel itself is always included

        # Find all of the points connected to the center
        # labeled_mask, num_features = sp.ndimage.label(input=threshold_mask)
        # if num_features == 0:
        #     return np.sum(np.prod(gaussian_kernel.flat, subset_image.flat))
        #
        # center_label = labeled_mask[center_index]
        # floodfill_mask = labeled_mask == center_label
        floodfill_mask = threshold_mask
        gaussian_kernel_subset = gaussian_kernel[
            floodfill_mask]  # Set output=threshold_mask before checking in for performance
        gaussian_sum = np.sum(gaussian_kernel_subset)
        gaussian_kernel_subset /= gaussian_sum
        image_and_kernel = subset_image[floodfill_mask] * gaussian_kernel_subset
        # image_subset = subset_image[floodfill_mask] * gaussian_kernel_subset
        # image_and_kernel = image_subset * gaussian_kernel_subset
        return np.sum(image_and_kernel)

    test = smart_kernel(image[0:config.kernel_size, 0:config.kernel_size].flat)

    image = nornir_imageregistration.ImageParamToImageArray(image)
    # Check if the image is already grayscale
    if len(image.shape) == 3:
        raise ValueError("Input image must be a grayscale image")

    # test = skimage.feature.blob_doh(image, min_sigma=10, max_sigma=300)

    # Apply Gaussian blur to the grayscale image
    blurred = ndimage.generic_filter(image, function=smart_kernel, size=config.kernel_size)
    return blurred
