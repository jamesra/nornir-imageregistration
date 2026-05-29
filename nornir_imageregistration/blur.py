"""
Functions related to blurring images and low-pass filtering
"""

import types

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration


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


def create_gaussian_kernel(size: int, sigma: float, xp: types.ModuleType, *, dtype) -> NDArray:
    """
    Create a normalized 2D Gaussian kernel on ``xp`` (NumPy or CuPy).

    Parameters:
    - size: Size of the kernel (should be odd).
    - sigma: Standard deviation for Gaussian kernel.
    - xp: Array module (``numpy`` or ``cupy``).
    - dtype: dtype for kernel coefficients (must match working image dtype).

    Returns:
    - Gaussian kernel as a 2D array on the same backend as ``xp``.
    """
    if size % 2 == 0:
        raise ValueError("Size of the kernel should be odd")

    ax = xp.arange(-size // 2 + 1.0, size // 2 + 1.0, dtype=dtype)
    xx, yy = xp.meshgrid(ax, ax)
    sigma_t = xp.asarray(sigma, dtype=dtype)
    kernel = xp.exp(-0.5 * (xp.square(xx) + xp.square(yy)) / xp.square(sigma_t))
    kernel = kernel / xp.sum(kernel)
    return kernel


def _sliding_window_view_2d(x: NDArray, window_shape: tuple[int, int], xp: types.ModuleType) -> NDArray:
    """2D sliding window; NumPy branch avoids relying on ``cupy_thunk`` for stride_tricks."""
    if xp is np:
        return np.lib.stride_tricks.sliding_window_view(x, window_shape)
    return xp.lib.stride_tricks.sliding_window_view(x, window_shape)


def smart_blur(image: nornir_imageregistration.ImageLike,  # type: ignore[valid-type]
               config: SmartBlurConfig) -> nornir_imageregistration.ImageLike:  # type: ignore[valid-type]
    """
    Apply a smart blur to the image.  This filter only includes pixels that are within a threshold range of the center point in the gaussian kernel
    """
    image = nornir_imageregistration.ImageParamToImageArray(image)
    if image.ndim == 3:
        raise ValueError("Input image must be a grayscale image")

    xp = cp.get_array_module(image)
    k = config.kernel_size
    pad = k // 2

    idt = np.dtype(image.dtype)
    if idt.kind == "f" and idt.itemsize >= 8:
        work = image.astype(xp.float64)
    else:
        work = image.astype(xp.float32)

    padded = xp.pad(work, ((pad, pad), (pad, pad)), mode="reflect")
    patches = _sliding_window_view_2d(padded, (k, k), xp)

    gaussian_kernel = create_gaussian_kernel(k, config.sigma, xp, dtype=work.dtype)
    kc = k // 2
    center = patches[:, :, kc, kc][..., xp.newaxis, xp.newaxis]
    low = xp.asarray(config.low_threshold, dtype=work.dtype)
    mask = xp.abs(patches - center) < low
    mask[..., kc, kc] = True

    weights = gaussian_kernel * mask.astype(work.dtype)
    wsum = xp.sum(weights, axis=(-2, -1), keepdims=True)
    weights = weights / wsum
    out = xp.sum(patches * weights, axis=(-2, -1))

    if np.issubdtype(image.dtype, np.floating) and out.dtype != image.dtype:
        out = out.astype(image.dtype, copy=False)
    return out
