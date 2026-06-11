"""
Python implementation of legacy ir-blob.

The filter emphasizes blob-like low-variance regions by comparing local variance
to the image-wide median local variance.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
from PIL import Image

import nornir_imageregistration

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

try:
    import cupyx
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupyx_thunk as cupyx


@dataclass(frozen=True)
class BlobFilterDiagnostics:
    backend: str
    radius: int
    median_radius: int
    max_value: float
    global_median_variance: float
    masked_pixel_count: int
    used_numpy_fallback: bool


def _validate_radius(name: str, value: int) -> int:
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value_i


def _as_backend_mask(mask: Any, xp: Any, image_shape: tuple[int, int]):
    if mask is None:
        return None

    mask_array = nornir_imageregistration.ImageParamToImageArray(mask)
    if mask_array.shape != image_shape:
        raise ValueError(f"Mask shape {mask_array.shape} must match image shape {image_shape}")

    if cp.get_array_module(mask_array) is not xp:
        mask_array = xp.asarray(mask_array)

    return mask_array.astype(xp.bool_, copy=False)


def _ensure_working_float(image: Any, xp: Any):
    image_dtype = np.dtype(image.dtype)
    if image_dtype.kind == "f" and image_dtype.itemsize >= 8:
        return image.astype(xp.float64, copy=False)
    return image.astype(xp.float32, copy=False)


def _compute_local_variance(image: Any, radius: int, valid_mask: Any, xp: Any, sp: Any):
    window = (radius * 2) + 1
    if window == 1:
        local_var = xp.zeros_like(image)
        if valid_mask is None:
            return local_var
        return xp.where(valid_mask, local_var, xp.zeros_like(local_var))

    if valid_mask is None:
        local_mean = sp.ndimage.uniform_filter(image, size=window, mode="reflect")
        local_mean_sq = sp.ndimage.uniform_filter(image * image, size=window, mode="reflect")
        return xp.maximum(local_mean_sq - local_mean * local_mean, 0.0)

    valid_f = valid_mask.astype(image.dtype, copy=False)
    count = sp.ndimage.uniform_filter(valid_f, size=window, mode="reflect")
    weighted_image = image * valid_f
    weighted_image_sq = image * image * valid_f
    sum_image = sp.ndimage.uniform_filter(weighted_image, size=window, mode="reflect")
    sum_image_sq = sp.ndimage.uniform_filter(weighted_image_sq, size=window, mode="reflect")

    eps = xp.asarray(1e-6, dtype=image.dtype)
    safe_count = xp.maximum(count, eps)
    local_mean = sum_image / safe_count
    local_mean_sq = sum_image_sq / safe_count
    local_var = xp.maximum(local_mean_sq - local_mean * local_mean, 0.0)
    return xp.where(count > eps, local_var, xp.zeros_like(local_var))


def _global_median_local_variance(local_var: Any, valid_mask: Any | None, xp: Any, work_dtype: Any):
    """Compute the image-wide median local variance, optionally restricted to valid mask pixels."""
    if valid_mask is None:
        return xp.median(local_var)

    if not bool(xp.any(valid_mask)):
        return xp.asarray(0.0, dtype=work_dtype)

    nan = xp.asarray(float("nan"), dtype=local_var.dtype)
    masked_values = xp.where(valid_mask, local_var, nan)
    return xp.nanmedian(masked_values)


def _blob_filter_impl(image: Any,
                      *,
                      radius: int,
                      median_radius: int,
                      max_value: float,
                      mask: Any = None) -> tuple[Any, BlobFilterDiagnostics]:
    image_arr = nornir_imageregistration.ImageParamToImageArray(image)
    if image_arr.ndim != 2:
        raise ValueError(f"Blob filter expects 2D grayscale images, got shape {image_arr.shape}")

    radius = _validate_radius("radius", radius)
    median_radius = _validate_radius("median_radius", median_radius)
    max_value = float(max_value)
    if max_value <= 0:
        raise ValueError(f"max_value must be > 0, got {max_value}")

    xp = cp.get_array_module(image_arr)
    sp = cupyx.scipy.get_array_module(image_arr)
    work = _ensure_working_float(image_arr, xp)
    valid_mask = _as_backend_mask(mask, xp, image_arr.shape)

    if median_radius > 0:
        med_window = (median_radius * 2) + 1
        work = sp.ndimage.median_filter(work, size=med_window, mode="reflect")

    local_var = _compute_local_variance(work, radius, valid_mask, xp, sp)
    global_median = _global_median_local_variance(local_var, valid_mask, xp, work.dtype)

    scalar_dtype = np.result_type(np.dtype(work.dtype), np.float32)
    one = xp.asarray(1.0, dtype=scalar_dtype)
    max_scalar = xp.asarray(max_value, dtype=scalar_dtype)
    response = (global_median.astype(scalar_dtype) + one) / (local_var.astype(scalar_dtype) + one)
    response = xp.clip(response, 0.0, max_scalar)
    response = response / max_scalar

    if valid_mask is not None:
        response = xp.where(valid_mask, response, xp.asarray(0.0, dtype=response.dtype))

    masked_count = 0
    if valid_mask is not None:
        masked_count = int(xp.size(valid_mask) - int(xp.sum(valid_mask)))

    diagnostics = BlobFilterDiagnostics(
        backend="cupy" if xp is cp else "numpy",
        radius=radius,
        median_radius=median_radius,
        max_value=max_value,
        global_median_variance=float(global_median.item()) if hasattr(global_median, "item") else float(global_median),
        masked_pixel_count=masked_count,
        used_numpy_fallback=False
    )
    return response.astype(np.float32, copy=False), diagnostics


def BlobFilter(image: Any,
               *,
               radius: int,
               median_radius: int,
               max_value: float,
               mask: Any = None,
               return_diagnostics: bool = False):
    """
    Apply ir-blob style filtering to an image array.

    Input and output stay on the same backend (NumPy or CuPy) unless a CuPy
    fallback to NumPy is required because of missing CuPyX functionality.
    """
    image_arr = nornir_imageregistration.ImageParamToImageArray(image)
    xp_in = cp.get_array_module(image_arr)
    mask_arr = None if mask is None else nornir_imageregistration.ImageParamToImageArray(mask)

    try:
        output, diagnostics = _blob_filter_impl(
            image_arr,
            radius=radius,
            median_radius=median_radius,
            max_value=max_value,
            mask=mask_arr)
    except (AttributeError, NotImplementedError, TypeError, ValueError, RuntimeError):
        if xp_in is not cp:
            raise

        image_np = nornir_imageregistration.EnsureNumpyArray(image_arr)
        mask_np = None if mask_arr is None else nornir_imageregistration.EnsureNumpyArray(mask_arr).astype(bool, copy=False)
        output_np, diagnostics = _blob_filter_impl(
            image_np,
            radius=radius,
            median_radius=median_radius,
            max_value=max_value,
            mask=mask_np)
        output = cp.asarray(output_np)
        diagnostics = BlobFilterDiagnostics(
            backend="cupy",
            radius=diagnostics.radius,
            median_radius=diagnostics.median_radius,
            max_value=diagnostics.max_value,
            global_median_variance=diagnostics.global_median_variance,
            masked_pixel_count=diagnostics.masked_pixel_count,
            used_numpy_fallback=True
        )

    if return_diagnostics:
        return output, diagnostics

    return output


def _should_use_cupy_from_files(load_path: str, min_pixels_for_gpu: int) -> bool:
    if min_pixels_for_gpu < 0:
        min_pixels_for_gpu = 0

    if not nornir_imageregistration.HasCupy():
        return False

    with Image.open(load_path, "r") as im:
        width, height = im.size
    return (width * height) >= int(min_pixels_for_gpu)


def BlobFilterImageFile(load_path: str,
                        save_path: str,
                        *,
                        radius: int,
                        median_radius: int,
                        max_value: float,
                        mask_path: str | None = None,
                        min_pixels_for_gpu: int = 1024 * 1024,
                        return_diagnostics: bool = False):
    """
    Run blob filtering from input/output file paths.

    If CuPy is active and available, GPU processing is used for sufficiently
    large images (controlled by ``min_pixels_for_gpu``).
    """
    backend = "cupy" if _should_use_cupy_from_files(load_path, min_pixels_for_gpu) else "numpy"

    image = nornir_imageregistration.LoadImage(load_path, dtype=np.float32, backend=backend)
    mask = None
    if mask_path is not None and os.path.exists(mask_path):
        mask = nornir_imageregistration.LoadImage(mask_path, dtype=bool, backend=backend)

    output, diagnostics = BlobFilter(
        image,
        radius=radius,
        median_radius=median_radius,
        max_value=max_value,
        mask=mask,
        return_diagnostics=True)

    nornir_imageregistration.SaveImage(save_path, output, bpp=8)

    if return_diagnostics:
        return diagnostics

    return None
