"""
Python implementation of legacy ir-blob.

The filter emphasizes blob-like low-variance regions by comparing local variance
to the image-wide median local variance, then normalizes for 8-bit output.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

import nornir_imageregistration

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

_VARIANCE_SENTINEL = np.float32(np.finfo(np.float32).max)
_NORMALIZE_CLIP_MIN = -3.0
_NORMALIZE_CLIP_MAX = 3.0
_OUTPUT_MIN = 0.0
_OUTPUT_MAX = 255.0


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
    """Validate a non-negative integer radius parameter."""
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value_i


def _window_bounds(size: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-index clipped window bounds along one image axis."""
    centers = np.arange(size, dtype=np.int64)
    starts = centers - radius
    starts = np.maximum(starts, 0)

    window = (radius * 2) + 1
    ends = starts + window
    overflow = ends - size
    overflow = np.maximum(overflow, 0)
    starts = starts - overflow
    ends = ends - overflow
    starts = np.maximum(starts, 0)
    return starts, ends


def _integral_image(values: np.ndarray) -> np.ndarray:
    """Build a padded cumulative-sum integral image for O(1) rectangle sums."""
    height, width = values.shape
    cumulative = np.cumsum(np.cumsum(values, axis=0, dtype=np.float64), axis=1)
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = cumulative
    return integral


def _rect_sum(integral: np.ndarray,
              y0: np.ndarray,
              y1: np.ndarray,
              x0: np.ndarray,
              x1: np.ndarray) -> np.ndarray:
    """Sum ``values`` over axis-aligned rectangles using an integral image."""
    return (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def _median_prefilter(image: np.ndarray, median_radius: int) -> np.ndarray:
    """Apply ITK-equivalent median prefiltering (nearest boundary, no-op at radius 0)."""
    if median_radius <= 0:
        return image

    window = (median_radius * 2) + 1
    return ndimage.median_filter(image, size=window, mode="nearest")


def _calc_variance_map(image: np.ndarray, mask: np.ndarray | None, radius: int) -> np.ndarray:
    """Port legacy ``calc_variance`` using integral images for masked population variance."""
    if radius <= 0:
        variance_map = np.zeros(image.shape, dtype=np.float32)
        if mask is None:
            return variance_map
        return np.where(mask, variance_map, _VARIANCE_SENTINEL).astype(np.float32, copy=False)

    height, width = image.shape
    image_f = image.astype(np.float64, copy=False)
    if mask is None:
        mask_f = np.ones((height, width), dtype=np.float64)
    else:
        mask_f = mask.astype(np.float64, copy=False)

    weighted = image_f * mask_f
    weighted_sq = weighted * weighted

    integral_count = _integral_image(mask_f)
    integral_sum = _integral_image(weighted)
    integral_sq = _integral_image(weighted_sq)

    y0, y1 = _window_bounds(height, radius)
    x0, x1 = _window_bounds(width, radius)

    y0_grid = y0[:, np.newaxis]
    y1_grid = y1[:, np.newaxis]
    x0_grid = x0[np.newaxis, :]
    x1_grid = x1[np.newaxis, :]

    mass = _rect_sum(integral_count, y0_grid, y1_grid, x0_grid, x1_grid)
    total = _rect_sum(integral_sum, y0_grid, y1_grid, x0_grid, x1_grid)
    total_sq = _rect_sum(integral_sq, y0_grid, y1_grid, x0_grid, x1_grid)

    variance_map = np.full((height, width), _VARIANCE_SENTINEL, dtype=np.float32)
    measured = mass > 0.0
    if not np.any(measured):
        return variance_map

    mean = np.zeros_like(mass, dtype=np.float64)
    mean[measured] = total[measured] / mass[measured]
    variance = np.zeros_like(mass, dtype=np.float64)
    variance[measured] = (total_sq[measured] / mass[measured]) - (mean[measured] * mean[measured])
    variance[measured] = np.maximum(variance[measured], 0.0)
    variance_map[measured] = variance[measured].astype(np.float32, copy=False)
    return variance_map


def _global_variance_median(variance_map: np.ndarray) -> float:
    """Return the legacy qsort median over measured variance samples only."""
    valid = variance_map[variance_map != _VARIANCE_SENTINEL]
    if valid.size == 0:
        return 0.0

    valid_sorted = np.sort(valid, axis=None)
    return float(valid_sorted[valid_sorted.size // 2])


def _enhance_blobs(variance_map: np.ndarray,
                   threshold: float,
                   mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Port legacy ``enhance_blobs`` metric generation and mean invalid fill."""
    global_median = _global_variance_median(variance_map)
    metric = np.empty(variance_map.shape, dtype=np.float64)

    measured = variance_map != _VARIANCE_SENTINEL
    metric[measured] = np.minimum(
        threshold,
        (global_median + 1.0) / (variance_map[measured].astype(np.float64) + 1.0),
    )

    if np.any(measured):
        mean_metric = float(np.mean(metric[measured]))
    else:
        mean_metric = 0.0

    metric[~measured] = mean_metric

    return metric.astype(np.float32, copy=False), global_median


def _masked_mean_sigma(values: np.ndarray, mask: np.ndarray | None) -> tuple[float, float]:
    """Compute masked mean and unbiased sigma matching ``StatisticsImageFilterWithMask``."""
    if mask is None:
        samples = values.astype(np.float64, copy=False).ravel()
    else:
        samples = values[mask].astype(np.float64, copy=False)

    count = samples.size
    if count == 0:
        return 0.0, 1.0
    if count == 1:
        return float(samples[0]), 1.0

    mean = float(np.mean(samples))
    variance = float(np.sum((samples - mean) ** 2) / (count - 1))
    sigma = float(np.sqrt(max(variance, 0.0)))
    if sigma <= 0.0:
        sigma = 1.0
    return mean, sigma


def _normalize_blob_output(metric: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Port ``normalize(image, 1, 1, 0, 255, mask)`` for the single-tile case."""
    mean, sigma = _masked_mean_sigma(metric, mask)
    normalized = (metric.astype(np.float64, copy=False) - mean) / sigma
    normalized = np.clip(normalized, _NORMALIZE_CLIP_MIN, _NORMALIZE_CLIP_MAX)

    value_min = float(np.min(normalized))
    value_max = float(np.max(normalized))
    value_range = value_max - value_min
    if value_range <= 0.0:
        return np.zeros_like(metric, dtype=np.float32)

    remapped = _OUTPUT_MIN + ((normalized - value_min) / value_range) * (_OUTPUT_MAX - _OUTPUT_MIN)
    return remapped.astype(np.float32, copy=False)


def _blob_filter_numpy(image: np.ndarray,
                       *,
                       radius: int,
                       median_radius: int,
                       max_value: float,
                       mask: np.ndarray | None) -> tuple[np.ndarray, BlobFilterDiagnostics]:
    """Run the legacy-equivalent blob filter on a NumPy array."""
    if image.ndim != 2:
        raise ValueError(f"Blob filter expects 2D grayscale images, got shape {image.shape}")

    radius = _validate_radius("radius", radius)
    median_radius = _validate_radius("median_radius", median_radius)
    max_value = float(max_value)
    if max_value <= 0:
        raise ValueError(f"max_value must be > 0, got {max_value}")

    if mask is not None:
        mask_bool = mask.astype(bool, copy=False)
        if mask_bool.shape != image.shape:
            raise ValueError(f"Mask shape {mask_bool.shape} must match image shape {image.shape}")
    else:
        mask_bool = None

    work = image.astype(np.float32, copy=False)
    work = _median_prefilter(work, median_radius)

    variance_map = _calc_variance_map(work, mask_bool, radius)
    metric, global_median = _enhance_blobs(variance_map, max_value, mask_bool)
    output = _normalize_blob_output(metric, mask_bool)

    masked_count = 0
    if mask_bool is not None:
        masked_count = int(mask_bool.size - int(np.count_nonzero(mask_bool)))

    diagnostics = BlobFilterDiagnostics(
        backend="numpy",
        radius=radius,
        median_radius=median_radius,
        max_value=max_value,
        global_median_variance=global_median,
        masked_pixel_count=masked_count,
        used_numpy_fallback=False,
    )
    return output, diagnostics


def BlobFilter(image: Any,
               *,
               radius: int,
               median_radius: int,
               max_value: float,
               mask: Any = None,
               return_diagnostics: bool = False):
    """
    Apply ir-blob style filtering to an image array.

    The reference implementation runs on NumPy to match legacy ``ir-blob`` output.
    """
    image_arr = nornir_imageregistration.ImageParamToImageArray(image)
    xp_in = cp.get_array_module(image_arr)
    image_np = nornir_imageregistration.EnsureNumpyArray(image_arr).astype(np.float32, copy=False)
    mask_np = None
    if mask is not None:
        mask_np = nornir_imageregistration.EnsureNumpyArray(mask).astype(bool, copy=False)

    output_np, diagnostics = _blob_filter_numpy(
        image_np,
        radius=radius,
        median_radius=median_radius,
        max_value=max_value,
        mask=mask_np,
    )

    if xp_in is cp:
        output = cp.asarray(output_np)
        diagnostics = BlobFilterDiagnostics(
            backend="cupy",
            radius=diagnostics.radius,
            median_radius=diagnostics.median_radius,
            max_value=diagnostics.max_value,
            global_median_variance=diagnostics.global_median_variance,
            masked_pixel_count=diagnostics.masked_pixel_count,
            used_numpy_fallback=True,
        )
    else:
        output = output_np
        diagnostics = diagnostics

    if return_diagnostics:
        return output, diagnostics

    return output


def _load_legacy_tile_image(load_path: str) -> np.ndarray:
    """Load an 8-bit mosaic tile as float32 0–255, matching legacy ``std_tile`` / ``load<image_t>``."""
    with Image.open(load_path, "r") as im:
        return np.array(im, dtype=np.float32)


def _load_legacy_tile_mask(mask_path: str) -> np.ndarray:
    """Load a mask PNG as a boolean array (nonzero pixels are valid)."""
    with Image.open(mask_path, "r") as im:
        return np.array(im) > 0


def BlobFilterImageFile(load_path: str,
                        save_path: str,
                        *,
                        radius: int,
                        median_radius: int,
                        max_value: float,
                        mask_path: str | None = None,
                        min_pixels_for_gpu: int = -1,
                        return_diagnostics: bool = False):
    """
    Run blob filtering from input/output file paths.

    Processing uses the NumPy reference path for legacy parity regardless of size.
    """
    del min_pixels_for_gpu  # parity-critical path always uses NumPy reference

    image = _load_legacy_tile_image(load_path)
    mask = None
    if mask_path is not None and os.path.exists(mask_path):
        mask = _load_legacy_tile_mask(mask_path)

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
