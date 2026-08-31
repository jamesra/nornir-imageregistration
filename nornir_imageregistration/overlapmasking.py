"""
Created on Sep 14, 2018

@author: u0490822
"""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration import Rectangle, ShapeLike

# Collection of masks we have already calculated (host + device), byte-capped LRU.
__known_overlap_masks: OrderedDict[tuple, NDArray] = OrderedDict()
# Host masks uploaded once per shape key when using CuPy (avoids cp.asarray per find_peak call).
__known_overlap_masks_device: OrderedDict[tuple, NDArray] = OrderedDict()

_DEFAULT_DEVICE_CACHE_MB = 256
_DEFAULT_HOST_CACHE_MB = 512


def _cache_budget_bytes(env_name: str, default_mb: int) -> int:
    """Return the byte budget from *env_name* (megabytes), or *default_mb*."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default_mb * 1024 * 1024
    try:
        return max(0, int(float(raw) * 1024 * 1024))
    except ValueError:
        return default_mb * 1024 * 1024


def _array_nbytes(arr: NDArray) -> int:
    nbytes = getattr(arr, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return int(np.prod(arr.shape)) * int(getattr(arr.dtype, "itemsize", 1))


def _lru_get(cache: OrderedDict[tuple, NDArray], key: tuple) -> NDArray | None:
    cached = cache.get(key)
    if cached is None:
        return None
    cache.move_to_end(key)
    return cached


def _lru_put(cache: OrderedDict[tuple, NDArray], key: tuple, value: NDArray,
             budget_bytes: int) -> None:
    """Insert *value* and evict least-recently-used entries until under *budget_bytes*."""
    if key in cache:
        cache.move_to_end(key)
        cache[key] = value
    else:
        cache[key] = value

    if budget_bytes <= 0:
        cache.clear()
        return

    total = sum(_array_nbytes(arr) for arr in cache.values())
    while total > budget_bytes and cache:
        _evicted_key, evicted = cache.popitem(last=False)
        total -= _array_nbytes(evicted)


def overlap_mask_cache_stats() -> dict[str, Any]:
    """Return host/device cache sizes for diagnostics and tests."""
    host_bytes = sum(_array_nbytes(arr) for arr in __known_overlap_masks.values())
    device_bytes = sum(_array_nbytes(arr) for arr in __known_overlap_masks_device.values())
    return {
        "host_entries": len(__known_overlap_masks),
        "host_bytes": host_bytes,
        "host_budget_bytes": _cache_budget_bytes("NORNIR_OVERLAP_MASK_HOST_CACHE_MB",
                                                 _DEFAULT_HOST_CACHE_MB),
        "device_entries": len(__known_overlap_masks_device),
        "device_bytes": device_bytes,
        "device_budget_bytes": _cache_budget_bytes("NORNIR_OVERLAP_MASK_CACHE_MB",
                                                   _DEFAULT_DEVICE_CACHE_MB),
    }


def __CreateMaskLookupIndex(target_image_shape: NDArray[np.integer],
                            source_image_shape: NDArray[np.integer],
                            correlation_image_shape: NDArray[np.integer],
                            min_overlap: float, max_overlap: float) -> tuple[
    int, int, int, int, int, int, float, float]:
    """
    Create an index into a dictionary for a overlap mask
    """

    dimensions = np.concatenate((target_image_shape, source_image_shape, correlation_image_shape))
    full_index = list(dimensions) + [min_overlap, max_overlap]
    return tuple(full_index)


def GetOverlapMask(target_image_shape: ShapeLike,
                   source_image_shape: ShapeLike,
                   correlation_image_size: ShapeLike,
                   MinOverlap: float = 0.0, MaxOverlap: float = 1.0):
    """Defines a mask that determines which peaks should be considered.

    Host-only: Rectangle geometry. Use ``GetOverlapMaskOnDevice`` to place the mask on CuPy.
    :param NDArray target_image_shape: Shape of fixed image, before padding
    :param NDArray source_image_shape: Shape of moving image, before padding
    :param NDArray correlation_image_size: Shape of correlation image, which will be equal to size of largest padded image dimensions
    :param float MinOverlap: The minimum amount of overlap between the fixed and moving images, area based
    :param float MaxOverlap: The maximum amount of overlap between the fixed and moving images, area based
    :return: An mxn image mask, with 1 indicating allowed peak locations
    """

    global __known_overlap_masks

    target_image_shape = np.asarray([int(x) for x in target_image_shape], dtype=np.int64)
    source_image_shape = np.asarray([int(x) for x in source_image_shape], dtype=np.int64)
    correlation_image_size = np.asarray([int(x) for x in correlation_image_size], dtype=np.int64)

    if MinOverlap == 0.0 and MaxOverlap == 1.0:  # and np.array_equal(FixedImageSize, MovingImageSize) and np.array_equal(FixedImageSize, CorrelationImageSize):
        return None

    MaskIndex = __CreateMaskLookupIndex(target_image_shape, source_image_shape, correlation_image_size, MinOverlap,
                                        MaxOverlap)

    cached = _lru_get(__known_overlap_masks, MaskIndex)
    if cached is not None:
        return cached

    mask = __CreateOverlapMaskBruteForce(target_image_shape, source_image_shape, correlation_image_size, MinOverlap,
                                         MaxOverlap)
    budget = _cache_budget_bytes("NORNIR_OVERLAP_MASK_HOST_CACHE_MB", _DEFAULT_HOST_CACHE_MB)
    _lru_put(__known_overlap_masks, MaskIndex, mask, budget)

    return mask


def GetOverlapMaskOnDevice(target_image_shape: ShapeLike,
                           source_image_shape: ShapeLike,
                           correlation_image_size: ShapeLike,
                           MinOverlap: float = 0.0,
                           MaxOverlap: float = 1.0,
                           xp=np):
    """Return an overlap mask on the same array module as *xp*.

    The host mask is built once and cached in ``GetOverlapMask``. When *xp* is CuPy,
    the mask is uploaded to the device once per shape key and reused across calls
    (for example thousands of ``find_peak`` invocations sharing the same geometry).
    Device entries are evicted LRU-style when the byte budget
    (``NORNIR_OVERLAP_MASK_CACHE_MB``, default 256) is exceeded.
    """
    mask = GetOverlapMask(target_image_shape, source_image_shape, correlation_image_size,
                          MinOverlap, MaxOverlap)
    if mask is None:
        return None

    if xp is np:
        return mask

    target_image_shape = np.asarray([int(x) for x in target_image_shape], dtype=np.int64)
    source_image_shape = np.asarray([int(x) for x in source_image_shape], dtype=np.int64)
    correlation_image_size = np.asarray([int(x) for x in correlation_image_size], dtype=np.int64)
    mask_index = __CreateMaskLookupIndex(target_image_shape, source_image_shape, correlation_image_size,
                                         MinOverlap, MaxOverlap)

    cached = _lru_get(__known_overlap_masks_device, mask_index)
    if cached is not None:
        return cached

    device_mask = xp.asarray(mask)
    budget = _cache_budget_bytes("NORNIR_OVERLAP_MASK_CACHE_MB", _DEFAULT_DEVICE_CACHE_MB)
    _lru_put(__known_overlap_masks_device, mask_index, device_mask, budget)
    return device_mask


def clear_overlap_mask_caches() -> None:
    """Clear cached overlap masks (host and device). Intended for tests."""
    global __known_overlap_masks, __known_overlap_masks_device
    __known_overlap_masks = OrderedDict()
    __known_overlap_masks_device = OrderedDict()


def __CreateFullMaskFromQuadrant(Mask: NDArray[np.bool_],
                                 isOddDimension: NDArray[np.bool_]):
    """
    Given the top right quadrant of a mask, replicates the mask symetrically around both the X and Y axis to create a full mask
    :param array isOddDimension: True if the axis has an odd dimension in the input.
    """
    MaskUpRight = Mask

    if isOddDimension[1]:
        MaskUpLeft = np.fliplr(Mask[:, 1:])
    else:
        MaskUpLeft = np.fliplr(Mask)

    UpperMask = np.hstack((MaskUpLeft, MaskUpRight))

    if isOddDimension[0]:
        LowerMask = np.flipud(UpperMask[1:, :])
        # MaskDownLeft = np.flipud(MaskUpLeft[1:,:])
        # MaskDownRight = np.fliplr(MaskUpRight[1:,:])
    else:
        LowerMask = np.flipud(UpperMask)
        # MaskDownLeft = np.flipud(MaskUpLeft)
        # MaskDownRight = np.fliplr(MaskUpRight)

    #    LowerMask = np.hstack((MaskDownLeft, MaskDownRight))

    Mask = np.vstack((LowerMask, UpperMask))

    return Mask


def __CreateOverlapMaskBruteForce(FixedImageSize: ShapeLike,
                                  MovingImageSize: ShapeLike,
                                  CorrelationImageSize: ShapeLike,
                                  MinOverlap: float = 0.0,
                                  MaxOverlap: float = 1.0):
    """Defines a mask that determines which peaks should be considered
    :param array FixedImageSize: Shape of fixed image, before padding
    :param array MovingImageSize: Shape of moving image, before padding
    :param array CorrelationImageSize: Shape of correlation image, which will be equal to size of largest padded image dimensions
    :param float MinOverlap: The minimum amount of overlap between the fixed and moving images, area based
    :param float MaxOverlap: The maximum amount of overlap between the fixed and moving images, area based
    :return: An mxn image mask, with 1 indicating allowed peak locations
    """

    if MinOverlap is None:
        MinOverlap = 0.0

    if MaxOverlap is None:
        MaxOverlap = 1.0

    if MinOverlap >= MaxOverlap:
        raise ValueError("Minimum overlap must be less than maximum overlap")

    QuadrantSize = np.asarray((CorrelationImageSize[0] / 2.0, CorrelationImageSize[1] / 2.0), dtype=np.float32)
    isOddDimension = np.mod(QuadrantSize, 1) > 0
    QuadrantSize = np.ceil(QuadrantSize).astype(np.int32, copy=False)
    Mask = np.zeros(QuadrantSize, dtype=bool)

    Mask = _PopulateMaskQuadrantOptimized(Mask, FixedImageSize, MovingImageSize, MinOverlap, MaxOverlap)  # type: ignore[arg-type]
    #     for ix in range(0, HalfCorrelationSize[1]):
    #         for iy in range(0, HalfCorrelationSize[0]):
    #             WarpedImageRect = Rectangle.CreateFromCenterPointAndArea((iy, ix), MovingImageSize)
    #
    #             overlap = Rectangle.overlap(WarpedImageRect, FixedImageRect)
    #             Mask[iy, ix] = overlap >= MinOverlap and overlap <= MaxOverlap
    return __CreateFullMaskFromQuadrant(Mask, isOddDimension)


def _PopulateMaskQuadrantBruteForce(Mask: NDArray[np.bool_],
                                    FixedImageSize: NDArray[np.integer],
                                    MovingImageSize: NDArray[np.integer],
                                    MinOverlap: float = 0.0,
                                    MaxOverlap: float = 1.0) -> NDArray[np.bool_]:
    FixedImageRect = Rectangle.CreateFromCenterPointAndArea((0, 0), FixedImageSize)  # type: ignore[arg-type]
    WarpedImageRect = None

    # We cannot overlap more than the minimum of each dimension.
    #
    # Axis 0, not axis 1: the min is taken per dimension across the two images, not per image
    # across its own two dimensions. Axis 1 gave [min(FixedH, FixedW), min(MovingH, MovingW)],
    # which is not an area either rectangle can intersect. Measured over 4000 random size
    # pairs, the true largest intersection exceeded it in a third of them, by up to 10.2x, so
    # the overlap "fraction" could pass 1.0 and be masked out by MaxOverlap (#126).
    #
    # The two axis-0 implementations below were always right; only this reference was wrong,
    # so no production mask changes. The two agree for equal squares, which is the single
    # geometry the parity test used, which is why this went unseen.
    maxPossibleOverlap = np.min(np.vstack((FixedImageSize, MovingImageSize)), 0)
    maxPossibleOverlapArea = np.prod(maxPossibleOverlap)

    Overlap = np.zeros(Mask.shape, dtype=np.float32)

    for ix in range(0, Mask.shape[1]):
        for iy in range(0, Mask.shape[0]):
            WarpedImageRect = Rectangle.CreateFromCenterPointAndArea((iy, ix), MovingImageSize)  # type: ignore[arg-type]

            overlap_rect = Rectangle.overlap_rect(WarpedImageRect, FixedImageRect)
            overlap = 0
            if overlap_rect is not None:
                overlap = overlap_rect.Area / maxPossibleOverlapArea

            Overlap[iy, ix] = overlap

    Mask = np.logical_and(Overlap >= MinOverlap, Overlap <= MaxOverlap)
    return Mask


def _PopulateMaskQuadrantBruteForceOptimized(Mask: NDArray[np.bool_],
                                             FixedImageSize: NDArray[np.integer],
                                             MovingImageSize: NDArray[np.integer],
                                             MinOverlap: float = 0.0,
                                             MaxOverlap: float = 1.0) -> NDArray[np.bool_]:
    FixedImageRect = Rectangle.CreateFromCenterPointAndArea((0, 0), FixedImageSize)  # type: ignore[arg-type]
    WarpedImageRect = None

    # We cannot overlap more than the minimum of each dimension
    maxPossibleOverlap = np.min(np.vstack((FixedImageSize, MovingImageSize)), 0)
    maxPossibleOverlapArea = np.prod(maxPossibleOverlap)

    for ix in range(0, Mask.shape[1]):
        for iy in range(0, Mask.shape[0]):
            WarpedImageRect = Rectangle.CreateFromCenterPointAndArea((iy, ix), MovingImageSize)  # type: ignore[arg-type]

            overlap_rect = Rectangle.overlap_rect(WarpedImageRect, FixedImageRect)
            overlap = 0
            if overlap_rect is not None:
                overlap = overlap_rect.Area / maxPossibleOverlapArea

            # Overlap[iy, ix] = overlap #Rectangle.overlap(WarpedImageRect, FixedImageRect)
            Mask[iy, ix] = MinOverlap <= overlap <= MaxOverlap

            if overlap < MinOverlap:
                Mask[iy:Mask.shape[0], ix] = False
                break

    # Mask = np.logical_and(Overlap >= MinOverlap, Overlap <= MaxOverlap)
    return Mask


def _PopulateMaskQuadrantOptimized(Mask: NDArray[np.bool_],
                                   FixedImageSize: NDArray[np.integer],
                                   MovingImageSize: NDArray[np.integer],
                                   MinOverlap: float = 0.0,
                                   MaxOverlap: float = 1.0) -> NDArray[np.bool_]:
    FixedImageRect = Rectangle.CreateFromCenterPointAndArea((0, 0), FixedImageSize)  # type: ignore[arg-type]
    WarpedImageRect = None

    # We cannot overlap more than the minimum of each dimension
    maxPossibleOverlap = np.min(np.vstack((FixedImageSize, MovingImageSize)), 0)
    maxPossibleOverlapArea = np.prod(maxPossibleOverlap)

    # Array of increasing values for X,Y size of Mask
    Px = np.arange(0, Mask.shape[1], 1)
    Py = np.arange(0, Mask.shape[0], 1)

    # Where the corresponding boundary of the rectangle lies for any given point P
    Mx_Left = Px - (MovingImageSize[1] / 2.0)
    My_Bottom = Py - (MovingImageSize[0] / 2.0)
    Mx_Right = Px + (MovingImageSize[1] / 2.0)
    My_Top = Py + (MovingImageSize[0] / 2.0)

    # Where the overlapping rectangle boundary lies for any given point P
    Ox_Left = np.fmax(Mx_Left.astype(np.float64, copy=False), FixedImageRect.MinX)
    Ox_Bottom = np.fmax(My_Bottom.astype(np.float64, copy=False), FixedImageRect.MinY)
    Ox_Right = np.fmin(Mx_Right.astype(np.float64, copy=False), FixedImageRect.MaxX)
    Ox_Top = np.fmin(My_Top.astype(np.float64, copy=False), FixedImageRect.MaxY)

    Ox_Width = Ox_Right - Ox_Left
    Ox_Height = Ox_Top - Ox_Bottom

    Ox_Width[Ox_Width < 0] = 0
    Ox_Height[Ox_Height < 0] = 0

    overlap = np.zeros(Mask.shape)

    for iy in range(0, Mask.shape[0]):
        if Ox_Height[iy] > 0:
            overlap[iy, :] = Ox_Width * Ox_Height[iy]

    fraction = overlap / maxPossibleOverlapArea

    # Mask = fraction > MinOverlap

    #     for ix in range(0, Mask.shape[1]):
    #         for iy in range(0, Mask.shape[0]):
    #             WarpedImageRect = Rectangle.CreateFromCenterPointAndArea((iy, ix), MovingImageSize)
    #
    #             overlap_rect = Rectangle.overlap_rect(WarpedImageRect, FixedImageRect)
    #             overlap = 0
    #             if overlap_rect is not None:
    #                 overlap = overlap_rect.Area / maxPossibleOverlapArea
    #
    #             #Overlap[iy, ix] = overlap #Rectangle.overlap(WarpedImageRect, FixedImageRect)
    #             Mask[iy, ix] = overlap >= MinOverlap and overlap <= MaxOverlap
    #
    #             if overlap < MinOverlap:
    #                 Mask[iy:Mask.shape[0], ix] = False
    #                 break

    Mask = np.logical_and(fraction >= MinOverlap, fraction <= MaxOverlap)
    return Mask
