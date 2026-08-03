"""

"""
from __future__ import annotations

import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing
from numpy.typing import NDArray

import nornir_imageregistration

# Shared pool for optional background extrema/stats; callers must opt in via prefetch.
_EXTREMA_PREFETCH_POOL: ThreadPoolExecutor | None = None
_EXTREMA_PREFETCH_POOL_LOCK = threading.Lock()


def _get_extrema_prefetch_pool() -> ThreadPoolExecutor:
    """Return the shared extrema prefetch pool, creating it on first use."""
    global _EXTREMA_PREFETCH_POOL
    with _EXTREMA_PREFETCH_POOL_LOCK:
        if _EXTREMA_PREFETCH_POOL is None:
            _EXTREMA_PREFETCH_POOL = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="img-extrema-prefetch",
            )
        return _EXTREMA_PREFETCH_POOL


class ImagePermutationHelper:
    """
    A helper class that takes an image and optional mask.  It exposes the image, mask, a version
     of the image with random noise where the mask is over the image.  It also exposes a version
     of the mask with image extrema values added to the mask
    :param object:
    :return:
    """
    _image: NDArray
    _mask: NDArray | None
    _blended_mask: NDArray | None
    _stats: nornir_imageregistration.ImageStats | None
    _image_with_mask_as_noise: NDArray | None
    _extrema_size_cutoff_in_pixels: int
    _extrema_future: Future[None] | None
    _extrema_lock: threading.Lock

    @property
    def shape(self) -> tuple[int, int]:
        return self._image.shape

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
    def Mask(self) -> NDArray | None:
        """
        :return:  The mask passed to the constructor, may be None
        """
        return self._mask

    @property
    def BlendedMask(self) -> NDArray:
        """
        :return: The mask combined with the extrema mask.  Is only the extrema mask if there was no Mask passed
        """
        self._ensure_blended_mask_and_stats()
        assert self._blended_mask is not None
        return self._blended_mask

    @property
    def Stats(self) -> nornir_imageregistration.ImageStats:
        """
        :return: Statistics for unmasked portion of the image
        """
        self._ensure_blended_mask_and_stats()
        assert self._stats is not None
        return self._stats

    @property
    def ImageWithMaskAsNoise(self) -> NDArray:
        """
        :return:  The image with random noise over the masked regions
        """
        if self._image_with_mask_as_noise is None:
            self._image_with_mask_as_noise = nornir_imageregistration.RandomNoiseMask(
                self._image,
                self.BlendedMask,
                imagestats=self.Stats,
                Copy=True,
            )
        return self._image_with_mask_as_noise

    def prefetch_extrema_async(self, executor: Executor | None = None) -> None:
        """Start background extrema/stats if not already computed or in flight.

        Safe to call multiple times. Accessors join any in-flight future.
        Does nothing if results already exist.
        """
        with self._extrema_lock:
            if self._blended_mask is not None and self._stats is not None:
                return
            if self._extrema_future is not None and not self._extrema_future.done():
                return
            pool = executor if executor is not None else _get_extrema_prefetch_pool()
            self._extrema_future = pool.submit(self._compute_blended_mask_and_stats)

    def _compute_blended_mask_and_stats(self) -> None:
        """Compute extrema blend and unmasked stats (may run on a worker thread)."""
        if self._blended_mask is not None and self._stats is not None:
            return

        extrema_mask = nornir_imageregistration.CreateExtremaMask(
            self._image,
            self._mask,
            size_cutoff=self._extrema_size_cutoff_in_pixels,
        )
        blended_mask = (
            np.logical_and(self._mask, extrema_mask)
            if self._mask is not None
            else extrema_mask
        )
        # CreateExtremaMask can mark an entire small ROI as excluded extrema (all False).
        # Keep tissue (or the raw image) rather than computing stats on an empty selection
        # and then noise-filling the whole ROI for registration.
        if not np.any(blended_mask):
            if self._mask is not None and np.any(self._mask):
                blended_mask = self._mask
            elif self._image.size > 0:
                blended_mask = np.ones(self._image.shape, dtype=bool)
            else:
                raise ValueError("Image has no data")

        stats = nornir_imageregistration.ImageStats.Create(self._image[blended_mask])
        with self._extrema_lock:
            if self._blended_mask is None:
                self._blended_mask = blended_mask
            if self._stats is None:
                self._stats = stats

    def _ensure_blended_mask_and_stats(self) -> None:
        """Build extrema blend and unmasked stats on first use, joining any prefetch."""
        if self._blended_mask is not None and self._stats is not None:
            return

        future: Future[None] | None
        with self._extrema_lock:
            if self._blended_mask is not None and self._stats is not None:
                return
            future = self._extrema_future

        if future is not None:
            future.result()
            return

        self._compute_blended_mask_and_stats()

    def __init__(self,
                 img: nornir_imageregistration.ImageLike,  # type: ignore[reportInvalidTypeForm]
                 mask: nornir_imageregistration.ImageLike | None = None,  # type: ignore[reportInvalidTypeForm]
                 extrema_mask_size_cuttoff: float | int | None = None,  # type: ignore[reportInvalidTypeForm]
                 dtype: Any = None):

        if dtype is None:
            try:
                dtype = img.dtype if np.issubdtype(img.dtype,
                                                   np.floating) else nornir_imageregistration.default_image_dtype()
            except:
                dtype = nornir_imageregistration.default_image_dtype()

        self._image_with_mask_as_noise = None
        self._blended_mask = None
        self._stats = None
        self._extrema_future = None
        self._extrema_lock = threading.Lock()

        img = nornir_imageregistration.ImageParamToImageArray(img, dtype=dtype)
        mask = nornir_imageregistration.ImageParamToImageArray(mask, dtype=bool) if mask is not None else None

        # Check if mask is multi-channel, if it is, take the first channel as the mask
        if mask is not None and len(mask.shape) > 2:
            mask = np.any(mask, axis=2)

        if mask is not None and img.shape != mask.shape:
            img, mask = nornir_imageregistration.EnsureMatchingImageMaskShape(img, mask)

        extrema_pixels: int
        if extrema_mask_size_cuttoff is None:
            extrema_mask_size_cuttoff = 0.01

        if isinstance(extrema_mask_size_cuttoff, np.ndarray):
            extrema_pixels = int(np.prod(extrema_mask_size_cuttoff))
        elif isinstance(extrema_mask_size_cuttoff, float):
            extrema_pixels = int(np.prod(img.shape) * extrema_mask_size_cuttoff)
        elif isinstance(extrema_mask_size_cuttoff, int):
            extrema_pixels = extrema_mask_size_cuttoff
        else:
            raise ValueError(f"extrema_mask_size_cutoff")
        self._extrema_size_cutoff_in_pixels = extrema_pixels

        self._image = img.astype(dtype, copy=False)
        self._mask = mask
