"""Source-side content cache for sticky LOW_CONTENT measure skips."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.cell_validity import (
    cell_intensity_std,
    low_content_std_min_threshold,
)

from nornir_imageregistration import cp


class SourceContentCache:
    """Per-refine cache of source ROI std keyed by grid ID.

    Source points are fixed on the source image for a given grid ID, so the
    content metric is computed once and reused. IDs below the std threshold are
    sticky measure-skips for the rest of the refine.
    """

    _std_by_id: dict[tuple[int, int], float]
    _low_content_ids: set[tuple[int, int]]
    min_std: float

    def __init__(self, min_std: float | None = None) -> None:
        self._std_by_id = {}
        self._low_content_ids = set()
        self.min_std = float(
            min_std if min_std is not None else low_content_std_min_threshold())

    @property
    def low_content_ids(self) -> set[tuple[int, int]]:
        """Grid IDs with sticky source low-content."""
        return self._low_content_ids

    def get(self, key: tuple[int, int]) -> float | None:
        """Return cached std or None if not yet measured."""
        return self._std_by_id.get(key)

    def remember(self, key: tuple[int, int], std: float) -> bool:
        """Store *std* for *key*; return True when content is sufficient."""
        value = float(std)
        self._std_by_id[key] = value
        if value < self.min_std:
            self._low_content_ids.add(key)
            return False
        return True

    def is_low_content(self, key: tuple[int, int]) -> bool:
        """True when *key* is known source-low-content."""
        return key in self._low_content_ids

    def ensure_from_roi(self, key: tuple[int, int], roi: NDArray) -> bool:
        """Cache std from *roi* if missing; return True when alignable."""
        if key in self._std_by_id:
            return key not in self._low_content_ids
        return self.remember(key, cell_intensity_std(roi))

    def as_mapping(self) -> Mapping[tuple[int, int], float]:
        """Read-only view of cached std values."""
        return self._std_by_id

    def clear(self) -> None:
        """Drop cached stds. Required when refine cell_size changes."""
        self._std_by_id.clear()
        self._low_content_ids.clear()


def crop_source_cell_std(
        source_image: NDArray,
        source_point: NDArray[np.floating] | tuple[float, float],
        cell_size: NDArray[np.integer] | tuple[int, int] | Sequence[int],
) -> float:
    """Crop an axis-aligned source cell and return its intensity std.

    Accepts NumPy or CuPy; ops follow ``cp.get_array_module``.
    """
    import nornir_imageregistration

    stds = crop_source_cell_stds_batched(
        source_image,
        nornir_imageregistration.EnsurePointsAre2DArray(source_point),
        cell_size)
    return float(stds[0])


def crop_source_cell_stds_batched(
        source_image: NDArray,
        source_points: NDArray[np.floating],
        cell_size: NDArray[np.integer] | tuple[int, int] | Sequence[int],
) -> NDArray[np.floating]:
    """Crop axis-aligned source cells and return intensity stds in one vectorized pass.

    Accepts NumPy or CuPy; ops follow ``cp.get_array_module``.

    Stays on the input image's array module (no forced host→device upload of the
    full mosaic). Stacks ROI crops and computes ``std`` along spatial axes so a
    CuPy session does not pay per-cell ``float(xp.std(...))`` syncs when the
    source is already on device. Host NumPy sources use a vectorized ``np.std``.
    """
    import nornir_imageregistration

    xp = cp.get_array_module(source_image)
    points = xp.asarray(source_points, dtype=xp.float64).reshape(-1, 2)
    num = int(points.shape[0])
    if num == 0:
        return np.zeros((0,), dtype=np.float64)

    area = np.asarray(cell_size, dtype=np.int64).ravel()[:2]
    h = int(area[0])
    w = int(area[1])
    half = xp.asarray((h / 2.0, w / 2.0), dtype=points.dtype)
    origins_host = nornir_imageregistration.EnsureNumpyArray(xp.floor(points - half))

    crops: list[NDArray] = []
    for i in range(num):
        bot = int(origins_host[i, 0])
        left = int(origins_host[i, 1])
        roi = nornir_imageregistration.CropImage(
            source_image, left, bot, w, h, cval=0)
        crops.append(xp.asarray(roi, dtype=xp.float64))

    stack = xp.stack(crops, axis=0)
    finite = xp.isfinite(stack)
    counts = xp.count_nonzero(finite, axis=(1, 2))
    filled = xp.where(finite, stack, xp.asarray(0.0, dtype=stack.dtype))
    stds = xp.std(filled, axis=(1, 2))
    stds = xp.where(counts >= 2, stds, xp.asarray(0.0, dtype=stds.dtype))
    return np.asarray(nornir_imageregistration.EnsureNumpyArray(stds), dtype=np.float64).reshape(-1)
