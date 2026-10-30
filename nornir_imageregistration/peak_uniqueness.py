"""Peak uniqueness (masked primary / 2nd-peak ratio) for false-peak detection."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

# Default exclusion radius around the primary peak before searching for a 2nd peak.
DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS: int = 3

# When no competing peak remains after masking, treat the primary as unique.
_UNIQUE_PEAK_RATIO: float = 1.0e6


def masked_peak_ratio(
        image: NDArray[np.floating],
        peak_row: int,
        peak_col: int,
        *,
        exclusion_radius: int = DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
        overlap_mask: Optional[NDArray[np.bool_]] = None,
        primary_value: float | None = None,
) -> float:
    """Return primary / 2nd-peak ratio after zeroing a square around the primary.

    :param image: 2D correlation image (preferably normalized to ``[0, 1]``).
    :param peak_row: Integer row of the primary peak.
    :param peak_col: Integer column of the primary peak.
    :param exclusion_radius: Half-width of the square cleared around the primary.
    :param overlap_mask: Optional eligible-pixel mask (True = keep).
    :param primary_value: Optional primary peak intensity; defaults to ``image[peak]``.
    :return: Ratio ``>= 1`` when a second peak exists; ``_UNIQUE_PEAK_RATIO`` when
        nothing competes; ``0`` when the primary is non-positive / invalid.
    """
    xp = cp.get_array_module(image)
    image = xp.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    h, w = int(image.shape[0]), int(image.shape[1])
    pr = int(np.clip(peak_row, 0, h - 1))
    pc = int(np.clip(peak_col, 0, w - 1))
    r = max(0, int(exclusion_radius))

    if primary_value is None:
        primary = float(image[pr, pc])
    else:
        primary = float(primary_value)
    if not np.isfinite(primary) or primary <= 0.0:
        return 0.0

    search = image.astype(image.dtype, copy=True)
    r0 = max(0, pr - r)
    r1 = min(h, pr + r + 1)
    c0 = max(0, pc - r)
    c1 = min(w, pc + r + 1)
    search[r0:r1, c0:c1] = xp.asarray(-xp.inf, dtype=search.dtype)

    if overlap_mask is not None:
        mask = xp.asarray(overlap_mask, dtype=xp.bool_)
        if mask.shape != image.shape:
            raise ValueError("overlap_mask must match image shape")
        search = xp.where(mask, search, xp.asarray(-xp.inf, dtype=search.dtype))

    second = float(search.max())
    if not np.isfinite(second) or second <= 0.0:
        return float(_UNIQUE_PEAK_RATIO)
    return float(primary / max(second, 1e-6))


def batched_masked_peak_ratios(
        images: NDArray[np.floating],
        peak_rows: NDArray[np.integer],
        peak_cols: NDArray[np.integer],
        *,
        exclusion_radius: int = DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
        overlap_mask: Optional[NDArray[np.bool_]] = None,
        primary_values: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Vectorized ``masked_peak_ratio`` over a ``(N, h, w)`` correlation stack.

    :return: ``(N,)`` ratios on the same array module as ``images``.
    """
    xp = cp.get_array_module(images)
    images = xp.asarray(images)
    if images.ndim != 3:
        raise ValueError("images must be a (N, h, w) stack")

    n, h, w = images.shape
    peak_rows = xp.asarray(peak_rows, dtype=xp.int64).reshape(n)
    peak_cols = xp.asarray(peak_cols, dtype=xp.int64).reshape(n)
    peak_rows = xp.clip(peak_rows, 0, h - 1)
    peak_cols = xp.clip(peak_cols, 0, w - 1)
    r = max(0, int(exclusion_radius))

    if primary_values is None:
        flat = images.reshape(n, -1)
        idx = peak_rows * w + peak_cols
        primary = flat[xp.arange(n), idx]
    else:
        primary = xp.asarray(primary_values, dtype=images.dtype).reshape(n)

    row_coords = xp.arange(h, dtype=xp.int64)[None, :, None]
    col_coords = xp.arange(w, dtype=xp.int64)[None, None, :]
    excluded = (
        (xp.abs(row_coords - peak_rows[:, None, None]) <= r)
        & (xp.abs(col_coords - peak_cols[:, None, None]) <= r)
    )
    neg_inf = xp.asarray(-xp.inf, dtype=images.dtype)
    search = xp.where(excluded, neg_inf, images)

    if overlap_mask is not None:
        mask = xp.asarray(overlap_mask, dtype=xp.bool_)
        if mask.shape != (h, w):
            raise ValueError("overlap_mask must be (h, w)")
        search = xp.where(mask[None, :, :], search, neg_inf)

    second = search.reshape(n, -1).max(axis=1)
    safe_second = xp.maximum(second, xp.asarray(1e-6, dtype=images.dtype))
    unique = xp.asarray(_UNIQUE_PEAK_RATIO, dtype=images.dtype)
    zero = xp.asarray(0.0, dtype=images.dtype)

    primary_ok = xp.isfinite(primary) & (primary > 0)
    second_ok = xp.isfinite(second) & (second > 0)
    ratios = xp.where(
        primary_ok & second_ok,
        primary / safe_second,
        xp.where(primary_ok, unique, zero),
    )
    return ratios
