"""Cell validity helpers shared by mosaic and STOS refinement."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

# Default minimum std-dev of ROI intensities (float64) for alignable content.
# Override: NORNIR_REFINE_LOW_CONTENT_STD_MIN (unset = this default).
# Source cells below this are sticky measure-skip + REJECT(LOW_CONTENT).
DEFAULT_LOW_CONTENT_STD_MIN: float = 1e-3


def low_content_std_min_threshold() -> float:
    """Return the min ROI std for alignable content.

    Meaning
        Minimum spatial standard deviation of a cell ROI (float64 intensities,
        after casting) required to attempt phase correlation.

    Default
        ``DEFAULT_LOW_CONTENT_STD_MIN`` (1e-3) when the env is unset or invalid.

    Valid values
        Non-negative finite float.

    Effect
        Source ROI below threshold → sticky measure-skip for the refine and
        ``RejectReason.LOW_CONTENT``. Does not permanently blacklist peak-ambiguous
        cells (those remasure). Moving-only flat with structured source may still
        remasure after the control transform improves.

    Reads the cached config. ``is_alignable_cell`` calls this once per cell, twice
    per measurement, and ``refresh=True`` re-reads every refine env var and rebuilds
    the config -- 8 us against 42 ns cached, which also meant the ``lru_cache``
    never served anything. Refine entry points refresh once per pass, so a
    mid-process change still lands within one pass.
    """
    return float(get_runtime_config().low_content_std_min)


def cell_intensity_std(cell: NDArray, *, mask: NDArray | None = None) -> float:
    """Return std of *cell* intensities (float64), optionally masked."""
    if cell is None or cell.size == 0:
        return 0.0
    xp = cp.get_array_module(cell)
    arr = xp.asarray(cell, dtype=xp.float64)
    if mask is not None:
        valid = xp.asarray(mask, dtype=bool) & xp.isfinite(arr)
        n = int(xp.count_nonzero(valid))
        if n < 2:
            return 0.0
        return float(xp.std(arr[valid]))
    finite = xp.isfinite(arr)
    if not bool(xp.any(finite)):
        return 0.0
    return float(xp.std(arr[finite]))


def _alignable_on_device(
        cell_arr: NDArray,
        xp,
        threshold: float,
        mask: NDArray | None) -> bool:
    """Evaluate the whole alignability test on device, syncing once at the end.

    The host path reads three separate device scalars -- ``amin == amax``,
    ``amax == 0``, then ``float(std)`` -- and each comparison drags a 0-d array
    back across the bus. ``cell_intensity_std`` adds more: ``count_nonzero`` and
    the ``arr[valid]`` boolean gather both need the element count on the host.

    So every reduction is kept as a device scalar and combined into one boolean,
    and the std is computed arithmetically rather than by gathering the valid
    elements, which avoids a size that only the host would know.
    """
    amin = cell_arr.min()
    amax = cell_arr.max()
    # NaN compares unequal to everything, so a cell containing NaN passes both of
    # these, exactly as the host path's == comparisons did.
    ok = (amin != amax) & (amax != 0)

    if threshold > 0.0:
        arr = cell_arr.astype(xp.float64, copy=False)
        valid = xp.isfinite(arr)
        if mask is not None:
            valid = valid & xp.asarray(mask, dtype=bool)
        count = valid.sum()
        # where() rather than a multiply: NaN * False is NaN, not zero.
        divisor = xp.maximum(count, 1)
        mean = xp.where(valid, arr, xp.float64(0.0)).sum() / divisor
        deviation = xp.where(valid, arr - mean, xp.float64(0.0))
        std = xp.sqrt((deviation * deviation).sum() / divisor)
        # Fewer than two valid samples gave std 0.0 on the host, which rejects
        # whenever the threshold is positive.
        ok = ok & (count >= 2) & (std >= threshold)

    return bool(ok)


def is_alignable_cell(
        cell: NDArray,
        *,
        min_std: float | None = None,
        mask: NDArray | None = None,
) -> bool:
    """Return True when *cell* has enough contrast for phase correlation / ROI align.

    Rejects empty arrays, constant (pure-color) cells, all-zero cells, and cells
    whose intensity std is below ``min_std`` (default from
    ``NORNIR_REFINE_LOW_CONTENT_STD_MIN`` / ``DEFAULT_LOW_CONTENT_STD_MIN``).
    """
    if cell is None or cell.size == 0:
        return False
    xp = cp.get_array_module(cell)
    cell_arr = xp.asarray(cell)

    if xp is not np:
        threshold = float(
            min_std if min_std is not None else low_content_std_min_threshold())
        return _alignable_on_device(cell_arr, xp, threshold, mask)

    # Host arrays have no sync to amortise, so keep short-circuiting instead and
    # skip the std entirely for constant or all-zero cells.
    amin = cell_arr.min()
    amax = cell_arr.max()
    if amin == amax:
        return False
    if amax == 0:
        return False
    threshold = float(min_std if min_std is not None else low_content_std_min_threshold())
    if threshold > 0.0 and cell_intensity_std(cell_arr, mask=mask) < threshold:
        return False
    return True
