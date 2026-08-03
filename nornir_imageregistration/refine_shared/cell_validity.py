"""Cell validity helpers shared by mosaic and STOS refinement."""

from __future__ import annotations

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
    """
    return float(get_runtime_config(refresh=True).low_content_std_min)


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
