"""Cell validity helpers shared by mosaic and STOS refinement."""

from __future__ import annotations

from numpy.typing import NDArray

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp


def is_alignable_cell(cell: NDArray) -> bool:
    """Return True when *cell* has enough contrast for phase correlation / ROI align.

    Rejects empty arrays, constant (pure-color) cells, and all-zero cells.
    """
    if cell is None or cell.size == 0:
        return False
    xp = cp.get_array_module(cell)
    cell = xp.asarray(cell)
    amin = cell.min()
    amax = cell.max()
    if amin == amax:
        return False
    if amax == 0:
        return False
    return True
