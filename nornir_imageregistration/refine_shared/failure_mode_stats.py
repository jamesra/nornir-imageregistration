"""Diagnostics helpers for Grid refine failure-mode analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def identity_lock_mask(
        locked: NDArray[np.bool_],
        travel: NDArray[np.floating],
        travel_eps: float = 0.5,
) -> NDArray[np.bool_]:
    """True where a cell is locked with near-zero peak travel (identity lock)."""
    locked_b = np.asarray(locked, dtype=bool)
    travel_f = np.asarray(travel, dtype=np.float64)
    return locked_b & (travel_f < float(travel_eps))


def peak_direction_coherence(
        peak_y: NDArray[np.floating],
        peak_x: NDArray[np.floating],
        *,
        mask: NDArray[np.bool_] | None = None,
        min_travel: float = 1.0,
) -> float:
    """Return ‖mean unit peak direction‖ in [0, 1] (1 = all peaks agree).

    Used to detect coherent residual translation when few cells lock
    (e.g. Grid16 240-241 unique peaks).
    """
    py = np.asarray(peak_y, dtype=np.float64).reshape(-1)
    px = np.asarray(peak_x, dtype=np.float64).reshape(-1)
    if mask is None:
        mask_b = np.ones(py.shape[0], dtype=bool)
    else:
        mask_b = np.asarray(mask, dtype=bool).reshape(-1)
    peaks = np.stack([py[mask_b], px[mask_b]], axis=1)
    if peaks.shape[0] == 0:
        return 0.0
    norms = np.linalg.norm(peaks, axis=1)
    keep = norms >= float(min_travel)
    if not np.any(keep):
        return 0.0
    unit = peaks[keep] / norms[keep, None]
    return float(np.linalg.norm(unit.mean(axis=0)))


def spatial_half_stats(
        source_x: NDArray[np.floating],
        locked: NDArray[np.bool_],
        travel: NDArray[np.floating],
) -> dict[str, dict[str, float]]:
    """Summarize lock/travel for low-x vs high-x halves (Composite L/R proxy)."""
    sx = np.asarray(source_x, dtype=np.float64).reshape(-1)
    locked_b = np.asarray(locked, dtype=bool).reshape(-1)
    travel_f = np.asarray(travel, dtype=np.float64).reshape(-1)
    mid = float(np.median(sx))
    out: dict[str, dict[str, float]] = {}
    for name, mask in (('low_x', sx <= mid), ('high_x', sx > mid)):
        if not np.any(mask):
            out[name] = {'n': 0.0, 'lock_frac': 0.0, 'travel_med': float('nan'),
                         'identity_lock_frac': 0.0}
            continue
        id_lock = identity_lock_mask(locked_b[mask], travel_f[mask])
        out[name] = {
            'n': float(mask.sum()),
            'lock_frac': float(locked_b[mask].mean()),
            'travel_med': float(np.median(travel_f[mask])),
            'identity_lock_frac': float(id_lock.mean()),
        }
    return out
