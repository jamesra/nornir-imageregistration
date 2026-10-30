"""Spatial regularization of per-vertex displacement fields (mosaic legacy port)."""

from __future__ import annotations

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray


def regularize_displacements(
        shifts: NDArray[np.floating],
        measured: NDArray[np.bool_],
        mesh_dims: tuple[int, int],
        median_radius: int = 1) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Port of legacy ``regularize_displacements`` (mosaic_refinement_common.cxx).

    Stages: median filter (radius ``median_radius``) on the measured displacement fields,
    radius-1 ring gap-fill for unmeasured vertices, then Gaussian blur (sigma=1) over the
    entire fields. Returns regularized per-vertex (y, x) shifts and the measured/filled
    flags (the legacy ``db`` image, accumulated into ``mass`` by the caller).
    """
    mesh_rows, mesh_cols = int(mesh_dims[0]), int(mesh_dims[1])
    dy = np.asarray(shifts[:, 0], dtype=np.float64).reshape(mesh_rows, mesh_cols)
    dx = np.asarray(shifts[:, 1], dtype=np.float64).reshape(mesh_rows, mesh_cols)
    db = np.asarray(measured, dtype=np.float64).reshape(mesh_rows, mesh_cols)

    if median_radius > 0:
        size = 2 * int(median_radius) + 1
        dy = scipy.ndimage.median_filter(dy, size=size, mode='nearest')
        dx = scipy.ndimage.median_filter(dx, size=size, mode='nearest')

    dy_filled = dy.copy()
    dx_filled = dx.copy()
    db_filled = db.copy()
    unmeasured_rows, unmeasured_cols = np.nonzero(db == 0)
    for row, col in zip(unmeasured_rows.tolist(), unmeasured_cols.tolist()):
        py = 0.0
        px = 0.0
        w = 0.0
        r = 1
        x0, x1 = col - r, col + r
        y0, y1 = row - r, row + r
        d = 2 * r + 1
        for o in range(d):
            for cx, cy in ((x0, y0 + o + 1), (x1, y0 + o), (x0 + o, y0), (x0 + o + 1, y1)):
                if 0 <= cx < mesh_cols and 0 <= cy < mesh_rows and db[cy, cx] != 0:
                    px += dx[cy, cx]
                    py += dy[cy, cx]
                    w += 1.0
        if w != 0.0:
            dy_filled[row, col] = py / w
            dx_filled[row, col] = px / w
            db_filled[row, col] = 1.0

    dy_filled = scipy.ndimage.gaussian_filter(dy_filled, sigma=1.0, mode='nearest', truncate=2.0)
    dx_filled = scipy.ndimage.gaussian_filter(dx_filled, sigma=1.0, mode='nearest', truncate=2.0)

    out_shifts = np.column_stack((dy_filled.reshape(-1), dx_filled.reshape(-1)))
    return out_shifts, db_filled.reshape(-1)
