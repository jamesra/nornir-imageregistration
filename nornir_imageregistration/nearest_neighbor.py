"""
Nearest-neighbor index abstraction with optional CuVS (GPU) backend.

CuVS brute-force nearest neighbor is O(N²). In 2D, scipy ``cKDTree`` is faster
for typical transform meshes (hundreds to a few thousand control points).
Measured crossover on this stack is about 4096 points (build + k=1 query):

* N=256:  cKDTree 0.28 ms vs CuVS 1.33 ms
* N=1024: cKDTree 0.33 ms vs CuVS 1.29 ms
* N=4096: cKDTree 1.63 ms vs CuVS 1.38 ms
* N=10000: cKDTree 4.38 ms vs CuVS 2.69 ms

So this module uses ``cKDTree`` below ``CUVS_NN_MIN_POINTS`` (default 4096) and
CuVS brute-force at that size or above when CuPy and CuVS are available.
Override the gate with ``NORNIR_CUVS_NN_MIN_POINTS``.

Pairwise ``cdist`` is a different problem (full N×M matrix, no tree) and stays
on CuVS whenever inputs are already on the GPU; see ``spatial_distance.cdist``.

Callers use the same ``.query(points, k=1)`` API regardless of backend.
"""
from __future__ import annotations

import os
from typing import Any, cast
import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.computational_lib import HasCuVS, UsingCupy

# Default gate: 2D cKDTree wins below this; CuVS brute-force at or above.
CUVS_NN_MIN_POINTS_DEFAULT: int = 4096
_CUVS_NN_MIN_POINTS_ENV: str = "NORNIR_CUVS_NN_MIN_POINTS"

# Optional CuVS import only when needed
_cuvs_brute_force = None
if UsingCupy() and HasCuVS():
    try:
        from cuvs.neighbors import brute_force as _cuvs_brute_force  # type: ignore[import-untyped]
    except Exception:
        pass


def cuvs_nn_min_points() -> int:
    """Return the point-count gate for CuVS brute-force nearest neighbor."""
    raw = os.environ.get(_CUVS_NN_MIN_POINTS_ENV)
    if raw is None or raw.strip() == "":
        return CUVS_NN_MIN_POINTS_DEFAULT
    try:
        return max(1, int(raw))
    except ValueError:
        return CUVS_NN_MIN_POINTS_DEFAULT


def _n_points(points: NDArray) -> int:
    """Number of points in an (N, D) array; a 1-D vector counts as one point."""
    if getattr(points, "ndim", 1) == 1:
        return 1
    return int(points.shape[0])


def _ensure_host_float32(points: NDArray) -> np.ndarray:
    """Return points as numpy float32 on host."""
    if hasattr(points, 'get'):  # type: ignore[union-attr]
        points = points.get()  # type: ignore[union-attr]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def _ensure_cupy_float32(points: NDArray):
    """Return points as a C-contiguous cupy float32 ``(N, D)`` array.

    ``cp.asarray`` normalizes dtype but not strides, and it is a no-op on an
    array that is already float32 CuPy. Control-point callers pass a (N, 2)
    view of a (N, 4) array (float32 strides (16, 4)); CuVS brute-force assumes
    packed rows and would otherwise read interleaved neighbors as coordinates.
    Mirrors the same guard in ``spatial_distance.cdist``.
    """
    import cupy as cp
    points = cp.asarray(points, dtype=cp.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return cp.ascontiguousarray(points)


class _ScipyNNIndex:
    """Wrapper around scipy.spatial.cKDTree with .query(points, k=1) compatible with callers."""

    def __init__(self, points: np.ndarray):
        from scipy.spatial import cKDTree  # type: ignore[attr-defined]
        self._tree = cKDTree(points)

    def query(self, points: NDArray, k: int = 1):
        # Match output backend to the query array (not the process-wide UsingCupy flag).
        try:
            import cupy as cp
            xp = cp.get_array_module(points)
        except Exception:
            xp = np

        host_points = _ensure_host_float32(points)
        distances, indices = self._tree.query(host_points, k=k)
        if xp is not np:
            distances = xp.asarray(distances)
            indices = xp.asarray(np.asarray(indices, dtype=np.intp))
        if k == 1 and host_points.shape[0] == 1:
            dist_scalar = float(cast(Any, distances).item()) if hasattr(distances, "item") else float(distances)
            index_scalar = int(cast(Any, indices).item()) if hasattr(indices, "item") else int(indices)
            return dist_scalar, index_scalar
        if k == 1:
            return distances, indices
        return distances, indices


class _CuVSNNIndex:
    """Wrapper around CuVS brute_force index with .query(points, k=1) matching scipy behavior."""

    def __init__(self, points):
        import cupy as cp
        assert _cuvs_brute_force is not None
        points = _ensure_cupy_float32(points)
        self._index = _cuvs_brute_force.build(points, metric="sqeuclidean")
        self._cp = cp

    def query(self, points, k: int = 1):
        cp = self._cp
        points = _ensure_cupy_float32(points)
        distances, neighbors = _cuvs_brute_force.search(self._index, points, k)  # type: ignore[union-attr]
        distances = cp.asarray(distances)
        # CuVS uses sqeuclidean; convert to Euclidean to match scipy cKDTree
        distances = cp.sqrt(cp.maximum(distances, 0.0))
        neighbors = cp.asarray(neighbors)
        if k == 1:
            distances = distances.ravel()
            neighbors = neighbors.ravel()
            if distances.size == 1:
                return float(distances.item()), int(neighbors.item())
        return distances, neighbors


def _use_cuvs_nn(points: NDArray) -> bool:
    """True when CuVS brute-force NN is expected to beat a 2D cKDTree."""
    if _cuvs_brute_force is None or not HasCuVS():
        return False
    if _n_points(points) < cuvs_nn_min_points():
        return False
    if UsingCupy():
        return True
    try:
        import cupy as cp
        return cp.get_array_module(points) is cp
    except Exception:
        return False


def build_nearest_neighbor_index(points: NDArray):
    """
    Build a nearest-neighbor index from a 2D point set (N x 2).

    Uses scipy ``cKDTree`` below ``cuvs_nn_min_points()`` (default 4096). CuVS
    brute-force is O(N²) and slower than a 2D tree at typical mesh sizes; see
    the module docstring. At the gate or above, uses a GPU index when CuPy and
    CuVS are available.

    The returned object supports ``.query(points, k=1)`` returning
    ``(distances, indices)`` in the same style as ``cKDTree.query``.

    :param points: Nx2 array of points (numpy or cupy, float32/float64).
    :return: Index-like object with .query(points, k=1) method.
    """
    if _use_cuvs_nn(points):
        return _CuVSNNIndex(points)
    points_host = _ensure_host_float32(points)
    return _ScipyNNIndex(points_host)
