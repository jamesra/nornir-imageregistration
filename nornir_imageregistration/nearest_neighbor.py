"""
Nearest-neighbor index abstraction with optional CuVS (GPU) backend.

When CuPy is active and CuVS is available, builds a GPU index for fast NN search.
Otherwise uses scipy.spatial.cKDTree. Callers use the same .query(points, k=1) API.
"""
from __future__ import annotations

from typing import Any, cast
import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.computational_lib import HasCuVS, UsingCupy

# Optional CuVS import only when needed
_cuvs_brute_force = None
if UsingCupy() and HasCuVS():
    try:
        from cuvs.neighbors import brute_force as _cuvs_brute_force  # type: ignore[import-untyped]
    except Exception:
        pass


def _ensure_host_float32(points: NDArray) -> np.ndarray:
    """Return points as numpy float32 on host."""
    if hasattr(points, 'get'):  # type: ignore[union-attr]
        points = points.get()  # type: ignore[union-attr]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def _ensure_cupy_float32(points: NDArray):
    """Return points as cupy float32."""
    import cupy as cp
    if not hasattr(points, 'get'):
        points = cp.asarray(points, dtype=cp.float32)
    else:
        points = cp.asarray(points, dtype=cp.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


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


def build_nearest_neighbor_index(points: NDArray):
    """
    Build a nearest-neighbor index from a 2D point set (N x 2).

    When CuPy is active and CuVS is available, uses a GPU index. Otherwise
    uses scipy.spatial.cKDTree. The returned object supports .query(points, k=1)
    returning (distances, indices) in the same style as cKDTree.query.

    :param points: Nx2 array of points (numpy or cupy, float32/float64).
    :return: Index-like object with .query(points, k=1) method.
    """
    if UsingCupy() and HasCuVS() and _cuvs_brute_force is not None:
        return _CuVSNNIndex(points)
    points_host = _ensure_host_float32(points)
    return _ScipyNNIndex(points_host)
