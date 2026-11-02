"""Extrapolate a control-point warp outside its hull with the nearest hull triangle's affine.

Mesh and grid transforms are piecewise affine inside the convex hull of their control
points and fall back to an RBF outside it. For image warping (assembly, STOS overlays)
that RBF was the cost: RC2 Grid32 overlays spent most of each worker in
``OneWayRBFTransform._GetMatrixWeightSums`` for the pixels outside the hull, a cost that
grows with pixels times control points (2026-11-02). James chose the nearest triangle's
affine for warping instead (2026-11-02): a point outside the hull is mapped by the affine
map of the hull triangle nearest to it, so the warp continues linearly past the edge,
matches the mesh on the boundary, and reproduces a globally affine transform exactly.

Registration and point transforms keep the RBF fallback; only the image warping path in
``assemble.write_to_target_roi_coords`` uses this.

Host-only: SciPy Delaunay and cKDTree.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np
import scipy.spatial
from numpy.typing import NDArray

# Hull edges checked exactly per point, after a KD-tree over edge midpoints proposes them.
_CANDIDATE_EDGES = 8
_CHUNK_POINTS = 1 << 20


class HullAffineExtrapolator:
    """Maps points outside the hull of *from_points* by the affine of the nearest hull triangle.

    *from_points* and *to_points* are matching ``(N, 2)`` control points in (Y, X); the
    affine of each hull triangle maps its *from* vertices onto its *to* vertices.
    """

    _edge_starts: NDArray[np.float64]
    _edge_vectors: NDArray[np.float64]
    _edge_lengths_sq: NDArray[np.float64]
    _affines: NDArray[np.float64]
    _midpoint_tree: scipy.spatial.cKDTree

    def __init__(self, from_points: NDArray[np.floating], to_points: NDArray[np.floating]) -> None:
        from_points = np.asarray(from_points, dtype=np.float64).reshape(-1, 2)
        to_points = np.asarray(to_points, dtype=np.float64).reshape(-1, 2)
        triangulation = scipy.spatial.Delaunay(from_points)

        starts, ends, triangles = [], [], []
        for simplex_index, (simplex, neighbors) in enumerate(zip(triangulation.simplices,
                                                                 triangulation.neighbors)):
            for k in range(3):
                if neighbors[k] == -1:  # the edge opposite vertex k is on the hull
                    a, b = (v for j, v in enumerate(simplex) if j != k)
                    starts.append(from_points[a])
                    ends.append(from_points[b])
                    triangles.append(simplex_index)
        starts_arr = np.asarray(starts)
        self._edge_starts = starts_arr
        self._edge_vectors = np.asarray(ends) - starts_arr
        self._edge_lengths_sq = np.maximum(np.einsum('ij,ij->i', self._edge_vectors, self._edge_vectors), 1e-300)
        self._midpoint_tree = scipy.spatial.cKDTree(starts_arr + self._edge_vectors / 2.0)

        # Affine per hull triangle, applied as [y, x, 1] @ A -> (y', x').
        affines = np.empty((len(triangles), 3, 2), dtype=np.float64)
        for i, simplex_index in enumerate(triangles):
            vertices = triangulation.simplices[simplex_index]
            homogeneous = np.hstack((from_points[vertices], np.ones((3, 1))))
            affines[i] = np.linalg.solve(homogeneous, to_points[vertices])
        self._affines = affines

    def __call__(self, points: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map *points* (``(M, 2)``, Y X) by their nearest hull triangle's affine."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = np.empty_like(points)
        k = min(_CANDIDATE_EDGES, len(self._edge_starts))
        for start in range(0, points.shape[0], _CHUNK_POINTS):
            chunk = points[start:start + _CHUNK_POINTS]
            _, candidates = self._midpoint_tree.query(chunk, k=k)
            candidates = candidates.reshape(chunk.shape[0], k)
            # Exact distance from each point to each candidate segment.
            rel = chunk[:, None, :] - self._edge_starts[candidates]
            vec = self._edge_vectors[candidates]
            t = np.clip(np.einsum('pkd,pkd->pk', rel, vec) / self._edge_lengths_sq[candidates], 0.0, 1.0)
            nearest = rel - t[..., None] * vec
            best = candidates[np.arange(chunk.shape[0]), np.argmin(np.einsum('pkd,pkd->pk', nearest, nearest), axis=1)]
            homogeneous = np.hstack((chunk, np.ones((chunk.shape[0], 1))))
            out[start:start + chunk.shape[0]] = np.einsum('pi,pij->pj', homogeneous, self._affines[best])
        return out


_lock = threading.Lock()
_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def inverse_extrapolator(transform) -> HullAffineExtrapolator:
    """Target -> source extrapolator for *transform*, built once per transform and set of points."""
    target = np.asarray(_host(transform.TargetPoints), dtype=np.float64)
    source = np.asarray(_host(transform.SourcePoints), dtype=np.float64)
    signature = (target.shape, hash(target.tobytes()), hash(source.tobytes()))
    with _lock:
        cached = _cache.get(transform)
        if cached is not None and cached[0] == signature:
            return cached[1]
    extrapolator = HullAffineExtrapolator(target, source)
    with _lock:
        _cache[transform] = (signature, extrapolator)
    return extrapolator


def supports_hull_extrapolation(transform) -> bool:
    """True for mesh and grid transforms whose outside-hull fallback is an RBF."""
    name = type(transform).__name__
    return ('WithRBFFallback' in name and hasattr(transform, 'TargetPoints')
            and hasattr(transform, 'SourcePoints'))


def _host(array):
    get = getattr(array, 'get', None)
    return get() if callable(get) else array
