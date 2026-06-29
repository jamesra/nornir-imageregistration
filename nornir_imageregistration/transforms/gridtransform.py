import logging
from typing import Any, cast

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

# Optional: older CuPy / builds may lack cupyx.scipy.interpolate (use SciPy on CPU below).
cuRegularGridInterpolator: Any | None = None
cuLinearNDInterpolator: Any | None = None
try:
    from cupyx.scipy.interpolate import RegularGridInterpolator as cuRegularGridInterpolator
    from cupyx.scipy.interpolate import LinearNDInterpolator as cuLinearNDInterpolator
except ImportError:
    pass

from numpy.typing import NDArray
import scipy
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import scipy.spatial

import nornir_imageregistration
from nornir_imageregistration.nearest_neighbor import build_nearest_neighbor_index
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms import float_to_shortest_string
from nornir_imageregistration.transforms.controlpointbase import ControlPointBase, ControlPointBase_GPUComponent
from .base import ITransformScaling, ITransformRelativeScaling, \
    ITransformTranslation, \
    TransformType, ITransformTargetRotation, ITargetSpaceControlPointEdit, IGridTransform, \
    ITriangulatedTargetSpace

_logger = logging.getLogger(__name__)


def _is_cupy_degenerate_triangulation_error(exc: BaseException) -> bool:
    """Return True when cupyx Delaunay rejected control points as degenerate or coplanar."""
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).lower()
    return 'degenerate' in message or 'coplanar' in message


# Cache of simplex (triangle) vertex-index arrays keyed by (rows, cols).
# A regular grid's triangulation topology depends only on its dimensions, so the
# connectivity can be derived once analytically and reused across every transform
# that shares those dimensions.
_GRID_SIMPLEX_CACHE: dict[tuple[int, int], NDArray[np.intp]] = {}


def _analytic_grid_simplices(rows: int, cols: int) -> NDArray[np.intp]:
    """Return the Delaunay-equivalent simplices for a row-major ``rows`` x ``cols`` regular grid.

    Each grid cell is split into two triangles. Vertex indices reference a
    row-major flattened lattice (index = row * cols + col), matching how
    ``GridTransform`` stores its control points. The result is cached per
    ``(rows, cols)`` because it never changes for a given grid shape.
    """
    key = (rows, cols)
    cached = _GRID_SIMPLEX_CACHE.get(key)
    if cached is not None:
        return cached

    if rows < 2 or cols < 2:
        # A degenerate 1xN / Nx1 grid (or smaller) has no triangles.
        simplices = np.empty((0, 3), dtype=np.intp)
        _GRID_SIMPLEX_CACHE[key] = simplices
        return simplices

    i, j = np.mgrid[0:rows - 1, 0:cols - 1]
    tl = (i * cols + j).ravel()  # top-left
    tr = (i * cols + j + 1).ravel()  # top-right
    bl = ((i + 1) * cols + j).ravel()  # bottom-left
    br = ((i + 1) * cols + j + 1).ravel()  # bottom-right
    tri1 = np.column_stack([tl, tr, bl])
    tri2 = np.column_stack([tr, br, bl])
    simplices = np.vstack([tri1, tri2]).astype(np.intp)
    _GRID_SIMPLEX_CACHE[key] = simplices
    return simplices


class _GridTopologyLinearInterpolator:
    """Linear (barycentric) interpolator over a regular grid's known triangulation.

    Drop-in replacement for ``scipy.interpolate.LinearNDInterpolator`` for the case
    where the control points form a ``rows`` x ``cols`` regular lattice. Because the
    triangulation topology is derived analytically from the grid dimensions, this
    avoids Qhull entirely (and therefore the CuPy degenerate/coplanar failure mode).

    Operates on NumPy arrays. CuPy inputs are converted to NumPy at a single
    boundary in ``__init__``/``__call__`` and NumPy is returned, matching the
    behavior callers already expect from the SciPy fallback path.
    """

    # Number of nearest candidate triangles probed per query point. The containing
    # triangle of an interior point is, for a structured grid, always among the very
    # nearest centroids; this margin is generous for robustness near distorted cells.
    _DEFAULT_CANDIDATES = 32

    # Tolerance on barycentric coordinates so points exactly on a shared edge/vertex
    # are accepted by at least one triangle.
    _BARYCENTRIC_TOL = 1e-9

    def __init__(self,
                 target_points: NDArray[np.floating],
                 source_values: NDArray[np.floating],
                 grid_dims: tuple[int, int]):
        rows, cols = int(grid_dims[0]), int(grid_dims[1])
        self._target_points = np.ascontiguousarray(
            nornir_imageregistration.EnsureNumpyArray(target_points), dtype=np.float64)
        self._source_values = np.ascontiguousarray(
            nornir_imageregistration.EnsureNumpyArray(source_values), dtype=np.float64)
        self._simplices = _analytic_grid_simplices(rows, cols)

        verts = self._target_points[self._simplices]  # (ntri, 3, 2)
        self._v2 = verts[:, 2, :]  # reference vertex per triangle (ntri, 2)
        # Edge matrix columns: (v0 - v2) and (v1 - v2). Invert per-triangle so a
        # query point maps directly to its first two barycentric weights.
        e0 = verts[:, 0, :] - self._v2
        e1 = verts[:, 1, :] - self._v2
        a = e0[:, 0]
        c = e0[:, 1]
        b = e1[:, 0]
        d = e1[:, 1]
        det = a * d - b * c
        self._valid = np.abs(det) > 0  # collinear/NaN triangles are unusable
        safe_det = np.where(self._valid, det, 1.0)
        inv = np.empty((self._simplices.shape[0], 2, 2), dtype=np.float64)
        inv[:, 0, 0] = d / safe_det
        inv[:, 0, 1] = -b / safe_det
        inv[:, 1, 0] = -c / safe_det
        inv[:, 1, 1] = a / safe_det
        self._inv = inv

        self._centroids = verts.mean(axis=1)  # (ntri, 2)
        if self._centroids.shape[0] > 0:
            self._tree = scipy.spatial.cKDTree(self._centroids)
        else:
            self._tree = None

    def __call__(self, xi: NDArray[np.floating]) -> NDArray[np.floating]:
        pts = np.ascontiguousarray(
            nornir_imageregistration.EnsureNumpyArray(xi), dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        n_out = pts.shape[0]
        m = self._source_values.shape[1]
        result = np.full((n_out, m), np.nan, dtype=np.float64)

        if self._tree is None or n_out == 0:
            return result

        k = min(self._DEFAULT_CANDIDATES, self._centroids.shape[0])
        _, candidates = self._tree.query(pts, k=k)
        if candidates.ndim == 1:
            candidates = candidates.reshape(-1, 1)

        found = np.zeros(n_out, dtype=bool)
        tol = -self._BARYCENTRIC_TOL
        for j in range(candidates.shape[1]):
            remaining = ~found
            if not np.any(remaining):
                break

            cand = candidates[:, j]
            inv_c = self._inv[cand]  # (n, 2, 2)
            delta = pts - self._v2[cand]  # (n, 2)
            b0 = inv_c[:, 0, 0] * delta[:, 0] + inv_c[:, 0, 1] * delta[:, 1]
            b1 = inv_c[:, 1, 0] * delta[:, 0] + inv_c[:, 1, 1] * delta[:, 1]
            b2 = 1.0 - b0 - b1
            inside = (self._valid[cand]
                      & (b0 >= tol) & (b1 >= tol) & (b2 >= tol)
                      & remaining)

            idx_in = np.nonzero(inside)[0]
            if idx_in.size == 0:
                continue

            weights = np.stack([b0[idx_in], b1[idx_in], b2[idx_in]], axis=1)  # (p, 3)
            tri_vals = self._source_values[self._simplices[cand[idx_in]]]  # (p, 3, m)
            result[idx_in] = np.einsum('pk,pkm->pm', weights, tri_vals)
            found[idx_in] = True

        return result


class _CuGridTopologyInterpolator:
    """GPU analytic barycentric interpolator for regular grids with known triangulation topology.

    Drop-in GPU replacement for :class:`_GridTopologyLinearInterpolator`.  The known grid
    topology lets us skip KD-tree construction entirely: for each query point we compute an
    approximate grid-cell index from the bounding box, check the two triangles in that cell,
    then fall back to the 8 surrounding cells for the small fraction of edge/boundary cases.

    All per-point work is vectorized in CuPy; no Python loop over individual query points.
    Only two device→host syncs are needed per ``__call__``: one to test whether the
    approximate lookup resolved all points, and one (if needed) to gather remaining indices
    for the neighbor expansion.
    """

    _BARYCENTRIC_TOL: float = 1e-9
    # Cardinal neighbors first, then diagonals.
    _NEIGHBOR_DI: tuple[int, ...] = (-1, 1, 0, 0, -1, -1, 1, 1)
    _NEIGHBOR_DJ: tuple[int, ...] = (0, 0, -1, 1, -1, 1, -1, 1)

    def __init__(
            self,
            target_points: NDArray[np.floating],
            source_values: NDArray[np.floating],
            grid_dims: tuple[int, int],
    ) -> None:
        rows, cols = int(grid_dims[0]), int(grid_dims[1])
        self._rows = rows
        self._cols = cols

        simplices_np = _analytic_grid_simplices(rows, cols)  # cached by (rows, cols)
        tgt_np = np.ascontiguousarray(
            nornir_imageregistration.EnsureNumpyArray(target_points), dtype=np.float64)
        src_np = np.ascontiguousarray(
            nornir_imageregistration.EnsureNumpyArray(source_values), dtype=np.float64)

        ntri = simplices_np.shape[0]
        if ntri > 0:
            verts = tgt_np[simplices_np]       # (ntri, 3, 2)
            v2_np = verts[:, 2, :]             # (ntri, 2) reference vertex
            e0 = verts[:, 0, :] - v2_np
            e1 = verts[:, 1, :] - v2_np
            a, c = e0[:, 0], e0[:, 1]
            b, d = e1[:, 0], e1[:, 1]
            det = a * d - b * c
            valid_np = np.abs(det) > 0
            safe_det = np.where(valid_np, det, 1.0)
            inv_np = np.empty((ntri, 2, 2), dtype=np.float64)
            inv_np[:, 0, 0] = d / safe_det
            inv_np[:, 0, 1] = -b / safe_det
            inv_np[:, 1, 0] = -c / safe_det
            inv_np[:, 1, 1] = a / safe_det
        else:
            v2_np = np.empty((0, 2), dtype=np.float64)
            valid_np = np.empty(0, dtype=bool)
            inv_np = np.empty((0, 2, 2), dtype=np.float64)

        # Single host→device transfer; tensors reused across __call__ invocations.
        self._v2 = cp.asarray(v2_np)
        self._inv = cp.asarray(inv_np)
        self._valid = cp.asarray(valid_np)
        self._simplices = cp.asarray(simplices_np)
        self._source_vals = cp.asarray(src_np)

        # Approximate cell geometry (bounding box divided by grid step).
        self._min_y = float(tgt_np[:, 0].min()) if len(tgt_np) > 0 else 0.0
        self._min_x = float(tgt_np[:, 1].min()) if len(tgt_np) > 0 else 0.0
        self._cell_h = ((float(tgt_np[:, 0].max()) - self._min_y) / (rows - 1)
                        if rows > 1 else 1.0)
        self._cell_w = ((float(tgt_np[:, 1].max()) - self._min_x) / (cols - 1)
                        if cols > 1 else 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_tri(
            self,
            pts: 'cp.ndarray',
            tri_idx: 'cp.ndarray',
            result: 'cp.ndarray',
            found: 'cp.ndarray',
            tol: float,
    ) -> None:
        """Vectorized barycentric evaluation for all pts against their candidate triangle.

        Updates ``result`` and ``found`` in place; only overwrites positions where the
        point falls inside the triangle AND has not already been found.
        """
        inv_c = self._inv[tri_idx]         # (n, 2, 2)
        delta = pts - self._v2[tri_idx]    # (n, 2)
        b0 = inv_c[:, 0, 0] * delta[:, 0] + inv_c[:, 0, 1] * delta[:, 1]
        b1 = inv_c[:, 1, 0] * delta[:, 0] + inv_c[:, 1, 1] * delta[:, 1]
        b2 = 1.0 - b0 - b1
        inside = (self._valid[tri_idx]
                  & (b0 >= tol) & (b1 >= tol) & (b2 >= tol)
                  & ~found)
        weights = cp.stack([b0, b1, b2], axis=1)                   # (n, 3)
        tri_verts = self._source_vals[self._simplices[tri_idx]]     # (n, 3, m)
        interp = cp.einsum('pk,pkm->pm', weights, tri_verts)        # (n, m)
        result[:] = cp.where(inside[:, None], interp, result)
        found[:] = found | inside

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, xi: NDArray) -> 'cp.ndarray':
        pts = cp.ascontiguousarray(cp.asarray(xi, dtype=cp.float64))
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        n = pts.shape[0]
        m = self._source_vals.shape[1]
        rows, cols = self._rows, self._cols
        tol = -self._BARYCENTRIC_TOL

        if n == 0 or self._simplices.shape[0] == 0:
            return cp.full((n, m), cp.nan, dtype=cp.float64)

        result = cp.full((n, m), cp.nan, dtype=cp.float64)

        # Approximate grid-cell index for every query point (fully on-device).
        i_approx = cp.clip(
            cp.floor((pts[:, 0] - self._min_y) / self._cell_h).astype(cp.int32),
            0, rows - 2)
        j_approx = cp.clip(
            cp.floor((pts[:, 1] - self._min_x) / self._cell_w).astype(cp.int32),
            0, cols - 2)

        found = cp.zeros(n, dtype=cp.bool_)
        base_cell = i_approx * (cols - 1) + j_approx  # (n,)

        # Pass 1: approximate cell only (2 triangles).
        # For well-behaved grids this resolves >99 % of query points.
        for dt in (0, 1):
            self._eval_tri(pts, base_cell * 2 + dt, result, found, tol)

        # Single sync: check whether any points remain unresolved.
        n_remaining: int = int(cp.sum(~found))
        if n_remaining == 0:
            return result

        # Pass 2: 3×3 neighborhood expansion for the rare unresolved points.
        # Gather them once to keep the batch small.
        remain_idx = cp.nonzero(~found)[0]          # (n_remaining,) on device
        pts_r = pts[remain_idx]
        i_r = i_approx[remain_idx]
        j_r = j_approx[remain_idx]
        result_r = result[remain_idx].copy()
        found_r = found[remain_idx].copy()

        for di, dj in zip(self._NEIGHBOR_DI, self._NEIGHBOR_DJ):
            ci = cp.clip(i_r + di, 0, rows - 2)
            cj = cp.clip(j_r + dj, 0, cols - 2)
            base_r = (ci * (cols - 1) + cj)
            for dt in (0, 1):
                self._eval_tri(pts_r, base_r * 2 + dt, result_r, found_r, tol)

        result[remain_idx] = result_r
        return result


def _build_scipy_linear_nd_interpolator(
        target_points: NDArray[np.floating],
        source_values: NDArray[np.floating],
        *,
        grid_dims: tuple[int, int] | None = None,
) -> LinearNDInterpolator | _GridTopologyLinearInterpolator | None:
    """Build a CPU LinearNDInterpolator (or analytic grid interpolator).

    When ``grid_dims`` is provided the control points are a regular grid, so the
    triangulation is derived analytically (no Qhull). Otherwise SciPy's Qhull-based
    ``LinearNDInterpolator`` is used and ``None`` is returned if it cannot triangulate.
    """
    try:
        target_np = nornir_imageregistration.EnsureNumpyArray(target_points)
        source_np = nornir_imageregistration.EnsureNumpyArray(source_values)
        if grid_dims is not None:
            return _GridTopologyLinearInterpolator(target_np, source_np, grid_dims)
        valid = np.isfinite(target_np).all(axis=1) & np.isfinite(source_np).all(axis=1)
        if not np.any(valid):
            return None
        target_np = target_np[valid]
        source_np = source_np[valid]
        tri = scipy.spatial.Delaunay(target_np)
        return LinearNDInterpolator(tri, source_np)
    except (scipy.spatial.QhullError, ValueError) as exc:
        _logger.warning(
            'SciPy LinearNDInterpolator failed (%s); transform queries will return NaN',
            exc,
        )
        return None


def _build_linear_nd_interpolator(
        target_points: NDArray[np.floating],
        source_values: NDArray[np.floating],
        *,
        force_scipy: bool = False,
        grid_dims: tuple[int, int] | None = None,
) -> tuple[Any | None, bool]:
    """Build GPU LinearNDInterpolator when possible; fall back to an analytic grid or SciPy Qhull interpolator.

    ``grid_dims`` is passed through to the CPU fallback so the analytic triangulation is used if
    ``cuLinearNDInterpolator`` is unavailable or raises a degenerate-triangulation error.

    Note: ``grid_dims`` no longer short-circuits to CPU unconditionally.  The original gate was
    added to avoid Qhull collinearity failures, but those occur only when the *query domain*
    (``target_points``) forms a regular grid — which is true for the **forward** direction
    (SourcePoints as domain) but not for the **inverse** direction (TargetPoints as domain, which
    are deformed fixed-space control points and are not collinear).  Letting ``cuLinearNDInterpolator``
    run first keeps the inverse transform on-device; the existing degenerate-error fallback below
    handles any edge cases.
    """
    if force_scipy or cuLinearNDInterpolator is None:
        return _build_scipy_linear_nd_interpolator(target_points, source_values, grid_dims=grid_dims), True

    try:
        target_pts = cp.asarray(target_points, dtype=np.float64)
        source_pts = cp.asarray(source_values, dtype=np.float64)
        return cuLinearNDInterpolator(target_pts, source_pts), False
    except ValueError as exc:
        if not _is_cupy_degenerate_triangulation_error(exc):
            raise
        _logger.warning(
            'CuPy LinearNDInterpolator failed (%s); using CPU analytic grid interpolator',
            exc,
        )
        # #region agent log
        try:
            import json
            import time
            with open("/workspace/.cursor/debug-5c3155.log", "a", encoding="utf-8") as _fh:
                _fh.write(json.dumps({
                    "sessionId": "5c3155",
                    "timestamp": int(time.time() * 1000),
                    "location": "gridtransform.py:_build_linear_nd_interpolator",
                    "message": "degenerate culinear -> CPU analytic",
                    "data": {"grid_dims": grid_dims, "n_control": int(len(target_points))},
                    "hypothesisId": "H6",
                }, default=str) + "\n")
        except OSError:
            pass
        # #endregion
        # Degenerate cuLinearND often coincides with highly warped TargetPoints.  The GPU
        # analytic path (_CuGridTopologyInterpolator) uses bounding-box cell geometry and can
        # return mostly NaN/wrong inverses on those tiles (e.g. TEM section edge tiles),
        # while the CPU _GridTopologyLinearInterpolator matches the reference.
        return _build_scipy_linear_nd_interpolator(target_points, source_values, grid_dims=grid_dims), True


def _inverse_transform_with_linear_nd_fallback(
        transform: Any,
        points: NDArray[np.floating],
        *,
        scipy_flag_attr: str,
        interpolator_property: str,
        output_dtype: Any,
        target_points_attr: str = 'TargetPoints',
        source_points_attr: str = 'SourcePoints',
) -> NDArray[np.floating]:
    """Run inverse transform with optional SciPy fallback after CuPy triangulation failure."""
    points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
    private_attr = f'_{interpolator_property}'
    retried = False
    while True:
        interp = getattr(transform, interpolator_property)
        if interp is None:
            trans_points = cp.empty(points.shape, dtype=output_dtype)
            trans_points[:] = cp.nan
            return trans_points

        try:
            if getattr(transform, scipy_flag_attr):
                pn = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
                trans_points = interp(pn)
                if output_dtype is not None and output_dtype is not points.dtype:
                    trans_points = trans_points.astype(output_dtype, copy=False)
                return cp.asarray(trans_points)
            result = interp(points)
            if output_dtype is not None:
                return result.astype(output_dtype, copy=False)
            return result
        except ValueError as exc:
            if (not retried and not getattr(transform, scipy_flag_attr)
                    and _is_cupy_degenerate_triangulation_error(exc)):
                retried = True
                interp, uses_scipy = _build_linear_nd_interpolator(
                    getattr(transform, target_points_attr),
                    getattr(transform, source_points_attr),
                    force_scipy=True,
                    grid_dims=getattr(transform, 'grid_dims', None),
                )
                setattr(transform, private_attr, interp)
                setattr(transform, scipy_flag_attr, uses_scipy)
                continue
        except Exception:
            pass

        log = logging.getLogger(str(transform.__class__))
        log.warning("Could not transform points: " + str(points))
        setattr(transform, private_attr, None)
        trans_points = cp.empty(points.shape, dtype=output_dtype)
        trans_points[:] = cp.nan
        return trans_points


def _forward_transform_with_linear_nd_fallback(
        transform: Any,
        points: NDArray[np.floating],
        *,
        scipy_flag_attr: str,
        interpolator_property: str,
        output_dtype: Any,
        target_points_attr: str = 'SourcePoints',
        source_points_attr: str = 'TargetPoints',
) -> NDArray[np.floating]:
    """Run forward transform with optional SciPy fallback after CuPy triangulation failure."""
    points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
    private_attr = f'_{interpolator_property}'
    retried = False
    while True:
        interp = getattr(transform, interpolator_property)
        if interp is None:
            trans_points = cp.empty(points.shape, dtype=output_dtype)
            trans_points[:] = cp.nan
            return trans_points

        try:
            if getattr(transform, scipy_flag_attr):
                pn = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
                trans_points = interp(pn).astype(output_dtype, copy=False)
                return cp.asarray(trans_points)
            return interp(points).astype(output_dtype, copy=False)
        except ValueError as exc:
            if (not retried and not getattr(transform, scipy_flag_attr)
                    and _is_cupy_degenerate_triangulation_error(exc)):
                retried = True
                interp, uses_scipy = _build_linear_nd_interpolator(
                    getattr(transform, target_points_attr),
                    getattr(transform, source_points_attr),
                    force_scipy=True,
                    grid_dims=getattr(transform, 'grid_dims', None),
                )
                setattr(transform, private_attr, interp)
                setattr(transform, scipy_flag_attr, uses_scipy)
                continue
        except Exception:
            pass

        log = logging.getLogger(str(transform.__class__))
        log.warning("Could not transform points: " + str(points))
        setattr(transform, private_attr, None)
        trans_points = cp.empty(points.shape, dtype=output_dtype)
        trans_points[:] = cp.nan
        return trans_points


class GridTransform(ITransformScaling, ITransformRelativeScaling, ITransformTranslation,
                    ITransformTargetRotation, ITargetSpaceControlPointEdit,
                    IGridTransform, ITriangulatedTargetSpace, ControlPointBase):

    @property
    def type(self) -> TransformType:
        return TransformType.GRID

    @property
    def grid(self) -> ITKGridDivision:
        return self._grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        rows, cols = self._grid.grid_dims
        return int(rows), int(cols)

    def Load(self, TransformString: str, pixelSpacing=None):
        """
        Creates an instance of the transform from the TransformString
        """
        return nornir_imageregistration.transforms.factory.LoadTransform(TransformString, pixelSpacing)

    def __getstate__(self):
        odict = {'_points': self._points,
                 '_grid': self._grid}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)  # type: ignore[attr-defined]
        self.OnChangeEventListeners = []
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        self._WarpedKDTree = None
        self._fixedtri = None
        self.OnTransformChanged()

    def __init__(self,
                 grid: ITKGridDivision):

        self._grid = grid
        try:
            control_points = np.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid:\n{grid.TargetPoints}\n\n{grid.SourcePoints}')
            raise

        super(GridTransform, self).__init__(control_points)

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        self._WarpedKDTree = None
        self._fixedtri = None
        pass

    def ToITKString(self):
        numPoints = self.SourcePoints.shape[0]
        (bottom, left, top, right) = self.MappedBoundingBox.ToTuple()
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self.grid.grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self.grid.grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        if self.points.shape[0] != self.grid.grid_dims.prod():
            raise ValueError("Grid transform number of points does not match grid dimensions")

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded += 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    @property
    def WarpedKDTree(self):
        if self._WarpedKDTree is None:
            self._WarpedKDTree = build_nearest_neighbor_index(self.SourcePoints)

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = build_nearest_neighbor_index(self.TargetPoints)

        return self._FixedKDTree

    @property
    def fixedtri(self):
        if self._fixedtri is None:
            # The source points are a regular grid, so triangulating them never hits
            # Qhull's degenerate/coplanar failure that deformed TargetPoints can. The
            # triangulation topology is shared between source and target spaces, so we
            # swap in the target geometry to obtain the fixed-space triangulation.
            source_np = nornir_imageregistration.EnsureNumpyArray(self.SourcePoints)
            tri = scipy.spatial.Delaunay(source_np, incremental=False)
            tri.points = nornir_imageregistration.EnsureNumpyArray(self.TargetPoints)
            self._fixedtri = tri

        return self._fixedtri

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.fixedtri

    def NearestFixedPoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return self.FixedKDTree.query(points)

    def NearestTargetPoint(self, points: NDArray[np.floating]):
        """Return the target points nearest to the query points
        :return: Distance, Index
        """
        return self.FixedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return self.WarpedKDTree.query(points)

    def NearestSourcePoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return self.WarpedKDTree.query(points)

    def Scale(self, scalar):
        """Scale both warped and control space by scalar"""
        self._points *= scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar):
        """Scale source space control points by scalar"""
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar):
        """Scale target space control points by scalar"""
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[np.floating]):
        """Translate all fixed points by the specified amount"""

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        """Translate all warped points by the specified amount"""
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def GetPointPairsInRect(self, points: NDArray[np.floating],
                            bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        OutputPoints = None

        bounds = nornir_imageregistration.Rectangle.PrimitiveToRectangle(bounds).ToArray()

        included_rows = []
        for iPoint in range(0, points.shape[0]):
            y, x = points[iPoint, :]
            if nornir_imageregistration.Rectangle.contains(bounds, (y, x)):
                included_rows.append(iPoint)

        OutputPoints = self._points[included_rows, :]

        if OutputPoints is not None:
            if OutputPoints.ndim == 1:
                OutputPoints = np.reshape(OutputPoints, (1, OutputPoints.shape[0]))

        return OutputPoints

    def GetFixedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        """Return the warped points from a set of target-source point pairs"""
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        """Return the target points from a set of target-source point pairs"""
        return points[:, 0:2]

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = RegularGridInterpolator(self._grid.axis_points,
                                                                np.reshape(self.TargetPoints, (
                                                                    self._grid.grid_dims[0], self._grid.grid_dims[1],
                                                                    2)),
                                                                bounds_error=False)

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            # Source points form a regular grid; derive the triangulation analytically
            # (no Qhull) instead of LinearNDInterpolator(self.fixedtri, ...).
            self._InverseInterpolator = _GridTopologyLinearInterpolator(
                self.TargetPoints, self.SourcePoints, self.grid_dims)

        return self._InverseInterpolator

    def Transform(self, points, **kwargs):
        """Map points from the warped space to fixed space"""
        transPoints = None

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        transPoints = self.ForwardInterpolator(points)
        return transPoints

    def InverseTransform(self, points, **kwargs):
        """Map points from the fixed space to the warped space"""
        transPoints = None

        method = kwargs.get('method', 'linear')

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.InverseInterpolator(points)
        except Exception as e:  # This is usually a scipy.spatial._qhull.QhullError:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None
            self._InverseInterpolator = None

            # This was added for the case where all points in the triangulation are colinear.
            transPoints = np.empty(points.shape)
            transPoints[:] = np.nan

        return transPoints

    @property
    def FixedTriangles(self):
        return self.fixedtri.simplices

    def GetFixedCentroids(self, triangles=None):
        """Centroids of fixed triangles"""
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[np.floating] | None):
        """Rotate all warped points about a center by a given angle"""
        self._points[:, 0:2] = ControlPointBase.RotatePoints(self.TargetPoints, rangle, rotationCenter)
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], point: NDArray[np.floating]) -> int | NDArray[
        np.integer]:
        self._points[index, 0:2] = point
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], points: NDArray[np.floating]) -> int | \
                                                                                                              NDArray[
                                                                                                                  np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        distance, index = self.NearestFixedPoint(old_points)
        return self.UpdateTargetPointsByIndex(cast(int | NDArray[np.integer], index), points)

    def OnFixedPointChanged(self):
        super(GridTransform, self).OnFixedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._fixedtri = None
        self._FixedKDTree = None
        super(GridTransform, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        raise NotImplementedError("Grid transforms have a fixed grid of points, they should not change")

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""
        super(GridTransform, self).ClearDataStructures()
        self._fixedtri = None
        self._FixedKDTree = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None


class GridTransform_GPUComponent(ITransformScaling, ITransformRelativeScaling, ITransformTranslation,
                                 ITransformTargetRotation, ITargetSpaceControlPointEdit,
                                 IGridTransform, ITriangulatedTargetSpace, ControlPointBase):

    @property
    def type(self) -> TransformType:
        return TransformType.GRID

    @property
    def grid(self) -> ITKGridDivision:
        return self._grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        rows, cols = self._grid.grid_dims
        return int(rows), int(cols)

    def Load(self, TransformString: str, pixelSpacing=None):
        """
        Creates an instance of the transform from the TransformString
        """
        return nornir_imageregistration.transforms.factory.LoadTransform(TransformString, pixelSpacing)

    def __getstate__(self):
        odict = {'_points': self._points,
                 '_grid': self._grid}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)  # type: ignore[attr-defined]
        self.OnChangeEventListeners = []
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        self._WarpedKDTree = None
        self._fixedtri = None
        self.OnTransformChanged()

    def __init__(self,
                 grid: ITKGridDivision):

        self._grid = grid
        try:
            control_points = np.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid:\n{grid.TargetPoints}\n\n{grid.SourcePoints}')
            raise

        super(GridTransform_GPUComponent, self).__init__(control_points)

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        self._WarpedKDTree = None
        self._fixedtri = None
        self._scipy_forward_grid = False
        self._scipy_inverse_interp = False
        pass

    def ToITKString(self):
        numPoints = self.SourcePoints.shape[0]
        (bottom, left, top, right) = self.MappedBoundingBox.ToTuple()
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self.grid.grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self.grid.grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded += 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    @property
    def WarpedKDTree(self):
        if self._WarpedKDTree is None:
            self._WarpedKDTree = build_nearest_neighbor_index(self.SourcePoints)

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = build_nearest_neighbor_index(self.TargetPoints)

        return self._FixedKDTree

    @property
    def fixedtri(self):
        if self._fixedtri is None:
            # The source points are a regular grid, so triangulating them never hits
            # Qhull's degenerate/coplanar failure that deformed TargetPoints can. The
            # triangulation topology is shared between source and target spaces, so we
            # swap in the target geometry to obtain the fixed-space triangulation.
            source_np = nornir_imageregistration.EnsureNumpyArray(self.SourcePoints)
            tri = scipy.spatial.Delaunay(source_np, incremental=False)
            tri.points = nornir_imageregistration.EnsureNumpyArray(self.TargetPoints)
            self._fixedtri = tri

        return self._fixedtri

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.fixedtri

    def NearestFixedPoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return self.FixedKDTree.query(points)

    def NearestTargetPoint(self, points: NDArray[np.floating]):
        """Return the target points nearest to the query points
        :return: Distance, Index
        """
        return self.FixedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return self.WarpedKDTree.query(points)

    def NearestSourcePoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return self.WarpedKDTree.query(points)

    def Scale(self, scalar):
        """Scale both warped and control space by scalar"""
        self._points *= scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar):
        """Scale source space control points by scalar"""
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar):
        """Scale target space control points by scalar"""
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[np.floating]):
        """Translate all fixed points by the specified amount"""

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        """Translate all warped points by the specified amount"""
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def GetPointPairsInRect(self, points: NDArray[np.floating],
                            bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        OutputPoints = None

        bounds = nornir_imageregistration.Rectangle.PrimitiveToRectangle(bounds).ToArray()

        for iPoint in range(0, points.shape[0]):
            y, x = points[iPoint, :]
            if nornir_imageregistration.Rectangle.contains(bounds, (y, x)):
                PointPair = self._points[iPoint, :]
                if OutputPoints is None:
                    OutputPoints = PointPair
                else:
                    OutputPoints = cp.vstack((OutputPoints, PointPair))

        if OutputPoints is not None:
            if OutputPoints.ndim == 1:
                OutputPoints = cp.reshape(OutputPoints, (1, OutputPoints.shape[0]))

        return OutputPoints

    def GetFixedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        """Return the warped points from a set of target-source point pairs"""
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        """Return the target points from a set of target-source point pairs"""
        return points[:, 0:2]

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            if cuRegularGridInterpolator is not None:
                # axis_points is a list of 1d axis samples (different lengths); never cp.array() the whole list.
                axes = tuple(cp.asarray(x, dtype=np.float64) for x in self._grid.axis_points)
                vals = cp.reshape(
                    cp.asarray(self.TargetPoints, dtype=np.float64),
                    (int(self._grid.grid_dims[0]), int(self._grid.grid_dims[1]), 2),
                )
                self._ForwardInterpolator = cuRegularGridInterpolator(
                    axes,
                    vals,
                    bounds_error=False,
                )
                self._scipy_forward_grid = False
            else:
                axes = tuple(np.asarray(x) for x in self._grid.axis_points)
                vals = np.reshape(
                    nornir_imageregistration.EnsureNumpyArray(self.TargetPoints),
                    (int(self._grid.grid_dims[0]), int(self._grid.grid_dims[1]), 2),
                )
                self._ForwardInterpolator = RegularGridInterpolator(axes, vals, bounds_error=False)
                self._scipy_forward_grid = True

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            interp, uses_scipy = _build_linear_nd_interpolator(
                self.TargetPoints,
                self.SourcePoints,
                grid_dims=self.grid_dims,
            )
            self._InverseInterpolator = interp
            self._scipy_inverse_interp = uses_scipy

        return self._InverseInterpolator

    def Transform(self, points, **kwargs):
        """Map points from the warped space to fixed space"""
        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        interp = self.ForwardInterpolator
        if self._scipy_forward_grid:
            pn = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
            return cp.asarray(interp(pn))
        return interp(points)

    def InverseTransform(self, points, **kwargs):
        """Map points from the fixed space to the warped space"""
        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        return _inverse_transform_with_linear_nd_fallback(
            self,
            points,
            scipy_flag_attr='_scipy_inverse_interp',
            interpolator_property='InverseInterpolator',
            output_dtype=cp.float32,
        )

    @property
    def FixedTriangles(self):
        return self.fixedtri.simplices

    def GetFixedCentroids(self, triangles=None):
        """Centroids of fixed triangles"""
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[np.floating] | None):
        """Rotate all warped points about a center by a given angle"""
        self._points[:, 0:2] = ControlPointBase.RotatePoints(self.TargetPoints, rangle, rotationCenter)
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], point: NDArray[np.floating]) -> int | NDArray[
        np.integer]:
        self._points[index, 0:2] = point
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], points: NDArray[np.floating]) -> int | \
                                                                                                              NDArray[
                                                                                                                  np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        distance, index = self.NearestFixedPoint(old_points)
        return self.UpdateTargetPointsByIndex(cast(int | NDArray[np.integer], index), points)

    def OnFixedPointChanged(self):
        super(GridTransform_GPUComponent, self).OnFixedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._fixedtri = None
        self._FixedKDTree = None
        super(GridTransform_GPUComponent, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        raise NotImplementedError("Grid transforms have a fixed grid of points, they should not change")

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""
        super(GridTransform_GPUComponent, self).ClearDataStructures()
        self._fixedtri = None
        self._FixedKDTree = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
