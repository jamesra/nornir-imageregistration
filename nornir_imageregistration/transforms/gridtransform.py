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
cuRBFInterpolator: Any | None = None
cuLinearNDInterpolator: Any | None = None
try:
    from cupyx.scipy.interpolate import RegularGridInterpolator as cuRegularGridInterpolator
    from cupyx.scipy.interpolate import RBFInterpolator as cuRBFInterpolator
    from cupyx.scipy.interpolate import LinearNDInterpolator as cuLinearNDInterpolator
except ImportError:
    pass

from numpy.typing import NDArray
import scipy
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

try:
    from scipy.interpolate import RBFInterpolator as SciPyRBFInterpolator
except ImportError:
    SciPyRBFInterpolator = None
import scipy.spatial

import nornir_imageregistration
from nornir_imageregistration.env_flags import env_flag, is_basic_falsey, is_basic_truthy
from nornir_imageregistration.nearest_neighbor import build_nearest_neighbor_index
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms import float_to_shortest_string
from nornir_imageregistration.transforms.controlpointbase import ControlPointBase, ControlPointBase_GPUComponent
from .base import ITransformScaling, ITransformRelativeScaling, \
    ITransformTranslation, \
    TransformType, ITransformTargetRotation, ITargetSpaceControlPointEdit, IGridTransform, \
    ITriangulatedTargetSpace

_logger = logging.getLogger(__name__)


def itk_control_point_strings(points: NDArray) -> list[str]:
    """``" X Y"`` for each control point's target position, as ``ToITKString`` writes them.

    One host copy, then Python floats: iterating a CuPy array row by row read every
    element from the device, and MoveMosaicsToZeroOrigin spent ~4-5 s per RC3 section
    in it (srv, 2026-11-02). Python floats format exactly as the NumPy/CuPy scalars did.
    """
    target_yx = nornir_imageregistration.EnsureNumpyArray(points)[:, :2].tolist()
    return [f" {float_to_shortest_string(x, 3)} {float_to_shortest_string(y, 3)}" for y, x in target_yx]


def _assemble_inverse_use_scipy() -> bool:
    """When True on CuPy, build grid inverse interpolators with SciPy Qhull instead of CuPy Delaunay.

    Production default (unset ``NORNIR_ASSEMBLE_INVERSE_SCIPY``): **CuPy** ``cuLinearNDInterpolator``
    (faster). Set ``NORNIR_ASSEMBLE_INVERSE_SCIPY=1`` to restore SciPy Qhull on the host.

    **Pixel effect (CuPy default vs SciPy):** inverse maps each output pixel to slightly different
    subpixel source coordinates. Image warps still use cubic ``map_coordinates``; differences
    quantize to **±1 DN** on 8-bit tiles (RPC3 601: ~84% of tiles have >=1 changed pixel;
    within affected tiles up to ~22% of pixels, all |delta|=1). Degenerate meshes still fall back to SciPy.
    """
    raw = env_flag('NORNIR_ASSEMBLE_INVERSE_SCIPY')
    if is_basic_falsey(raw):
        return False
    if is_basic_truthy(raw):
        return True
    # When inverse-scipy is unset, stay on CuPy even if grid extrapolation is on.
    if is_basic_truthy(env_flag('NORNIR_ASSEMBLE_GRID_EXTRAPOLATE')):
        return False
    return False


def _is_cupy_degenerate_triangulation_error(exc: BaseException) -> bool:
    """Return True when cupyx Delaunay rejected control points as degenerate or coplanar."""
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).lower()
    return 'degenerate' in message or 'coplanar' in message


def _orient2d(points: NDArray[np.floating], simplices: NDArray[np.integer]) -> NDArray[np.floating]:
    """Signed parallelogram area for each triangle (positive = CCW)."""
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _incircle2d(
        a: NDArray[np.floating],
        b: NDArray[np.floating],
        c: NDArray[np.floating],
        d: NDArray[np.floating],
) -> NDArray[np.floating]:
    """InCircle determinant; positive when *d* is inside circumcircle of CCW triangle abc."""
    ad = a - d
    bd = b - d
    cd = c - d
    return (
            (ad[:, 0] * ad[:, 0] + ad[:, 1] * ad[:, 1]) * (bd[:, 0] * cd[:, 1] - bd[:, 1] * cd[:, 0])
            - (bd[:, 0] * bd[:, 0] + bd[:, 1] * bd[:, 1]) * (ad[:, 0] * cd[:, 1] - ad[:, 1] * cd[:, 0])
            + (cd[:, 0] * cd[:, 0] + cd[:, 1] * cd[:, 1]) * (ad[:, 0] * bd[:, 1] - ad[:, 1] * bd[:, 0])
    )


def delaunay_topology_still_valid(
        points: NDArray[np.floating],
        simplices: NDArray[np.integer],
        neighbors: NDArray[np.integer],
        *,
        orient_eps: float = 1e-14,
        incircle_eps: float = 1e-12,
) -> bool:
    """Return True when *simplices* remain a non-inverted Delaunay triangulation of *points*.

    Cost is O(number of triangles): orientation of every simplex plus the Lawson
    InCircle test across every internal edge. That is always cheaper than rebuilding
    Delaunay (O(n log n) typical, much worse on near-degenerate TargetPoints), so
    callers should run this whenever a cached triangulation exists — there is no
    useful "too many points moved, skip the check" cutoff.
    """
    pts = np.asarray(points, dtype=np.float64)
    sim = np.asarray(simplices, dtype=np.int64)
    neigh = np.asarray(neighbors, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 2 or sim.ndim != 2 or sim.shape[1] != 3:
        return False
    if not np.all(np.isfinite(pts)):
        return False

    orient = _orient2d(pts, sim)
    if np.any(np.abs(orient) <= orient_eps):
        return False

    # Normalize to CCW so InCircle sign is consistent.
    sim_ccw = sim.copy()
    cw = orient < 0
    if np.any(cw):
        sim_ccw[cw, 1], sim_ccw[cw, 2] = sim[cw, 2], sim[cw, 1]

    # neighbors[t, i] is the triangle across the edge opposite vertex simplices[t, i].
    for apex_slot in range(3):
        opp = neigh[:, apex_slot]
        internal = opp >= 0
        if not np.any(internal):
            continue
        tri_idx = np.flatnonzero(internal)
        nbr_idx = opp[internal].astype(np.int64, copy=False)
        a = pts[sim_ccw[tri_idx, (apex_slot + 1) % 3]]
        b = pts[sim_ccw[tri_idx, (apex_slot + 2) % 3]]
        c = pts[sim_ccw[tri_idx, apex_slot]]
        # Foreign vertex = the neighbor simplex vertex not shared with this triangle.
        nbr_verts = sim[nbr_idx]
        tri_verts = sim[tri_idx]
        foreign = np.full(tri_idx.shape[0], -1, dtype=np.int64)
        for j in range(3):
            cand = nbr_verts[:, j]
            shared = (
                    (cand == tri_verts[:, 0])
                    | (cand == tri_verts[:, 1])
                    | (cand == tri_verts[:, 2])
            )
            foreign = np.where((foreign < 0) & ~shared, cand, foreign)
        if np.any(foreign < 0):
            return False
        d = pts[foreign]
        if np.any(_incircle2d(a, b, c, d) > incircle_eps):
            return False
    return True


class _FixedSimplicesLinearND:
    """Vector LinearND using predetermined triangles (no Qhull / cupyx Delaunay).

    SciPy's ``Delaunay`` cannot safely accept new vertex coordinates without
    rebuilding ``equations`` / ``find_simplex`` caches, so topology-preserving
    updates bind values through matplotlib's ``Triangulation`` instead.
    """

    tri: Any | None
    _ys: Any
    _xs: Any

    def __init__(
            self,
            points: NDArray[np.floating],
            values: NDArray[np.floating],
            simplices: NDArray[np.integer],
    ) -> None:
        from matplotlib.tri import LinearTriInterpolator, Triangulation

        pts = np.asarray(points, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64)
        sim = np.asarray(simplices, dtype=np.int32)
        # Control points are (Y, X); matplotlib triangulation is (x, y).
        triang = Triangulation(x=pts[:, 1], y=pts[:, 0], triangles=sim)
        self._ys = LinearTriInterpolator(triang, vals[:, 0])
        self._xs = LinearTriInterpolator(triang, vals[:, 1])
        self.tri = None

    def __call__(self, query: NDArray[np.floating]) -> NDArray[np.floating]:
        q = np.asarray(query, dtype=np.float64)
        single = q.ndim == 1
        if single:
            q = q.reshape(1, 2)
        y_out = np.ma.filled(self._ys(q[:, 1], q[:, 0]), np.nan).astype(np.float64, copy=False)
        x_out = np.ma.filled(self._xs(q[:, 1], q[:, 0]), np.nan).astype(np.float64, copy=False)
        out = np.column_stack([np.asarray(y_out, dtype=np.float64),
                               np.asarray(x_out, dtype=np.float64)])
        return out[0] if single else out


def _linear_nd_from_fixed_simplices(
        points: NDArray[np.floating],
        values: NDArray[np.floating],
        simplices: NDArray[np.integer],
) -> _FixedSimplicesLinearND:
    """Build an inverse interpolator that reuses *simplices* with new vertex coords."""
    return _FixedSimplicesLinearND(points, values, simplices)


def _rebuild_fixedtri_scipy(transform: Any) -> None:
    """Replace ``_fixedtri`` with a fresh SciPy Delaunay of TargetPoints."""
    try:
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        if target_np.shape[0] < 3 or not np.all(np.isfinite(target_np)):
            transform._fixedtri = None
            return
        transform._fixedtri = scipy.spatial.Delaunay(target_np, incremental=False)
    except Exception:
        _logger.debug('SciPy Delaunay rebuild after target edit failed', exc_info=True)
        transform._fixedtri = None


def _fixedtri_kept_across_clear(transform: Any) -> Any | None:
    """Return the cached Delaunay when its simplices remain valid for TargetPoints.

    ``OnTransformChanged`` calls ``ClearDataStructures``, which nulls ``_fixedtri``.
    Run the Lawson check *before* that clear, then assign the returned object back
    onto the transform afterward. Interpolation uses live TargetPoints with
    ``_linear_nd_from_fixed_simplices``, not ``tri.points``.
    """
    tri = getattr(transform, '_fixedtri', None)
    if tri is None:
        return None
    try:
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        if not delaunay_topology_still_valid(target_np, tri.simplices, tri.neighbors):
            return None
        return tri
    except Exception:
        _logger.debug('Failed to preserve fixed triangulation; will rebuild', exc_info=True)
        return None


def _cache_fixedtri_from_inverse(transform: Any, interp: Any, uses_scipy: bool) -> None:
    """Store a SciPy Delaunay on *transform* for later topology-preserving updates."""
    if getattr(transform, '_fixedtri', None) is not None or interp is None:
        return
    try:
        if uses_scipy and getattr(interp, 'tri', None) is not None:
            transform._fixedtri = interp.tri
            return
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        transform._fixedtri = scipy.spatial.Delaunay(target_np, incremental=False)
    except Exception:
        _logger.debug('Could not cache fixed triangulation after inverse build', exc_info=True)


def _build_scipy_linear_nd_interpolator(
        target_points: NDArray[np.floating],
        source_values: NDArray[np.floating],
) -> LinearNDInterpolator | None:
    """Build SciPy LinearNDInterpolator; return None when Qhull cannot triangulate."""
    try:
        target_np = nornir_imageregistration.EnsureNumpyArray(target_points)
        source_np = nornir_imageregistration.EnsureNumpyArray(source_values)
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


_scipy_interpolators_only: bool = False


def use_scipy_interpolators_in_this_process() -> None:
    """Build every LinearNDInterpolator of this process with SciPy (host), never CuPy.

    Process-pool workers run NumPy by design (the parent uploads their results). Without this each worker that can
    import CuPy opens its own CUDA context and holds a device triangulation, so GPU memory grows with the worker
    count until the card is full (RC3 Mosaic grid prewarp, RTX 4500 Ada 24 GB).
    """
    global _scipy_interpolators_only
    _scipy_interpolators_only = True


def _build_linear_nd_interpolator(
        target_points: NDArray[np.floating],
        source_values: NDArray[np.floating],
        *,
        force_scipy: bool = False,
) -> tuple[Any | None, bool]:
    """Build GPU LinearNDInterpolator when possible; fall back to SciPy Qhull on degenerate CuPy triangulation."""
    if force_scipy or _scipy_interpolators_only or cuLinearNDInterpolator is None:
        return _build_scipy_linear_nd_interpolator(target_points, source_values), True

    try:
        target_pts = cp.asarray(target_points, dtype=np.float64)
        source_pts = cp.asarray(source_values, dtype=np.float64)
        return cuLinearNDInterpolator(target_pts, source_pts), False
    except ValueError as exc:
        if not _is_cupy_degenerate_triangulation_error(exc):
            raise
        # Regular lattices are coplanar to cupyx Delaunay; SciPy Qhull is the
        # expected path. DEBUG so nornir-build -debug can still trace it.
        _logger.debug(
            'CuPy LinearNDInterpolator failed (%s); falling back to SciPy Qhull',
            exc,
        )
        return _build_scipy_linear_nd_interpolator(target_points, source_values), True


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
    private_attr = f'_{interpolator_property}'
    retried = False
    while True:
        interp = getattr(transform, interpolator_property)
        uses_scipy = bool(getattr(transform, scipy_flag_attr, False))
        if uses_scipy:
            points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        else:
            points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        if interp is None:
            if uses_scipy:
                trans_points = np.empty(points.shape, dtype=output_dtype or np.float64)
                trans_points[:] = np.nan
                return trans_points
            trans_points = cp.empty(points.shape, dtype=output_dtype)
            trans_points[:] = cp.nan
            return trans_points

        try:
            if uses_scipy:
                trans_points = interp(points)
                if output_dtype is not None and output_dtype is not points.dtype:
                    trans_points = trans_points.astype(output_dtype, copy=False)
                return trans_points
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
                )
                setattr(transform, private_attr, interp)
                setattr(transform, scipy_flag_attr, uses_scipy)
                continue
            # Non-retryable ValueError (e.g. Qhull): soft-fail below.

        log = logging.getLogger(str(transform.__class__))
        log.warning("Could not transform points: " + str(points))
        setattr(transform, private_attr, None)
        if uses_scipy:
            trans_points = np.empty(points.shape, dtype=output_dtype or np.float64)
            trans_points[:] = np.nan
            return trans_points
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
                )
                setattr(transform, private_attr, interp)
                setattr(transform, scipy_flag_attr, uses_scipy)
                continue
            # Non-retryable ValueError (e.g. Qhull): soft-fail below.

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
            control_points = grid.pack_control_point_pairs(on_device=False)
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
        output.extend(itk_control_point_strings(self.points))

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
            self._fixedtri = scipy.spatial.Delaunay(
                nornir_imageregistration.EnsureNumpyArray(self.TargetPoints), incremental=False)

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
            self._InverseInterpolator = LinearNDInterpolator(self.fixedtri, self.SourcePoints)

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

    def UpdateTargetPointsByIndex(
            self,
            index: int | NDArray[np.integer],
            point: NDArray[np.floating],
            *,
            remove_duplicates: bool = True,
    ) -> int | NDArray[np.integer]:
        """Move target-space control points at *index*.

        *remove_duplicates* is accepted for interface compatibility and is ignored;
        a grid never unique-collapses (source lattice identity).
        """
        xp = cp.get_array_module(self._points)
        point_xp = xp.asarray(point)
        if not isinstance(index, (int, np.integer)):
            index = xp.asarray(index)
        self._points[index, 0:2] = point_xp
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(
            self,
            old_points: NDArray[np.floating],
            points: NDArray[np.floating],
            *,
            remove_duplicates: bool = True,
    ) -> int | NDArray[np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        distance, index = self.NearestFixedPoint(old_points)
        return self.UpdateTargetPointsByIndex(
            cast(int | NDArray[np.integer], index), points, remove_duplicates=remove_duplicates)

    def OnFixedPointChanged(self):
        super(GridTransform, self).OnFixedPointChanged()
        # Lawson check before OnTransformChanged → ClearDataStructures nulls _fixedtri.
        kept_tri = _fixedtri_kept_across_clear(self)
        # Interactive-edit OnTransformChanged skips ClearDataStructures; still drop
        # interpolators so the next query rebinds to relocated vertices / new values.
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        try:
            from nornir_imageregistration import interactive_edit
            if interactive_edit.in_progress():
                super(ControlPointBase, self).OnTransformChanged()
            else:
                super(GridTransform, self).OnTransformChanged()
        except ImportError:
            super(GridTransform, self).OnTransformChanged()
        if kept_tri is not None:
            self._fixedtri = kept_tri
        else:
            _rebuild_fixedtri_scipy(self)

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
                                 IGridTransform, ITriangulatedTargetSpace, ControlPointBase_GPUComponent):

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
            control_points = grid.pack_control_point_pairs(on_device=True)
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
        output.extend(itk_control_point_strings(self.points))

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
            self._fixedtri = scipy.spatial.Delaunay(
                nornir_imageregistration.EnsureNumpyArray(self.TargetPoints), incremental=False)

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

        self._points[:, 0:2] = self._points[:, 0:2] + cp.asarray(offset)
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        """Translate all warped points by the specified amount"""
        self._points[:, 2:4] = self._points[:, 2:4] + cp.asarray(offset)
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
            # Topology-preserving path: reuse cached simplices with live TargetPoints
            # (no Qhull / cupyx Delaunay). Used after Spacebar / point edits when the
            # Lawson certificate kept ``_fixedtri``.
            if self._fixedtri is not None:
                self._InverseInterpolator = _linear_nd_from_fixed_simplices(
                    nornir_imageregistration.EnsureNumpyArray(self.TargetPoints),
                    nornir_imageregistration.EnsureNumpyArray(self.SourcePoints),
                    self._fixedtri.simplices,
                )
                self._scipy_inverse_interp = True
                return self._InverseInterpolator

            force_scipy = (
                    nornir_imageregistration.GetActiveComputationLib()
                    == nornir_imageregistration.ComputationLib.cupy
                    and _assemble_inverse_use_scipy())
            interp, uses_scipy = _build_linear_nd_interpolator(
                self.TargetPoints,
                self.SourcePoints,
                force_scipy=force_scipy,
            )
            self._InverseInterpolator = interp
            self._scipy_inverse_interp = uses_scipy
            _cache_fixedtri_from_inverse(self, interp, uses_scipy)

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
        if getattr(self, '_scipy_inverse_interp', False):
            points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        elif self._InverseInterpolator is None:
            _ = self.InverseInterpolator
            if getattr(self, '_scipy_inverse_interp', False):
                points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
            else:
                points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        else:
            points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        return _inverse_transform_with_linear_nd_fallback(
            self,
            points,
            scipy_flag_attr='_scipy_inverse_interp',
            interpolator_property='InverseInterpolator',
            output_dtype=points.dtype,
        )

    @property
    def FixedTriangles(self):
        return self.fixedtri.simplices

    def GetFixedCentroids(self, triangles=None):
        """Centroids of fixed triangles"""
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        xp = cp.get_array_module(fixedTriangleVerticies)
        swappedTriangleVerticies = xp.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = xp.mean(swappedTriangleVerticies, 1)
        return xp.swapaxes(Centroids, 0, 1)

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[np.floating] | None):
        """Rotate all warped points about a center by a given angle"""
        xp = cp.get_array_module(self.TargetPoints)
        center = rotationCenter if rotationCenter is not None else xp.mean(self.TargetPoints, axis=0)
        self._points[:, 0:2] = ControlPointBase_GPUComponent.RotatePoints(self.TargetPoints, rangle, center)
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(
            self,
            index: int | NDArray[np.integer],
            point: NDArray[np.floating],
            *,
            remove_duplicates: bool = True,
    ) -> int | NDArray[np.integer]:
        """Move target-space control points at *index*.

        *remove_duplicates* is accepted for interface compatibility and is ignored;
        a grid never unique-collapses (source lattice identity).
        """
        xp = cp.get_array_module(self._points)
        point_xp = xp.asarray(point)
        if not isinstance(index, (int, np.integer)):
            index = xp.asarray(index)
        self._points[index, 0:2] = point_xp
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(
            self,
            old_points: NDArray[np.floating],
            points: NDArray[np.floating],
            *,
            remove_duplicates: bool = True,
    ) -> int | NDArray[np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        distance, index = self.NearestFixedPoint(old_points)
        return self.UpdateTargetPointsByIndex(
            cast(int | NDArray[np.integer], index), points, remove_duplicates=remove_duplicates)

    def OnFixedPointChanged(self):
        super(GridTransform_GPUComponent, self).OnFixedPointChanged()
        # Lawson check before OnTransformChanged → ClearDataStructures nulls _fixedtri.
        kept_tri = _fixedtri_kept_across_clear(self)
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        try:
            from nornir_imageregistration import interactive_edit
            if interactive_edit.in_progress():
                super(ControlPointBase_GPUComponent, self).OnTransformChanged()
            else:
                super(GridTransform_GPUComponent, self).OnTransformChanged()
        except ImportError:
            super(GridTransform_GPUComponent, self).OnTransformChanged()
        if kept_tri is not None:
            self._fixedtri = kept_tri
        else:
            _rebuild_fixedtri_scipy(self)

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


class GridTransform_GPU(ITransformScaling, ITransformRelativeScaling, ITransformTranslation,
                        ITransformTargetRotation, ITargetSpaceControlPointEdit,
                        IGridTransform, ControlPointBase_GPUComponent):

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
            control_points = grid.pack_control_point_pairs(on_device=True)
        except:
            print(f'Invalid grid:\n{grid.TargetPoints}\n\n{grid.SourcePoints}')
            raise

        super(GridTransform_GPU, self).__init__(control_points)

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._scipy_forward_grid = False
        self._scipy_inverse_interp = False
        pass

    def ToITKString(self):
        numPoints = self.SourcePoints.shape[0]
        bottom, left, top, right = cast(tuple[float, float, float, float], self.MappedBoundingBox.ToTuple())
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self.grid.grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self.grid.grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        output.extend(itk_control_point_strings(self.points))

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    def NearestFixedPoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return None

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        """Return the fixed points nearest to the query points
        :return: Distance, Index
        """
        return None

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

        self._points[:, 0:2] = self._points[:, 0:2] + cp.asarray(offset)
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        """Translate all warped points by the specified amount"""
        self._points[:, 2:4] = self._points[:, 2:4] + cp.asarray(offset)
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
            if cuRBFInterpolator is not None:
                self._InverseInterpolator = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)
                self._scipy_inverse_interp = False
            else:
                tgt = nornir_imageregistration.EnsureNumpyArray(self.TargetPoints)
                src = nornir_imageregistration.EnsureNumpyArray(self.SourcePoints)
                if SciPyRBFInterpolator is not None:
                    try:
                        self._InverseInterpolator = SciPyRBFInterpolator(tgt, src)
                    except Exception:
                        tri = scipy.spatial.Delaunay(tgt)
                        self._InverseInterpolator = LinearNDInterpolator(tri, src)
                else:
                    tri = scipy.spatial.Delaunay(tgt)
                    self._InverseInterpolator = LinearNDInterpolator(tri, src)
                self._scipy_inverse_interp = True

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
        interp = self.InverseInterpolator
        if self._scipy_inverse_interp:
            pn = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
            return cp.asarray(interp(pn))
        return interp(points)

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[np.floating] | None):
        """Rotate all warped points about a center by a given angle"""
        xp = cp.get_array_module(self.TargetPoints)
        center = rotationCenter if rotationCenter is not None else xp.mean(self.TargetPoints, axis=0)
        self._points[:, 0:2] = ControlPointBase_GPUComponent.RotatePoints(self.TargetPoints, rangle, center)
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(
            self,
            index: int | NDArray[np.integer],
            point: NDArray[np.floating],
            *,
            remove_duplicates: bool = True,
    ) -> int | NDArray[np.integer]:
        """Move target-space control points at *index*.

        *remove_duplicates* is accepted for interface compatibility and is ignored;
        a grid never unique-collapses (source lattice identity).
        """
        xp = cp.get_array_module(self._points)
        point_xp = xp.asarray(point)
        if not isinstance(index, (int, np.integer)):
            index = xp.asarray(index)
        self._points[index, 0:2] = point_xp
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(
            self,
            old_points: NDArray[np.floating],
            points: NDArray[np.floating],
            *,
            remove_duplicates: bool = True,
    ) -> int | NDArray[np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        query_result = cast(
            tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
            self.NearestFixedPoint(old_points),
        )
        distance, index = query_result
        return self.UpdateTargetPointsByIndex(
            cast(int | NDArray[np.integer], index), points, remove_duplicates=remove_duplicates)

    def OnFixedPointChanged(self):
        super(GridTransform_GPU, self).OnFixedPointChanged()
        try:
            from nornir_imageregistration import interactive_edit
            if interactive_edit.in_progress():
                self._ForwardInterpolator = None
                self._InverseInterpolator = None
                super(ControlPointBase_GPUComponent, self).OnTransformChanged()
                return
        except ImportError:
            pass
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        super(GridTransform_GPU, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        raise NotImplementedError("Grid transforms have a fixed grid of points, they should not change")

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""
        super(GridTransform_GPU, self).ClearDataStructures()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
