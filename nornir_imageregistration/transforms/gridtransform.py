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


def _build_linear_nd_interpolator(
        target_points: NDArray[np.floating],
        source_values: NDArray[np.floating],
        *,
        force_scipy: bool = False,
) -> tuple[Any | None, bool]:
    """Build GPU LinearNDInterpolator when possible; fall back to SciPy Qhull on degenerate CuPy triangulation."""
    if force_scipy or cuLinearNDInterpolator is None:
        return _build_scipy_linear_nd_interpolator(target_points, source_values), True

    try:
        target_pts = cp.asarray(target_points, dtype=np.float64)
        source_pts = cp.asarray(source_values, dtype=np.float64)
        return cuLinearNDInterpolator(target_pts, source_pts), False
    except ValueError as exc:
        if not _is_cupy_degenerate_triangulation_error(exc):
            raise
        _logger.warning(
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
            # try:
            # self._fixedtri = Delaunay(self.TargetPoints, incremental =True)
            # except:
            self._fixedtri = scipy.spatial.Delaunay(self.TargetPoints, incremental=False)

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
            # try:
            # self._fixedtri = Delaunay(self.TargetPoints, incremental =True)
            # except:
            self._fixedtri = scipy.spatial.Delaunay(self.TargetPoints, incremental=False)

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
            control_points = cp.hstack((grid.TargetPoints, grid.SourcePoints))
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
        center = rotationCenter if rotationCenter is not None else np.mean(self.TargetPoints, axis=0)
        self._points[:, 0:2] = ControlPointBase_GPUComponent.RotatePoints(self.TargetPoints, rangle, center)
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
        query_result = cast(
            tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
            self.NearestFixedPoint(old_points),
        )
        distance, index = query_result
        return self.UpdateTargetPointsByIndex(cast(int | NDArray[np.integer], index), points)

    def OnFixedPointChanged(self):
        super(GridTransform_GPU, self).OnFixedPointChanged()
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
