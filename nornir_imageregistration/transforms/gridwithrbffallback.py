"""
Created on Oct 18, 2012

@author: Jamesan
"""

import numpy as np
from typing import Any, cast

try:
    import cupy as cp
    from cupyx.scipy.interpolate import RegularGridInterpolator as cuRegularGridInterpolator

    # import cupyx
    # from cupyx.scipy.interpolate import RegularGridInterpolator as cuRegularGridInterpolator
    # from cupyx.scipy.interpolate import RBFInterpolator as cuRBFInterpolator
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
import scipy.spatial
from scipy.interpolate import RegularGridInterpolator as RegularGridInterpolator
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import float_to_shortest_string
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms.base import IDiscreteTransform, ITransformScaling, \
    ITransformRelativeScaling, ITransformTargetRotation, ITargetSpaceControlPointEdit, IControlPoints, IGridTransform, \
    ITriangulatedTargetSpace, ITransformTranslation 
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents
from nornir_imageregistration.transforms.transform_type import TransformType
from nornir_imageregistration.transforms.landmark import Landmark_GPU, Landmark_CPU
from . import utils


def _fixed_points_for_extrapolation_fill(
    trans_points: NDArray[np.floating], fixed_points: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Discrete grid interpolators may return NumPy while the RBF path returns CuPy; inplace
    ``trans[invalid] = fixed`` must not mix backends (CuPy disallows implicit NumPy conversion).
    """
    xp = cp.get_array_module(trans_points)
    if xp is np:
        return nornir_imageregistration.EnsurePointsAre2DNumpyArray(fixed_points)
    return nornir_imageregistration.EnsurePointsAre2DCuPyArray(fixed_points)


def _defer_continuous_rbf(model: Any) -> None:
    """Mark the RBF fallback stale so the single prewarm worker can replace it."""
    model._continuous_stale = True


def _update_fallback_target_by_index(
        model: Any,
        index: int | NDArray[np.integer],
        point: NDArray[np.floating] | None) -> int | NDArray[np.integer]:
    """Update discrete target points; RBF reconstruction is always deferred."""
    if point is None:
        raise ValueError("point cannot be None")
    result = model._discrete_transform.UpdateTargetPointsByIndex(index, point)
    _defer_continuous_rbf(model)
    model.OnTransformChanged()
    return result


def _update_fallback_target_by_position(
        model: Any,
        index: NDArray[np.floating],
        point: NDArray[np.floating] | None) -> int | NDArray[np.integer]:
    """Update discrete target points by position; RBF reconstruction is always deferred."""
    if point is None:
        raise ValueError("point cannot be None")
    result = model._discrete_transform.UpdateTargetPointsByPosition(index, point)
    _defer_continuous_rbf(model)
    model.OnTransformChanged()
    return result


def _build_refreshed_continuous(model: Any, continuous_ctor: Any) -> Any:
    """Build a replacement RBF fallback from a snapshot of the discrete grid."""
    src = utils.host_copy_points(model._discrete_transform.SourcePoints)
    tgt = utils.host_copy_points(model._discrete_transform.TargetPoints)
    continuous = continuous_ctor(src, tgt)
    initialize = getattr(continuous, "InitializeDataStructures", None)
    if callable(initialize):
        initialize()
    return continuous


def _initialize_fallback_data_structures(model: Any, continuous_ctor: Any) -> None:
    """Rebuild RBF weights, recreating the continuous transform if drag left it stale."""
    if getattr(model, "_continuous_stale", False):
        model._continuous_transform = continuous_ctor(
            model._discrete_transform.SourcePoints,
            model._discrete_transform.TargetPoints,
        )
        model._continuous_stale = False
    model._continuous_transform.InitializeDataStructures()


class GridWithRBFFallback(IDiscreteTransform, IControlPoints, ITransformScaling, ITransformRelativeScaling,
                          ITransformTargetRotation, ITransformTranslation,
                          ITargetSpaceControlPointEdit, IGridTransform, ITriangulatedTargetSpace,
                          DefaultTransformChangeEvents):
    """
    classdocs
    """
    _continuous_stale: bool = False

    @property
    def type(self) -> TransformType:
        return self._discrete_transform.type

    @property
    def grid(self) -> ITKGridDivision:
        return self._discrete_transform.grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        rows, cols = self._discrete_transform.grid_dims
        return int(rows), int(cols)

    def ToITKString(self) -> str:
        return self._discrete_transform.ToITKString()

    def __getstate__(self):
        odict = super(GridWithRBFFallback, self).__getstate__()
        odict['_discrete_transform'] = self._discrete_transform
        odict['_continuous_transform'] = self._continuous_transform
        return odict

    def __setstate__(self, dictionary):
        super(GridWithRBFFallback, self).__setstate__(dictionary)
        self._discrete_transform = dictionary['_discrete_transform']
        self._continuous_transform = dictionary['_continuous_transform']
        self._continuous_stale = dictionary.get('_continuous_stale', False)

    def InitializeDataStructures(self):
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection)
        _initialize_fallback_data_structures(self, twoway_ctor)
        # self._discrete_transform.InitializeDataStructures() Grid does not have an Initialize data structures call

    def build_refreshed_continuous(self) -> Any:
        """Return a new RBF fallback built from the current discrete grid (off-UI)."""
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection)
        return _build_refreshed_continuous(self, twoway_ctor)

    def apply_refreshed_continuous(self, continuous: Any) -> None:
        """Install an off-UI RBF fallback onto this live grid."""
        self._continuous_transform = continuous
        self._continuous_stale = False

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""
        self._continuous_transform.ClearDataStructures()
        self._discrete_transform.ClearDataStructures()

    def OnFixedPointChanged(self):
        self._continuous_transform.OnFixedPointChanged()
        self._discrete_transform.OnFixedPointChanged()
        self.OnTransformChanged()

    def OnWarpedPointChanged(self):
        self._continuous_transform.OnWarpedPointChanged()
        self._discrete_transform.OnWarpedPointChanged()
        self.OnTransformChanged()

    def Transform(self, points: NDArray[np.floating], **kwargs) -> NDArray[np.floating]:
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=points.dtype)

        TransformedPoints = self._discrete_transform.Transform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if len(points) > 1:
                # print invalid_mask;
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points

        BadPoints = np.asarray(BadPoints, dtype=np.float32)
        if not (BadPoints.dtype == np.float32 or BadPoints.dtype == np.float64):
            BadPoints = np.asarray(BadPoints, dtype=np.float32)

        FixedPoints = self._continuous_transform.Transform(BadPoints)

        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points: NDArray[np.floating], **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=points.dtype)

        TransformedPoints = self._discrete_transform.InverseTransform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if points.ndim > 1:
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        if not (BadPoints.dtype == np.float32 or BadPoints.dtype == np.float64):
            BadPoints = np.asarray(BadPoints, dtype=np.float32)

        FixedPoints = self._continuous_transform.InverseTransform(BadPoints)

        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def __init__(self,
                 grid: ITKGridDivision):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(GridWithRBFFallback, self).__init__()

        self._discrete_transform = nornir_imageregistration.transforms.GridTransform(grid)
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection)
        self._continuous_transform = twoway_ctor(grid.SourcePoints, grid.TargetPoints)
        self._continuous_stale = False

    def AddTransform(self, mappedTransform: IControlPoints, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return nornir_imageregistration.transforms.AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance,
                                                                 create_copy=create_copy)

    @staticmethod
    def Load(TransformString: str, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseGridTransform(TransformString, pixelSpacing)

    @property
    def MappedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """Bounding box of mapped space points"""
        return self._discrete_transform.MappedBoundingBox

    @property
    def SourceBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._discrete_transform.SourceBoundingBox

    @property
    def FixedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._discrete_transform.FixedBoundingBox

    @property
    def TargetBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._discrete_transform.TargetBoundingBox

    @property
    def SourcePoints(self) -> NDArray[np.floating]:
        return self._discrete_transform.SourcePoints

    @property
    def TargetPoints(self) -> NDArray[np.floating]:
        return self._discrete_transform.TargetPoints

    @property
    def points(self) -> NDArray[np.floating]:
        return self._discrete_transform.points

    @property
    def NumControlPoints(self) -> int:
        return self._discrete_transform.NumControlPoints

    def NearestTargetPoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestTargetPoint(points))

    def NearestFixedPoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestFixedPoint(points))

    def NearestSourcePoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestSourcePoint(points))

    def NearestWarpedPoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestWarpedPoint(points))

    def GetFixedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in target space'''
        return self._discrete_transform.GetPointPairsInTargetRect(bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in source space'''
        return self._discrete_transform.GetPointPairsInSourceRect(bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return self._discrete_transform.PointPairsToWarpedPoints(points)

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return self._discrete_transform.PointPairsToTargetPoints(points)

    @property
    def fixedtri(self) -> scipy.spatial.Delaunay:
        return cast(scipy.spatial.Delaunay, self._discrete_transform.FixedTriangles)

    @property
    def FixedTriangles(self) -> scipy.spatial.Delaunay:
        return cast(scipy.spatial.Delaunay, self._discrete_transform.FixedTriangles)

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self._discrete_transform.target_space_trianglulation

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._discrete_transform.TranslateFixed(offset)
        _defer_continuous_rbf(self)
        self.OnTransformChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._discrete_transform.TranslateWarped(offset)
        _defer_continuous_rbf(self)
        self.OnTransformChanged()

    def Scale(self, scalar: float):
        '''Scale both warped and control space by scalar'''
        self._discrete_transform.Scale(scalar)
        self._continuous_transform.Scale(scalar)
        self.OnTransformChanged()

    def ScaleWarped(self, scalar: float):
        '''Scale source space control points by scalar'''
        self._discrete_transform.ScaleWarped(scalar)
        self._continuous_transform.ScaleWarped(scalar)
        self.OnTransformChanged()

    def ScaleFixed(self, scalar: float):
        '''Scale target space control points by scalar'''
        self._discrete_transform.ScaleFixed(scalar)
        self._continuous_transform.ScaleFixed(scalar)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        if rotation_center is None:
            rotation_center = self.FixedBoundingBox.Center

        self._discrete_transform.RotateTargetPoints(rangle, rotation_center)
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection)
        self._continuous_transform = twoway_ctor(self._discrete_transform.SourcePoints, self._discrete_transform.TargetPoints)

        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], point: NDArray[np.floating] | None) -> int | \
                                                                                                                 NDArray[
                                                                                                                     np.integer]:
        # Using this may cause errors since the discrete and continuous transforms are not guaranteed to use the same index
        return _update_fallback_target_by_index(self, index, point)

    def UpdateTargetPointsByPosition(self, index: NDArray[np.floating], point: NDArray[np.floating] | None) -> int | \
                                                                                                               NDArray[
                                                                                                                   np.integer]:
        return _update_fallback_target_by_position(self, index, point)


class GridWithRBFFallback_GPUComponent(IDiscreteTransform, IControlPoints, ITransformScaling,
                                       ITransformRelativeScaling, ITransformTargetRotation,
                                       ITransformTranslation,
                                       ITargetSpaceControlPointEdit, IGridTransform, ITriangulatedTargetSpace,
                                       DefaultTransformChangeEvents):
    """
    classdocs
    """
    _continuous_stale: bool = False

    @property
    def type(self) -> TransformType:
        return self._discrete_transform.type

    @property
    def grid(self) -> ITKGridDivision:
        return self._discrete_transform.grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        rows, cols = self._discrete_transform.grid_dims
        return int(rows), int(cols)

    def ToITKString(self) -> str:
        return self._discrete_transform.ToITKString()

    def __getstate__(self):
        odict = super(GridWithRBFFallback_GPUComponent, self).__getstate__()
        odict['_discrete_transform'] = self._discrete_transform
        odict['_continuous_transform'] = self._continuous_transform
        return odict

    def __setstate__(self, dictionary):
        super(GridWithRBFFallback_GPUComponent, self).__setstate__(dictionary)
        self._discrete_transform = dictionary['_discrete_transform']
        self._continuous_transform = dictionary['_continuous_transform']
        self._continuous_stale = dictionary.get('_continuous_stale', False)

    def InitializeDataStructures(self):
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection_GPUComponent)
        _initialize_fallback_data_structures(self, twoway_ctor)
        # self._discrete_transform.InitializeDataStructures() Grid does not have an Initialize data structures call

    def build_refreshed_continuous(self) -> Any:
        """Return a new RBF fallback built from the current discrete grid (off-UI)."""
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection_GPUComponent)
        return _build_refreshed_continuous(self, twoway_ctor)

    def apply_refreshed_continuous(self, continuous: Any) -> None:
        """Install an off-UI RBF fallback onto this live grid."""
        self._continuous_transform = continuous
        self._continuous_stale = False

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""
        self._continuous_transform.ClearDataStructures()
        self._discrete_transform.ClearDataStructures()

    def OnFixedPointChanged(self):
        self._continuous_transform.OnFixedPointChanged()
        self._discrete_transform.OnFixedPointChanged()
        self.OnTransformChanged()

    def OnWarpedPointChanged(self):
        self._continuous_transform.OnWarpedPointChanged()
        self._discrete_transform.OnWarpedPointChanged()
        self.OnTransformChanged()

    def Transform(self, points: NDArray[np.floating], **kwargs) -> NDArray[np.floating]:
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        if points.shape[0] == 0:
            return cp.empty((0, 2), dtype=points.dtype)

        TransformedPoints = self._discrete_transform.Transform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if len(points) > 1:
                # print invalid_mask;
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points

        # BadPoints = cp.asarray(BadPoints, dtype=np.float32)
        if not (BadPoints.dtype == np.float32 or BadPoints.dtype == np.float64):
            BadPoints = cp.asarray(BadPoints, dtype=np.float32)

        FixedPoints = self._continuous_transform.Transform(BadPoints)
        FixedPoints = _fixed_points_for_extrapolation_fill(TransformedPoints, FixedPoints)

        TransformedPoints[invalid_mask] = FixedPoints

        # Ensure discrete-transform outputs remain on the GPU when callers expect CuPy arrays.
        TransformedPoints = nornir_imageregistration.EnsurePointsAre2DCuPyArray(TransformedPoints)
        return TransformedPoints

    def InverseTransform(self, points: NDArray[np.floating], **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        if points.shape[0] == 0:
            return cp.empty((0, 2), dtype=points.dtype)


        TransformedPoints = self._discrete_transform.InverseTransform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if points.ndim > 1:
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        if not (BadPoints.dtype == np.float32 or BadPoints.dtype == np.float64):
            BadPoints = cp.asarray(BadPoints, dtype=np.float32)

        FixedPoints = self._continuous_transform.InverseTransform(BadPoints)
        FixedPoints = _fixed_points_for_extrapolation_fill(TransformedPoints, FixedPoints)
        TransformedPoints[invalid_mask] = FixedPoints

        # Ensure discrete-transform outputs remain on the GPU when callers expect CuPy arrays.
        TransformedPoints = nornir_imageregistration.EnsurePointsAre2DCuPyArray(TransformedPoints)
        return TransformedPoints

    def __init__(self,
                 grid: ITKGridDivision):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(GridWithRBFFallback_GPUComponent, self).__init__()

        # self._discrete_transform = nornir_imageregistration.transforms.GridTransform(grid)
        self._discrete_transform = nornir_imageregistration.transforms.GridTransform_GPUComponent(grid)
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection_GPUComponent)
        self._continuous_transform = twoway_ctor(grid.SourcePoints, grid.TargetPoints)
        self._continuous_stale = False

    def AddTransform(self, mappedTransform: IControlPoints, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return nornir_imageregistration.transforms.AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance,
                                                                 create_copy=create_copy)

    @staticmethod
    def Load(TransformString: str, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseGridTransform(TransformString, pixelSpacing)

    @property
    def MappedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """Bounding box of mapped space points"""
        return self._discrete_transform.MappedBoundingBox

    @property
    def FixedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._discrete_transform.FixedBoundingBox

    @property
    def SourceBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._discrete_transform.MappedBoundingBox

    @property
    def TargetBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._discrete_transform.FixedBoundingBox

    @property
    def SourcePoints(self) -> NDArray[np.floating]:
        return self._discrete_transform.SourcePoints

    @property
    def TargetPoints(self) -> NDArray[np.floating]:
        return self._discrete_transform.TargetPoints

    @property
    def points(self) -> NDArray[np.floating]:
        return self._discrete_transform.points

    @property
    def NumControlPoints(self) -> int:
        return self._discrete_transform.NumControlPoints

    def NearestTargetPoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestTargetPoint(points))

    def NearestFixedPoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestFixedPoint(points))

    def NearestSourcePoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestSourcePoint(points))

    def NearestWarpedPoint(self, points: NDArray[np.floating]) -> tuple[float | NDArray[np.floating], int | NDArray[np.integer]]:
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        return cast(tuple[float | NDArray[np.floating], int | NDArray[np.integer]],
                    self._discrete_transform.NearestWarpedPoint(points))

    def GetFixedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self._discrete_transform.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in target space'''
        return self._discrete_transform.GetPointPairsInTargetRect(bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in source space'''
        return self._discrete_transform.GetPointPairsInSourceRect(bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return self._discrete_transform.PointPairsToWarpedPoints(points)

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return self._discrete_transform.PointPairsToTargetPoints(points)

    @property
    def fixedtri(self) -> scipy.spatial.Delaunay:
        return cast(scipy.spatial.Delaunay, self._discrete_transform.FixedTriangles)

    @property
    def FixedTriangles(self) -> scipy.spatial.Delaunay:
        return cast(scipy.spatial.Delaunay, self._discrete_transform.FixedTriangles)

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self._discrete_transform.target_space_trianglulation

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._discrete_transform.TranslateFixed(offset)
        _defer_continuous_rbf(self)
        self.OnTransformChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._discrete_transform.TranslateWarped(offset)
        _defer_continuous_rbf(self)
        self.OnTransformChanged()

    def Scale(self, scalar: float):
        '''Scale both warped and control space by scalar'''
        self._discrete_transform.Scale(scalar)
        self._continuous_transform.Scale(scalar)
        self.OnTransformChanged()

    def ScaleWarped(self, scalar: float):
        '''Scale source space control points by scalar'''
        self._discrete_transform.ScaleWarped(scalar)
        self._continuous_transform.ScaleWarped(scalar)
        self.OnTransformChanged()

    def ScaleFixed(self, scalar: float):
        '''Scale target space control points by scalar'''
        self._discrete_transform.ScaleFixed(scalar)
        self._continuous_transform.ScaleFixed(scalar)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        if rotation_center is None:
            rotation_center = self.FixedBoundingBox.Center

        self._discrete_transform.RotateTargetPoints(rangle, rotation_center)
        twoway_ctor = cast(Any, nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection)
        self._continuous_transform = twoway_ctor(self._discrete_transform.SourcePoints, self._discrete_transform.TargetPoints)

        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], point: NDArray[np.floating] | None) -> int | \
                                                                                                                 NDArray[
                                                                                                                     np.integer]:
        # Using this may cause errors since the discrete and continuous transforms are not guaranteed to use the same index
        return _update_fallback_target_by_index(self, index, point)

    def UpdateTargetPointsByPosition(self, index: NDArray[np.floating], point: NDArray[np.floating] | None) -> int | \
                                                                                                               NDArray[
                                                                                                                   np.integer]:
        return _update_fallback_target_by_position(self, index, point)


class GridWithRBFInterpolator_Direct_GPU(Landmark_GPU):
    """
    classdocs
    """

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

    def ToITKString(self) -> str:
        numPoints = self.SourcePoints.shape[0]
        bottom, left, top, right = cast(tuple[float, float, float, float], self.MappedBoundingBox.ToTuple())
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self._grid._grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self._grid._grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded = NumAdded + 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    def __getstate__(self):

        odict = super(GridWithRBFInterpolator_Direct_GPU, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(GridWithRBFInterpolator_Direct_GPU, self).__setstate__(dictionary)

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = super(GridWithRBFInterpolator_Direct_GPU, self).InverseInterpolator

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = super(GridWithRBFInterpolator_Direct_GPU, self).ForwardInterpolator

        return self._ForwardRBFInstance

    # def InitializeDataStructures(self):
    #
    #     self._ForwardRBFInstance = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)
    #     self._ReverseRBFInstance = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)
    #
    #
    # def ClearDataStructures(self):
    #     """Something about the transform has changed, for example the points.
    #        Clear out our data structures so we do not use bad data"""
    #
    #     super(GridWithRBFInterpolator_Direct_GPU, self).ClearDataStructures()
    #
    #     self._ForwardRBFInstance = None
    #     self._ReverseRBFInstance = None

    def OnFixedPointChanged(self):
        super(GridWithRBFInterpolator_Direct_GPU, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(GridWithRBFInterpolator_Direct_GPU, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """
        print("GridWithRBFInterpolator_Direct_GPU -> TRANSFORM()")
        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        TransformedPoints = super(GridWithRBFInterpolator_Direct_GPU, self).Transform(points)
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """
        print("GridWithRBFInterpolator_Direct_GPU -> INVERSETRANSFORM()")

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        iTransformedPoints = super(GridWithRBFInterpolator_Direct_GPU, self).InverseTransform(points)
        return iTransformedPoints

    def __init__(self, grid: ITKGridDivision):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        self._grid = grid
        try:
            control_points = cp.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid: {grid.TargetPoints} {grid.SourcePoints}')
            raise

        super(GridWithRBFInterpolator_Direct_GPU, self).__init__(control_points)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString: str, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseGridTransform(TransformString, pixelSpacing)


class GridWithRBFInterpolator_Direct_CPU(Landmark_CPU):
    """
    classdocs
    """

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

    def ToITKString(self) -> str:
        numPoints = self.SourcePoints.shape[0]
        bottom, left, top, right = cast(tuple[float, float, float, float], self.MappedBoundingBox.ToTuple())
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self._grid._grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self._grid._grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded = NumAdded + 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    def __getstate__(self):

        odict = super(GridWithRBFInterpolator_Direct_CPU, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(GridWithRBFInterpolator_Direct_CPU, self).__setstate__(dictionary)

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = super(GridWithRBFInterpolator_Direct_CPU, self).InverseInterpolator

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = super(GridWithRBFInterpolator_Direct_CPU, self).ForwardInterpolator

        return self._ForwardRBFInstance

    # def InitializeDataStructures(self):
    #
    #     self._ForwardRBFInstance = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)
    #     self._ReverseRBFInstance = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)
    #
    #
    # def ClearDataStructures(self):
    #     """Something about the transform has changed, for example the points.
    #        Clear out our data structures so we do not use bad data"""
    #
    #     super(GridWithRBFInterpolator_Direct_CPU, self).ClearDataStructures()
    #
    #     self._ForwardRBFInstance = None
    #     self._ReverseRBFInstance = None

    def OnFixedPointChanged(self):
        super(GridWithRBFInterpolator_Direct_CPU, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(GridWithRBFInterpolator_Direct_CPU, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        TransformedPoints = super(GridWithRBFInterpolator_Direct_CPU, self).Transform(points)
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        iTransformedPoints = super(GridWithRBFInterpolator_Direct_CPU, self).InverseTransform(points)
        return iTransformedPoints

    def __init__(self, grid: ITKGridDivision):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        self._grid = grid
        try:
            control_points = np.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid: {grid.TargetPoints} {grid.SourcePoints}')
            raise

        super(GridWithRBFInterpolator_Direct_CPU, self).__init__(control_points)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString: str, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseGridTransform(TransformString, pixelSpacing)


class GridWithRBFInterpolator_GPU(Landmark_GPU):
    """
    classdocs
    """

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

    def ToITKString(self) -> str:
        numPoints = self.SourcePoints.shape[0]
        bottom, left, top, right = cast(tuple[float, float, float, float], self.MappedBoundingBox.ToTuple())
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self._grid._grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self._grid._grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded = NumAdded + 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    def __getstate__(self):

        odict = super(GridWithRBFInterpolator_GPU, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        odict['_discrete_transform'] = self._discrete_transform  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(GridWithRBFInterpolator_GPU, self).__setstate__(dictionary)

    @property
    def discrete_transform(self):
        if self._discrete_transform is None:
            axes = tuple(cp.asarray(x, dtype=np.float64) for x in self._grid.axis_points)
            vals = cp.reshape(
                cp.asarray(self.TargetPoints, dtype=np.float64),
                (int(self._grid.grid_dims[0]), int(self._grid.grid_dims[1]), 2),
            )
            self._discrete_transform = cuRegularGridInterpolator(axes, vals, bounds_error=False)

        return self._discrete_transform

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = super(GridWithRBFInterpolator_GPU, self).InverseInterpolator

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = super(GridWithRBFInterpolator_GPU, self).ForwardInterpolator

        return self._ForwardRBFInstance

    def InitializeDataStructures(self):

        super(GridWithRBFInterpolator_GPU, self).InitializeDataStructures()
        # self._ForwardRBFInstance = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)
        # self._ReverseRBFInstance = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)

        # self._discrete_transform.InitializeDataStructures() Grid does not have an Initialize data structures call

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""

        super(GridWithRBFInterpolator_GPU, self).ClearDataStructures()

        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None
        self._discrete_transform = None

    def OnFixedPointChanged(self):
        super(GridWithRBFInterpolator_GPU, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(GridWithRBFInterpolator_GPU, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """
        print("GridWithRBFInterpolator_GPU -> TRANSFORM()")
        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        if points.shape[0] == 0:
            return cp.empty((0, 2), dtype=points.dtype)

        TransformedPoints = self.discrete_transform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if len(points) > 1:
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points

        # BadPoints = cp.asarray(BadPoints, dtype=np.float32)
        if not (BadPoints.dtype == np.float32 or BadPoints.dtype == np.float64):
            BadPoints = cp.asarray(BadPoints, dtype=np.float32)

        FixedPoints = super(GridWithRBFInterpolator_GPU, self).Transform(BadPoints)

        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """
        print("GridWithRBFInterpolator_GPU -> INVERSETRANSFORM()")
        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        iTransformedPoints = super(GridWithRBFInterpolator_GPU, self).InverseTransform(points)
        return iTransformedPoints

    def __init__(self, grid: ITKGridDivision):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        self._grid = grid
        try:
            control_points = cp.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid: {grid.TargetPoints} {grid.SourcePoints}')
            raise

        super(GridWithRBFInterpolator_GPU, self).__init__(control_points)
        axes = tuple(cp.asarray(x, dtype=np.float64) for x in self._grid.axis_points)
        vals = cp.reshape(
            cp.asarray(self._grid.TargetPoints, dtype=np.float64),
            (int(self._grid.grid_dims[0]), int(self._grid.grid_dims[1]), 2),
        )
        self._discrete_transform = cuRegularGridInterpolator(axes, vals, bounds_error=False)
        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString: str, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseGridTransform(TransformString, pixelSpacing)


class GridWithRBFInterpolator_CPU(Landmark_CPU):
    """
    classdocs
    """

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

    def ToITKString(self) -> str:
        numPoints = self.SourcePoints.shape[0]
        bottom, left, top, right = cast(tuple[float, float, float, float], self.MappedBoundingBox.ToTuple())
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self._grid._grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self._grid._grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded = NumAdded + 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    def __getstate__(self):

        odict = super(GridWithRBFInterpolator_CPU, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        odict['_discrete_transform'] = self._discrete_transform  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(GridWithRBFInterpolator_CPU, self).__setstate__(dictionary)

    @property
    def discrete_transform(self):
        if self._discrete_transform is None:
            self._discrete_transform = RegularGridInterpolator(self._grid.axis_points,
                                                               np.reshape(self.TargetPoints, (
                                                                   self._grid.grid_dims[0], self._grid.grid_dims[1],
                                                                   2)),
                                                               bounds_error=False)

        return self._discrete_transform

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = super(GridWithRBFInterpolator_CPU, self).InverseInterpolator

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = super(GridWithRBFInterpolator_CPU, self).ForwardInterpolator

        return self._ForwardRBFInstance

    def InitializeDataStructures(self):

        super(GridWithRBFInterpolator_CPU, self).InitializeDataStructures()
        # self._ForwardRBFInstance = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)
        # self._ReverseRBFInstance = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)

        # self._discrete_transform.InitializeDataStructures() Grid does not have an Initialize data structures call

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""

        super(GridWithRBFInterpolator_CPU, self).ClearDataStructures()

        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None
        self._discrete_transform = None

    def OnFixedPointChanged(self):
        super(GridWithRBFInterpolator_CPU, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(GridWithRBFInterpolator_CPU, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """
        print("GridWithRBFInterpolator_CPU -> TRANSFORM()")
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=points.dtype)

        TransformedPoints = self.discrete_transform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if len(points) > 1:
                # print invalid_mask;
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points

        if not (BadPoints.dtype == np.float32 or BadPoints.dtype == np.float64):
            BadPoints = np.asarray(BadPoints, dtype=np.float32)

        FixedPoints = super(GridWithRBFInterpolator_CPU, self).Transform(BadPoints)

        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """
        print("GridWithRBFInterpolator_CPU -> INVERSETRANSFORM()")
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        iTransformedPoints = super(GridWithRBFInterpolator_CPU, self).InverseTransform(points)
        return iTransformedPoints

    def __init__(self, grid: ITKGridDivision):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        self._grid = grid
        try:
            control_points = np.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid: {grid.TargetPoints} {grid.SourcePoints}')
            raise

        super(GridWithRBFInterpolator_CPU, self).__init__(control_points)
        self._discrete_transform = RegularGridInterpolator(self._grid.axis_points,
                                                           np.reshape(self._grid.TargetPoints, (
                                                               self._grid.grid_dims[0], self._grid.grid_dims[1], 2)),
                                                           bounds_error=False)
        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString: str, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseGridTransform(TransformString, pixelSpacing)


if __name__ == '__main__':
    p = np.array([[0, 0, 0, 0],
                  [0, 10, 0, -10],
                  [10, 0, -10, 0],
                  [10, 10, -10, -10]])

    (Fixed, Moving) = np.hsplit(p, 2)
    T: Any = nornir_imageregistration.transforms.OneWayRBFWithLinearCorrection(Fixed, Moving)

    warpedPoints = [[0, 0], [-5, -5]]
    fp = T.ViewTransform(warpedPoints)
    print(("__Transform " + str(warpedPoints) + " to " + str(fp)))
    wp = T.InverseTransform(fp)

    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    T.AddPoint([5, 5, -5, -5])
    print("\nPoint added")
    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    T.AddPoint([5, 5, 5, 5])
    print("\nDuplicate Point added")
    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    warpedPoint = [[-5, -5]]
    fp = T.ViewTransform(warpedPoint)
    print(("__Transform " + str(warpedPoint) + " to " + str(fp)))
    wp = T.InverseTransform(fp)

    T.UpdatePoint(3, [10, 15, -10, -15])
    print("\nPoint updated")
    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    warpedPoint = [[-9, -14]]
    fp = T.ViewTransform(warpedPoint)
    print(("__Transform " + str(warpedPoint) + " to " + str(fp)))
    wp = T.InverseTransform(fp)

    T.RemovePoint(1)
    print("\nPoint removed")
    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    print("\nFixedPointsInRect")
    print(T.GetFixedPointsRect([-1, -1, 14, 4]))

