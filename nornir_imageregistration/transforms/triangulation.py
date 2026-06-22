'''
Created on Oct 18, 2012

@author: Jamesan
'''
from collections.abc import Iterable
import logging
from multiprocessing import Value
from typing import Any, cast

import numpy as np

try:
    import cupy as cp
    # import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx
from numpy.typing import NDArray
import scipy
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator

cuLinearNDInterpolator: Any | None = None
try:
    from cupyx.scipy.interpolate import LinearNDInterpolator as cuLinearNDInterpolator
except ImportError:
    pass

import nornir_imageregistration
from nornir_imageregistration.nearest_neighbor import build_nearest_neighbor_index
from nornir_imageregistration.transforms.gridtransform import (
    _build_linear_nd_interpolator,
    _forward_transform_with_linear_nd_fallback,
    _inverse_transform_with_linear_nd_fallback,
)
import nornir_pools
from . import TransformType
from .base import ITransform, ITransformScaling, ITransformRelativeScaling, ITransformTranslation, \
    IControlPointEdit, ITransformSourceRotation, ITransformTargetRotation, ITriangulatedTargetSpace, \
    ITriangulatedSourceSpace, IControlPointAddRemove
from .controlpointbase import ControlPointBase

class Triangulation(ITransformScaling, ITransformRelativeScaling, ITransformTranslation, IControlPointEdit,
                    ITransformSourceRotation,
                    ITransformTargetRotation, ITriangulatedTargetSpace, ITriangulatedSourceSpace,
                    IControlPointAddRemove, ControlPointBase):
    '''
    Triangulation transform has a nx4 array of points, with rows organized as
    [controlx controly warpedx warpedy]
    '''

    _points: NDArray[np.floating]
    _ForwardInterpolator: LinearNDInterpolator | None
    _InverseInterpolator: LinearNDInterpolator | None
    _fixedtri: scipy.spatial.Delaunay | None
    _warpedtri: scipy.spatial.Delaunay | None
    _WarpedKDTree: Any  # nearest-neighbor index (scipy or CuVS backend)
    _FixedKDTree: Any

    @property
    def type(self) -> TransformType:
        return nornir_imageregistration.transforms.transform_type.TransformType.MESH

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
    def fixedtri(self) -> scipy.spatial.Delaunay:
        if self._fixedtri is None:
            # try:
            # self._fixedtri = Delaunay(self.TargetPoints, incremental =True)
            # except:
            self._fixedtri = scipy.spatial.Delaunay(self.TargetPoints, incremental=False)

        return self._fixedtri

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.fixedtri

    @property
    def warpedtri(self) -> scipy.spatial.Delaunay:
        if self._warpedtri is None:
            # try:
            # self._warpedtri = Delaunay(self.SourcePoints, incremental =True)
            # except:
            self._warpedtri = scipy.spatial.Delaunay(self.SourcePoints, incremental=False)

        return self._warpedtri

    @property
    def source_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.warpedtri

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            # self._ForwardInterpolator = CloughTocher2DInterpolator(self.warpedtri, self.TargetPoints)
            self._ForwardInterpolator = LinearNDInterpolator(self.warpedtri, self.TargetPoints)

        return cast(LinearNDInterpolator, self._ForwardInterpolator)

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            # self._InverseInterpolator = CloughTocher2DInterpolator(self.fixedtri, self.SourcePoints)
            self._InverseInterpolator = LinearNDInterpolator(self.fixedtri, self.SourcePoints)

        return cast(LinearNDInterpolator, self._InverseInterpolator)

    def AddTransform(self, mappedTransform: ITransform, EnrichTolerance: float | None = None, create_copy: bool = True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return nornir_imageregistration.transforms.AddTransforms(cast(ITransform, self), cast(Any, mappedTransform),
                                                                 EnrichTolerance=EnrichTolerance,
                                                                 create_copy=create_copy)

    def Transform(self, points: NDArray[np.floating], **kwargs):
        '''Map points from the warped space to fixed space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        out_xp = cp.get_array_module(points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.ForwardInterpolator(points).astype(np.float32, copy=False)
        except Exception as e:  # This is usually a scipy.spatial._qhull.QhullError:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            self._ForwardInterpolator = None

            # This was added for the case where all points in the triangulation are colinear.
            transPoints = np.empty(points.shape)
            transPoints[:] = np.nan

        # When CuPy support was first added, there was no support for LinearNDInterpolator, but the rest of the Cupy paths expect a Cupy array, so convert the array to CuPy if needed
        transPoints = transPoints if out_xp is np else nornir_imageregistration.EnsurePointsAre2DCuPyArray(
            transPoints)

        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        out_xp = cp.get_array_module(points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.InverseInterpolator(points).astype(np.float32, copy=False)
        except Exception as e:  # This is usually a scipy.spatial._qhull.QhullError:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None
            self._InverseInterpolator = None

            # This was added for the case where all points in the triangulation are colinear.
            transPoints = np.empty(points.shape)
            transPoints[:] = np.nan

        # When CuPy support was first added, there was no support for LinearNDInterpolator, but the rest of the Cupy paths expect a Cupy array, so convert the array to CuPy if needed
        transPoints = transPoints if out_xp is np else nornir_imageregistration.EnsurePointsAre2DCuPyArray(
            transPoints)

        return transPoints

    def AddPoints(self, new_points: NDArray[np.floating]):
        '''Add the point and return the index'''
        numPts = self.NumControlPoints
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(new_points)

        duplicates = np.atleast_1d(self.FindDuplicateFixedPoints(new_points[:, 0:2]))
        duplicates = np.ravel(duplicates)
        new_points = new_points[~duplicates, :]
        if new_points.ndim == 1:
            new_points = np.reshape(new_points, (1, 4))

        if new_points.shape[0] == 0:
            return

        self._points = np.append(self.points, new_points, 0)
        # self._points = Triangulation.RemoveDuplicates(self._points)

        # We won't see a change in the number of points if the new point was a duplicate
        if self.NumControlPoints != numPts:
            self.OnPointsAddedToTransform(new_points)

        return

    def AddPoint(self, pointpair: NDArray[np.floating]) -> int:
        '''Add the point and return the index'''
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(pointpair)
        self.AddPoints(new_points)

        Distance, index = self.NearestFixedPoint((new_points[0, 0], new_points[0, 1]))
        return index

    def UpdatePointPair(self, index: int, pointpair: NDArray[np.floating]):
        self._points[index, :] = pointpair
        self._points = Triangulation.RemoveDuplicateControlPoints(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint((pointpair[0], pointpair[1]))
        return index

    def UpdateFixedPoints(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._points[index, 0:2] = points
        self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
        self.OnFixedPointChanged()

        distance, index = self.NearestFixedPoint(points)
        return index

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], new_points: NDArray[np.floating]) -> int | \
                                                                                                               NDArray[
                                                                                                                   np.integer]:
        return self.UpdateFixedPoints(index, new_points)

    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], new_points: NDArray[np.floating]) -> int | \
                                                                                                                  NDArray[
                                                                                                                      np.integer]:
        Distance, index = self.NearestTargetPoint(old_points)
        return self.UpdateTargetPointsByIndex(index, new_points)

    def UpdateWarpedPoints(self, index: int | NDArray[np.integer],
                           points: NDArray[np.floating]) -> int | NDArray[
        np.integer]:
        self._points[index, 2:4] = points
        self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
        self.OnWarpedPointChanged()

        distance, index = self.NearestWarpedPoint(points)
        return cast(int | NDArray[np.integer], index)

    def UpdateSourcePointsByIndex(self, index: int | NDArray[np.integer], new_points: NDArray[np.floating]) -> int | \
                                                                                                               NDArray[
                                                                                                                   np.integer]:
        return self.UpdateWarpedPoints(index, new_points)

    def UpdateSourcePointsByPosition(self, old_points: NDArray[np.floating], new_points: NDArray[np.floating]) -> int | \
                                                                                                                  NDArray[
                                                                                                                      np.integer]:
        distance, index = self.NearestSourcePoint(old_points)
        return self.UpdateSourcePointsByIndex(index, new_points)

    def RemovePoint(self, index: int | NDArray[np.integer]):
        nToRemove = 1
        if isinstance(index, Iterable):
            nToRemove = len(index)

        if self._points.shape[0] - nToRemove < 3:
            raise ValueError("Cannot remove points, must have at least three points")

        xp = cp.get_array_module(self._points)
        keep = xp.ones(self._points.shape[0], dtype=bool)
        keep[index] = False
        self._points = self._points[keep, :].copy()
        # self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
        self.OnTransformChanged()

    def InitializeDataStructures(self):
        '''This optional method performs all computationally intense data structure creation
           If not run these data structures should be initialized in a lazy fashion by the class
           If it is known that the data structures will be needed this function can be faster
           since computations can be performed in parallel'''

        MPool = nornir_pools.GetGlobalMultithreadingPool()
        TPool = nornir_pools.GetGlobalThreadPool()
        FixedTriTask = MPool.add_task("Fixed Triangle Delaunay", scipy.spatial.Delaunay, self.TargetPoints)
        WarpedTriTask = MPool.add_task("Warped Triangle Delaunay", scipy.spatial.Delaunay, self.SourcePoints)

        # Cannot pickle KDTree, so use Python's thread pool

        self._WarpedKDTree = build_nearest_neighbor_index(self.SourcePoints)
        self._FixedKDTree = build_nearest_neighbor_index(self.TargetPoints)

        self._fixedtri = FixedTriTask.wait_return()
        self._warpedtri = WarpedTriTask.wait_return()

    def OnPointsAddedToTransform(self, new_points):
        '''Similiar to OnTransformChanged, but optimized to handle the case of points being added'''

        self.OnTransformChanged()
        return

    #
    #         if(self._fixedtri is None or
    #            self._warpedtri is None):
    #             self.OnTransformChanged()
    #             return
    #
    #         self._WarpedKDTree = None
    #         self._FixedKDTree = None
    #         self._FixedBoundingBox = None
    #         self._MappedBoundingBox = None
    #         self._ForwardInterpolator = None
    #         self._InverseInterpolator = None
    #
    #         self._fixedtri.add_points(new_points[:,0:2])
    #         self._warpedtri.add_points(new_points[:,2:4])
    #         super(Triangulation, self).OnTransformChanged()

    def OnFixedPointChanged(self):
        super(Triangulation, self).OnFixedPointChanged()
        self._FixedKDTree = None
        self._fixedtri = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

        super(Triangulation, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        super(Triangulation, self).OnWarpedPointChanged()
        self._WarpedKDTree = None
        self._warpedtri = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        super(Triangulation, self).OnTransformChanged()

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points. 
           Clear out our data structures, so we do not use stale data'''
        super(Triangulation, self).ClearDataStructures()
        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def NearestTargetPoint(self, points: NDArray[np.floating]):
        return self.FixedKDTree.query(points)

    def NearestFixedPoint(self, points: NDArray[np.floating] | tuple[float, float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestSourcePoint(self, points: NDArray[np.floating] | tuple[float, float]):
        return self.WarpedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        '''Return the warped points nearest to the query points
        :return: Distance, Index'''
        return self.WarpedKDTree.query(points)

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 2:4] = ControlPointBase.RotatePoints(self.SourcePoints, rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 0:2] = ControlPointBase.RotatePoints(self.TargetPoints, rangle, rotation_center)
        self.OnTransformChanged()

    def FlipWarped(self, flip_center=None):
        '''
        Flips the X coordinates along the vertical line passing through flip_center.  If flip_center is None the center of the bounding box of the points is used.
        '''
        if flip_center is None:
            flip_center = self.MappedBoundingBox.Center

        temp = self.points[:, 2:4] - flip_center
        temp[:, 1] = -temp[:, 1]
        temp = temp + flip_center[1]
        self.points[:, 2:4] = temp
        self.OnTransformChanged()

    def Scale(self, scalar: float):
        '''Scale both warped and control space by scalar'''
        self._points *= scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar: float):
        '''Scale source space control points by scalar'''
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar: float):
        '''Scale target space control points by scalar'''
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    @property
    def MappedBoundingBoxHeight(self):
        raise DeprecationWarning("MappedBoundingBoxHeight is deprecated.  Use MappedBoundingBox.Height instead")
        return self.MappedBoundingBox.Height

    @property
    def FixedTriangles(self):
        return self.fixedtri.simplices

    @property
    def WarpedTriangles(self):
        return self.warpedtri.simplices

    def GetFixedCentroids(self, triangles=None):
        '''Centroids of fixed triangles'''
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def GetWarpedCentroids(self, triangles=None):
        '''Centroids of warped triangles'''
        if triangles is None:
            triangles = self.WarpedTriangles

        warpedTriangleVerticies = self.SourcePoints[triangles]
        swappedTriangleVerticies = np.swapaxes(warpedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def __init__(self, pointpairs: NDArray[np.floating]):
        '''
        Constructor requires at least three point pairs
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX] 
        '''
        super(Triangulation, self).__init__(pointpairs)

        if self._points.shape[0] < 3:
            raise ValueError("Triangulation transform must have at least three points to function")

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None

    def ToITKString(self) -> str:
        return nornir_imageregistration.transforms.factory._MeshTransformToIRToolsString(self, self.MappedBoundingBox)

    def Load(self, TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)

    @classmethod
    def load(cls, variableParams, fixedParams):

        points = np.array.fromiter(variableParams)
        points.reshape(variableParams / 2, 2)


class Triangulation_GPUComponent(ITransformScaling, ITransformRelativeScaling, ITransformTranslation, IControlPointEdit,
                                 ITransformSourceRotation,
                                 ITransformTargetRotation, ITriangulatedTargetSpace, ITriangulatedSourceSpace,
                                 IControlPointAddRemove, ControlPointBase):
    '''
    Triangulation transform has an nx4 array of points, with rows organized as
    [controlx controly warpedx warpedy]
    '''

    @property
    def type(self) -> TransformType:
        return nornir_imageregistration.transforms.transform_type.TransformType.MESH

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
    def fixedtri(self) -> scipy.spatial.Delaunay:
        if self._fixedtri is None:
            # try:
            # self._fixedtri = Delaunay(self.TargetPoints, incremental =True)
            # except:
            self._fixedtri = scipy.spatial.Delaunay(self.TargetPoints, incremental=False)

        return self._fixedtri

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.fixedtri

    @property
    def warpedtri(self) -> scipy.spatial.Delaunay:
        if self._warpedtri is None:
            # try:
            # self._warpedtri = Delaunay(self.SourcePoints, incremental =True)
            # except:
            self._warpedtri = scipy.spatial.Delaunay(self.SourcePoints, incremental=False)

        return self._warpedtri

    @property
    def source_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.warpedtri

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            interp, uses_scipy = _build_linear_nd_interpolator(
                self.SourcePoints,
                self.TargetPoints,
            )
            self._ForwardInterpolator = interp
            self._scipy_forward_interp = uses_scipy

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

    def AddTransform(self, mappedTransform, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return nornir_imageregistration.transforms.AddTransforms(cast(ITransform, self), cast(Any, mappedTransform),
                                                                 EnrichTolerance=EnrichTolerance, create_copy=create_copy)

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        return _forward_transform_with_linear_nd_fallback(
            self,
            points,
            scipy_flag_attr='_scipy_forward_interp',
            interpolator_property='ForwardInterpolator',
            output_dtype=cp.float32,
        )

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        return _inverse_transform_with_linear_nd_fallback(
            self,
            points,
            scipy_flag_attr='_scipy_inverse_interp',
            interpolator_property='InverseInterpolator',
            output_dtype=cp.float32,
        )

    def AddPoints(self, new_points: NDArray[np.floating]):
        '''Add the point and return the index'''
        numPts = self.NumControlPoints
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(new_points)

        duplicates = self.FindDuplicateFixedPoints(new_points[:, 0:2])
        new_points = new_points[~duplicates, :]

        if new_points.shape[0] == 0:
            return

        self._points = np.append(self.points, new_points, 0)
        # self._points = Triangulation_GPUComponent.RemoveDuplicates(self._points)

        # We won't see a change in the number of points if the new point was a duplicate
        if self.NumControlPoints != numPts:
            self.OnPointsAddedToTransform(new_points)

        return

    def AddPoint(self, pointpair: NDArray[np.floating]) -> int:
        '''Add the point and return the index'''
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(pointpair)
        self.AddPoints(new_points)

        Distance, index = self.NearestFixedPoint((float(pointpair[0]), float(pointpair[1])))
        return cast(int, index)

    def UpdatePointPair(self, index: int, pointpair: NDArray[np.floating]):
        self._points[index, :] = pointpair
        self._points = Triangulation_GPUComponent.RemoveDuplicateControlPoints(self.points)
        self.OnTransformChanged()

        Distance, nearest_index = self.NearestFixedPoint((float(pointpair[0]), float(pointpair[1])))
        return cast(int, nearest_index)

    def UpdateFixedPoints(self, index: Any, points: NDArray[np.floating]) -> int | NDArray[np.integer]:
        self._points[index, 0:2] = points
        self._points = Triangulation_GPUComponent.RemoveDuplicateControlPoints(self._points)
        self.OnFixedPointChanged()

        distance, index = self.NearestFixedPoint(points)
        return cast(int | NDArray[np.integer], index)

    def UpdateTargetPointsByIndex(self, index: Any, points: NDArray[np.floating]) -> int | \
                                                                                                           NDArray[
                                                                                                               np.integer]:
        return self.UpdateFixedPoints(index, points)

    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], points: NDArray[np.floating]) -> int | \
                                                                                                              NDArray[
                                                                                                                  np.integer]:
        Distance, index = self.NearestTargetPoint(old_points)
        return self.UpdateTargetPointsByIndex(index, points)

    def UpdateWarpedPoints(self, index: Any,
                           points: NDArray[np.floating]) -> int | NDArray[
        np.integer]:
        self._points[index, 2:4] = points
        self._points = Triangulation_GPUComponent.RemoveDuplicateControlPoints(self._points)
        self.OnWarpedPointChanged()

        distance, index = self.NearestWarpedPoint(points)
        return cast(int | NDArray[np.integer], index)

    def UpdateSourcePointsByIndex(self, index: Any, point: NDArray[np.floating]) -> int | NDArray[
        np.integer]:
        return self.UpdateWarpedPoints(index, point)

    def UpdateSourcePointsByPosition(self, old_points: NDArray[np.floating], points: NDArray[np.floating]) -> int | \
                                                                                                              NDArray[
                                                                                                                  np.integer]:
        distance, index = self.NearestSourcePoint(old_points)
        return self.UpdateSourcePointsByIndex(index, points)

    def RemovePoint(self, index: int | NDArray[np.integer]):
        if self._points.shape[0] <= 3:
            return  # Cannot have fewer than three points

        xp = cp.get_array_module(self._points)
        keep = xp.ones(self._points.shape[0], dtype=bool)
        keep[index] = False
        self._points = self._points[keep, :].copy()
        # self._points = Triangulation_GPUComponent.RemoveDuplicateControlPoints(self._points)
        self.OnTransformChanged()

    def InitializeDataStructures(self):
        '''This optional method performs all computationally intense data structure creation
           If not run these data structures should be initialized in a lazy fashion by the class
           If it is known that the data structures will be needed this function can be faster
           since computations can be performed in parallel'''

        MPool = nornir_pools.GetGlobalMultithreadingPool()
        TPool = nornir_pools.GetGlobalThreadPool()
        FixedTriTask = MPool.add_task("Fixed Triangle Delaunay", scipy.spatial.Delaunay, self.TargetPoints)
        WarpedTriTask = MPool.add_task("Warped Triangle Delaunay", scipy.spatial.Delaunay, self.SourcePoints)

        # Cannot pickle KDTree, so use Python's thread pool

        self._WarpedKDTree = build_nearest_neighbor_index(self.SourcePoints)
        self._FixedKDTree = build_nearest_neighbor_index(self.TargetPoints)

        self._fixedtri = FixedTriTask.wait_return()
        self._warpedtri = WarpedTriTask.wait_return()

    def OnPointsAddedToTransform(self, new_points):
        '''Similiar to OnTransformChanged, but optimized to handle the case of points being added'''

        self.OnTransformChanged()
        return

    #
    #         if(self._fixedtri is None or
    #            self._warpedtri is None):
    #             self.OnTransformChanged()
    #             return
    #
    #         self._WarpedKDTree = None
    #         self._FixedKDTree = None
    #         self._FixedBoundingBox = None
    #         self._MappedBoundingBox = None
    #         self._ForwardInterpolator = None
    #         self._InverseInterpolator = None
    #
    #         self._fixedtri.add_points(new_points[:,0:2])
    #         self._warpedtri.add_points(new_points[:,2:4])
    #         super(Triangulation_GPUComponent, self).OnTransformChanged()

    def OnFixedPointChanged(self):
        super(Triangulation_GPUComponent, self).OnFixedPointChanged()
        self._FixedKDTree = None
        self._fixedtri = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

        super(Triangulation_GPUComponent, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        super(Triangulation_GPUComponent, self).OnWarpedPointChanged()
        self._WarpedKDTree = None
        self._warpedtri = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        super(Triangulation_GPUComponent, self).OnTransformChanged()

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data'''
        super(Triangulation_GPUComponent, self).ClearDataStructures()
        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def NearestTargetPoint(self, points: NDArray[np.floating]):
        return self.FixedKDTree.query(points)

    def NearestFixedPoint(self, points: NDArray[np.floating] | tuple[float, float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        query_points = points if isinstance(points, np.ndarray) else np.asarray(points, dtype=np.float32)
        return self.FixedKDTree.query(query_points)

    def NearestSourcePoint(self, points: NDArray[np.floating]):
        return self.WarpedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        '''Return the warped points nearest to the query points
        :return: Distance, Index'''
        return self.WarpedKDTree.query(points)

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 2:4] = ControlPointBase.RotatePoints(self.SourcePoints, rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 0:2] = ControlPointBase.RotatePoints(self.TargetPoints, rangle, rotation_center)
        self.OnTransformChanged()

    def FlipWarped(self, flip_center=None):
        '''
        Flips the X coordinates along the vertical line passing through flip_center.  If flip_center is None the center of the bounding box of the points is used.
        '''
        if flip_center is None:
            flip_center = self.MappedBoundingBox.Center

        temp = self.points[:, 2:4] - flip_center
        temp[:, 1] = -temp[:, 1]
        temp = temp + flip_center[1]
        self.points[:, 2:4] = temp
        self.OnTransformChanged()

    def Scale(self, scalar: float):
        '''Scale both warped and control space by scalar'''
        self._points = self._points * scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar: float):
        '''Scale source space control points by scalar'''
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar: float):
        '''Scale target space control points by scalar'''
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    @property
    def MappedBoundingBoxHeight(self):
        raise DeprecationWarning("MappedBoundingBoxHeight is deprecated.  Use MappedBoundingBox.Height instead")
        return self.MappedBoundingBox.Height

    @property
    def FixedTriangles(self):
        return self.fixedtri.simplices

    @property
    def WarpedTriangles(self):
        return self.warpedtri.simplices

    def GetFixedCentroids(self, triangles=None):
        '''Centroids of fixed triangles'''
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def GetWarpedCentroids(self, triangles=None):
        '''Centroids of warped triangles'''
        if triangles is None:
            triangles = self.WarpedTriangles

        warpedTriangleVerticies = self.SourcePoints[triangles]
        swappedTriangleVerticies = np.swapaxes(warpedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def __init__(self, pointpairs: NDArray[np.floating]):
        '''
        Constructor requires at least three point pairs
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        '''
        super(Triangulation_GPUComponent, self).__init__(pointpairs)

        if self._points.shape[0] < 3:
            raise ValueError("Triangulation_GPUComponent transform must have at least three points to function")

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None
        self._scipy_forward_interp = cuLinearNDInterpolator is None
        self._scipy_inverse_interp = cuLinearNDInterpolator is None

    def ToITKString(self) -> str:
        return nornir_imageregistration.transforms.factory._MeshTransformToIRToolsString(self, self.MappedBoundingBox)

    def Load(self, TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)

    @classmethod
    def load(cls, variableParams, fixedParams):

        points = cp.array.fromiter(variableParams)
        points.reshape(variableParams / 2, 2)
