'''
Created on Oct 18, 2012

@author: Jamesan
'''

import logging

import numpy as np
from numpy.typing import NDArray

import nornir_pools
import nornir_imageregistration 
from . import utils, TransformType
from .base import IDiscreteTransform, ITransformScaling, ITransformRelativeScaling, ITransformTranslation, \
 IControlPointEdit, ITransformSourceRotation, ITransformTargetRotation, ITriangulatedTargetSpace, \
    ITriangulatedSourceSpace, IControlPointAddRemove

import scipy
from scipy.interpolate import LinearNDInterpolator
import scipy.spatial

from .addition import AddTransforms
from .controlpointbase import ControlPointBase


class Triangulation(ITransformScaling, ITransformRelativeScaling, ITransformTranslation, IControlPointEdit, ITransformSourceRotation,
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
            self._WarpedKDTree = scipy.spatial.cKDTree(self.SourcePoints)

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = scipy.spatial.cKDTree(self.TargetPoints)

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
    def target_space_trianglulation(self)->scipy.spatial.Delaunay:
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
            #self._ForwardInterpolator = CloughTocher2DInterpolator(self.warpedtri, self.TargetPoints)
            self._ForwardInterpolator = LinearNDInterpolator(self.warpedtri, self.TargetPoints)

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            #self._InverseInterpolator = CloughTocher2DInterpolator(self.fixedtri, self.SourcePoints)
            self._InverseInterpolator = LinearNDInterpolator(self.fixedtri, self.SourcePoints)

        return self._InverseInterpolator

    def AddTransform(self, mappedTransform, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space''' 
        return AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance, create_copy=create_copy)

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try: 
            transPoints = self.ForwardInterpolator(points).astype(np.float32, copy=False)
        except Exception as e: # This is usually a scipy.spatial._qhull.QhullError:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            self._ForwardInterpolator = None

            #This was added for the case where all points in the triangulation are colinear.
            transPoints = np.empty(points.shape)
            transPoints[:] = np.NaN

        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.InverseInterpolator(points).astype(np.float32, copy=False)
        except Exception as e: # This is usually a scipy.spatial._qhull.QhullError:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None
            self._InverseInterpolator = None

            # This was added for the case where all points in the triangulation are colinear.
            transPoints = np.empty(points.shape)
            transPoints[:] = np.NaN

        return transPoints

    def AddPoints(self, new_points: NDArray[float]):
        '''Add the point and return the index'''
        numPts = self.NumControlPoints
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(new_points)

        duplicates = self.FindDuplicateFixedPoints(new_points[:, 0:2])
        new_points = new_points[~duplicates, :]

        if new_points.shape[0] == 0:
            return

        self._points = np.append(self.points, new_points, 0)
        # self._points = Triangulation.RemoveDuplicates(self._points)

        # We won't see a change in the number of points if the new point was a duplicate
        if self.NumControlPoints != numPts:
            self.OnPointsAddedToTransform(new_points)

        return

    def AddPoint(self, pointpair: NDArray[float]) -> int:
        '''Add the point and return the index'''
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(pointpair)
        self.AddPoints(new_points)

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdatePointPair(self, index: int, pointpair: NDArray[float]):
        self._points[index, :] = pointpair
        self._points = Triangulation.RemoveDuplicateControlPoints(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdateFixedPoints(self, index: int, points: NDArray[float]):
        self._points[index, 0:2] = points
        self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
        self.OnFixedPointChanged()

        distance, index = self.NearestFixedPoint(points)
        return index

    def UpdateTargetPointsByIndex(self, index: int | NDArray[int], points: NDArray[float]) -> int | NDArray[int]:
        return self.UpdateFixedPoints(index, points)

    def UpdateTargetPointsByPosition(self, old_points: NDArray[float], points: NDArray[float]) -> int | NDArray[int]:
        Distance, index = self.NearestTargetPoint(old_points)
        return self.UpdateTargetPointsByIndex(index, points)

    def UpdateWarpedPoints(self, index: int | NDArray[int] | NDArray[float], points: NDArray[float]) -> int | NDArray[int]:
        self._points[index, 2:4] = points
        self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
        self.OnWarpedPointChanged()

        distance, index = self.NearestWarpedPoint(points)
        return index

    def UpdateSourcePointsByIndex(self, index: int | NDArray[int], point: NDArray[float]) -> int | NDArray[int]:
        return self.UpdateWarpedPoints(index, point)

    def UpdateSourcePointsByPosition(self, old_points: NDArray[float], points: NDArray[float]) -> int | NDArray[int]:
        distance, index = self.NearestSourcePoint(old_points)
        return self.UpdateSourcePointsByIndex(index, points)

    def RemovePoint(self, index: int | NDArray[int]):
        if self._points.shape[0] <= 3:
            return  # Cannot have fewer than three points

        self._points = np.delete(self._points, index, 0)
        #self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
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

        FixedKDTask = TPool.add_task("Fixed KDTree", scipy.spatial.cKDTree, self.TargetPoints)
        # WarpedKDTask = TPool.add_task("Warped KDTree", KDTree, self.SourcePoints)

        self._WarpedKDTree = scipy.spatial.cKDTree(self.SourcePoints)

        # MPool.wait_completion()

        self._FixedKDTree = FixedKDTask.wait_return()

        # self._FixedKDTree = cKDTree(self.TargetPoints)

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
           Clear out our data structures so we do not use bad data'''
        super(Triangulation, self).ClearDataStructures()
        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def NearestTargetPoint(self, points: NDArray[float]):
        return self.FixedKDTree.query(points)

    def NearestFixedPoint(self, points: NDArray[float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestSourcePoint(self, points: NDArray[float]):
        return self.WarpedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[float]):
        '''Return the warped points nearest to the query points
        :return: Distance, Index'''
        return self.WarpedKDTree.query(points)

    def TranslateFixed(self, offset: NDArray[float]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[float]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[float] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 2:4] = ControlPointBase.RotatePoints(self.SourcePoints, rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[float] | None):
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
        temp[:,1] = -temp[:,1] 
        temp = temp + flip_center[1]
        self.points[:,2:4] = temp
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
        return self.fixedtri.vertices

    @property
    def WarpedTriangles(self):
        return self.warpedtri.vertices

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

    def __init__(self, pointpairs: NDArray[float]):
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
            self._WarpedKDTree = scipy.spatial.cKDTree(self.SourcePoints)

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = scipy.spatial.cKDTree(self.TargetPoints)

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

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            # self._InverseInterpolator = CloughTocher2DInterpolator(self.fixedtri, self.SourcePoints)
            self._InverseInterpolator = LinearNDInterpolator(self.fixedtri, self.SourcePoints)

        return self._InverseInterpolator

    def AddTransform(self, mappedTransform, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance, create_copy=create_copy)

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        # Convert points back to numpy for ForwardInterpolator function
        points = points.get()
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.ForwardInterpolator(points).astype(np.float32, copy=False)
        except Exception as e:  # This is usually a scipy.spatial._qhull.QhullError:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            self._ForwardInterpolator = None

            # This was added for the case where all points in the triangulation are colinear.
            transPoints = np.empty(points.shape)
            transPoints[:] = np.NaN

        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        # Convert points to numpy for InverseInterpolator function
        points = points.get()
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
            transPoints[:] = np.NaN

        return transPoints

    def AddPoints(self, new_points: NDArray[float]):
        '''Add the point and return the index'''
        numPts = self.NumControlPoints
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(new_points)

        duplicates = self.FindDuplicateFixedPoints(new_points[:, 0:2])
        new_points = new_points[~duplicates, :]

        if new_points.shape[0] == 0:
            return

        self._points = np.append(self.points, new_points, 0)
        # self._points = Triangulation.RemoveDuplicates(self._points)

        # We won't see a change in the number of points if the new point was a duplicate
        if self.NumControlPoints != numPts:
            self.OnPointsAddedToTransform(new_points)

        return

    def AddPoint(self, pointpair: NDArray[float]) -> int:
        '''Add the point and return the index'''
        new_points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(pointpair)
        self.AddPoints(new_points)

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdatePointPair(self, index: int, pointpair: NDArray[float]):
        self._points[index, :] = pointpair
        self._points = Triangulation.RemoveDuplicateControlPoints(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdateFixedPoints(self, index: int, points: NDArray[float]):
        self._points[index, 0:2] = points
        self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
        self.OnFixedPointChanged()

        distance, index = self.NearestFixedPoint(points)
        return index

    def UpdateTargetPointsByIndex(self, index: int | NDArray[int], points: NDArray[float]) -> int | NDArray[int]:
        return self.UpdateFixedPoints(index, points)

    def UpdateTargetPointsByPosition(self, old_points: NDArray[float], points: NDArray[float]) -> int | NDArray[int]:
        Distance, index = self.NearestTargetPoint(old_points)
        return self.UpdateTargetPointsByIndex(index, points)

    def UpdateWarpedPoints(self, index: int | NDArray[int] | NDArray[float], points: NDArray[float]) -> int | NDArray[
        int]:
        self._points[index, 2:4] = points
        self._points = Triangulation.RemoveDuplicateControlPoints(self._points)
        self.OnWarpedPointChanged()

        distance, index = self.NearestWarpedPoint(points)
        return index

    def UpdateSourcePointsByIndex(self, index: int | NDArray[int], point: NDArray[float]) -> int | NDArray[int]:
        return self.UpdateWarpedPoints(index, point)

    def UpdateSourcePointsByPosition(self, old_points: NDArray[float], points: NDArray[float]) -> int | NDArray[int]:
        distance, index = self.NearestSourcePoint(old_points)
        return self.UpdateSourcePointsByIndex(index, points)

    def RemovePoint(self, index: int | NDArray[int]):
        if self._points.shape[0] <= 3:
            return  # Cannot have fewer than three points

        self._points = np.delete(self._points, index, 0)
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

        FixedKDTask = TPool.add_task("Fixed KDTree", scipy.spatial.cKDTree, self.TargetPoints)
        # WarpedKDTask = TPool.add_task("Warped KDTree", KDTree, self.SourcePoints)

        self._WarpedKDTree = scipy.spatial.cKDTree(self.SourcePoints)

        # MPool.wait_completion()

        self._FixedKDTree = FixedKDTask.wait_return()

        # self._FixedKDTree = cKDTree(self.TargetPoints)

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

    def NearestTargetPoint(self, points: NDArray[float]):
        return self.FixedKDTree.query(points)

    def NearestFixedPoint(self, points: NDArray[float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestSourcePoint(self, points: NDArray[float]):
        return self.WarpedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[float]):
        '''Return the warped points nearest to the query points
        :return: Distance, Index'''
        return self.WarpedKDTree.query(points)

    def TranslateFixed(self, offset: NDArray[float]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[float]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[float] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 2:4] = ControlPointBase.RotatePoints(self.SourcePoints, rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[float] | None):
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
        return self.fixedtri.vertices

    @property
    def WarpedTriangles(self):
        return self.warpedtri.vertices

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

    def __init__(self, pointpairs: NDArray[float]):
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

    def ToITKString(self) -> str:
        return nornir_imageregistration.transforms.factory._MeshTransformToIRToolsString(self, self.MappedBoundingBox)

    def Load(self, TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)

    @classmethod
    def load(cls, variableParams, fixedParams):

        points = np.array.fromiter(variableParams)
        points.reshape(variableParams / 2, 2)

        self._FixedKDTree = None

    def ToITKString(self) -> str:
        return nornir_imageregistration.transforms.factory._MeshTransformToIRToolsString(self, self.MappedBoundingBox)

    def Load(self, TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)

    @classmethod
    def load(cls, variableParams, fixedParams):

        points = np.array.fromiter(variableParams)
        points.reshape(variableParams / 2, 2)
