'''
Created on September 2023

@author: clementsan
'''

import logging

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from . import utils, TransformType
from .base import IDiscreteTransform, ITransformScaling, ITransformRelativeScaling, ITransformTranslation, \
 IControlPointEdit, ITransformSourceRotation, ITransformTargetRotation, ITriangulatedTargetSpace, \
    ITriangulatedSourceSpace, IControlPointAddRemove

from scipy.interpolate import RBFInterpolator as RBFInterpolator

import cupy as cp
from cupyx.scipy.interpolate import RBFInterpolator as cuRBFInterpolator


from .addition import AddTransforms
from .controlpointbase import ControlPointBase, ControlPointBase_GPUComponent


class Landmark_GPU(ITransformScaling, ITransformRelativeScaling, ITransformTranslation, IControlPointEdit,
                        ITransformSourceRotation, ITransformTargetRotation,
                        IControlPointAddRemove, ControlPointBase_GPUComponent):
    '''
    Landmark transform has an nx4 array of points, with rows organized as
    [controlx controly warpedx warpedy]
    '''

    @property
    def type(self) -> TransformType:
        return nornir_imageregistration.transforms.transform_type.TransformType.MESH

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            self._InverseInterpolator = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)

        return self._InverseInterpolator

    def AddTransform(self, mappedTransform, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance, create_copy=create_copy)

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        # transPoints = self.ForwardInterpolator(points).astype(np.float32, copy=False)
        transPoints = self.ForwardInterpolator(points)
        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        # transPoints = self.InverseInterpolator(points).astype(np.float32, copy=False)
        transPoints = self.InverseInterpolator(points)
        return transPoints

    def AddPoints(self, new_points: NDArray[float]):
        '''Add the point and return the index'''
        numPts = self.NumControlPoints
        new_points = nornir_imageregistration.EnsurePointsAre4xN_CuPyArray(new_points)

        duplicates = self.FindDuplicateFixedPoints(new_points[:, 0:2])
        new_points = new_points[~duplicates, :]

        if new_points.shape[0] == 0:
            return

        self._points = cp.append(self.points, new_points, 0)
        # self._points = Landmark_GPU.RemoveDuplicates(self._points)

        # We won't see a change in the number of points if the new point was a duplicate
        if self.NumControlPoints != numPts:
            self.OnPointsAddedToTransform(new_points)

        return

    def AddPoint(self, pointpair: NDArray[float]) -> int:
        '''Add the point and return the index'''
        new_points = nornir_imageregistration.EnsurePointsAre4xN_CuPyArray(pointpair)
        self.AddPoints(new_points)

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdatePointPair(self, index: int, pointpair: NDArray[float]):
        self._points[index, :] = pointpair
        self._points = Landmark_GPU.RemoveDuplicateControlPoints(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdateFixedPoints(self, index: int, points: NDArray[float]):
        self._points[index, 0:2] = points
        self._points = Landmark_GPU.RemoveDuplicateControlPoints(self._points)
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
        self._points = Landmark_GPU.RemoveDuplicateControlPoints(self._points)
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
        # self._points = Landmark_GPU.RemoveDuplicateControlPoints(self._points)
        self.OnTransformChanged()

    def InitializeDataStructures(self):
        '''This optional method performs all computationally intense data structure creation
           If not run these data structures should be initialized in a lazy fashion by the class
           If it is known that the data structures will be needed this function can be faster
           since computations can be performed in parallel'''

        self._ForwardInterpolator = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)
        self._InverseInterpolator = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)

    def OnPointsAddedToTransform(self, new_points):
        '''Similiar to OnTransformChanged, but optimized to handle the case of points being added'''

        self.OnTransformChanged()
        return

    def OnFixedPointChanged(self):
        super(Landmark_GPU, self).OnFixedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

        super(Landmark_GPU, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        super(Landmark_GPU, self).OnWarpedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        super(Landmark_GPU, self).OnTransformChanged()

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data'''
        super(Landmark_GPU, self).ClearDataStructures()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def NearestTargetPoint(self, points: NDArray[float]):
        return None

    def NearestFixedPoint(self, points: NDArray[float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return None

    def NearestSourcePoint(self, points: NDArray[float]):
        return None

    def NearestWarpedPoint(self, points: NDArray[float]):
        '''Return the warped points nearest to the query points
        :return: Distance, Index'''
        return None

    def TranslateFixed(self, offset: NDArray[float]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + cp.asarray(offset)
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[float]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + cp.asarray(offset)
        self.OnWarpedPointChanged()

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[float] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 2:4] = ControlPointBase_GPUComponent.RotatePoints(self.SourcePoints, rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[float] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 0:2] = ControlPointBase_GPUComponent.RotatePoints(self.TargetPoints, rangle, rotation_center)
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

    def __init__(self, pointpairs: NDArray[float]):
        '''
        Constructor requires at least three point pairs
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        '''
        super(Landmark_GPU, self).__init__(pointpairs)

        if self._points.shape[0] < 3:
            raise ValueError("Landmark_GPU transform must have at least three points to function")

        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def ToITKString(self) -> str:
        return nornir_imageregistration.transforms.factory._MeshTransformToIRToolsString(self, self.MappedBoundingBox)

    def Load(self, TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing, use_cp=True)

    @classmethod
    def load(cls, variableParams, fixedParams):

        points = cp.array.fromiter(variableParams)
        points.reshape(variableParams / 2, 2)


class Landmark_CPU(ITransformScaling, ITransformRelativeScaling, ITransformTranslation, IControlPointEdit,
                        ITransformSourceRotation, ITransformTargetRotation,
                        IControlPointAddRemove, ControlPointBase):
    '''
    Landmark transform has an nx4 array of points, with rows organized as
    [controlx controly warpedx warpedy]
    '''

    @property
    def type(self) -> TransformType:
        return nornir_imageregistration.transforms.transform_type.TransformType.MESH

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = RBFInterpolator(self.SourcePoints, self.TargetPoints)

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            self._InverseInterpolator = RBFInterpolator(self.TargetPoints, self.SourcePoints)

        return self._InverseInterpolator

    def AddTransform(self, mappedTransform, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance, create_copy=create_copy)

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        # transPoints = self.ForwardInterpolator(points).astype(np.float32, copy=False)
        transPoints = self.ForwardInterpolator(points)
        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        # transPoints = self.InverseInterpolator(points).astype(np.float32, copy=False)
        transPoints = self.InverseInterpolator(points)
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
        # self._points = Landmark_CPU.RemoveDuplicates(self._points)

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
        self._points = Landmark_CPU.RemoveDuplicateControlPoints(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdateFixedPoints(self, index: int, points: NDArray[float]):
        self._points[index, 0:2] = points
        self._points = Landmark_CPU.RemoveDuplicateControlPoints(self._points)
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
        self._points = Landmark_CPU.RemoveDuplicateControlPoints(self._points)
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
        # self._points = Landmark_CPU.RemoveDuplicateControlPoints(self._points)
        self.OnTransformChanged()

    def InitializeDataStructures(self):
        '''This optional method performs all computationally intense data structure creation
           If not run these data structures should be initialized in a lazy fashion by the class
           If it is known that the data structures will be needed this function can be faster
           since computations can be performed in parallel'''

        self._ForwardInterpolator = RBFInterpolator(self.SourcePoints, self.TargetPoints)
        self._InverseInterpolator = RBFInterpolator(self.TargetPoints, self.SourcePoints)

    def OnPointsAddedToTransform(self, new_points):
        '''Similiar to OnTransformChanged, but optimized to handle the case of points being added'''

        self.OnTransformChanged()
        return

    def OnFixedPointChanged(self):
        super(Landmark_CPU, self).OnFixedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

        super(Landmark_CPU, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        super(Landmark_CPU, self).OnWarpedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        super(Landmark_CPU, self).OnTransformChanged()

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data'''
        super(Landmark_CPU, self).ClearDataStructures()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def NearestTargetPoint(self, points: NDArray[float]):
        return None

    def NearestFixedPoint(self, points: NDArray[float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return None

    def NearestSourcePoint(self, points: NDArray[float]):
        return None

    def NearestWarpedPoint(self, points: NDArray[float]):
        '''Return the warped points nearest to the query points
        :return: Distance, Index'''
        return None

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

    def __init__(self, pointpairs: NDArray[float]):
        '''
        Constructor requires at least three point pairs
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        '''
        super(Landmark_CPU, self).__init__(pointpairs)

        if self._points.shape[0] < 3:
            raise ValueError("Landmark_CPU transform must have at least three points to function")

        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def ToITKString(self) -> str:
        return nornir_imageregistration.transforms.factory._MeshTransformToIRToolsString(self, self.MappedBoundingBox)

    def Load(self, TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing, use_cp=False)

    @classmethod
    def load(cls, variableParams, fixedParams):

        points = np.array.fromiter(variableParams)
        points.reshape(variableParams / 2, 2)
