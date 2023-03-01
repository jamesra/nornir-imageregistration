"""
Created on Oct 18, 2012

@author: Jamesan
"""

import math
import numpy
from typing import Callable
from numpy.typing import NDArray
import scipy.interpolate
import nornir_imageregistration

import nornir_pools
import scipy.linalg
import scipy.spatial

from . import utils
from .triangulation import Triangulation
import nornir_imageregistration.transforms
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms.transform_type import TransformType
from nornir_imageregistration.transforms.base import IDiscreteTransform, IControlPoints, ITransformScaling, ITransform,\
    ITransformTargetRotation, ITargetSpaceControlPointEdit, IControlPoints, IGridTransform, ITriangulatedTargetSpace
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents

class GridWithRBFFallback(IDiscreteTransform, IControlPoints, ITransformScaling, ITransformTargetRotation,
                          ITargetSpaceControlPointEdit, IGridTransform, ITriangulatedTargetSpace, DefaultTransformChangeEvents):
    """
    classdocs
    """

    @property
    def type(self) -> TransformType:
        return self._discrete_transform.type

    @property
    def grid(self) -> ITKGridDivision:
        return self._discrete_transform.grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        return self._grid._grid_dims

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

    def InitializeDataStructures(self):
        self._continuous_transform.InitializeDataStructures()
        self._discrete_transform.InitializeDataStructures()

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

    def Transform(self, points: NDArray[float], **kwargs) -> NDArray[float]:
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return numpy.empty((0, 2), dtype=points.dtype)

        TransformedPoints = self._discrete_transform.Transform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (GoodPoints, InvalidIndicies) = utils.InvalidIndicies(TransformedPoints)

        if len(InvalidIndicies) == 0:
            return TransformedPoints
        else:
            if len(points) > 1:
                # print InvalidIndicies;
                BadPoints = points[InvalidIndicies]
            else:
                BadPoints = points

        BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32)
        if not (BadPoints.dtype == numpy.float32 or BadPoints.dtype == numpy.float64):
            BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32)
            
        FixedPoints = self._continuous_transform.Transform(BadPoints)

        TransformedPoints[InvalidIndicies] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points: NDArray[float], **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return numpy.empty((0, 2), dtype=points.dtype)

        TransformedPoints = self._discrete_transform.InverseTransform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (GoodPoints, InvalidIndicies) = utils.InvalidIndicies(TransformedPoints)

        if len(InvalidIndicies) == 0:
            return TransformedPoints
        else:
            if points.ndim > 1:
                BadPoints = points[InvalidIndicies]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        if not (BadPoints.dtype == numpy.float32 or BadPoints.dtype == numpy.float64):
            BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32)

        FixedPoints = self._continuous_transform.InverseTransform(BadPoints)

        TransformedPoints[InvalidIndicies] = FixedPoints
        return TransformedPoints

    def __init__(self,
                 grid: ITKGridDivision):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(GridWithRBFFallback, self).__init__()

        self._discrete_transform = nornir_imageregistration.transforms.GridTransform(grid)
        self._continuous_transform = nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection(grid.SourcePoints, grid.TargetPoints)

    def AddTransform(self, mappedTransform: IControlPoints, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        return nornir_imageregistration.transforms.AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance, create_copy=create_copy)
        
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
    def SourcePoints(self) -> NDArray[float]:
        return self._discrete_transform.SourcePoints

    @property
    def TargetPoints(self) -> NDArray[float]:
        return self._discrete_transform.TargetPoints

    @property
    def points(self) -> NDArray[float]:
        return self._discrete_transform.points

    @property
    def NumControlPoints(self) -> int:
        return self._discrete_transform.NumControlPoints

    def NearestTargetPoint(self, points: NDArray[float]) -> tuple((float | NDArray[float], int | NDArray[int])):
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self._discrete_transform.NearestTargetPoint(points)

    def NearestFixedPoint(self, points: NDArray[float]) -> tuple((float | NDArray[float], int | NDArray[int])):
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self._discrete_transform.NearestFixedPoint(points)

    def NearestSourcePoint(self, points: NDArray[float]) -> tuple((float | NDArray[float], int | NDArray[int])):
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        return self._discrete_transform.NearestSourcePoint(points)

    def NearestWarpedPoint(self, points: NDArray[float]) -> tuple((float | NDArray[float], int | NDArray[int])):
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        return self._discrete_transform.NearestWarpedPoint(points)

    @property
    def fixedtri(self) -> scipy.spatial.Delaunay:
        return self._discrete_transform.FixedTriangles

    @property
    def FixedTriangles(self) -> scipy.spatial.Delaunay:
        return self._discrete_transform.FixedTriangles
    
    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self._discrete_transform.target_space_trianglulation

    def TranslateFixed(self, offset: NDArray[float]):
        '''Translate all fixed points by the specified amount'''

        self._discrete_transform.TranslateFixed(offset)
        self._continuous_transform.TranslateFixed(offset)
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[float]):
        '''Translate all warped points by the specified amount'''
        self._discrete_transform.TranslateWarped(offset)
        self._continuous_transform.TranslateWarped(offset)
        self.OnWarpedPointChanged()

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

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[float] | None):
        '''Rotate all warped points about a center by a given angle'''
        if rotation_center is None:
            rotation_center = self.FixedBoundingBox.Center

        self._discrete_transform.RotateTargetPoints(rangle, rotation_center)
        self._continuous_transform = nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection(
            self._discrete_transform.SourcePoints, self._discrete_transform.TargetPoints)

        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[int], point: NDArray[float] | None) -> int | NDArray[int]:
        #Using this may cause errors since the discrete and continuous transforms are not guaranteed to use the same index
        result = self._discrete_transform.UpdateTargetPointsByIndex(index, point)
        self._continuous_transform.UpdateTargetPointsByIndex(index, point)
        self.OnTransformChanged()
        return result

    def UpdateTargetPointsByPosition(self, index: NDArray[float], point: NDArray[float] | None) -> int | NDArray[int]:
        result = self._discrete_transform.UpdateTargetPointsByPosition(index, point)
        self._continuous_transform.UpdateTargetPointsByPosition(index, point)
        self.OnTransformChanged()
        return result

if __name__ == '__main__':
    p = numpy.array([[0, 0, 0, 0],
                  [0, 10, 0, -10],
                  [10, 0, -10, 0],
                  [10, 10, -10, -10]])

    (Fixed, Moving) = numpy.hsplit(p, 2)
    T = OneWayRBFWithLinearCorrection(Fixed, Moving)

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


