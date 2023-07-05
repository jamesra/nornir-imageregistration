"""
Created on Oct 18, 2012

@author: Jamesan
"""

import numpy
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.transforms.base import IDiscreteTransform, IControlPoints, ITransformScaling, ITransform, \
    ITransformRelativeScaling
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents
from nornir_imageregistration.transforms.transform_type import TransformType
from . import utils


class DiscreteWithContinuousFallback(IDiscreteTransform, IControlPoints, ITransformScaling, ITransformRelativeScaling,
                                     DefaultTransformChangeEvents):
    """
    classdocs
    """

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __getstate__(self):
        odict = super(DiscreteWithContinuousFallback, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance
        return odict

    def __setstate__(self, dictionary):
        super(DiscreteWithContinuousFallback, self).__setstate__(dictionary)

    def InitializeDataStructures(self):
        self._continous_transform.InitializeDataStructures()
        self._discrete_transform.InitializeDataStructures()

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""
        self._continous_transform.ClearDataStructures()
        self._discrete_transform.ClearDataStructures()

    def OnFixedPointChanged(self):
        self._continous_transform.OnFixedPointChanged()
        self._discrete_transform.OnFixedPointChanged()

    def OnWarpedPointChanged(self):
        self._continous_transform.OnWarpedPointChanged()
        self._discrete_transform.OnWarpedPointChanged()

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

    def __init__(self, continuous_transform: ITransform, discrete_transform: IDiscreteTransform):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX] 
        """
        if not isinstance(discrete_transform, IControlPoints):
            raise ValueError("Discrete transform should implement IControlPoints")

        if not isinstance(discrete_transform, IDiscreteTransform):
            raise ValueError("Discrete transform should implement IDiscreteTransform")

        if not (isinstance(discrete_transform, ITransformRelativeScaling) and isinstance(continuous_transform,
                                                                                         ITransformRelativeScaling)):
            raise ValueError("Discrete and continuous transform should implement ITransformRelativeScaling")

        if not (isinstance(discrete_transform, ITransformScaling) and isinstance(continuous_transform,
                                                                                 ITransformScaling)):
            raise ValueError("Discrete and continuous transform should implement ITransformScaling")

        super(DiscreteWithContinuousFallback, self).__init__()

        self._continuous_transform = continuous_transform
        self._discrete_transform = discrete_transform

    @staticmethod
    def Load(TransformString: str, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)

    @property
    def MappedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """Bounding box of mapped space points"""
        return self._discrete_transform.MappedBoundingBox

    @property
    def FixedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._discrete_transform.FixedBoundingBox

    @property
    def SourcePoints(self) -> NDArray:
        return self._discrete_transform.SourcePoints

    @property
    def TargetPoints(self) -> NDArray:
        return self._discrete_transform.TargetPoints

    @property
    def points(self) -> NDArray:
        return self._discrete_transform.points

    def NearestFixedPoint(self, points: NDArray) -> tuple((float | NDArray[float], int | NDArray[int])):
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self._discrete_transform.NearestFixedPoint(points)

    def NearestWarpedPoint(self, points: NDArray) -> tuple((float | NDArray[float], int | NDArray[int])):
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        return self._discrete_transform.NearestWarpedPoint(points)

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
