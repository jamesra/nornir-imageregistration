from typing import Callable
import numpy
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.transforms.base import ITransform, IDiscreteTransform, IControlPoints,  \
    ITransformChangeEvents,\
    ITransformTranslation, ITransformScaling, ITransformTargetRotation, \
    ITransformSourceRotation, IControlPointEdit
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents
from nornir_imageregistration.transforms.one_way_rbftransform import OneWayRBFWithLinearCorrection
from nornir_imageregistration.transforms.transform_type import TransformType



class TwoWayRBFWithLinearCorrection(ITransform, IControlPoints, ITransformScaling, ITransformTargetRotation,
                                    ITransformSourceRotation, IControlPointEdit, DefaultTransformChangeEvents):
    def __init__(self, WarpedPoints: NDArray[float], FixedPoints: NDArray[float], BasisFunction: Callable[[float], float] | None = None):
        self._forward_rbf = OneWayRBFWithLinearCorrection(WarpedPoints=WarpedPoints, FixedPoints=FixedPoints, BasisFunction=BasisFunction)
        self._inverse_rbf = OneWayRBFWithLinearCorrection(WarpedPoints=FixedPoints, FixedPoints=WarpedPoints, BasisFunction=BasisFunction)

    def _reset_inverse_transform(self):
        self._inverse_rbf = OneWayRBFWithLinearCorrection(WarpedPoints=self._forward_rbf.SourcePoints,
                                                          FixedPoints=self._forward_rbf.TargetPoints,
                                                          BasisFunction=self._forward_rbf.BasisFunction)

    def Transform(self, point: NDArray[float], **kwargs) -> NDArray:
        return self._forward_rbf.Transform(point, **kwargs)

    def InverseTransform(self, point: NDArray[float], **kwargs) -> NDArray:
        return self._inverse_rbf.Transform(point, **kwargs)

    def Load(self, TransformString: str, pixelSpacing=None):
        raise NotImplementedError

    @property
    def type(self) -> TransformType:
        return nornir_imageregistration.transforms.transform_type.TransformType.RBF

    @property
    def SourcePoints(self) -> NDArray:
        return self._forward_rbf.SourcePoints

    @property
    def TargetPoints(self) -> NDArray:
        return self._forward_rbf.TargetPoints

    @property
    def points(self) -> NDArray:
        return self._forward_rbf.points

    @property
    def NumControlPoints(self) -> int:
        return self._forward_rbf.NumControlPoints

    @property
    def MappedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """Bounding box of mapped space points"""
        return self._forward_rbf.MappedBoundingBox

    @property
    def FixedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self._forward_rbf.FixedBoundingBox

    def NearestFixedPoint(self, points: NDArray) -> tuple((float | NDArray[float], int | NDArray[int])):
        return self._forward_rbf.NearestFixedPoint(points)

    def NearestWarpedPoint(self, points: NDArray) -> tuple((float | NDArray[float], int | NDArray[int])):
        return self._forward_rbf.NearestWarpedPoint(points)

    def Scale(self, scalar: float):
        '''Scale both warped and control space by scalar'''
        self._forward_rbf.Scale(scalar)
        self._inverse_rbf.Scale(scalar)
        self.OnTransformChanged()

    def ScaleWarped(self, scalar: float):
        '''Scale source space control points by scalar'''
        self._forward_rbf.ScaleWarped(scalar)
        self._inverse_rbf.ScaleFixed(scalar)
        self.OnTransformChanged()

    def ScaleFixed(self, scalar: float):
        '''Scale target space control points by scalar'''
        self._forward_rbf.ScaleFixed(scalar)
        self._inverse_rbf.ScaleWarped(scalar)
        self.OnTransformChanged()

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[float] | None):
        """Rotate all warped points by the specified amount
        If rotation center is not specified the transform chooses"""
        self._forward_rbf.RotateSourcePoints(rangle, rotation_center)
        self._inverse_rbf.RotateTargetPoints(rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[float] | None):
        """Rotate all fixed points by the specified amount
        If rotation center is not specified the transform chooses"""
        self._forward_rbf.RotateTargetPoints(rangle, rotation_center)
        self._inverse_rbf.RotateSourcePoints(rangle, rotation_center)
        self.OnTransformChanged()

    def AddPoint(self, pointpair: NDArray[float]):
        self._forward_rbf.AddPoint(pointpair)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def AddPoints(self, new_points: NDArray[float]):
        self._forward_rbf.AddPoints(new_points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdatePointPair(self, index: int, pointpair: NDArray[float]):
        self._forward_rbf.UpdatePointPair(index, pointpair)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateSourcePoints(self, index: int | NDArray[int], points: NDArray[float]):
        self._forward_rbf.UpdateSourcePoints(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateTargetPoints(self, index: int | NDArray[int], points: NDArray[float]):
        self._forward_rbf.UpdateTargetPoints(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def RemovePoint(self, index: int):
        self._forward_rbf.RemovePoint(index)
        self._inverse_rbf.RemovePoint(index)
        self.OnTransformChanged()