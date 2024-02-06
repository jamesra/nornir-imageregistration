from typing import Callable

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.transforms.base import ITransform, IControlPoints, \
    ITransformScaling, ITransformRelativeScaling, ITransformTargetRotation, \
    ITransformSourceRotation, IControlPointEdit
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents
from nornir_imageregistration.transforms.one_way_rbftransform import OneWayRBFWithLinearCorrection, \
    OneWayRBFWithLinearCorrection_GPUComponent
from nornir_imageregistration.transforms.transform_type import TransformType


class TwoWayRBFWithLinearCorrection(ITransform, IControlPoints, ITransformScaling, ITransformRelativeScaling,
                                    ITransformTargetRotation,
                                    ITransformSourceRotation, IControlPointEdit, DefaultTransformChangeEvents):
    def __init__(self, WarpedPoints: NDArray[np.floating], FixedPoints: NDArray[np.floating],
                 BasisFunction: Callable[[float], float] | None = None):
        super(TwoWayRBFWithLinearCorrection, self).__init__()
        self._forward_rbf = OneWayRBFWithLinearCorrection(WarpedPoints=WarpedPoints, FixedPoints=FixedPoints,
                                                          BasisFunction=BasisFunction)
        self._inverse_rbf = OneWayRBFWithLinearCorrection(WarpedPoints=FixedPoints, FixedPoints=WarpedPoints,
                                                          BasisFunction=BasisFunction)

    def __getstate__(self):
        odict = super(TwoWayRBFWithLinearCorrection, self).__getstate__()
        odict['_forward_rbf'] = self._forward_rbf
        odict['_inverse_rbf'] = self._inverse_rbf
        return odict

    def __setstate__(self, dictionary):
        super(TwoWayRBFWithLinearCorrection, self).__setstate__(dictionary)
        self._forward_rbf = dictionary['_forward_rbf']
        self._inverse_rbf = dictionary['_inverse_rbf']

    def _reset_inverse_transform(self):
        self._inverse_rbf = OneWayRBFWithLinearCorrection(WarpedPoints=self._forward_rbf.SourcePoints,
                                                          FixedPoints=self._forward_rbf.TargetPoints,
                                                          BasisFunction=self._forward_rbf.BasisFunction)

    def Transform(self, point: NDArray[np.floating], **kwargs) -> NDArray:
        return self._forward_rbf.Transform(point, **kwargs)

    def InverseTransform(self, point: NDArray[np.floating], **kwargs) -> NDArray:
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

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in target space'''
        return self._forward_rbf.GetPointPairsInTargetRect(bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in source space'''
        return self._forward_rbf.GetPointPairsInSourceRect(bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return self._forward_rbf.PointPairsToWarpedPoints(points)

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return self._forward_rbf.PointPairsToTargetPoints(points)

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

    def NearestTargetPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        return self._forward_rbf.NearestTargetPoint(points)

    def NearestFixedPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        return self._forward_rbf.NearestFixedPoint(points)

    def NearestSourcePoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        return self._forward_rbf.NearestSourcePoint(points)

    def NearestWarpedPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
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

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all warped points by the specified amount
        If rotation center is not specified the transform chooses"""
        self._forward_rbf.RotateSourcePoints(rangle, rotation_center)
        self._inverse_rbf.RotateTargetPoints(rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all fixed points by the specified amount
        If rotation center is not specified the transform chooses"""
        self._forward_rbf.RotateTargetPoints(rangle, rotation_center)
        self._inverse_rbf.RotateSourcePoints(rangle, rotation_center)
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._forward_rbf.TranslateFixed(offset)
        self._inverse_rbf.TranslateWarped(offset)
        self.OnTransformChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._forward_rbf.TranslateWarped(offset)
        self._inverse_rbf.TranslateFixed(offset)
        self.OnTransformChanged()

    def AddPoint(self, pointpair: NDArray[np.floating]):
        self._forward_rbf.AddPoint(pointpair)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def AddPoints(self, new_points: NDArray[np.floating]):
        self._forward_rbf.AddPoints(new_points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdatePointPair(self, index: int, pointpair: NDArray[np.floating]):
        self._forward_rbf.UpdatePointPair(index, pointpair)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateSourcePointsByIndex(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateSourcePointsByIndex(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateSourcePointsByPosition(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateSourcePointsByPosition(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateTargetPointsByIndex(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateTargetPointsByPosition(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateTargetPointsByPosition(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def RemovePoint(self, index: int):
        self._forward_rbf.RemovePoint(index)
        self._inverse_rbf.RemovePoint(index)
        self.OnTransformChanged()

    def OnFixedPointChanged(self):
        self._forward_rbf.OnFixedPointChanged()
        self._inverse_rbf.OnWarpedPointChanged()
        self.OnTransformChanged()

    def OnWarpedPointChanged(self):
        self._forward_rbf.OnWarpedPointChanged()
        self._inverse_rbf.OnFixedPointChanged()
        self.OnTransformChanged()

    def InitializeDataStructures(self):
        self._forward_rbf.InitializeDataStructures()
        self._inverse_rbf.InitializeDataStructures()


class TwoWayRBFWithLinearCorrection_GPUComponent(ITransform, IControlPoints, ITransformScaling,
                                                 ITransformRelativeScaling, ITransformTargetRotation,
                                                 ITransformSourceRotation, IControlPointEdit,
                                                 DefaultTransformChangeEvents):
    def __init__(self, WarpedPoints: NDArray[np.floating], FixedPoints: NDArray[np.floating],
                 BasisFunction: Callable[[float], float] | None = None):
        super(TwoWayRBFWithLinearCorrection_GPUComponent, self).__init__()
        self._forward_rbf = OneWayRBFWithLinearCorrection_GPUComponent(WarpedPoints=WarpedPoints,
                                                                       FixedPoints=FixedPoints,
                                                                       BasisFunction=BasisFunction)
        self._inverse_rbf = OneWayRBFWithLinearCorrection_GPUComponent(WarpedPoints=FixedPoints,
                                                                       FixedPoints=WarpedPoints,
                                                                       BasisFunction=BasisFunction)

    def __getstate__(self):
        odict = super(TwoWayRBFWithLinearCorrection_GPUComponent, self).__getstate__()
        odict['_forward_rbf'] = self._forward_rbf
        odict['_inverse_rbf'] = self._inverse_rbf
        return odict

    def __setstate__(self, dictionary):
        super(TwoWayRBFWithLinearCorrection_GPUComponent, self).__setstate__(dictionary)
        self._forward_rbf = dictionary['_forward_rbf']
        self._inverse_rbf = dictionary['_inverse_rbf']

    def _reset_inverse_transform(self):
        self._inverse_rbf = OneWayRBFWithLinearCorrection_GPUComponent(WarpedPoints=self._forward_rbf.SourcePoints,
                                                                       FixedPoints=self._forward_rbf.TargetPoints,
                                                                       BasisFunction=self._forward_rbf.BasisFunction)

    def Transform(self, point: NDArray[np.floating], **kwargs) -> NDArray:
        return self._forward_rbf.Transform(point, **kwargs)

    def InverseTransform(self, point: NDArray[np.floating], **kwargs) -> NDArray:
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

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in target space'''
        return self._forward_rbf.GetPointPairsInTargetRect(bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle):
        '''Return the point pairs inside the rectangle defined in source space'''
        return self._forward_rbf.GetPointPairsInSourceRect(bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return self._forward_rbf.PointPairsToWarpedPoints(points)

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return self._forward_rbf.PointPairsToTargetPoints(points)

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

    def NearestTargetPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        return self._forward_rbf.NearestTargetPoint(points)

    def NearestFixedPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        return self._forward_rbf.NearestFixedPoint(points)

    def NearestSourcePoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        return self._forward_rbf.NearestSourcePoint(points)

    def NearestWarpedPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
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

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all warped points by the specified amount
        If rotation center is not specified the transform chooses"""
        self._forward_rbf.RotateSourcePoints(rangle, rotation_center)
        self._inverse_rbf.RotateTargetPoints(rangle, rotation_center)
        self.OnTransformChanged()

    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all fixed points by the specified amount
        If rotation center is not specified the transform chooses"""
        self._forward_rbf.RotateTargetPoints(rangle, rotation_center)
        self._inverse_rbf.RotateSourcePoints(rangle, rotation_center)
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._forward_rbf.TranslateFixed(offset)
        self._inverse_rbf.TranslateWarped(offset)
        self.OnTransformChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._forward_rbf.TranslateWarped(offset)
        self._inverse_rbf.TranslateFixed(offset)
        self.OnTransformChanged()

    def AddPoint(self, pointpair: NDArray[np.floating]):
        self._forward_rbf.AddPoint(pointpair)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def AddPoints(self, new_points: NDArray[np.floating]):
        self._forward_rbf.AddPoints(new_points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdatePointPair(self, index: int, pointpair: NDArray[np.floating]):
        self._forward_rbf.UpdatePointPair(index, pointpair)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateSourcePointsByIndex(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateSourcePointsByIndex(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateSourcePointsByPosition(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateSourcePointsByPosition(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateTargetPointsByIndex(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def UpdateTargetPointsByPosition(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
        self._forward_rbf.UpdateTargetPointsByPosition(index, points)
        self._reset_inverse_transform()
        self.OnTransformChanged()

    def RemovePoint(self, index: int):
        self._forward_rbf.RemovePoint(index)
        self._inverse_rbf.RemovePoint(index)
        self.OnTransformChanged()

    def OnFixedPointChanged(self):
        self._forward_rbf.OnFixedPointChanged()
        self._inverse_rbf.OnWarpedPointChanged()
        self.OnTransformChanged()

    def OnWarpedPointChanged(self):
        self._forward_rbf.OnWarpedPointChanged()
        self._inverse_rbf.OnFixedPointChanged()
        self.OnTransformChanged()

    def InitializeDataStructures(self):
        self._forward_rbf.InitializeDataStructures()
        self._inverse_rbf.InitializeDataStructures()
