"""
Created on Oct 18, 2012

@author: Jamesan
"""
import abc
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import *
import scipy.spatial

import nornir_imageregistration
from nornir_imageregistration.transforms.transform_type import TransformType


class ITransform(ABC):

    @abstractmethod
    def Transform(self, point: NDArray[np.floating], **kwargs) -> NDArray:
        """Map points from the mapped space to fixed space. Nornir is gradually transitioning to a source space to target space naming convention."""
        raise NotImplementedError()

    @abstractmethod
    def InverseTransform(self, point: NDArray[np.floating], **kwargs) -> NDArray:
        """Map points from the fixed space to mapped space. Nornir is gradually transitioning to a target space to source space naming convention."""
        raise NotImplementedError()

    @abc.abstractmethod
    def Load(self, TransformString: str, pixelSpacing=None):
        """
        Creates an instance of the transform from the TransformString
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def type(self) -> TransformType:
        raise NotImplementedError()


class ITransformChangeEvents(ABC):
    @abstractmethod
    def AddOnChangeEventListener(self, func):
        """
        Call func whenever the transform changes
        """
        raise NotImplementedError()

    @abstractmethod
    def RemoveOnChangeEventListener(self, func):
        """
        Stop calling func whenever the transform changes
        """
        raise NotImplementedError()


class ITransformTranslation(ABC):
    @abstractmethod
    def TranslateFixed(self, offset: NDArray):
        """Translate all fixed points by the specified amount"""
        raise NotImplementedError()

    @abstractmethod
    def TranslateWarped(self, offset: NDArray):
        """Translate all warped points by the specified amount"""
        raise NotImplementedError()


class ITransformSourceRotation(ABC):
    @abstractmethod
    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all warped points by the specified amount
        If rotation center is not specified the transform chooses"""
        raise NotImplementedError()


class ITransformTargetRotation(ABC):
    @abstractmethod
    def RotateTargetPoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all fixed points by the specified amount
        If rotation center is not specified the transform chooses"""
        raise NotImplementedError()


class ITransformScaling(ABC):
    '''Supports scaling target and source space together (changing image downsample level for example)'''

    @abstractmethod
    def Scale(self, scalar: float) -> None:
        """Scale both spaces by the specified amount"""
        raise NotImplementedError()


class ITransformRelativeScaling(ABC):
    '''Supports scaling of target space or source space independently of each other'''

    @abstractmethod
    def ScaleFixed(self, scalar: float) -> None:
        """Scale all fixed points by the specified amount"""
        raise NotImplementedError()

    @abstractmethod
    def ScaleWarped(self, scalar: float) -> None:
        """Scale all warped points by the specified amount"""
        raise NotImplementedError()


class IDiscreteTransform(ITransform, ABC):
    @property
    @abc.abstractmethod
    def MappedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """Bounding box of mapped space points"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def FixedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """Bounding box of fixed space points"""
        raise NotImplementedError()


class IGridTransform(ITransform, ABC):
    @property
    @abc.abstractmethod
    def grid(self) -> nornir_imageregistration.IGrid:
        raise NotImplementedError()


class IControlPoints(ABC):
    @property
    @abc.abstractmethod
    def SourcePoints(self) -> NDArray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def TargetPoints(self) -> NDArray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def points(self) -> NDArray:
        raise NotImplementedError()

    @abc.abstractmethod
    def NearestFixedPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        '''
        Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def NearestWarpedPoint(self, points: NDArray) -> tuple((float | NDArray[np.floating], int | NDArray[np.integer])):
        '''
        Return the warped points nearest to the query points
        :return: Distance, Index
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle) -> NDArray[np.floating]:
        '''Return the point pairs inside the rectangle defined in target space'''
        raise NotImplementedError()

    @abc.abstractmethod
    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle) -> NDArray[np.floating]:
        '''Return the point pairs inside the rectangle defined in source space'''
        raise NotImplementedError()

    @abc.abstractmethod
    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        '''Return the warped points from a set of target-source point pairs'''
        raise NotImplementedError()

    @abc.abstractmethod
    def PointPairsToTargetPoints(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        '''Return the target points from a set of target-source point pairs'''
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def NumControlPoints(self) -> int:
        raise NotImplementedError()


class ITriangulatedTargetSpace(ABC):
    @property
    @abc.abstractmethod
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        raise NotImplementedError()


class ITriangulatedSourceSpace(ABC):
    @property
    @abc.abstractmethod
    def source_space_trianglulation(self) -> scipy.spatial.Delaunay:
        raise NotImplementedError()


class IControlPointAddRemove(ABC):
    """Interface for control point based transforms that can add/remove control points"""

    @abc.abstractmethod
    def AddPoint(self, pointpair: NDArray[np.floating]) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def AddPoints(self, new_points: NDArray[np.floating]):
        raise NotImplementedError()

    @abc.abstractmethod
    def RemovePoint(self, index: int | NDArray[np.integer]):
        raise NotImplementedError()


class ITargetSpaceControlPointEdit(ABC):
    """Transforms where the source space side of control points can be moved.
       Originally added for grid transforms where source points are unmovable"""

    @abc.abstractmethod
    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], points: NDArray[np.floating]) -> int | \
                                                                                                           NDArray[
                                                                                                               np.integer]:
        """:return: The new index of the points"""
        raise NotImplementedError()

    @abc.abstractmethod
    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], new_points: NDArray[np.floating]) -> int | \
                                                                                                                  NDArray[
                                                                                                                      int]:
        """Move the points closest to old_points to positions at new_points
        :return: The new index of the points
        """
        raise NotImplementedError()


class ISourceSpaceControlPointEdit(ABC):
    """Transforms where the source space side of control points can be moved"""

    @abc.abstractmethod
    def UpdateSourcePointsByIndex(self, index: int | NDArray[np.integer], points: NDArray[np.floating]) -> int | \
                                                                                                           NDArray[
                                                                                                               np.integer]:
        """:return: The new index of the points"""
        raise NotImplementedError()

    @abc.abstractmethod
    def UpdateSourcePointsByPosition(self, old_points: NDArray[np.floating], new_points: NDArray[np.floating]) -> int | \
                                                                                                                  NDArray[
                                                                                                                      int]:
        """Move the points closest to old_points to positions at new_points
        :return: The new index of the points
        """
        raise NotImplementedError()


class IControlPointEdit(ITargetSpaceControlPointEdit, ISourceSpaceControlPointEdit, ABC):
    """Control point transforms where source and target control points can be edited"""

    @abc.abstractmethod
    def UpdatePointPair(self, index: int, pointpair: NDArray[np.floating]):
        raise NotImplementedError()


class IRigidTransform(ITransform, ABC):
    """A transform that encodes a rigid transformation:
    The order of operations should be:
    1. Scaling
    2. Rotation
    3. Translation
    4. Flip
    """

    @property
    @abc.abstractmethod
    def angle(self) -> float:
        """Angle of rotation in radians"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def target_offset(self) -> NDArray[np.floating]:
        """Translation from source to target space"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def source_space_center_of_rotation(self) -> NDArray[np.floating]:
        """Center of rotation in source space"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def flip_ud(self) -> bool:
        """Whether to flip the Y axis"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def scalar(self) -> float:
        """Scaling factor"""
        raise NotImplementedError()


class Base(ITransform, ITransformTranslation, ABC):
    """Base class of all transforms"""
    pass
