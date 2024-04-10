# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
from __future__ import annotations

import typing
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
    import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

from nornir_imageregistration import VectorLike
from nornir_imageregistration.spatial import Rectangle
import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import DefaultTransformChangeEvents, TransformType, base
import nornir_imageregistration.transforms.defaulttransformchangeevents

from nornir_imageregistration.transforms.utils import IdentityMatrix, RotationMatrix, TranslateMatrixXY, ScaleMatrixXY, \
    FlipMatrixY, FlipMatrixX


class RigidNoRotation(base.ITransformScaling, base.ITransformTranslation,
                      base.IRigidTransform, DefaultTransformChangeEvents):
    """This class is legacy and probably needs a deprecation warning"""

    _target_offset: NDArray[np.floating]  # Amount to translate the points after centering and rotation
    _source_space_center_of_rotation: NDArray[np.floating]  # Where the center of rotation lies
    _angle: float = 0  # Angle in radians, zero for this base class

    @property
    def angle(self):
        """
        :return: The angle of rotation in radians
        """
        return self._angle

    @property
    def flip_ud(self):
        return False

    @property
    def type(self) -> TransformType:
        return nornir_imageregistration.transforms.TransformType.RIGID

    def _transform_rectangle(self, rect: nornir_imageregistration.spatial.Rectangle | None):
        if rect is None:
            return None

        # Transform the other bounding box
        mapped_corners = self.Transform(rect.Corners)
        return Rectangle.CreateBoundingRectangleForPoints(mapped_corners)

    def _inverse_transform_rectangle(self, rect: nornir_imageregistration.spatial.Rectangle | None):
        if rect is None:
            return None

        # Transform the other bounding box
        mapped_corners = self.InverseTransform(rect.Corners)
        return Rectangle.CreateBoundingRectangleForPoints(mapped_corners)

    def TranslateFixed(self, offset: NDArray[np.floating]):
        """Translate all fixed points by the specified amount"""
        self._target_offset = self._target_offset + nornir_imageregistration.EnsurePointsAre1DArray(offset)
        self.OnTransformChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        """Translate all warped points by the specified amount"""
        self._target_offset = self._target_offset - nornir_imageregistration.EnsurePointsAre1DArray(offset)
        self.OnTransformChanged()

    def Scale(self, scalar: float):
        """Scale both warped and control space by scalar"""
        self._target_offset *= scalar
        self._source_space_center_of_rotation *= scalar
        self.OnTransformChanged()

    @property
    def target_offset(self) -> NDArray[np.floating]:
        return self._target_offset

    @property
    def source_space_center_of_rotation(self) -> NDArray[np.floating]:
        return self._source_space_center_of_rotation

    @property
    def scalar(self) -> float:
        return 1.0

    def __init__(self, target_offset: VectorLike,
                 source_rotation_center: VectorLike | None = None,
                 angle: float | None = None):
        """
        Creates a Rigid Transformation.  If used only one BoundingBox parameter needs to be specified
        :param tuple target_offset:  The amount to offset points in mapped (source) space to translate them to fixed (target) space
        :param tuple source_rotation_center: The (Y,X) center of rotation in mapped space
        :param float angle: The angle to rotate, in radians
        :param Rectangle FixedBoundingBox:  Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        :param Rectangle MappedBoundingBox: Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        """
        super(RigidNoRotation, self).__init__()

        if angle is None:
            angle = 0.0

        if source_rotation_center is None:
            source_rotation_center = (0.0, 0.0)

        self._target_offset = nornir_imageregistration.EnsurePointsAre1DArray(target_offset)
        self._source_space_center_of_rotation = nornir_imageregistration.EnsurePointsAre1DArray(
            source_rotation_center)
        self._angle = angle  # type: float

    def __getstate__(self):

        cp_arrays = cp.get_array_module(self._target_offset) == cp
        tgt_offset = self._target_offset if not cp_arrays else self._target_offset.get()
        sscr = self._source_space_center_of_rotation if not cp_arrays else self._source_space_center_of_rotation.get()

        odict = {'_angle': self._angle, '_target_offset': (tgt_offset[0], tgt_offset[1]),
                 '_source_space_center_of_rotation': (sscr[0],
                                                      sscr[1])}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)

        xp = nornir_imageregistration.GetComputationModule()

        # Check for legacy .pickle data by looking for non-underscore attributes.
        # Legacy .pickle files  are used by TestMosaicTilesetTileOffsets.test_Alignment_RC3_0001 test
        if 'target_offset' in dictionary:
            self._target_offset = dictionary['target_offset']

        if 'source_space_center_of_rotation' in dictionary:
            self._source_space_center_of_rotation = dictionary['source_space_center_of_rotation']

        self._target_offset = xp.asarray((self._target_offset[0], self._target_offset[1]), dtype=np.float32)
        self._source_space_center_of_rotation = xp.asarray((self._source_space_center_of_rotation[0],
                                                            self._source_space_center_of_rotation[1]), dtype=np.float32)

        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    def __repr__(self):
        return f"Offset: {self._target_offset[0]:03g}y,{self._target_offset[1]:03g}x"

    @staticmethod
    def Load(TransformString: typing.Sequence[str], pixelSpacing: float | int | None = None):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString, pixelSpacing)

    def ToITKString(self) -> str:
        # TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return f"Rigid2DTransf" \
               f"orm_double_2_2 vp 3 {self._angle} {self._target_offset[1]} {self._target_offset[0]} fp 2 {self._source_space_center_of_rotation[1]} {self._source_space_center_of_rotation[0]}"

    def Transform(self, points: NDArray[np.floating], **kwargs):

        if not (self._angle is None or self._angle == 0):
            # Look at GetTransformedRigidCornerPoints for a possible implementation
            raise NotImplemented("Rotation is not implemented")

        points = nornir_imageregistration.EnsurePointsAre2DArray(points)
        transformed = points + self._target_offset
        return transformed

    def InverseTransform(self, points: NDArray[np.floating], **kwargs):

        if not (self._angle is None or self._angle == 0):
            # Look at GetTransformedRigidCornerPoints for a possible implementation
            raise NotImplemented("Rotation is not implemented")

        points = nornir_imageregistration.EnsurePointsAre2DArray(points)
        itransformed = points - self._target_offset
        return itransformed


class Rigid(base.ITransformSourceRotation, RigidNoRotation):
    """
    Applies a rotation+translation transform
    The order of operations is:
    1. Scaling
    2. Rotation
    3. Translation
    4. Flip

    Remember that matrix multiplacation is applied in reverse order, so the last operation is the first to be applied
    """

    @property
    def flip_ud(self) -> bool:
        return self._flip_ud

    _flip_ud: bool  # True if the transform should flip the Y axis
    _scalar: float = 1  # The relative scale difference between source and target space
    source_space_center_of_rotation: NDArray[np.floating]  # The center of rotation in the source space

    @property
    def scalar(self) -> float:
        """The relative scale difference between source and target space"""
        return self._scalar

    @property
    def _forward_rotation_matrix(self) -> NDArray[np.floating]:
        """rotation matrix to move points to the target space"""
        return RotationMatrix(self.angle)

    @property
    def _inverse_rotation_matrix(self) -> NDArray[np.floating]:
        """inverse rotation to move points to the source space"""
        return RotationMatrix(-self.angle)

    @property
    def _forward_translation_matrix(self) -> NDArray[np.floating]:
        """translation matrix to move points to the target space"""
        return TranslateMatrixXY(self._target_offset)

    @property
    def _inverse_translation_matrix(self) -> NDArray[np.floating]:
        """inverse translation to move points to the source space"""
        return TranslateMatrixXY(-self._target_offset)

    @property
    def _forward_center_of_rotation_translation(self) -> NDArray[np.floating]:
        """Matrix to translate the center of rotation to 0,0 origin in the source space"""
        return TranslateMatrixXY(self.source_space_center_of_rotation)

    @property
    def _inverse_center_of_rotation_translation(self) -> NDArray[np.floating]:
        """Matrix to translate the center of rotation to original position in the source space"""
        return TranslateMatrixXY(-self.source_space_center_of_rotation)

    @property
    def _flip_y_matrix(self) -> NDArray[np.floating]:
        """Matrix to flip the Y axis, identity if matrix is not flipped"""
        return FlipMatrixY() if self._flip_ud else IdentityMatrix()

    @property
    def _forward_scale_matrix(self) -> NDArray[np.floating]:
        """Matrix to scale the points"""
        return ScaleMatrixXY(self._scalar)

    @property
    def _inverse_scale_matrix(self) -> NDArray[np.floating]:
        """Matrix to scale the points"""
        return ScaleMatrixXY(1.0 / self._scalar)

    forward_matrix: NDArray[np.floating]  # The composite matrix to perform the transform in a single step
    inverse_matrix: NDArray[np.floating]  # The composite matrix to perform the inverse transform in a single step

    def __getstate__(self):
        data = super(Rigid, self).__getstate__()
        data['flip_ud'] = self._flip_ud
        return data

    def __setstate__(self, dictionary: dict):
        super(Rigid, self).__setstate__(dictionary)
        self._flip_ud = dictionary['flip_ud']
        self._update_transform_matrix()

    def __init__(self, target_offset: VectorLike,
                 source_rotation_center: VectorLike | None = None,
                 angle: float | None = None, flip_ud: bool = False):
        """
        Creates a Rigid Transformation.  If used only one BoundingBox parameter needs to be specified
        :param tuple target_offset:  The amount to offset points in mapped (source) space to translate them to fixed (target) space
        :param tuple source_rotation_center: The (Y,X) center of rotation in mapped (source) space
        :param float angle: The angle to rotate, in radians
        :param Rectangle FixedBoundingBox:  Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        :param Rectangle MappedBoundingBox: Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        """
        super(Rigid, self).__init__(target_offset, source_rotation_center, angle)

        self._flip_ud = flip_ud
        self._update_transform_matrix()

    def _update_transform_matrix(self):
        """Update the forward and inverse matrices"""
        self.forward_matrix = self._forward_translation_matrix @ self._forward_center_of_rotation_translation @ \
                              self._flip_y_matrix @ self._forward_rotation_matrix @ \
                              self._inverse_center_of_rotation_translation @ self._forward_scale_matrix
        self.inverse_matrix = self._inverse_scale_matrix @ self._forward_center_of_rotation_translation @ \
                              self._inverse_rotation_matrix @ self._flip_y_matrix @ \
                              self._inverse_center_of_rotation_translation @ self._inverse_translation_matrix

    @staticmethod
    def Load(TransformString: typing.Sequence[str], pixelSpacing: float | None = None) -> Rigid:
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString, pixelSpacing)

    def ToITKString(self) -> str:
        # TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        # This is horrible, but we negate the angle to be compatible with ITK, then reverse it again on loading
        return "Rigid2DTransform_double_2_2 vp 3 {0} {1} {2} fp 2 {3} {4}".format(self.angle, self._target_offset[1],
                                                                                  self._target_offset[0],
                                                                                  self.source_space_center_of_rotation[
                                                                                      1],
                                                                                  self.source_space_center_of_rotation[
                                                                                      0])

    def Transform(self, points: NDArray[np.floating], **kwargs):

        xp = cp.get_array_module(points)
        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

        if self.angle == 0 and self.scalar == 1 and not self.flip_ud:
            transformed = points + self._target_offset
            return transformed

        num_points = points.shape[0]
        centered_points = xp.hstack((points, xp.ones((num_points, 1))))
        output_points = xp.transpose(xp.matmul(self.forward_matrix, xp.transpose(centered_points)))
        output_points = output_points[:, 0:2]
        itransformed = xp.around(output_points, nornir_imageregistration.RoundingPrecision(output_points.dtype))
        return itransformed

    def InverseTransform(self, points: NDArray[np.floating], **kwargs):

        xp = cp.get_array_module(points)
        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

        if self.angle == 0 and self.scalar == 1 and not self.flip_ud:
            itransformed = points - self._target_offset
            return itransformed

        num_points = points.shape[0]
        centered_points = xp.hstack((points, xp.ones((num_points, 1))))
        output_points = xp.transpose(xp.matmul(self.inverse_matrix, xp.transpose(centered_points)))
        output_points = output_points[:, 0:2]
        itransformed = xp.around(output_points, nornir_imageregistration.RoundingPrecision(output_points.dtype))
        return itransformed

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all warped points by the specified amount"""
        self._angle = self._angle + rangle
        xp = nornir_imageregistration.GetComputationModule()

        if rotation_center is not None:
            self.source_space_center_of_rotation = xp.array(rotation_center)

        self._update_transform_matrix()
        self.OnTransformChanged()

    def Scale(self, value: float):

        # We aren't changing the relative scale of either space compared to the other
        # We are changing the scale of both spaces, so simply adjust the target and source space offsets
        # Do not call super, this method is a replacement
        self._scalar *= value
        self._update_transform_matrix()
        self.OnTransformChanged()

    def __repr__(self):
        return f"Offset: {self._target_offset[0]:03g}y,{self._target_offset[1]:03g}x Flip: {self.flip_ud} Angle: {self.angle:03g}r Rot Center: {self.source_space_center_of_rotation[0]:03g}y,{self.source_space_center_of_rotation[1]:03g}x"


class CenteredSimilarity2DTransform(Rigid, base.ITransformRelativeScaling):
    """
    Applies a scaling+rotation+translation transform
    The order of operations is:
    1. Scaling
    2. Rotation
    3. Translation
    4. Flip

    Remember that matrix multiplacation is applied in reverse order, so the last operation is the first to be applied
    """

    def __getstate__(self):
        odict = super(CenteredSimilarity2DTransform, self).__getstate__()
        odict['_scalar'] = self._scalar
        return odict

    def __setstate__(self, dictionary: dict):
        self._scalar = dictionary['_scalar']
        super(CenteredSimilarity2DTransform, self).__setstate__(dictionary)

    def __init__(self, target_offset: VectorLike,
                 source_rotation_center: VectorLike | None = None,
                 angle: float = None, scalar: float = None, flip_ud: bool = False):
        """
        Creates a Rigid Transformation.  If used only one BoundingBox parameter needs to be specified
        :param tuple target_offset:  The amount to offset points in mapped (source) space to translate them to fixed (target) space
        :param tuple source_rotation_center: The (Y,X) center of rotation in mapped space
        :param float angle: The angle to rotate, in radians
        :param Rectangle FixedBoundingBox:  Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        :param Rectangle MappedBoundingBox: Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        """
        self._scalar = 1.0 if scalar is None else scalar
        super(CenteredSimilarity2DTransform, self).__init__(target_offset, source_rotation_center, angle, flip_ud)

    @staticmethod
    def Load(TransformString: typing.Sequence[str], pixelSpacing: float | None = None) -> Rigid:
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString, pixelSpacing)

    def ToITKString(self) -> str:
        return "CenteredSimilarity2DTransform_double_2_2 vp 6 {0} {1} {2} {3} {4} {5} fp 0".format(self._scalar,
                                                                                                   self.angle,
                                                                                                   self.source_space_center_of_rotation[
                                                                                                       1],
                                                                                                   self.source_space_center_of_rotation[
                                                                                                       0],
                                                                                                   self._target_offset[
                                                                                                       1],
                                                                                                   self._target_offset[
                                                                                                       0])

    def ScaleWarped(self, scalar: float):
        """Scale source space control points by scalar"""
        self._scalar /= scalar
        self._update_transform_matrix()
        self.OnTransformChanged()

    def ScaleFixed(self, scalar: float):
        """Scale target space control points by scalar"""
        self._scalar *= scalar
        self._update_transform_matrix()
        self.OnTransformChanged()

    def __repr__(self):
        return super(CenteredSimilarity2DTransform, self).__repr__() + " scale: {0}:04g".format(self.scalar)
