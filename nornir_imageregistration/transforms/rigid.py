import typing
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp
    import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

from nornir_imageregistration.spatial import Rectangle
import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import DefaultTransformChangeEvents, TransformType, base
import nornir_imageregistration.transforms.defaulttransformchangeevents


class RigidNoRotation(base.ITransform, base.ITransformScaling, base.ITransformTranslation,
                      DefaultTransformChangeEvents):
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

    def __init__(self, target_offset=tuple[float, float] | list[float] | NDArray[np.floating],
                 source_rotation_center: tuple[float, float] | typing.Sequence[float] | NDArray[
                     np.floating] | None = None,
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

    flip_ud: bool  # True if the transform should flip the Y axis

    forward_rotation_matrix: NDArray[np.floating]
    inverse_rotation_matrix: NDArray[np.floating]

    forward_translation_matrix: NDArray[np.floating]
    inverse_translation_matrix: NDArray[np.floating]

    forward_matrix: NDArray[np.floating]
    inverse_matrix: NDArray[np.floating]

    def __getstate__(self):
        data = super(Rigid, self).__getstate__()
        data['flip_ud'] = self.flip_ud
        return data

    def __setstate__(self, dictionary: dict):
        super(Rigid, self).__setstate__(dictionary)
        self.flip_ud = dictionary['flip_ud']
        self.update_rotation_matrix()

    def __init__(self, target_offset: tuple[float, float] | list[float, float] | NDArray,
                 source_rotation_center: tuple[float, float] | list[float, float] | NDArray | None = None,
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

        self.flip_ud = flip_ud
        self.update_rotation_matrix()

    def update_rotation_matrix(self):

        xp = nornir_imageregistration.GetComputationModule()

        # forward_matrix = nornir_imageregistration.transforms.utils.IdentityMatrix()
        # inverse_matrix = nornir_imageregistration.transforms.utils.IdentityMatrix()
        #
        self.forward_translation_matrix = nornir_imageregistration.transforms.utils.TranslateMatrixXY(
            self._target_offset)
        self.inverse_translation_matrix = nornir_imageregistration.transforms.utils.TranslateMatrixXY(
            -self._target_offset)

        self.forward_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(self.angle)
        self.inverse_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(-self.angle)

        self.forward_matrix = xp.matmul(self.forward_translation_matrix, self.forward_rotation_matrix)
        self.inverse_matrix = xp.matmul(self.inverse_translation_matrix, self.inverse_rotation_matrix)
        #

        if self.flip_ud:
            flip_y_matrix = nornir_imageregistration.transforms.utils.FlipMatrixY()
            self.forward_matrix = xp.matmul(flip_y_matrix, self.forward_matrix)
            self.inverse_matrix = xp.matmul(flip_y_matrix, self.inverse_matrix)

    @staticmethod
    def Load(TransformString):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString)

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

        if self.angle == 0 and not self.flip_ud:
            transformed = points + self._target_offset
            return transformed

        numPoints = points.shape[0]

        centered_points = points - self.source_space_center_of_rotation
        # centered_points = np.transpose(centered_points)
        centered_points = xp.hstack((centered_points, xp.ones((numPoints, 1))))

        # centered_rotated_points = (self.forward_rotation_matrix @ centered_points.T).T
        centered_rotated_points = xp.transpose(xp.matmul(self.forward_rotation_matrix, xp.transpose(centered_points)))
        # centered_rotated_points = np.transpose(centered_rotated_points)
        centered_rotated_points = centered_rotated_points[:, 0:2]

        rotated_points = centered_rotated_points + self.source_space_center_of_rotation
        output_points = rotated_points + self._target_offset

        if self.flip_ud:
            output_points[:, 0] = -output_points[:, 0]

        transformed = xp.around(output_points, nornir_imageregistration.RoundingPrecision(output_points.dtype))
        return transformed

    def InverseTransform(self, points: NDArray[np.floating], **kwargs):

        xp = cp.get_array_module(points)
        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

        if self.angle == 0 and not self.flip_ud:
            itransformed = points - self._target_offset
            return itransformed

        numPoints = points.shape[0]
        input_points = np.copy(points)

        if self.flip_ud:
            input_points[:, 0] = -input_points[:, 0]

        input_points = input_points - self._target_offset
        centered_points = input_points - self.source_space_center_of_rotation

        # centered_points = np.transpose(centered_points)
        centered_points = xp.hstack((centered_points, xp.zeros((numPoints, 1))))
        # rotated_points = centered_points @ self.inverse_rotation_matrix
        # rotated_points = (self.inverse_rotation_matrix @ centered_points.T).T
        rotated_points = xp.transpose(xp.matmul(self.inverse_rotation_matrix, xp.transpose(centered_points)))
        # rotated_points = np.transpose(rotated_points)
        rotated_points = rotated_points[:, 0:2]

        output_points = rotated_points + self.source_space_center_of_rotation
        itransformed = xp.around(output_points, nornir_imageregistration.RoundingPrecision(output_points.dtype))
        return itransformed

    def RotateSourcePoints(self, rangle: float, rotation_center: NDArray[np.floating] | None):
        """Rotate all warped points by the specified amount"""
        self._angle = self._angle + rangle
        xp = nornir_imageregistration.GetComputationModule()

        if rotation_center is not None:
            self.source_space_center_of_rotation = xp.array(rotation_center)

        self.update_rotation_matrix()
        self.OnTransformChanged()

    def Scale(self, scalar: float):

        # We aren't changing the relative scale of either space compared to the other
        # We are changing the scale of both spaces, so simply adjust the target and source space offsets
        # Do not call super, this method is a replacement
        self._target_offset *= scalar
        self._source_space_center_of_rotation *= scalar
        self.OnTransformChanged()

    def __repr__(self):
        return f"Offset: {self._target_offset[0]:03g}y,{self._target_offset[1]:03g}x Flip: {self.flip_ud} Angle: {self.angle:03g}r Rot Center: {self.source_space_center_of_rotation[0]:03g}y,{self.source_space_center_of_rotation[1]:03g}x"


class CenteredSimilarity2DTransform(Rigid, base.ITransformRelativeScaling):
    """
    Applies a scale+rotation+translation transform
    """

    def __getstate__(self):
        odict = super(CenteredSimilarity2DTransform, self).__getstate__()
        odict['_scalar'] = self._scalar
        return odict

    def __setstate__(self, dictionary: dict):
        super(CenteredSimilarity2DTransform, self).__setstate__(dictionary)
        self._scalar = dictionary['_scalar']

    @property
    def scalar(self):
        """The relative scale difference between source and target space"""
        return self._scalar

    def __init__(self, target_offset: tuple[float, float] | Sequence[float] | NDArray,
                 source_rotation_center: tuple[float, float] | Sequence[float] | NDArray | None = None,
                 angle: float = None, scalar: float = None, flip_ud: bool = False):
        """
        Creates a Rigid Transformation.  If used only one BoundingBox parameter needs to be specified
        :param tuple target_offset:  The amount to offset points in mapped (source) space to translate them to fixed (target) space
        :param tuple source_rotation_center: The (Y,X) center of rotation in mapped space
        :param float angle: The angle to rotate, in radians
        :param Rectangle FixedBoundingBox:  Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        :param Rectangle MappedBoundingBox: Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        """
        super(CenteredSimilarity2DTransform, self).__init__(target_offset, source_rotation_center, angle, flip_ud)

        self._scalar = 1.0 if scalar is None else scalar

    @staticmethod
    def Load(transform_string: str):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(transform_string)

    def ToITKString(self) -> str:
        # TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        # This is horrible, but we negate the angle to be compatible with ITK, then reverse it again on loading
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
        self.source_space_center_of_rotation *= scalar
        self._scalar /= scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar: float):
        """Scale target space control points by scalar"""
        self._target_offset *= scalar
        self._scalar *= scalar
        self.OnTransformChanged()

    def Transform(self, points: NDArray[np.floating], **kwargs):

        xp = cp.get_array_module(points)
        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

        if self.angle == 0 and self._scalar == 1.0:
            transformed = points + self._target_offset
            return transformed

        numPoints = points.shape[0]

        centered_points = points - self.source_space_center_of_rotation
        # centered_points = np.transpose(centered_points)
        centered_points = xp.hstack(
            (centered_points, xp.ones((numPoints, 1))))  # Add a row so we can multiply the matrix

        if self._scalar != 1.0:
            centered_points = centered_points * self._scalar

        # centered_rotated_points = centered_points @ self.forward_rotation_matrix
        # centered_rotated_points = (self.forward_rotation_matrix @ centered_points.T).T
        centered_rotated_points = xp.transpose(xp.matmul(self.forward_rotation_matrix, xp.transpose(centered_points)))
        # centered_rotated_points = np.transpose(centered_rotated_points)
        centered_rotated_points = centered_rotated_points[:, 0:2]

        rotated_points = centered_rotated_points + self.source_space_center_of_rotation
        output_points = rotated_points + self._target_offset
        if self.flip_ud:
            output_points[:, 0] = -output_points[:, 0]

        return output_points

    def InverseTransform(self, points: NDArray[np.floating], **kwargs):

        xp = cp.get_array_module(points)
        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

        if self.angle == 0 and self._scalar == 1.0:
            itransformed = points - self._target_offset
            return itransformed

        if self.flip_ud:
            points[:, 0] = -points[:, 0]

        numPoints = points.shape[0]

        input_points = points - self._target_offset
        centered_points = input_points - self.source_space_center_of_rotation

        # centered_points = np.transpose(centered_points)
        centered_points = xp.hstack((centered_points, xp.ones((numPoints, 1))))

        if self._scalar != 1.0:
            centered_points = centered_points / self._scalar

        # rotated_points = centered_points @ self.inverse_rotation_matrix
        # rotated_points = (self.inverse_rotation_matrix @ centered_points.T).T
        rotated_points = xp.transpose(xp.matmul(self.inverse_rotation_matrix, xp.transpose(centered_points)))
        # rotated_points = np.transpose(rotated_points)
        rotated_points = rotated_points[:, 0:2]

        output_points = rotated_points + self.source_space_center_of_rotation
        return output_points

    def __repr__(self):
        return super(CenteredSimilarity2DTransform, self).__repr__() + " scale: {0}:04g".format(self.scalar)
