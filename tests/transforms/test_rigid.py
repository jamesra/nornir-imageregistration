import unittest
import datetime
import math

import hypothesis
import hypothesis.strategies as st
import numpy as np
from numpy.typing import NDArray

try:
    from transforms.data import OffsetTransformPoints, \
        TranslateRotateTransformPoints, IdentityFlippedUDTransformPoints, TranslateFlippedUDTransformPoints, \
        TranslateRotateFlippedTransformPoints
    from transforms.checks import TransformAgreementCheck, TransformCheck, TransformInverseCheck
except ImportError:
    from tests.transforms.data import OffsetTransformPoints, \
        TranslateRotateTransformPoints, IdentityFlippedUDTransformPoints, TranslateFlippedUDTransformPoints, \
        TranslateRotateFlippedTransformPoints
    from tests.transforms.checks import TransformAgreementCheck, TransformCheck, TransformInverseCheck

import nornir_imageregistration
import nornir_imageregistration.transforms
from setup_imagetest import ImageTestBase


def rotated_points(source_points, angle: float, source_rotation_center=None) -> NDArray[np.floating]:
    '''Rotates points around a circle without a matrix (used to test matrix implementation'''
    if source_rotation_center is None:
        source_rotation_center = (0, 0)
    source_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_points)
    source_rotation_center = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_rotation_center)

    source_points = source_points - source_rotation_center
    x_col = source_points[:, 1]
    y_col = source_points[:, 0]

    x = np.cos(angle) * x_col - np.sin(angle) * y_col
    y = np.sin(angle) * x_col + np.cos(angle) * y_col

    result = np.vstack((y, x)).T
    result = result + source_rotation_center
    return result


def _testRotate_simple(self: unittest.TestCase, T: nornir_imageregistration.transforms.Rigid):
    angle = T.angle

    sourcePoint = [[0, 1],
                   [0, 2]]  # [y, x]

    targetPoint = [[np.sin(angle), np.cos(angle)],
                   # This is not a rotation transform.  It is a manual rotation to create Y, X point outputs for angles
                   [np.sin(angle) * 2, np.cos(angle) * 2]]

    sourcePoint = np.asarray(sourcePoint)
    targetPoint = np.asarray(targetPoint)

    TransformCheck(self, T, sourcePoint, targetPoint)


class TestRigidTransforms(unittest.TestCase):

    # def test_rigid_transform_boundingboxes(self):
    #     A_fixed_center = (0, 75)
    #     B_fixed_center = (0,-75)
    #     A_shape = (100,100)
    #     B_shape = (100,100)
    #     A_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(A_fixed_center, A_shape)
    #     B_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(B_fixed_center, B_shape)
    #     shared_mapped_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea((0,0), A_shape)
    #     transform_A = nornir_imageregistration.transforms.Rigid(A_fixed_center, MappedBoundingBox=shared_mapped_bbox)
    #     transform_B = nornir_imageregistration.transforms.Rigid(B_fixed_center, MappedBoundingBox=shared_mapped_bbox)
    #
    #     np.testing.assert_array_equal(transform_A.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
    #     np.testing.assert_array_equal(transform_B.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
    #
    #     np.testing.assert_array_equal(transform_A.FixedBoundingBox.BoundingBox, A_target_bbox.BoundingBox)
    #     np.testing.assert_array_equal(transform_B.FixedBoundingBox.BoundingBox, B_target_bbox.BoundingBox)
    #     return

    def test_rotated_points(self, angle: float = None):
        if angle is None:
            angle = math.pi / 6

        source_points = np.array([[0, 1],
                                  [0, 2]])  # [y, x])

        expected_points = np.array([[np.sin(angle), np.cos(angle)],
                                    # This is not a rotation transform.  It is a manual rotation to create Y, X point outputs for angles
                                    [np.sin(angle) * 2, np.cos(angle) * 2]])

        check_points = rotated_points(source_points, angle)

        np.testing.assert_allclose(expected_points, check_points)

    def testIdentity(self):
        T = nornir_imageregistration.transforms.Rigid([0, 0], [0, 0], 0)

        warpedPoint = np.array([[0, 0],
                                [0.25, 0.25],
                                [1, 1],
                                [-1, -1]])
        TransformCheck(self, T, warpedPoint, warpedPoint)

    def testIdentityInverted(self):
        T = nornir_imageregistration.transforms.Rigid([0, 0], [0, 0], 0, flip_ud=True)

        source_point = IdentityFlippedUDTransformPoints[:, 2:4]
        target_point = IdentityFlippedUDTransformPoints[:, 0:2]

        TransformCheck(self, T, source_point, target_point)

    def testTranslate(self):
        T = nornir_imageregistration.transforms.Rigid([1, 1], [0, 0], 0)

        warpedPoint = OffsetTransformPoints[:, 2:4]
        controlPoint = OffsetTransformPoints[:, 0:2]

        TransformCheck(self, T, warpedPoint, controlPoint)

    def testTranslateInverted(self):
        T = nornir_imageregistration.transforms.Rigid([1, -2], [0, 0], 0, flip_ud=True)

        warpedPoint = TranslateFlippedUDTransformPoints[:, 2:4]
        controlPoint = TranslateFlippedUDTransformPoints[:, 0:2]
        TransformCheck(self, T, warpedPoint, controlPoint)

    def testRotate_simple(self):
        # Rotate a point at x=1, y= 0
        angle = np.pi / 6.0
        T = nornir_imageregistration.transforms.Rigid([0, 0], [0, 0], angle)
        self.assertTrue(angle == T.angle)

        _testRotate_simple(self, T)

    def test_Rotate90_standard_points(self):
        offset = np.array((1, 2))  # numpy.array(targetShape) / 2.0
        source_rotation_center = np.array((0, 0))

        angle = np.pi / 2.0
        T = nornir_imageregistration.transforms.Rigid(offset, source_rotation_center, angle)

        points = TranslateRotateTransformPoints
        target_points = points[:, 0:2]
        source_points = points[:, 2:]

        TransformCheck(self, T, source_points, target_points)

    def test_Rotate90_flipped_points(self):
        offset = np.array((1, 2))
        source_rotation_center = np.array((0, 0))

        angle = np.pi / 2.0
        T = nornir_imageregistration.transforms.Rigid(offset, source_rotation_center, angle, flip_ud=True)

        points = TranslateRotateFlippedTransformPoints
        target_points = points[:, 0:2]
        source_points = points[:, 2:]

        TransformCheck(self, T, source_points, target_points)

    def testOffsetRotate_Rigid(self):
        xp = nornir_imageregistration.GetComputationModule()
        offset = np.array((-2, 1))
        source_rotation_center = np.array([1, 0])
        angle = np.pi / 6.0
        T = nornir_imageregistration.transforms.Rigid(offset, source_rotation_center=source_rotation_center,
                                                      angle=angle)

        sourcePoint = [[0, 0],
                       [0, 10]]

        targetPoint = rotated_points(source_points=sourcePoint, angle=angle,
                                     source_rotation_center=source_rotation_center) + offset
        targetPoint = xp.array(targetPoint, dtype=float)
        sourcePoint = xp.asarray(sourcePoint, dtype=float)

        TransformCheck(self, T, sourcePoint, targetPoint)

    def testOffsetRotateTranslate_Rigid(self):
        angle = np.pi / 6.0
        T = nornir_imageregistration.transforms.Rigid([1, 2], source_rotation_center=[1, 0],
                                                      angle=angle)

        sourcePoint = [[0, 0],
                       [0, 10]]

        sourcePoint = np.asarray(sourcePoint)

        TransformInverseCheck(self, T, sourcePoint)


class TestTransforms_CenteredSimilarity(unittest.TestCase):

    # def test_rigid_transform_boundingboxes(self):
    #     A_fixed_center = (0, 75)
    #     B_fixed_center = (0,-75)
    #     A_shape = (100,100)
    #     B_shape = (100,100)
    #     A_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(A_fixed_center, A_shape)
    #     B_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(B_fixed_center, B_shape)
    #     shared_mapped_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea((0,0), A_shape)
    #     transform_A = nornir_imageregistration.transforms.Rigid(A_fixed_center, MappedBoundingBox=shared_mapped_bbox)
    #     transform_B = nornir_imageregistration.transforms.Rigid(B_fixed_center, MappedBoundingBox=shared_mapped_bbox)
    #
    #     np.testing.assert_array_equal(transform_A.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
    #     np.testing.assert_array_equal(transform_B.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
    #
    #     np.testing.assert_array_equal(transform_A.FixedBoundingBox.BoundingBox, A_target_bbox.BoundingBox)
    #     np.testing.assert_array_equal(transform_B.FixedBoundingBox.BoundingBox, B_target_bbox.BoundingBox)
    #     return

    def testIdentity(self):
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([0, 0], [0, 0], 0)

        warpedPoint = np.array([[0, 0],
                                [0.25, 0.25],
                                [1, 1],
                                [-1, -1]])
        TransformCheck(self, T, warpedPoint, warpedPoint)

    def testTranslate(self):
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([1, 1], [0, 0], 0)

        warpedPoint = OffsetTransformPoints[:, 2:4]

        controlPoint = OffsetTransformPoints[:, 0:2]

        TransformCheck(self, T, warpedPoint, controlPoint)

    def testRotate(self):
        angle = np.pi / 6.0
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([0, 0], [0, 0], angle)

        sourcePoint = [[0, 0],
                       [0, 10]]  # [y, x]

        targetPoint = [[0, 0],
                       [np.sin(angle) * 10, np.cos(angle) * 10]]

        sourcePoint = np.asarray(sourcePoint)
        targetPoint = np.asarray(targetPoint)

        TransformCheck(self, T, sourcePoint, targetPoint)

    def testRotate_simple(self):
        # Rotate a point at x=1, y= 0

        angle = np.pi / 6.0
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([0, 0], [0, 0], angle)
        self.assertTrue(angle == T.angle)

        _testRotate_simple(self, T)

    def test_Rotate90_standard_points(self):
        offset = np.array((1, 2))  # numpy.array(targetShape) / 2.0
        source_rotation_center = np.array((0, 0))

        angle = np.pi / 2.0
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(offset, source_rotation_center, angle)

        points = TranslateRotateTransformPoints
        target_points = points[:, 0:2]
        source_points = points[:, 2:]

        TransformCheck(self, T, source_points, target_points)

    def testOffsetRotate(self):
        xp = nornir_imageregistration.GetComputationModule()
        angle = np.pi / 6.0
        offset = np.array((0, 1), dtype=float)

        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([0, 0], source_rotation_center=offset,
                                                                              angle=angle)

        sourcePoint = xp.array([[0, 1],
                                [0, 10]])  # [y, x]
        #
        # t_source_point -= offset
        #
        # targetPoint = np.array([[np.sin(angle), np.cos(angle)],
        #                [np.sin(angle) * (sourcePoint[1, 0] - offset[0]), np.cos(angle) * (sourcePoint[1, 1] - offset[1])]])
        # targetPoint = np.asarray(targetPoint) + offset

        sourcePoint = xp.asarray(sourcePoint)

        TransformInverseCheck(self, T, sourcePoint)

    def testOffsetRotateTranslate(self):
        xp = nornir_imageregistration.GetComputationModule()
        angle = np.pi / 4.0
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([1, 2], source_rotation_center=[1, 0],
                                                                              angle=angle)

        sourcePoint = [[0, 0],
                       [10, 0]]

        sourcePoint = xp.asarray(sourcePoint)

        TransformInverseCheck(self, T, sourcePoint)

    def testOffsetRotateTranslateScale(self):
        xp = nornir_imageregistration.GetComputationModule()
        angle = np.pi / 4.0
        scale = 2
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([1, 2], source_rotation_center=[1, 0],
                                                                              angle=angle,
                                                                              scalar=scale)

        sourcePoint = [[0, 0],
                       [10, 0]]

        sourcePoint = xp.asarray(sourcePoint)

        TransformInverseCheck(self, T, sourcePoint)

    def testScale(self):
        """
        Check if a scale between source and target space is applied correctly
        :return:
        """
        xp = nornir_imageregistration.GetComputationModule()
        angle = 0
        scale = 10
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([0, 0], source_rotation_center=[0, 0],
                                                                              angle=angle,
                                                                              scalar=10)

        sourcePoint = xp.array([[0, 1],
                                [2, 0],
                                [-2, -1]])

        targetPoint = sourcePoint * scale

        TransformCheck(self, T, sourcePoint, targetPoint)

    def testScaleAboutNonZeroCenter(self):
        """ITK CenteredSimilarity: scale about c, then translate — center maps to c + t."""
        scale = 2.0
        center = np.asarray([100.0, 200.0], dtype=float)
        peak = np.asarray([3.0, -5.0], dtype=float)
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=peak,
            source_rotation_center=center,
            angle=0.0,
            scalar=scale)

        # Center must stay fixed under scale-about-center, then move by peak only.
        TransformCheck(self, T, center.reshape(1, 2), (center + peak).reshape(1, 2))

        # A point offset from center scales about center: c + s*(p-c) + t
        offset_pt = center + np.asarray([10.0, -20.0], dtype=float)
        expected = center + scale * (offset_pt - center) + peak
        TransformCheck(self, T, offset_pt.reshape(1, 2), expected.reshape(1, 2))

        TransformInverseCheck(self, T, np.vstack([center, offset_pt]))

    def testScaleAboutLargeCenterTemMagnitude(self):
        """Anti-regression: TEM1/TEM2-scale about a full-res center must not bias translate.

        Before the centered-scale matrix fix, Transform(c) was ~s*c (origin scale),
        an error of about (s-1)*c ≈ 19 px for s=1.038 and c≈500.
        """
        scale = 1.038
        center = np.asarray([499.5, 499.5], dtype=float)
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(0.0, 0.0),
            source_rotation_center=center,
            angle=0.0,
            scalar=scale)

        TransformCheck(self, T, center.reshape(1, 2), center.reshape(1, 2))

        offset_pt = center + np.asarray([40.0, -25.0], dtype=float)
        expected_centered = center + scale * (offset_pt - center)
        TransformCheck(self, T, offset_pt.reshape(1, 2), expected_centered.reshape(1, 2))

        # Document the old bug: origin-based scale would land far from the centered result.
        origin_scaled = offset_pt * scale
        err_if_origin_scale = float(np.linalg.norm(origin_scaled - expected_centered))
        self.assertGreater(err_if_origin_scale, 10.0,
                           "Fixture assumes origin-scale bias is large at this center/scale")

    def testScaleWithTranslation(self):
        """
        Check if we retain the scale between source and target space, but change the scale of both spaces simultaneously,
        as in changing image resolutions, that the transform still works correctly
        :return:
        """
        xp = nornir_imageregistration.GetComputationModule()
        angle = 0
        scale = 10
        offset = np.array((5, 5), dtype=float)
        source_rotation_center = np.array((0, 0), dtype=float)
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(offset,
                                                                              source_rotation_center=source_rotation_center,
                                                                              angle=angle,
                                                                              scalar=scale)

        sourcePoint = xp.array([[0, 1],
                                [2, 0],
                                [-2, -1]])

        targetPoint = (sourcePoint * scale) + xp.asarray(offset)

        TransformCheck(self, T, sourcePoint, targetPoint)

        T.Scale(1 / scale)
        adjusted_scale = 1 / scale
        targetPoint = (sourcePoint * scale) + (xp.asarray(offset) * adjusted_scale)
        TransformCheck(self, T, sourcePoint, targetPoint)

    @hypothesis.given(r_angle=st.floats(min_value=-np.pi, max_value=np.pi),
                      target_offset=st.tuples(st.floats(min_value=-15, max_value=15),
                                              st.floats(min_value=-15, max_value=15)),
                      source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
                                                       st.floats(min_value=-15, max_value=15)),
                      scale=st.floats(min_value=0.1, max_value=10),
                      flip_ud=st.booleans(),
                      source_points=st.lists(
                          st.tuples(st.floats(min_value=-15, max_value=15),
                                    st.floats(min_value=-15, max_value=15)),
                          min_size=1, max_size=12))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=20))  # Cupy Context takes a while to initialize
    def testCenteredSimilarityTransform(self,
                                        r_angle: float,
                                        target_offset: np.ndarray,
                                        source_rotation_center: np.ndarray,
                                        source_points: list[tuple[float, float]],
                                        scale: float,
                                        flip_ud: bool):
        source_point_array = np.array(source_points, dtype=float)
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=target_offset,
                                                                                      source_rotation_center=source_rotation_center,
                                                                                      angle=r_angle, scalar=scale,
                                                                                      flip_ud=flip_ud)
        TransformInverseCheck(self, transform, source_point_array)

    #
    # @hypothesis.given(r_angle=st.floats(min_value=-np.pi, max_value=np.pi),
    #                   target_offset=st.tuples(st.floats(min_value=-15, max_value=15),
    #                                           st.floats(min_value=-15, max_value=15)),
    #                   source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
    #                                                    st.floats(min_value=-15, max_value=15)),
    #                   scale=st.floats(min_value=0.1, max_value=10),
    #                   flip_ud=st.booleans(),
    #                   source_points=st.lists(
    #                       st.tuples(st.floats(min_value=-15, max_value=15),
    #                                 st.floats(min_value=-15, max_value=15)),
    #                       min_size=1, max_size=12))
    # @hypothesis.settings(deadline=datetime.timedelta(seconds=20))  # Cupy Context takes a while to initialize
    # def testCenteredSimilarityTransform_manual_inverse(self,
    #                                                    r_angle: float,
    #                                                    target_offset: np.ndarray,
    #                                                    source_rotation_center: np.ndarray,
    #                                                    source_points: list[tuple[float, float]],
    #                                                    scale: float,
    #                                                    flip_ud: bool):
    # """
    # This doesn't work yet, to make an inverse transform the scale, flip, and translate need to be calculated in the correct order from the initial transform inputs.
    # """
    #     flip_ud = False
    #     source_point_array = np.array(source_points, dtype=float)
    #     target_offset = np.array(target_offset, dtype=float)
    #     source_rotation_center = np.array(source_rotation_center, dtype=float)
    #
    #     transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=target_offset,
    #                                                                                   source_rotation_center=source_rotation_center,
    #                                                                                   angle=r_angle, scalar=scale,
    #                                                                                   flip_ud=flip_ud)
    #     inverse_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
    #         target_offset=-target_offset * (1 / scale),
    #         source_rotation_center=source_rotation_center,
    #         angle=-r_angle, scalar=1 / scale,
    #         flip_ud=flip_ud)
    #     TransformInverseCheck(self, transform, source_point_array)
    #     TransformInverseCheck(self, inverse_transform, source_point_array)
    #
    #     transformed_target_point_array = transform.Transform(source_point_array)
    #     transformed_source_point_array = inverse_transform.Transform(transformed_target_point_array)
    #     inverse_transformed_target_point_array = inverse_transform.InverseTransform(source_point_array)
    #
    #     np.testing.assert_allclose(source_point_array, transformed_source_point_array, atol=1e-3)
    #     np.testing.assert_allclose(transformed_target_point_array, inverse_transformed_target_point_array, atol=1e-3)

    @hypothesis.given(r_angle=st.floats(min_value=-np.pi, max_value=np.pi),
                      target_offset=st.tuples(st.floats(min_value=-15, max_value=15),
                                              st.floats(min_value=-15, max_value=15)),
                      source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
                                                       st.floats(min_value=-15, max_value=15)),
                      flip_ud=st.booleans(),
                      source_points=st.lists(
                          st.tuples(st.floats(min_value=-15, max_value=15),
                                    st.floats(min_value=-15, max_value=15)),
                          min_size=1, max_size=12))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=20))  # Cupy Context takes a while to initialize
    def testRigidTransform(self,
                           r_angle: float,
                           target_offset: np.ndarray,
                           source_rotation_center: np.ndarray,
                           flip_ud: bool,
                           source_points: list[tuple[float, float]]):
        source_point_array = np.array(source_points, dtype=float)
        transform = nornir_imageregistration.transforms.Rigid(target_offset=target_offset,
                                                              source_rotation_center=source_rotation_center,
                                                              angle=r_angle,
                                                              flip_ud=flip_ud)

        TransformInverseCheck(self, transform, source_point_array)

        target_point_array = transform.Transform(source_point_array)

        transform_similar = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=target_offset,
            source_rotation_center=source_rotation_center,
            angle=r_angle,
            flip_ud=flip_ud)

        TransformCheck(self, transform_similar, source_point_array, target_point_array)


class TestRigidFactory(ImageTestBase):
    __transform_tolerance = 1e-5

    def testRigidvsMeshFactory(self):
        xp = nornir_imageregistration.GetComputationModule()
        r = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=[10, 10],
                                                                             source_image_shape=[10, 10],
                                                                             rangle=0,
                                                                             warped_offset=[-10, 5],
                                                                             flip_ud=False)

        m = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=[10, 10],
                                                                                 source_image_shape=[10, 10],
                                                                                 rangle=0,
                                                                                 warped_offset=[-10, 5],
                                                                                 flip_ud=False)

        p1 = [[0, 0],
              [10, 10]]

        r1 = r.Transform(p1)
        m1 = m.Transform(p1)

        xp.testing.assert_allclose(r1, m1, atol=self.__transform_tolerance,
                                   err_msg="Mesh and Rigid Transform do not agree")

        ir1 = r.InverseTransform(r1)
        im1 = m.InverseTransform(m1)

        xp.testing.assert_allclose(ir1, p1, atol=self.__transform_tolerance,
                                   err_msg="Mesh and Rigid Transform do not agree")
        xp.testing.assert_allclose(im1, p1, atol=self.__transform_tolerance,
                                   err_msg="Mesh and Rigid Transform do not agree")
        xp.testing.assert_allclose(im1, ir1, atol=self.__transform_tolerance,
                                   err_msg="Mesh and Rigid Transform do not agree")

        TransformAgreementCheck(r, m, [10, -3])

    def testRigidvsMeshFactoryTranslate(self):
        d_angle = 0
        r_angle = (d_angle / 180.0) * np.pi
        r = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=[10, 10],
                                                                             source_image_shape=[10, 10],
                                                                             rangle=r_angle,
                                                                             warped_offset=[-10, 5],
                                                                             flip_ud=False)

        m = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=[10, 10],
                                                                                 source_image_shape=[10, 10],
                                                                                 rangle=r_angle,
                                                                                 warped_offset=[-10, 5],
                                                                                 flip_ud=False)

        p1 = [[0, 0],
              [10, 10]]

        TransformAgreementCheck(r, m, p1)

    def testRigidvsMeshFactoryRotation(self):
        d_angle = 90
        r_angle = (d_angle / 180.0) * np.pi
        self.runRigidvsMeshFactoryRotationTranslate(shape=(10, 10), source_offset=(0, 0), r_angle=r_angle)

    def testRigidvsMeshFactoryRotationTranslate(self):
        d_angle = 90
        r_angle = (d_angle / 180.0) * np.pi
        self.runRigidvsMeshFactoryRotationTranslate(shape=(10, 10), source_offset=(-10, 5), r_angle=r_angle)

    @hypothesis.given(shape=st.tuples(st.integers(min_value=1, max_value=15), st.integers(min_value=1, max_value=15)),
                      source_offset=st.tuples(st.floats(min_value=-15, max_value=15),
                                              st.floats(min_value=-15, max_value=15)),
                      r_angle=st.floats(min_value=-np.pi, max_value=np.pi))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=20))  # Cupy Context takes a while to initialize
    def testRigidvsMeshFactoryHypothesis(self, shape: tuple[int, int], source_offset: tuple[float, float],
                                         r_angle: float):
        self.runRigidvsMeshFactoryRotationTranslate(shape=shape, source_offset=source_offset, r_angle=r_angle)

    def runRigidvsMeshFactoryRotationTranslate(self, shape: tuple[int, int], source_offset: tuple[float, float],
                                               r_angle: float):
        xp = nornir_imageregistration.GetComputationModule()
        shape = np.array(shape, dtype=int)
        source_offset = np.array(source_offset, dtype=float)
        r = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=shape,
                                                                             source_image_shape=shape,
                                                                             rangle=r_angle,
                                                                             warped_offset=source_offset,
                                                                             flip_ud=False)

        m = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=shape,
                                                                                 source_image_shape=shape,
                                                                                 rangle=r_angle,
                                                                                 warped_offset=source_offset,
                                                                                 flip_ud=False)

        p1 = xp.array([(0, 0),
                       shape / 2.0,
                       shape])

        TransformAgreementCheck(r, m, p1)


class TestCenteredSimilarityToITKString(unittest.TestCase):
    """CenteredSimilarity2DTransform saves simpler Rigid2D strings when scalar is unity."""

    def test_unity_scale_translate_only_uses_rigid_string(self) -> None:
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(3.0, 7.0),
            source_rotation_center=(0, 0),
            angle=0,
            scalar=1,
        )
        itk_string = transform.ToITKString()
        self.assertTrue(itk_string.startswith("Rigid2DTransform_double_2_2"))
        self.assertFalse(itk_string.startswith("CenteredSimilarity2DTransform"))

    def test_unity_scale_with_rotation_uses_rigid_string(self) -> None:
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(1.0, 2.0),
            source_rotation_center=(10.0, 20.0),
            angle=0.25,
            scalar=1,
        )
        itk_string = transform.ToITKString()
        self.assertTrue(itk_string.startswith("Rigid2DTransform_double_2_2"))

    def test_non_unity_scale_uses_similarity_string(self) -> None:
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(1.0, 2.0),
            source_rotation_center=(0, 0),
            angle=0.1,
            scalar=1.05,
        )
        itk_string = transform.ToITKString()
        self.assertTrue(itk_string.startswith("CenteredSimilarity2DTransform_double_2_2"))
        self.assertNotIn("FixedCenterOfRotationAffine", itk_string)

    def test_unflipped_rigid_family_never_uses_affine(self) -> None:
        """Affine ITK strings are only for flip_ud; unflipped stay Rigid2D/CS2D."""
        cases = [
            nornir_imageregistration.transforms.RigidTranslation((1.0, 2.0)),
            nornir_imageregistration.transforms.Rigid(
                (1.0, 2.0), (3.0, 4.0), 0.2, flip_ud=False),
            nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
                (1.0, 2.0), (3.0, 4.0), 0.2, 1.0, flip_ud=False),
            nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
                (1.0, 2.0), (3.0, 4.0), 0.2, 1.05, flip_ud=False),
        ]
        for transform in cases:
            with self.subTest(type=type(transform).__name__, scalar=getattr(transform, "scalar", 1)):
                itk = transform.ToITKString()
                self.assertFalse(
                    itk.startswith("FixedCenterOfRotationAffineTransform"),
                    f"unflipped {type(transform).__name__} must not use Affine: {itk}")
                self.assertFalse(bool(getattr(transform, "flip_ud", False)))

    def test_flip_ud_uses_affine_string_and_round_trips(self) -> None:
        """flip_ud must persist via FixedCenterOfRotationAffine and restore flip_ud."""
        for scale in (1.0, 1.05):
            with self.subTest(scale=scale):
                transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
                    target_offset=(10.0, -20.0),
                    source_rotation_center=(100.0, 200.0),
                    angle=0.5,
                    scalar=scale,
                    flip_ud=True,
                )
                itk_string = transform.ToITKString()
                self.assertTrue(
                    itk_string.startswith("FixedCenterOfRotationAffineTransform_double_2_2"))
                ref_points = np.array(
                    [[0.0, 0.0], [50.0, 80.0], [100.0, 200.0], [30.0, 40.0]], dtype=float)
                ref_targets = transform.Transform(ref_points)
                loaded = nornir_imageregistration.transforms.LoadTransform(itk_string)
                self.assertTrue(getattr(loaded, "flip_ud", False))
                np.testing.assert_allclose(
                    nornir_imageregistration.EnsureNumpyArray(loaded.Transform(ref_points)),
                    nornir_imageregistration.EnsureNumpyArray(ref_targets),
                    atol=1e-5)

    def test_rigid_flip_ud_round_trips(self) -> None:
        transform = nornir_imageregistration.transforms.Rigid(
            target_offset=(3.0, -4.0),
            source_rotation_center=(50.0, 60.0),
            angle=0.3,
            flip_ud=True,
        )
        itk_string = transform.ToITKString()
        self.assertTrue(itk_string.startswith("FixedCenterOfRotationAffineTransform_double_2_2"))
        ref_points = np.array([[0.0, 0.0], [10.0, 20.0], [40.0, 50.0]], dtype=float)
        ref_targets = transform.Transform(ref_points)
        loaded = nornir_imageregistration.transforms.LoadTransform(itk_string)
        self.assertTrue(getattr(loaded, "flip_ud", False))
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(loaded.Transform(ref_points)),
            nornir_imageregistration.EnsureNumpyArray(ref_targets),
            atol=1e-5)

    def test_non_unity_scale_round_trip_preserves_geometry(self) -> None:
        """Scaled CS2D must round-trip through ITK strings without flipping angle sign."""
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(-92.46865844726562, 91.00515747070312),
            source_rotation_center=(234.70594787597656, 361.69439697265625),
            angle=7.613996145581269,
            scalar=1.0466634271776583,
        )
        ref_points = np.array([[0.0, 0.0], [100.0, 200.0], [30.0, 40.0]], dtype=float)
        ref_targets = transform.Transform(ref_points)
        loaded = nornir_imageregistration.transforms.LoadTransform(transform.ToITKString())
        np.testing.assert_allclose(loaded.angle, transform.angle, atol=1e-9)
        np.testing.assert_allclose(loaded.Transform(ref_points), ref_targets, atol=1e-5)

    def test_unity_scale_round_trip_preserves_geometry(self) -> None:
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(4.0, 5.0),
            source_rotation_center=(1.0, 2.0),
            angle=0.15,
            scalar=1,
        )
        ref_points = np.array([[0, 0], [10, 20], [30, 40]], dtype=float)
        ref_targets = transform.Transform(ref_points)
        loaded = nornir_imageregistration.transforms.LoadTransform(transform.ToITKString())
        np.testing.assert_allclose(loaded.Transform(ref_points), ref_targets, atol=1e-5)

    def test_non_unity_scale_round_trip_preserves_centered_geometry(self) -> None:
        """ITK string round-trip must keep scale-about-center (not origin) semantics."""
        center = np.array([100.0, 200.0], dtype=float)
        peak = np.array([3.0, -5.0], dtype=float)
        scale = 1.05
        # angle=0 isolates scale-about-center + translate geometry from rotation.
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=peak,
            source_rotation_center=center,
            angle=0.0,
            scalar=scale,
        )
        offset_pt = center + np.array([10.0, -8.0], dtype=float)
        ref_points = np.vstack([center, offset_pt, np.array([0.0, 0.0], dtype=float)])
        ref_targets = transform.Transform(ref_points)
        itk_string = transform.ToITKString()
        self.assertTrue(itk_string.startswith("CenteredSimilarity2DTransform_double_2_2"))
        loaded = nornir_imageregistration.transforms.LoadTransform(itk_string)
        np.testing.assert_allclose(loaded.Transform(ref_points), ref_targets, atol=1e-4)

        # Center moves by translation only; offset point uses centered scale (not s*p).
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(loaded.Transform(center.reshape(1, 2)))[0],
            center + peak,
            atol=1e-3)
        expected_offset = center + scale * (offset_pt - center) + peak
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(loaded.Transform(offset_pt.reshape(1, 2)))[0],
            expected_offset,
            atol=1e-3)

        # With rotation, center still round-trips (scale must not inject origin bias into t).
        rotated = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=peak,
            source_rotation_center=center,
            angle=0.2,
            scalar=scale,
        )
        rotated_loaded = nornir_imageregistration.transforms.LoadTransform(rotated.ToITKString())
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(rotated.Transform(center.reshape(1, 2)))[0],
            nornir_imageregistration.EnsureNumpyArray(rotated_loaded.Transform(center.reshape(1, 2)))[0],
            atol=1e-3)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(rotated_loaded.Transform(center.reshape(1, 2)))[0],
            center + peak,
            atol=1e-3)

    def test_unity_scale_save_load_save_is_idempotent(self) -> None:
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(2.0, 3.0),
            source_rotation_center=(0, 0),
            angle=0,
            scalar=1,
        )
        ref_points = np.array([[0, 0], [5, 10]], dtype=float)
        ref_targets = transform.Transform(ref_points)
        first_string = transform.ToITKString()
        reloaded = nornir_imageregistration.transforms.LoadTransform(first_string)
        second_string = nornir_imageregistration.transforms.TransformToIRToolsString(reloaded)
        self.assertTrue(first_string.startswith("Rigid2DTransform_double_2_2"))
        self.assertTrue(second_string.startswith("Rigid2DTransform_double_2_2"))
        np.testing.assert_allclose(reloaded.Transform(ref_points), ref_targets, atol=1e-5)


class TestConvertTransformToRigidTransform(unittest.TestCase):
    """Rigid conversions should emit CenteredSimilarity2DTransform for scaling support."""

    def test_plain_rigid_becomes_similarity(self) -> None:
        rigid = nornir_imageregistration.transforms.Rigid(
            target_offset=(1.0, 2.0),
            source_rotation_center=(3.0, 4.0),
            angle=0.2,
        )
        converted = nornir_imageregistration.transforms.ConvertTransformToRigidTransform(rigid)
        self.assertIsInstance(
            converted, nornir_imageregistration.transforms.CenteredSimilarity2DTransform)
        self.assertAlmostEqual(converted.scalar, 1.0)
        self.assertTrue(hasattr(converted, "ScaleWarped"))

    def test_rigid_translation_becomes_similarity(self) -> None:
        rigid = nornir_imageregistration.transforms.RigidTranslation(target_offset=(5.0, 6.0))
        converted = nornir_imageregistration.transforms.ConvertTransformToRigidTransform(rigid)
        self.assertIsInstance(
            converted, nornir_imageregistration.transforms.CenteredSimilarity2DTransform)
        self.assertAlmostEqual(converted.scalar, 1.0)


class TestScaleWarpedAboutSourcePoint(unittest.TestCase):
    """ScaleWarpedAboutSourcePoint pins the pivot in target space."""

    def test_pivot_unchanged_in_target_space(self) -> None:
        pivot = np.array([12.0, 34.0], dtype=np.float32)
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(5.0, 6.0),
            source_rotation_center=(0.0, 0.0),
            angle=0.2,
            scalar=1.0,
        )
        target_before = np.squeeze(transform.Transform(pivot.reshape(1, 2)))
        transform.ScaleWarpedAboutSourcePoint(1.1, pivot)
        target_after = np.squeeze(transform.Transform(pivot.reshape(1, 2)))
        np.testing.assert_allclose(target_before, target_after, atol=1e-4)

    def test_off_pivot_point_moves(self) -> None:
        pivot = np.array([0.0, 0.0], dtype=np.float32)
        off_pivot = np.array([10.0, 0.0], dtype=np.float32)
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=(0.0, 0.0),
            source_rotation_center=(50.0, 50.0),
            angle=0.0,
            scalar=1.0,
        )
        target_before = np.squeeze(transform.Transform(off_pivot.reshape(1, 2)))
        transform.ScaleWarpedAboutSourcePoint(2.0, pivot)
        target_after = np.squeeze(transform.Transform(off_pivot.reshape(1, 2)))
        self.assertFalse(np.allclose(target_before, target_after, atol=1e-4))
        self.assertAlmostEqual(transform.scalar, 0.5)
