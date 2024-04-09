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
    from test.transforms.data import OffsetTransformPoints, \
        TranslateRotateTransformPoints, IdentityFlippedUDTransformPoints, TranslateFlippedUDTransformPoints, \
        TranslateRotateFlippedTransformPoints
    from test.transforms.checks import TransformAgreementCheck, TransformCheck, TransformInverseCheck

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

    def testScaleWithTranslation(self):
        xp = nornir_imageregistration.GetComputationModule()
        angle = 0
        scale = 10
        offset = xp.array((5, 5))
        T = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(offset, source_rotation_center=[0, 0],
                                                                              angle=angle,
                                                                              scalar=10)

        sourcePoint = xp.array([[0, 1],
                                [2, 0],
                                [-2, -1]])

        targetPoint = (sourcePoint * scale) + offset

        TransformCheck(self, T, sourcePoint, targetPoint)


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
