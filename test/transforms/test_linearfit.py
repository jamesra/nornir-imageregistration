# -*- coding: utf-8 -*-

import unittest

from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import scipy
from scipy.stats import linregress

import nornir_imageregistration
import nornir_imageregistration.transforms

import unittest
import datetime
import math

import hypothesis
import hypothesis.strategies as st
import numpy as np
from numpy.typing import NDArray

try:
    from transforms.checks import TransformAgreementCheck, TransformCheck, TransformInverseCheck
except ImportError:
    from test.transforms.checks import TransformAgreementCheck, TransformCheck, TransformInverseCheck

epsilon = np.finfo(float).eps


class TestLinearFit(unittest.TestCase):

    @hypothesis.given(x=arrays(np.float64, 10, elements=st.floats(0, 10), unique=True),
                      y_noise=arrays(np.float64, 10, elements=st.floats(-1, 1), unique=True),
                      slope=st.floats(min_value=-10, max_value=10),
                      y_intercept=st.floats(min_value=-10, max_value=10))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=20))  # Cupy Context takes a while to initialize
    def test_linearFit1D(self, x: NDArray[float], y_noise: NDArray[float], slope: float, y_intercept: float):
        """
        Does a linear regression of a simple line with artificial noise
        :return:
        """
        points1 = x

        x = points1
        y = ((slope * x)) + y_intercept

        lr_slope, lr_intercept, lr_r_value, lr_p_value, lr_std_err = linregress(x, y)
        # distanceVector = ((-1 * y) + (lr_slope * x) + lr_intercept) / np.sqrt(abs(-1 ** 2 + lr_slope ** 2))
        distance = ((lr_slope * x) + lr_intercept) - y
        print(distance)
        sumOfDistances = np.sum(distance)

        # Testing if linear fit has all the distances of points to itself = 0
        np.testing.assert_allclose(sumOfDistances, 0, atol=1e-5)

        np.testing.assert_allclose(y_intercept, lr_intercept, atol=1e-5)
        np.testing.assert_allclose(slope, lr_slope, atol=1e-5)

        # =============================================================================

    #         print("SLOPE: ",slope)
    #         print("INTERCEPT: ",intercept)
    #         print(slope*0 + intercept)
    #         print("R-Squared: ",r_value**2)
    #         plt.plot(x,y,'o',label="original data")
    #         plt.plot(x,intercept+slope*x,'r',label="fitted data")
    #         plt.legend()
    #         plt.show()
    #
    # =============================================================================

    @hypothesis.given(points2D=arrays(np.float64, (20, 2), elements=st.floats(0, 30), unique=True),
                      translate=st.tuples(st.floats(min_value=-10, max_value=10),
                                          st.floats(min_value=-10, max_value=10)),
                      angle=st.floats(min_value=-np.pi, max_value=np.pi),
                      scale=st.floats(min_value=0.1, max_value=10),
                      source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
                                                       st.floats(min_value=-15, max_value=15)))
    def test_linearFit2D(self, points2D: NDArray[float],
                         translate: NDArray[float],
                         angle: float,
                         scale: float,
                         source_rotation_center: NDArray[float]):
        translate = np.array(translate)
        source_rotation_center = np.array(source_rotation_center)

        xv = np.arange(-5, 6)
        yv = np.arange(-5, 6)
        xx, yy = np.meshgrid(xv, yv)
        xx = xx.flatten()
        yy = yy.flatten()
        gridPoints = np.transpose(np.vstack((xx, yy)))
        # print(gridPoints)
        # plt.scatter(xx, yy)
        # plt.show()
        # print(points2D[:, 0])
        # print(points2D[:, 1])
        numPoints = len(gridPoints)

        # numPoints = len(points2D)
        # print(numPoints)
        # slope1,intercept1,r_value1,p_value1,std_err1 = linregress(points2D[:,0],points2D[:,1])
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(xx, yy)

        # translate =np.array([0,0])
        print("Translate: ", translate)
        # rotate = -np.pi / 2 + np.pi * np.random.rand()
        print("Rotate ", angle)
        # scale = np.random.randint(1, 10)
        print('Scaling by: ', scale)
        # center_rotation = np.random.randint(-10,10,size=2)
        # source_rotation_center = np.array([0, 0])
        print("Center of rotation: ", source_rotation_center)

        points2D1 = gridPoints - source_rotation_center
        points2D1 = np.transpose(points2D1)
        points2D1 = np.vstack((points2D1, np.ones((1, numPoints))))

        # points2D1 = points2D1 * scale
        rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(angle)
        # print(rotation_matrix)
        points2D1_rotated = rotation_matrix @ points2D1
        # print(points2D1_rotated)
        points2D1_rotated = np.transpose(points2D1_rotated)
        # print(points2D1_rotated)

        points2D1_rotated = points2D1_rotated[:, 0:2]
        points2D1_rotated += source_rotation_center

        output_points2D = points2D1_rotated + translate

        output_x = output_points2D[:, 0]
        output_y = output_points2D[:, 1]
        # print(output_x)
        # print(output_points2D)
        # print(output_points2D[:,0])
        # print(output_points2D.shape)

        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(output_x, output_y)

        # Took four sample points at different x coordinates.
        sample_x1 = 0
        sample_x2 = 1
        sample_x3 = -1
        sample_x4 = 2
        beforeFit1 = slope1 * sample_x1 + intercept1
        afterFit1 = slope2 * sample_x1 + intercept2
        beforeFit2 = slope1 * sample_x2 + intercept1
        afterFit2 = slope2 * sample_x2 + intercept2
        beforeFit3 = slope1 * sample_x3 + intercept1
        afterFit3 = slope2 * sample_x3 + intercept2
        beforeFit4 = slope1 * sample_x4 + intercept1
        afterFit4 = slope2 * sample_x4 + intercept2

        beforeVector = np.array([[0, 1, -1], [beforeFit1, beforeFit2, beforeFit3]])
        afterVector = np.array([[0, 1, -1], [afterFit1, afterFit2, afterFit3]])
        # A = np.dot(afterVector,np.linalg.inv(beforeVector))
        # print(A)

        centroid_before = np.mean(beforeVector, axis=1).reshape(-1, 1)
        centroid_after = np.mean(afterVector, axis=1).reshape(-1, 1)

        center_centroid_before = beforeVector - centroid_before
        center_centroid_after = afterVector - centroid_after

        H = np.matmul(center_centroid_after, np.transpose(center_centroid_before))
        # print(H)
        U, S, VH = np.linalg.svd(H)
        R = np.matmul(U, VH.T)
        # print(VH)
        # print(np.linalg.det(R))
        if np.linalg.det(R) < 0:
            print("Correcting for reflection!...")
            VH[:, 1] *= -1
            R = U @ VH.T
        print(f"Rotation Matrix {R}")
        # fit_rotation = np.arctan(R[1,1]/R[0,0])
        # print(fit_rotation)
        t = -R @ centroid_before + centroid_after
        estimated_angle = np.arctan2(R[0, 1], R[0, 0])
        self.assertAlmostEqual(angle, estimated_angle, places=3)
        print(f"Translate Estimate: {t}")

        np.testing.assert_allclose(translate, t, atol=1e-5)

    # =============================================================================
    #         fit_translation = t
    #         print(fit_translation)
    #         matrix = cv2.estimateAffinePartial2D(beforeVector.T,afterVector.T)
    #         print(matrix)
    # =============================================================================
    # =============================================================================
    #         plt.plot(points2D[0],points2D[1],'o')
    #         plt.plot(points2D[0],intercept+slope*points2D[0],'r',label="fitted line")
    #         plt.show()
    # =============================================================================
    @hypothesis.given(points2D=arrays(np.float64, (20, 2), elements=st.floats(0, 30), unique=True),
                      translate=st.tuples(st.floats(min_value=-10, max_value=10),
                                          st.floats(min_value=-10, max_value=10)),
                      angle=st.floats(min_value=-np.pi + epsilon, max_value=np.pi),
                      scale=st.floats(min_value=0.1, max_value=10),
                      source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
                                                       st.floats(min_value=-15, max_value=15)),
                      flip_ud=st.booleans())
    def test_linearFit2DKabschUmeyama(self, points2D: NDArray[float],
                                      translate: NDArray[float],
                                      angle: float,
                                      scale: float,
                                      source_rotation_center: NDArray[float],
                                      flip_ud: bool):
        """
        A set of initial random points is created and scipy's linregress function is used
        to get the linear fit.

        This is commented out and then a randomly generated array of size 20x2 is created
        with the first column containing x values and second containing y values.


        After applying rotation, translation and scaling to those points (randomly generated transformation components),
        linear fit is then generated on the new set of points. Later, taking a sample set of x coordinates,
        the corresponding y coordinates are found for both the fits. Now, with a set of points (warp: before and fixed: after),
        the Kabsch Umeyama algorithm is used to find optimal rotation, translation and scaling between the fits.

        Testing is then done to see if the factors obtained are the same as the initial randomly generated ones.

        """

        # warpPoints = arrays(np.float64, (20,2), elements=st.floats(0, 30),unique=True).example()
        # warpPoints = np.random.randint(-50, 50, size=(100, 2))
        warpPoints = points2D
        # refPoints = arrays(np.float64, (20,2), elements=st.floats(0, 30),unique=True).example()

        # =============================================================================
        #         xv = np.arange(-5,6)
        #         yv = np.arange(-5,6)
        #         xx,yy = np.meshgrid(xv,yv)
        #         xx = xx.flatten()
        #         yy = yy.flatten()
        # =============================================================================
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(warpPoints[:, 0], warpPoints[:, 1])
        # slope1,intercept1,r_value1,p_value1,std_err1 = linregress(xx,yy)

        # grid = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
        # print(grid)
        print("\n\nWarp points: ", warpPoints)
        # n,m = grid.shape
        n, m = warpPoints.shape

        # translate =np.array([0,0])
        print("Translate: ", translate)
        rotate = angle
        print("Rotate ", rotate)
        print('Scaling by: ', scale)
        center_rotation = source_rotation_center
        print("Center of rotation: ", center_rotation)

        # points2D1 = np.transpose(grid)
        points2D1 = np.transpose(warpPoints)
        points2D1 = np.vstack((points2D1, np.ones((1, n))))

        # points2D1 = points2D1 * scale
        # rotation_matrix = nornir_imageregistration.transforms.utils.IdentityMatrix()#nornir_imageregistration.transforms.utils.RotationMatrix(rotate)
        rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(rotate)
        # print(rotation_matrix)
        points2D1_rotated = rotation_matrix @ points2D1
        # print(points2D1_rotated)
        points2D1_rotated = np.transpose(points2D1_rotated)
        # print(points2D1_rotated)

        points2D1_rotated = points2D1_rotated[:, 0:2]

        # output_points2D = points2D1_rotated
        output_points2D = points2D1_rotated + translate

        # print(output_points2D)
        output_x = output_points2D[:, 0]
        output_y = output_points2D[:, 1]

        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(output_x, output_y)

        # Generating points on each of the linear fits in order to pass them to algorithm.
        sample_x_vector = np.arange(-100, 100)
        beforeFit = slope1 * sample_x_vector + intercept1
        afterFit = slope2 * sample_x_vector + intercept2
        beforeVector = np.hstack((sample_x_vector.reshape(-1, 1), beforeFit.reshape(-1, 1)))
        afterVector = np.hstack((sample_x_vector.reshape(-1, 1), afterFit.reshape(-1, 1)))

        # =============================================================================
        #         sample_x1 = 10
        #         sample_x2 = 20
        #         sample_x3 = -10
        #         beforeFit1 = slope1*sample_x1 + intercept1
        #         afterFit1 = slope2*sample_x1 + intercept2
        #         beforeFit2 = slope1*sample_x2 + intercept1
        #         afterFit2 = slope2*sample_x2 + intercept2
        #         beforeFit3 = slope1*sample_x3 + intercept1
        #         afterFit3 = slope2*sample_x3 + intercept2
        #
        #         beforeVector = np.array([[sample_x1,beforeFit1],[sample_x2,beforeFit2],[sample_x3,beforeFit3]])
        #         afterVector = np.array([[sample_x1,afterFit1],[sample_x2,afterFit2],[sample_x3,afterFit3]])
        # =============================================================================
        # print(beforeVector)
        # print(afterVector)

        centroid_before = np.mean(beforeVector, axis=0)
        centroid_after = np.mean(afterVector, axis=0)

        varianceAfter = np.mean(np.linalg.norm(afterVector - centroid_after, axis=1) ** 2)

        H = ((afterVector - centroid_after).T @ (beforeVector - centroid_before)) / n

        U, D, VT = scipy.linalg.svd(H)

        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))

        S = np.diag([1] * (m - 1) + [d])

        rotateResult = U @ S @ VT

        scaleResult = varianceAfter / np.trace(np.diag(D) @ S)

        translateResult = centroid_after - scaleResult * rotateResult @ centroid_before

        angleResult = np.arctan2(rotateResult[1, 0], rotateResult[0, 0])

        foundAngle = np.arctan2(rotateResult[1, 0], rotateResult[0, 0])
        print(f"Reflection: found={relface}")
        print(f"Rotation angle:\n\tfound={foundAngle}\n vs\n\toriginal={rotate}", )
        print(f"\nResulting rotation matrix: found={rotateResult} vs original={rotation_matrix}")
        print(f"Resulting scaling: found={scaleResult} vs original={scale}")
        print(f"Resulting translation: : found={translateResult} vs original={translate}")

        # self.assertAlmostEqual(rotate, angleResult, places=4)
        # self.assertAlmostEqual(scale, scaleResult, places=4)
        # np.testing.assert_allclose(translate.flatten(), translateResult.flatten())
        #
        # np.testing.assert_allclose(center_rotation.flatten(), rotateResult.flatten())
        # np.testing.assert_allclose(source_rotation_center.flatten(), r.source_rotation_center.flatten())
        # self.assertEqual(flip_ud, r.flip_ud)

    @hypothesis.given(source_points=arrays(np.float64, (20, 2), elements=st.floats(0, 30), unique=True),
                      translate=st.tuples(st.floats(min_value=-10, max_value=10),
                                          st.floats(min_value=-10, max_value=10)),
                      angle=st.floats(min_value=-np.pi, max_value=np.pi),
                      scale=st.floats(min_value=0.1, max_value=10),
                      source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
                                                       st.floats(min_value=-15, max_value=15)),
                      flip_ud=st.booleans())
    def test_EstimateScale(self, source_points: NDArray[float],
                           translate: NDArray[float],
                           angle: float,
                           scale: float,
                           source_rotation_center: NDArray[float],
                           flip_ud: bool):
        """
        A set of initial random points is created along with random rigid transform parameters.  The input
        points are then transformed to the target space.

        We attempt to reverse engineer the scale difference between the sets of points.
        """

        source_points = np.array(source_points)
        n, m = source_points.shape

        # translate =np.array([0,0])
        rotate = angle
        #
        # print("Translate: ", translate)
        # print("Rotate ", rotate)
        # print('Scaling by: ', scale)
        # print("Center of rotation: ", source_rotation_center)
        # print(f"Flip Up/Down: {flip_ud}")

        forward_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=translate,
            source_rotation_center=source_rotation_center,
            angle=rotate,
            scalar=scale,
            flip_ud=flip_ud)

        target_points = forward_transform.Transform(source_points)

        # print("\n\nTarget points: ", target_points)

        scale_estimate = nornir_imageregistration.transforms.converters.EstimateScale(source_points, target_points)
        self.assertAlmostEqual(scale, scale_estimate, places=4)

    # verbose_settings = hypothesis.settings(verbosity=hypothesis.Verbosity.verbose)

    @hypothesis.given(source_points=arrays(np.float64, (10, 2), elements=st.floats(-10, 10), unique=True),
                      translate=st.tuples(st.floats(min_value=-10, max_value=10),
                                          st.floats(min_value=-10, max_value=10)),
                      angle=st.floats(min_value=-np.pi + 0.0001, max_value=np.pi),
                      scale=st.floats(min_value=0.1, max_value=10),
                      source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
                                                       st.floats(min_value=-15, max_value=15)),
                      flip_ud=st.booleans())
    @hypothesis.settings(verbosity=hypothesis.Verbosity.verbose, )
    def test_linearFit_improved(self, source_points: NDArray[float],
                                translate: NDArray[float],
                                angle: float,
                                scale: float,
                                source_rotation_center: NDArray[float],
                                flip_ud: bool):
        """
        A set of initial random points is created along with random rigid transform parameters.  The input
        points are then transformed to the target space.

        We attempt to reverse engineer the rigid transform using source and target point pairs.

        1. Determine the scale
        2. Determine the translation to the center of rotation
        3. Determine the rotation matrix
        4. Determine if the points are inverted
        5. Determine the translation
        """

        if angle <= -math.pi:
            angle += math.pi * 2

        # grid = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
        # print(grid)
        source_points = np.array(source_points)
        # print("\n\nSource points: ", source_points)
        # n,m = grid.shape
        num_pts, m = source_points.shape

        # This is a rigid transform, so transform points to the origin to reduce floating point error
        source_points -= np.mean(source_points, axis=0)

        print("Translate: ", translate)
        print("Rotate ", angle)
        print('Scaling by: ', scale)
        print("Center of rotation: ", source_rotation_center)
        print(f"Flip Up/Down: {flip_ud}")

        forward_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=translate,
            source_rotation_center=source_rotation_center,
            angle=angle,
            scalar=scale,
            flip_ud=flip_ud)

        target_points = forward_transform.Transform(source_points)

        # print("\n\nTarget points: ", target_points)

        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)
        centered_source_points = source_points - source_center
        centered_target_points = target_points - source_center

        scale_estimate = nornir_imageregistration.transforms.converters.EstimateScale(centered_source_points,
                                                                                      centered_target_points)
        self.assertAlmostEqual(scale, scale_estimate, places=4)

        ###################################################################################
        # Past this point the scale is known, remove the scalar from the target_points, and
        # continue with the rest of the algorithm
        unscaled_target_points = target_points / scale_estimate
        unscaled_target_center = np.mean(unscaled_target_points, axis=0)
        unscaled_centered_target_points = unscaled_target_points - unscaled_target_center

        zeros_z_column = np.zeros((num_pts, 1))
        rotation = scipy.spatial.transform.Rotation.align_vectors(
            np.hstack((zeros_z_column, centered_source_points)),
            np.hstack(
                (zeros_z_column, unscaled_centered_target_points))
        )
        euler_angles = rotation[0].as_euler('zyx')
        estimated_angle = euler_angles[2]

        # my_rotation = nornir_imageregistration.transforms.converters._kabsch_umeyama(source_points=source_points,
        #                                                                              target_points=unscaled_target_points)
        my_rotation = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(
            source_points=source_points,
            target_points=unscaled_target_points)

        self.assertAlmostEqual(angle, estimated_angle, places=3)

        ###################################################################################
        # Determine if the transform is reflected
        relation = nornir_imageregistration.transforms.converters.calculate_control_points_relationship(source_points,
                                                                                                        target_points)
        reflected = relation == nornir_imageregistration.transforms.ControlPointRelation.FLIPPED

        if relation == nornir_imageregistration.transforms.ControlPointRelation.COLINEAR:
            hypothesis.note("Colinear points detected")
            return

        self.assertEqual(flip_ud, reflected)

        ###################################################################################
        # Past this point the angle is estimated.  We next remove the angle from the target
        # points and continue with the rest of the algorithm

        rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(estimated_angle)

        transform_without_translate = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=np.zeros((2,)),
            source_rotation_center=np.zeros((2,)),
            angle=estimated_angle,
            scalar=scale_estimate,
            flip_ud=reflected)

        untranslated_target_points = transform_without_translate.Transform(source_points)
        untranslated_center = np.mean(untranslated_target_points, axis=0)
        translation_estimate = np.mean(untranslated_target_points - source_points, axis=0)

        translation_estimate_2 = np.hstack((0, target_center)) - (
                scale * rotation_matrix @ np.hstack((0, source_center)))

        ################################################################################################
        # Past this point the reflection is known, we next remove the reflection from the target points

        estimated_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=translation_estimate_2[1:],
            source_rotation_center=np.zeros((2,)),
            angle=estimated_angle,
            scalar=scale_estimate,
            flip_ud=reflected)

        test_target_points = estimated_transform.Transform(source_points)
        np.testing.assert_allclose(target_points, test_target_points, atol=1e-5)

        return
 
    @hypothesis.given(r_angle=st.floats(min_value=-np.pi, max_value=np.pi),
                      target_offset=st.tuples(st.floats(min_value=-15, max_value=15),
                                              st.floats(min_value=-15, max_value=15)),
                      source_rotation_center=st.tuples(st.floats(min_value=-15, max_value=15),
                                                       st.floats(min_value=-15, max_value=15)),
                      flip_ud=st.booleans(),
                      source_points=st.lists(
                          st.tuples(st.integers(min_value=-15, max_value=15),
                                    st.integers(min_value=-15, max_value=15)),
                          unique=True, min_size=3, max_size=12))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=20))  # Cupy Context takes a while to initialize
    def testFitTransform(self,
                         r_angle: float,
                         target_offset: np.ndarray,
                         source_rotation_center: np.ndarray,
                         flip_ud: bool,
                         source_points: list[tuple[float, float]]):
        """Generates a set of points in source space and random rigid transform parameters.
        Verifies that the rigid transform can transform and invert the points to get the original points.
        Then uses the transformed target space points and source points to estimate a rigid transformation
        and checks if the estimated parameters are the same as the original ones"""
        source_point_array = np.array(source_points, dtype=float)
        source_rotation_center = np.array([source_rotation_center], dtype=float)
        target_offset = np.array([target_offset], dtype=float)
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

        similar_target_point_array = transform_similar.Transform(source_point_array)
        np.testing.assert_array_almost_equal(target_point_array, similar_target_point_array)

        TransformCheck(self, transform_similar, source_point_array, target_point_array)

        r = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(
            source_points=source_point_array,
            target_points=target_point_array)

        self.assertAlmostEqual(r_angle, r.angle, places=4)
        np.testing.assert_allclose(target_offset.flatten(), r.translation.flatten())
        np.testing.assert_allclose(source_rotation_center.flatten(), r.source_rotation_center.flatten())
        self.assertEqual(flip_ud, r.reflected)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
