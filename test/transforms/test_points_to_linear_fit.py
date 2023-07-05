# -*- coding: utf-8 -*-
import unittest

import hypothesis
import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration.transforms
from test.transforms import TranslateRotateTransformPoints


def angles_close(a_in: float, b_in: float, **kwargs) -> bool:
    """Returns true if two angles are approximately equal"""
    tau = np.pi * 2
    a_t = a_in / tau
    b_t = b_in / tau

    # Have to mod twice to handle rare floating point rounding errors
    a = a_t - int(a_t)
    b = b_t - int(b_t)

    if np.allclose(a, b, **kwargs):
        return True

    # Handle the wrap-around case where they are an epsilon apart
    if a < b:
        a += 1
    else:
        b += 1

    return np.allclose(a, b, **kwargs)


class TestGridFitting(unittest.TestCase):
    # Transformation components (c= Scaling, rangle= Rotation angle, R = rotation matrix, t = )

    point_value_strategy = hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=-1e6,
                                                        max_value=1e6, allow_subnormal=False)
    point_strategy = hypothesis.strategies.tuples(point_value_strategy, point_value_strategy)
    unique_points_strategy = hypothesis.strategies.lists(point_strategy, unique_by=lambda p: (int(p[0]), int(p[1])),
                                                         min_size=4, max_size=500)

    def test_reproduction1(self):
        self.runFit(rangle=1.0,
                    translate=(0.0, 0.0),
                    scale=1.0,
                    flip_ud=False,
                    points=[(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)],
                    hypothesis_test=False)

    @hypothesis.settings(verbosity=hypothesis.Verbosity.normal, deadline=None)
    @hypothesis.given(translate=point_strategy,
                      scale=hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=0.01,
                                                         max_value=100),
                      points=unique_points_strategy)
    # @hypothesis.example(rangle=0.0, translate=(0.0, 0.0), scale=1.0, flip_ud=True, points=[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    # @hypothesis.example(rangle=0.0, translate=(0.0, 0.0), scale=1.0, flip_ud=False, points=[(0.0, 0.0), (0.0, 2.0), (0.0, 1.0)])
    def test_hypothesis_no_rotation(self, translate: tuple[float, float], scale: float,
                                    points: list[tuple[float, float]]):
        self.runFit_norotation(translate, scale, False, points, hypothesis_test=True)

    @hypothesis.settings(verbosity=hypothesis.Verbosity.normal, deadline=None)
    @hypothesis.given(rangle=hypothesis.strategies.floats(-np.pi, np.pi, exclude_min=True),
                      translate=point_strategy,
                      scale=hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=0.01,
                                                         max_value=100),
                      points=unique_points_strategy)
    # @hypothesis.example(rangle=0.0, translate=(0.0, 0.0), scale=1.0, flip_ud=True, points=[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    # @hypothesis.example(rangle=0.0, translate=(0.0, 0.0), scale=1.0, flip_ud=False, points=[(0.0, 0.0), (0.0, 2.0), (0.0, 1.0)])
    def test_hypothesis_no_flip(self, rangle: float, translate: tuple[float, float], scale: float,
                                points: list[tuple[float, float]]):
        self.runFit(rangle, translate, scale, False, points, hypothesis_test=True)

    @hypothesis.settings(verbosity=hypothesis.Verbosity.normal, deadline=None)
    @hypothesis.given(rangle=hypothesis.strategies.floats(-np.pi, np.pi, exclude_min=True),
                      translate=point_strategy,
                      scale=hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=0.01,
                                                         max_value=100),
                      flip_ud=hypothesis.strategies.booleans(),
                      points=unique_points_strategy)
    # @hypothesis.example(rangle=0.0, translate=(0.0, 0.0), scale=1.0, flip_ud=True, points=[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    # @hypothesis.example(rangle=0.0, translate=(0.0, 0.0), scale=1.0, flip_ud=False, points=[(0.0, 0.0), (0.0, 2.0), (0.0, 1.0)])
    def test_hypothesis_with_flip(self, rangle: float, translate: tuple[float, float], scale: float, flip_ud: bool,
                                  points: list[tuple[float, float]]):
        self.runFit(rangle, translate, scale, flip_ud, points, hypothesis_test=True)

    @staticmethod
    def build_output_points(rangle: float, translate: tuple[float, float], scale: float, flip_ud: bool,
                            points: list[tuple[float, float]], hypothesis_test: bool) -> NDArray[float]:
        R = nornir_imageregistration.transforms.utils.RotationMatrix(rangle)
        # t = np.random.randint(1, 10, size=2)
        t = np.array(translate)
        # c = scale
        scale_matrix = nornir_imageregistration.transforms.utils.ScaleMatrixXY(scale)
        c = np.array((scale, scale, 1.0))
        points_array = np.array(points)

        # print(f'Starting angle: {rangle}')
        # print(f'Starting scale: {scale}')
        # print(R)
        # print(f'Starting translation: {t}')

        # B = np.random.randint(0, 4000, size=(num_pts, 2))
        # =============================================================================
        #         B = np.array([[232, 38],
        #                       [208, 32],
        #                       [181, 31],
        #                       [155, 45],
        #                       [142, 33],
        #                       [121, 59],
        #                       [139, 69]])
        # =============================================================================
        # Transforming the 2D grid (B) using rotation, scaling and translation.
        # numPoints = B.shape[0]
        # points2D1 = pts
        num_pts = points_array.shape[0]
        # points2D1 = np.transpose(points_array)
        points2D1 = np.array(points_array)
        points2D1 = np.hstack((points2D1, np.ones((num_pts, 1))))

        if flip_ud:
            points2D1 = np.flipud(points2D1)

            # points2D1 = np.transpose(points2D1)
        points2D1_scaled = (scale_matrix @ points2D1.T).T
        points2D1_rotated = (R @ points2D1_scaled.T).T

        # points2D1_rotated = np.transpose(points2D1_rotated)

        points2D1_rotated = points2D1_rotated[:, 0:2]

        output_points2D = points2D1_rotated + t

        if hypothesis_test:
            hypothesis.note(points)
            hypothesis.note(output_points2D)
        else:
            print(points)
            print(output_points2D)

        return output_points2D

    def runFit(self, rangle: float, translate: tuple[float, float], scale: float, flip_ud: bool,
               points: list[tuple[float, float]], hypothesis_test: bool):
        points_array = np.array(points)
        t = np.array(translate)
        num_pts = points_array.shape[0]
        output_points2D = TestGridFitting.build_output_points(rangle, translate, scale, flip_ud, points,
                                                              hypothesis_test)

        # =============================================================================
        # # Simplified code for applying transformation components on a grid
        # R_only = np.array([[np.cos(rangle),-np.sin(rangle)],[np.sin(rangle),np.cos(rangle)]])
        # output_points2D_OneLine = np.array([t + c * R_only @ b for b in B])
        # print(output_points2D_OneLine)
        # =============================================================================
        # Obtaining the rotation, scaling and translation factors back from the new grid and our original grid.
        calc_source_rotate_center, calc_rotate, calc_scale, calc_translate, calc_flipped = nornir_imageregistration.transforms.converters._kabsch_umeyama(
            output_points2D, points_array)

        calc_rotate_angle = np.arctan2(calc_rotate[0, 1], calc_rotate[0, 0])
        calc_rotate_angle = np.mod(calc_rotate_angle, np.pi * 2)
        # calc_rotate_angle = np.arcsin(calc_rotate[0, 1])
        # calc_rotate_angle = np.arctan2(calc_rotate[0, 1], calc_rotate[0, 0])
        if calc_rotate_angle < np.pi:
            calc_rotate_angle += np.pi * 2
        if calc_rotate_angle > np.pi:
            calc_rotate_angle -= np.pi * 2

        translate_output = np.squeeze(calc_translate)

        if hypothesis_test:
            if flip_ud:
                hypothesis.event(f'Flipped')

            if num_pts == 2:
                hypothesis.event(f'2 points')
            elif num_pts < 6:
                hypothesis.event(f'3-5 points')
            elif num_pts < 11:
                hypothesis.event(f'6-10 points')
            elif num_pts < 26:
                hypothesis.event(f'11-25 points')
            else:
                hypothesis.event('more than 25 points')

            hypothesis.note(f'Flipped U/D: {flip_ud} -> {calc_flipped}')
            hypothesis.note(f'Rotation Angle {rangle} -> {calc_rotate} -> {calc_rotate_angle}')
            hypothesis.note(f'Scale: {scale} -> {calc_scale}')
            hypothesis.note(f'Translation: {t} -> {translate_output}')
            hypothesis.note(f'Src Rot Center: {np.mean(points_array, 0)} -> {calc_source_rotate_center}')
        else:
            print(f'Rotation Angle {rangle} -> {calc_rotate} -> {calc_rotate_angle}')
            print(f'Scale: {scale} -> {calc_scale}')
            print(f'Translation: {t} -> {translate_output}')

        # Adjust rangle to wrap around

        self.assertEqual(flip_ud, calc_flipped, "Flipped not detected")
        self.assertTrue(angles_close(rangle, calc_rotate_angle, atol=1e-3))
        # np.testing.assert_allclose(rangle, calc_rotate_angle, atol=1e-3)
        self.assertTrue(np.allclose(scale, calc_scale, atol=1e-3), "scale incorrect")
        self.assertTrue(np.allclose(t, translate_output, atol=1e-3), "translation incorrect")
        self.assertTrue(np.allclose(calc_source_rotate_center, np.mean(points_array, 0)), "source center incorrect")
        return

    def runFit_norotation(self, translate: tuple[float, float], scale: float, flip_ud: bool,
                          points: list[tuple[float, float]], hypothesis_test: bool):
        points_array = np.array(points)
        t = np.array(translate)
        num_pts = points_array.shape[0]
        output_points2D = TestGridFitting.build_output_points(0, translate, scale, flip_ud, points, hypothesis_test)

        # =============================================================================
        # # Simplified code for applying transformation components on a grid
        # R_only = np.array([[np.cos(rangle),-np.sin(rangle)],[np.sin(rangle),np.cos(rangle)]])
        # output_points2D_OneLine = np.array([t + c * R_only @ b for b in B])
        # print(output_points2D_OneLine)
        # =============================================================================
        # Obtaining the rotation, scaling and translation factors back from the new grid and our original grid.
        calc_scale, calc_translate = nornir_imageregistration.transforms.converters._kabsch_umeyama_translation_scaling(
            output_points2D, points_array)

        translate_output = np.squeeze(calc_translate)

        if hypothesis_test:
            if flip_ud:
                hypothesis.event(f'Flipped')
            if num_pts == 2:
                hypothesis.event(f'2 points')
            elif num_pts < 6:
                hypothesis.event(f'3-5 points')
            elif num_pts < 11:
                hypothesis.event(f'6-10 points')
            elif num_pts < 26:
                hypothesis.event(f'11-25 points')
            else:
                hypothesis.event('more than 25 points')

            hypothesis.note(f'Scale: {scale} -> {calc_scale}')
            hypothesis.note(f'Translation: {t} -> {translate_output}')
        else:
            print(f'Scale: {scale} -> {calc_scale}')
            print(f'Translation: {t} -> {translate_output}')

        # Adjust rangle to wrap around

        # np.testing.assert_allclose(rangle, calc_rotate_angle, atol=1e-3)
        self.assertTrue(np.allclose(scale, calc_scale, atol=1e-3), "scale incorrect")
        self.assertTrue(np.allclose(t, translate_output, atol=1e-3), "translation incorrect")
        return

    def test_anglesclose(self):
        self.assertTrue(angles_close(0, 2 * np.pi), "Angles close works")
        self.assertTrue(angles_close(-2 * np.pi, 0), "Angles close works")
        self.assertTrue(angles_close(np.pi, 3 * np.pi), "Angles close works")

    def test_runSpecific(self):
        """A specific test case for a commonly used set of points in tests"""
        control_points = TranslateRotateTransformPoints
        target_points = control_points[:, 0:2]
        source_points = control_points[:, 2:]
        r = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(target_points,
                                                                                                    source_points)

        RT = nornir_imageregistration.transforms.Rigid(target_offset=r.translation,
                                                       source_rotation_center=r.source_rotation_center, angle=r.angle)

        transformed_target_points = RT.Transform(source_points)
        np.testing.assert_allclose(target_points, transformed_target_points)

    def test_runRotationMatrixTest(self):
        """A specific test case to ensure our rotation matrix works for the reversed x,y convention of our arrays"""
        control_points = np.array([[-.5, .5, -.5, -.5],
                                   [-.5, -.5, .5, -.5],
                                   [.5, -.5, .5, .5],
                                   [.5, .5, -.5, .5]])
        target_points = control_points[:, 0:2]
        source_points = control_points[:, 2:]
        angle = np.pi / 2.0

        input_points = np.hstack((source_points, np.zeros((source_points.shape[0], 1))))

        rot_mat = nornir_imageregistration.transforms.utils.RotationMatrix(angle)
        transformed_target_points = (rot_mat @ input_points.T).T
        transformed_target_points = transformed_target_points[:, 0:2]
        np.testing.assert_allclose(target_points, transformed_target_points)


if __name__ == "__main__":
    unittest.main()
