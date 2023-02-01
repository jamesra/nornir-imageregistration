# -*- coding: utf-8 -*-
import unittest
import hypothesis
import numpy as np
from numpy.typing import NDArray
import nornir_imageregistration.transforms


# Algorithm for getting the transformation factors from two sets of points.
def kabsch_umeyama(A: NDArray, B: NDArray):
    '''
    This function is used to get the translation, rotation and scaling factors when aligning
    points in B on reference points in A.
    
    The R,c,t componenets once return can be used to obtain B' 
    '''
    assert A.shape == B.shape
    num_pts, num_dims = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    centered_A = A - EA
    centered_B = B - EB
    VarB = np.mean(np.linalg.norm(centered_B, axis=1) ** 2)

    H = (centered_A.T @ centered_B) / num_pts
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (num_dims - 1) + [d])

    rotation_matrix = U @ S @ VT
    scale = np.trace(np.diag(D) @ S) / VarB
    translation = EA - scale * rotation_matrix @ EB

    return rotation_matrix, scale, translation


class TestGridFitting(unittest.TestCase):
    # Transformation components (c= Scaling, rangle= Rotation angle, R = rotation matrix, t = )

    point_value_strategy = hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6, allow_subnormal=False )
    point_strategy = hypothesis.strategies.tuples(point_value_strategy, point_value_strategy)

    @hypothesis.settings(verbosity=hypothesis.Verbosity.verbose)
    @hypothesis.given(rangle=hypothesis.strategies.floats(-np.pi, np.pi, exclude_min=True),
                      translate=point_strategy,
                      scale=hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=100),
                      points=hypothesis.strategies.lists(point_strategy, min_size=2, max_size=500, unique_by=(lambda x: x)))
    def test_runFit(self, rangle: float, translate: tuple[float, float], scale: float, points: list[tuple[float, float]]):
        # c = np.random.randint(1, 10)
        # rangle = -np.pi / 2 + np.pi * np.random.rand()

        #scale = 1.0

        R = nornir_imageregistration.transforms.utils.RotationMatrix(rangle)
        # t = np.random.randint(1, 10, size=2)
        t = np.array(translate)
        #c = scale
        scale_matrix = nornir_imageregistration.transforms.utils.ScaleMatrixXY(scale)
        c = np.array((scale, scale, 1.0))
        points_array = np.array(points)

        #print(f'Starting angle: {rangle}')
        #print(f'Starting scale: {scale}')
        #print(R)
        #print(f'Starting translation: {t}')

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
        #numPoints = B.shape[0]
        #points2D1 = pts
        num_pts = points_array.shape[0]
        #points2D1 = np.transpose(points_array)
        points2D1 = np.array(points_array)
        points2D1 = np.hstack((points2D1, np.ones((num_pts, 1))))
        #points2D1 = np.transpose(points2D1)
        points2D1_scaled = points2D1 @ scale_matrix
        points2D1_rotated = points2D1_scaled @ R

        #points2D1_rotated = np.transpose(points2D1_rotated)

        points2D1_rotated = points2D1_rotated[:, 0:2]

        output_points2D = points2D1_rotated + t

        hypothesis.note(points)
        hypothesis.note(output_points2D)

        # =============================================================================
        # # Simplified code for applying transformation components on a grid
        # R_only = np.array([[np.cos(rangle),-np.sin(rangle)],[np.sin(rangle),np.cos(rangle)]])
        # output_points2D_OneLine = np.array([t + c * R_only @ b for b in B])
        # print(output_points2D_OneLine)
        # =============================================================================
        # Obtaining the rotation, scaling and translation factors back from the new grid and our original grid.
        calc_rotate, calc_scale, calc_translate = kabsch_umeyama(output_points2D, points_array)

        #calc_rotate_angle = np.arctan2(calc_rotate[1, 0], calc_rotate[0, 0])
        #calc_rotate_angle = np.arcsin(calc_rotate[0, 1])
        calc_rotate_angle = np.arctan2(calc_rotate[0, 1], calc_rotate[0, 0])
        if calc_rotate_angle <= -np.pi * 2:
            calc_rotate_angle += np.pi * 2

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

        hypothesis.note(f'Rotation Angle {rangle} -> {calc_rotate} -> {calc_rotate_angle}')
        translate_output = np.squeeze(calc_translate)
        #translate_output = np.zeros((t.size))
        #for i in range(len(t)):
        #    translate_output[i] = calc_translate[0, i]
        hypothesis.note(f'Scale: {scale} -> {calc_scale}')
        hypothesis.note(f'Translation: {t} -> {translate_output}')

        #Adjust rangle to wrap around

        self.assertTrue(self.anglesclose(rangle, calc_rotate_angle, atol=1e-4))
        #np.testing.assert_allclose(rangle, calc_rotate_angle, atol=1e-3)
        np.testing.assert_allclose(scale, calc_scale, atol=1e-3)
        np.testing.assert_allclose(t, translate_output, atol=1e-3)

    def anglesclose(self, a_in: float, b_in: float, **kwargs) -> bool:
        """Returns true if two angles are approximately equal"""
        tau = np.pi * 2
        a_t = a_in / tau
        b_t = b_in / tau

        #Have to mod twice to handle rare floating point rounding errors
        a = (a_t % 1.0) % 1.0
        b = (b_t % 1.0) % 1.0

        if np.allclose(a, b, **kwargs):
            return True

        #Handle the wrap-around case where they are an epsilon apart
        if a < b:
            a += 1
        else:
            b += 1

        return np.allclose(a, b, **kwargs)


if __name__ == "__main__":
    unittest.main()
