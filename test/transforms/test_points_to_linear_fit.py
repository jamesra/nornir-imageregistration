# -*- coding: utf-8 -*-
import math
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
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarB = np.mean(np.linalg.norm(B - EB, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    rotation_matrix = U @ S @ VT
    scale = np.trace(np.diag(D) @ S) / VarB
    translation = EA - scale * rotation_matrix @ EB

    return rotation_matrix, scale, translation


class TestGridFitting(unittest.TestCase):
    # Transformation components (c= Scaling, rangle= Rotation angle, R = rotation matrix, t = )

    point_value_strategy = hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
    point_strategy = hypothesis.strategies.tuples(point_value_strategy, point_value_strategy)

    @hypothesis.given(rangle=hypothesis.strategies.floats(-math.pi, math.pi),
                      translate=point_strategy,
                      scale=hypothesis.strategies.floats(allow_nan=False, allow_infinity=False),
                      points=hypothesis.strategies.lists(point_strategy, min_size=2, max_size=500, unique_by=(lambda x: x)))
    def test_runFit(self, rangle: float, translate: tuple[float, float], scale: float, points: list[tuple[float, float]]):
        # c = np.random.randint(1, 10)
        # rangle = -np.pi / 2 + np.pi * np.random.rand()

        R = nornir_imageregistration.transforms.utils.RotationMatrix(rangle)
        # t = np.random.randint(1, 10, size=2)
        t = np.array(translate)
        #c = scale
        c = np.array((1.0, 1.0, 1.0))
        points_array = np.array(points)

        print(f'Starting angle: {rangle}')
        print(f'Starting scale: {c}')
        #print(R)
        print(f'Starting translation: {t}')

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
        points2D1_scaled = points2D1 * c

        points2D1_rotated = points2D1_scaled @ R

        #points2D1_rotated = np.transpose(points2D1_rotated)

        points2D1_rotated = points2D1_rotated[:, 0:2]

        output_points2D = points2D1_rotated + t

        print(output_points2D)

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
        print(f'Rotation Angle {calc_rotate} -> {calc_rotate_angle}')
        translate_output = np.squeeze(np.array(calc_translate[0, :]))
        #translate_output = np.zeros((t.size))
        #for i in range(len(t)):
        #    translate_output[i] = calc_translate[0, i]
        print(f'Scale: {scale}')
        print(f'Translation: {translate_output}')
        np.testing.assert_allclose(rangle, calc_rotate_angle, atol=1e-3)
        np.testing.assert_allclose(c, calc_scale, atol=1e-3)
        np.testing.assert_allclose(t, translate_output, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
