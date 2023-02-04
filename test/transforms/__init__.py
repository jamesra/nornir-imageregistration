__all__ = ['ForwardTransformCheck', 'TransformCheck', 'NearestFixedCheck', 'NearestWarpedCheck',
           'IdentityTransformPoints', 'MirrorTransformPoints', 'TranslateTransformPoints', 'OffsetTransformPoints']

import nornir_imageregistration
from nornir_imageregistration.transforms.base import IDiscreteTransform, ITransform
import numpy as np
from numpy.typing import NDArray

__transform_tolerance = 1e-5

def TransformInverseCheck(test, transform: ITransform, warpedPoint):
    """Ensures that a point can map to its expected transformed position and back again.
    Does not validate that the transformed point is an expected value"""
    fp = transform.Transform(warpedPoint)
    #np.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)
    wp = transform.InverseTransform(fp)
    np.testing.assert_allclose(wp, warpedPoint, atol=__transform_tolerance, rtol=0)

def ForwardTransformCheck(test, transform: ITransform, warpedPoint, fixedPoint):
    '''Ensures that a point can map to its expected transformed position and back again'''
    fp = transform.Transform(warpedPoint)
    np.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)


def TransformCheck(test, transform: ITransform, warpedPoint, fixedPoint):
    '''Ensures that a point can map to its expected transformed position and back again'''
    fp = transform.Transform(warpedPoint)
    np.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)
    wp = transform.InverseTransform(fp)
    np.testing.assert_allclose(wp, warpedPoint, atol=__transform_tolerance, rtol=0)


def NearestFixedCheck(test, transform: ITransform, fixedPoints, testPoints):
    '''Ensures that the nearest fixed point can be found for a test point'''
    distance, index = transform.NearestFixedPoint(testPoints)
    np.testing.assert_allclose(transform.TargetPoints[index, :], fixedPoints, atol=__transform_tolerance, rtol=0)


def NearestWarpedCheck(test, transform: ITransform, warpedPoints, testPoints):
    '''Ensures that the nearest warped point can be found for a test point'''
    distance, index = transform.NearestWarpedPoint(testPoints)
    np.testing.assert_allclose(transform.SourcePoints[index, :], warpedPoints, atol=__transform_tolerance, rtol=0)


def TransformAgreementCheck(t1: ITransform, t2: ITransform, points: NDArray[float] | None = None):
    '''Ensures that the nearest warped point can be found for a test point'''
    if points is None:
        points = np.array([0, 0], dtype=np.float32)
    else:
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

    r1 = t1.Transform(points)
    m1 = t2.Transform(points)

    np.testing.assert_allclose(r1, m1, err_msg="Pair of Transforms do not agree", atol=__transform_tolerance, rtol=0)

    ir1 = t1.InverseTransform(r1)
    im1 = t2.InverseTransform(m1)

    np.testing.assert_allclose(ir1, im1, err_msg="Pair of InverseTransforms do not agree", atol=__transform_tolerance,
                               rtol=0)
    np.testing.assert_allclose(ir1, points, err_msg="Pair of InverseTransforms do not agree",
                               atol=__transform_tolerance, rtol=0)
    np.testing.assert_allclose(im1, points, err_msg="Pair of InverseTransforms do not agree",
                               atol=__transform_tolerance, rtol=0)

    if isinstance(t1, IDiscreteTransform) and isinstance(t2, IDiscreteTransform):
        np.testing.assert_allclose(t1.MappedBoundingBox.Corners, t2.MappedBoundingBox.Corners,
                                   atol=__transform_tolerance)
        np.testing.assert_allclose(t1.FixedBoundingBox.Corners, t2.FixedBoundingBox.Corners, atol=__transform_tolerance)


### MirrorTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space                                    WarpedSpace           ###
# . . . . . . . . . . 2 . . . . . . . . . 3      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . 0 . . . . . . . . . 1      1 . . . . . . . . . 0 . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      3 . . . . . . . . . 2 . . . . . . . . . .
# Coordinates are CY, CX, MY, MX
MirrorTransformPoints = np.array([[0, 0, 0, 0],
                                  [0, 10, 0, -10],
                                  [10, 0, -10, 0],
                                  [10, 10, -10, -10]])

IdentityTransformPoints = np.array([[0, 0, 0, 0],
                                    [1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [1, 1, 1, 1]])

# Translate points by (1,2)
TranslateTransformPoints = np.array([[0, 0, 1, 2],
                                     [1, 0, 2, 2],
                                     [0, 1, 1, 3],
                                     [1, 1, 2, 3]])

# Translate points by (1,2) and rotate about (0,0) by 90
TranslateRotateTransformPoints = np.array([[1, -3, 1, 2],
                                           [1, -4, 2, 2],
                                           [2, -3, 1, 3],
                                           [2, -4, 2, 3]])

# Translate points by (1,2) rotate about (0,0) by 90 and scale by 1/2
TranslateRotateScaleTransformPoints = np.array([[3, -4, 1, 2],
                                                [3, -6, 2, 2],
                                                [5, -4, 1, 3],
                                                [5, -6, 2, 3]])
# Used to test IsOffsetAtZero
OffsetTransformPoints = np.array([[1, 1, 0, 0],
                                  [2, 1, 1, 0],
                                  [1, 2, 0, 1],
                                  [2, 2, 1, 1]])

# Describes a square where in one transform the top-right corner is moved to the center
CompressedTransformPoints = np.array([[0, 0, 0, 0],
                                      [10, 0, 10, 0],
                                      [0, 10, 0, 10],
                                      [10, 10, 7.5, 6]])
