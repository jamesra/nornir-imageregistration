import nornir_imageregistration

__all__ = ['ForwardTransformCheck', 'TransformCheck', 'NearestFixedCheck', 'NearestWarpedCheck',
           'IdentityTransformPoints', 'MirrorTransformPoints', 'TranslateTransformPoints', 'OffsetTransformPoints']

import numpy as np

__transform_tolerance = 1e-5

def ForwardTransformCheck(test, transform, warpedPoint, fixedPoint):
        '''Ensures that a point can map to its expected transformed position and back again'''
        fp = transform.Transform(warpedPoint)
        np.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)

def TransformCheck(test, transform, warpedPoint, fixedPoint):
        '''Ensures that a point can map to its expected transformed position and back again'''
        fp = transform.Transform(warpedPoint)
        np.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)
        wp = transform.InverseTransform(fp)
        np.testing.assert_allclose(wp, warpedPoint, atol=__transform_tolerance, rtol=0)

def NearestFixedCheck(test, transform, fixedPoints, testPoints):
        '''Ensures that the nearest fixed point can be found for a test point'''
        distance, index = transform.NearestFixedPoint(testPoints)
        np.testing.assert_allclose(transform.TargetPoints[index,:], fixedPoints, atol=__transform_tolerance, rtol=0)

def NearestWarpedCheck(test, transform, warpedPoints, testPoints):
        '''Ensures that the nearest warped point can be found for a test point'''
        distance, index = transform.NearestWarpedPoint(testPoints)
        np.testing.assert_allclose(transform.SourcePoints[index,:], warpedPoints, atol=__transform_tolerance, rtol=0)

def TransformAgreementCheck(t1, t2, points=None):
        '''Ensures that the nearest warped point can be found for a test point'''
        if points is None:
            points = np.array([0,0],dtype=np.float32)
        else:
            points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
            
        r1 = t1.Transform(points)
        m1 = t2.Transform(points)
        
        np.testing.assert_allclose(r1, m1, err_msg="Pair of Transforms do not agree", atol=__transform_tolerance, rtol=0)
        
        ir1 = t1.InverseTransform(r1)
        im1 = t2.InverseTransform(m1)
        
        np.testing.assert_allclose(ir1, im1, err_msg="Pair of InverseTransforms do not agree", atol=__transform_tolerance, rtol=0)
        np.testing.assert_allclose(ir1, points, err_msg="Pair of InverseTransforms do not agree", atol=__transform_tolerance, rtol=0)
        np.testing.assert_allclose(im1, points, err_msg="Pair of InverseTransforms do not agree", atol=__transform_tolerance, rtol=0)


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

# Used to test IsOffsetAtZero
OffsetTransformPoints = np.array([[1, 1, 0, 0],
                              [2, 1, 1, 0],
                              [1, 2, 0, 1],
                              [2, 2, 1, 1]])