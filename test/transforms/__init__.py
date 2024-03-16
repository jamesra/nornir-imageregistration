__all__ = ['ForwardTransformCheck', 'TransformCheck', 'NearestFixedCheck', 'NearestWarpedCheck',
           'IdentityTransformPoints', 'MirrorTransformPoints', 'TranslateTransformPoints', 'OffsetTransformPoints',
           'TranslateRotateTransformPoints', 'TranslateRotateScaleTransformPoints', 'TransformInverseCheck',
           'TransformAgreementCheck']

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

import nornir_imageregistration
from nornir_imageregistration.transforms.base import IDiscreteTransform, ITransform

__transform_tolerance = 1e-5


def TransformInverseCheck(test, transform: ITransform, warpedPoint: NDArray[np.floating]):
    """Ensures that a point can map to its expected transformed position and back again.
    Does not validate that the transformed point is an expected value"""
    xp = nornir_imageregistration.GetComputationModule()
    fp = transform.Transform(warpedPoint)
    # np.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)
    wp = transform.InverseTransform(fp)
    # wp = wp.get() if nornir_imageregistration.UsingCupy() else wp

    xp.testing.assert_allclose(wp, warpedPoint, atol=__transform_tolerance, rtol=0)


def ForwardTransformCheck(test, transform: ITransform, warpedPoint: NDArray[np.floating],
                          fixedPoint: NDArray[np.floating]):
    '''Ensures that a point can map to its expected transformed position and back again'''
    xp = nornir_imageregistration.GetComputationModule()
    fp = transform.Transform(warpedPoint)

    fp = fp.get() if nornir_imageregistration.UsingCupy() else fp

    xp.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)


def TransformCheck(test, transform: ITransform, source_point: NDArray[np.floating], target_point: NDArray[np.floating]):
    '''Ensures that a point can map to its expected transformed position and back again'''
    xp = nornir_imageregistration.GetComputationModule()

    source_point = nornir_imageregistration.EnsureNumpyArray(source_point)
    target_point = nornir_imageregistration.EnsureNumpyArray(target_point)

    fp = transform.Transform(source_point)
    wp = transform.InverseTransform(fp)

    # When Cupy was suppport was added, LinearNDInterpolator was not supported, so some transforms always returned numpy arrays even in Cupy mode
    fp = fp.get() if nornir_imageregistration.UsingCupy() else fp
    wp = wp.get() if nornir_imageregistration.UsingCupy() else wp

    xp.testing.assert_allclose(fp, target_point, atol=__transform_tolerance, rtol=0)
    xp.testing.assert_allclose(wp, source_point, atol=__transform_tolerance, rtol=0)


def NearestFixedCheck(test, transform: ITransform, fixedPoints: NDArray[np.floating], testPoints: NDArray[np.floating]):
    '''Ensures that the nearest fixed point can be found for a test point'''
    xp = nornir_imageregistration.GetComputationModule()
    distance, index = transform.NearestFixedPoint(testPoints)
    xp.testing.assert_allclose(transform.TargetPoints[index, :], fixedPoints, atol=__transform_tolerance, rtol=0)


def NearestWarpedCheck(test, transform: ITransform, warpedPoints: NDArray[np.floating],
                       testPoints: NDArray[np.floating]):
    '''Ensures that the nearest warped point can be found for a test point'''
    xp = nornir_imageregistration.GetComputationModule()
    distance, index = transform.NearestWarpedPoint(testPoints)
    xp.testing.assert_allclose(transform.SourcePoints[index, :], warpedPoints, atol=__transform_tolerance, rtol=0)


def TransformAgreementCheck(t1: ITransform, t2: ITransform, points: NDArray[np.floating] | None = None):
    '''Ensures that the nearest warped point can be found for a test point'''
    xp = nornir_imageregistration.GetComputationModule()
    if points is None:
        points = xp.array([0, 0], dtype=np.float32)
    else:
        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

    r1 = t1.Transform(points)
    m1 = t2.Transform(points)

    # When Cupy was suppport was added, LinearNDInterpolator was not supported, so some transforms always returned numpy arrays even in Cupy mode
    # r1_compare = r1 if xp.get_array_module(r1) == np else r1.get()
    # m1_compare = m1 if xp.get_array_module(m1) == np else m1.get()

    xp.testing.assert_allclose(r1, m1, err_msg="Pair of Transforms do not agree", atol=__transform_tolerance, rtol=0)

    ir1 = t1.InverseTransform(r1)
    im1 = t2.InverseTransform(m1)

    xp.testing.assert_allclose(ir1, im1, err_msg="Pair of InverseTransforms do not agree", atol=__transform_tolerance,
                               rtol=0)
    xp.testing.assert_allclose(ir1, points, err_msg="Pair of InverseTransforms do not agree",
                               atol=__transform_tolerance, rtol=0)
    xp.testing.assert_allclose(im1, points, err_msg="Pair of InverseTransforms do not agree",
                               atol=__transform_tolerance, rtol=0)

    if isinstance(t1, IDiscreteTransform) and isinstance(t2, IDiscreteTransform):
        xp.testing.assert_allclose(t1.MappedBoundingBox.Corners, t2.MappedBoundingBox.Corners,
                                   atol=__transform_tolerance)
        xp.testing.assert_allclose(t1.FixedBoundingBox.Corners, t2.FixedBoundingBox.Corners, atol=__transform_tolerance)


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

IdentityFlippedUDTransformPoints = np.array([[0, 0, 0, 0],
                                             [1, 0, -1, 0],
                                             [0, 1, 0, 1],
                                             [1, 1, -1, 1]])

### TranslateTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space (Target)                          WarpedSpace (Source)    ###
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 2 4 . . . . . . .
# . . . . . . . . . . 2 4 . . . . . . . . .      . . . . . . . . . . | . 1 3 . . . . . . .
# .-.-.-.-.-.-.-.-.-.-1-3-.-.-.-.-.-.-.-.-.      .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .

# Translate points by (1,2)
TranslateTransformPoints = np.array([[0, 0, 1, 2],
                                     [1, 0, 2, 2],
                                     [0, 1, 1, 3],
                                     [1, 1, 2, 3]])

### TranslateFlippedUDTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space (Target)                          WarpedSpace (Source)    ###
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 2 4 . . . . . . .
# . . . . . . . . . . 2 4 . . . . . . . . .      . . . . . . . . . . | . 1 3 . . . . . . .
# .-.-.-.-.-.-.-.-.-.-1-3-.-.-.-.-.-.-.-.-.      .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 1 3 . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 2 4 . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .

# Translate points by (-1,-2)
TranslateFlippedUDTransformPoints = np.array([[0, 0, 1, 2],
                                              [-1, 0, 2, 2],
                                              [0, 1, 1, 3],
                                              [-1, 1, 2, 3]])

### RotateTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space (Target)                          WarpedSpace (Source)    ###
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . 4 3 | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . 2 1 | . . . . . . . . . .      . . . . . . . . . . | . 2 4 . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 1 3 . . . . . . .
# .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.      .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | 1 2 . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | 3 4 . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .

# Translate points by (0,0) and rotate about (0,0) by 90
RotateTransformPoints = np.array([[2, -1, 1, 2],
                                  [2, -2, 2, 2],
                                  [3, -1, 1, 3],
                                  [3, -2, 2, 3]])

### TranslateRotateTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space (Target)                          WarpedSpace (Source)    ###
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . 4 3 . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . 2 1 . . | . . . . . . . . . .      
# .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.      .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . 3 1 . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . 4 2 . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .      

# Translate points by (1,2) and rotate about (0,0) by 90
TranslateRotateTransformPoints = np.array([[-2, 1, 1, -3],
                                           [-3, 1, 1, -4],
                                           [-2, 0, 2, -3],
                                           [-3, 0, 2, -4]])

### TranslateRotateFlippedTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space (Target)                          WarpedSpace (Source)    ###
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 2 4 . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 1 3 . . . . . . .
# .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.      .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . 2 1 . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . 4 3 . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .

# Translate points by (1,2) and rotate about (0,0) by 90
TranslateRotateFlippedTransformPoints = np.array([[-3, 1, 1, 2],
                                                  [-3, 0, 2, 2],
                                                  [-4, 1, 1, 3],
                                                  [-4, 0, 2, 3]])

### TranslateRotateScaleTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space (Target)                          WarpedSpace (Source)    ###
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . 4 . 3 . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . 2 . 1 . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 2 4 . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . 1 3 . . . . . . .
# .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.      .-.-.-.-.-.-.-.-.-.-+-.-.-.-.-.-.-.-.-.-.
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .
# . . . . . . . . . . | . . . . . . . . . .      . . . . . . . . . . | . . . . . . . . . .

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
