import numpy as np
from numpy._typing import NDArray

import nornir_imageregistration
from nornir_imageregistration import IDiscreteTransform, ITransform

__transform_tolerance = 1e-5


def TransformInverseCheck(test, transform: ITransform, warpedPoint: NDArray[np.floating]):
    """Ensures that a point can map to its expected transformed position and back again.
    Does not validate that the transformed point is an expected value"""
    xp = nornir_imageregistration.GetComputationModule()
    fp = transform.Transform(warpedPoint)
    # np.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)
    wp = transform.InverseTransform(fp)
    wp = nornir_imageregistration.EnsureNumpyArray(wp)
    warped_np = nornir_imageregistration.EnsureNumpyArray(warpedPoint)

    xp.testing.assert_allclose(wp, warped_np, atol=__transform_tolerance, rtol=0)


def ForwardTransformCheck(test, transform: ITransform, warpedPoint: NDArray[np.floating],
                          fixedPoint: NDArray[np.floating]):
    '''Ensures that a point can map to its expected transformed position and back again'''
    xp = nornir_imageregistration.GetComputationModule()
    fp = transform.Transform(warpedPoint)

    fp = nornir_imageregistration.EnsureNumpyArray(fp)

    xp.testing.assert_allclose(fp, fixedPoint, atol=__transform_tolerance, rtol=0)


def TransformCheck(test, transform: ITransform, source_point: NDArray[np.floating], target_point: NDArray[np.floating]):
    '''Ensures that a point can map to its expected transformed position and back again'''
    xp = nornir_imageregistration.GetComputationModule()

    source_point = nornir_imageregistration.EnsureNumpyArray(source_point)
    target_point = nornir_imageregistration.EnsureNumpyArray(target_point)

    fp = transform.Transform(source_point)
    wp = transform.InverseTransform(fp)

    # Transforms may return NumPy or CuPy depending on backend; compare on host.
    fp = nornir_imageregistration.EnsureNumpyArray(fp)
    wp = nornir_imageregistration.EnsureNumpyArray(wp)

    xp.testing.assert_allclose(fp, target_point, atol=__transform_tolerance, rtol=0)
    xp.testing.assert_allclose(wp, source_point, atol=__transform_tolerance, rtol=0)


def NearestFixedCheck(test, transform: ITransform, fixedPoints: NDArray[np.floating], testPoints: NDArray[np.floating]):
    '''Ensures that the nearest fixed point can be found for a test point'''
    xp = nornir_imageregistration.GetComputationModule()
    distance, index = transform.NearestFixedPoint(testPoints)
    index_np = np.atleast_1d(nornir_imageregistration.EnsureNumpyArray(index)).astype(np.intp, copy=False).ravel()
    subset = transform.TargetPoints[index_np]
    subset_np = nornir_imageregistration.EnsureNumpyArray(subset)
    fp = nornir_imageregistration.EnsureNumpyArray(fixedPoints)
    xp.testing.assert_allclose(subset_np, fp, atol=__transform_tolerance, rtol=0)


def NearestWarpedCheck(test, transform: ITransform, warpedPoints: NDArray[np.floating],
                       testPoints: NDArray[np.floating]):
    '''Ensures that the nearest warped point can be found for a test point'''
    xp = nornir_imageregistration.GetComputationModule()
    distance, index = transform.NearestWarpedPoint(testPoints)
    index_np = np.atleast_1d(nornir_imageregistration.EnsureNumpyArray(index)).astype(np.intp, copy=False).ravel()
    subset = transform.SourcePoints[index_np]
    subset_np = nornir_imageregistration.EnsureNumpyArray(subset)
    wp = nornir_imageregistration.EnsureNumpyArray(warpedPoints)
    xp.testing.assert_allclose(subset_np, wp, atol=__transform_tolerance, rtol=0)


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

    r1 = nornir_imageregistration.EnsureNumpyArray(r1)
    m1 = nornir_imageregistration.EnsureNumpyArray(m1)

    xp.testing.assert_allclose(r1, m1, err_msg="Pair of Transforms do not agree", atol=__transform_tolerance, rtol=0)

    ir1 = t1.InverseTransform(r1)
    im1 = t2.InverseTransform(m1)

    ir1 = nornir_imageregistration.EnsureNumpyArray(ir1)
    im1 = nornir_imageregistration.EnsureNumpyArray(im1)
    points_cmp = nornir_imageregistration.EnsureNumpyArray(points)

    xp.testing.assert_allclose(ir1, im1, err_msg="Pair of InverseTransforms do not agree", atol=__transform_tolerance,
                               rtol=0)
    xp.testing.assert_allclose(ir1, points_cmp, err_msg="Pair of InverseTransforms do not agree",
                               atol=__transform_tolerance, rtol=0)
    xp.testing.assert_allclose(im1, points_cmp, err_msg="Pair of InverseTransforms do not agree",
                               atol=__transform_tolerance, rtol=0)

    if isinstance(t1, IDiscreteTransform) and isinstance(t2, IDiscreteTransform):
        c1 = nornir_imageregistration.EnsureNumpyArray(t1.MappedBoundingBox.Corners)
        c2 = nornir_imageregistration.EnsureNumpyArray(t2.MappedBoundingBox.Corners)
        xp.testing.assert_allclose(c1, c2,
                                   atol=__transform_tolerance)
        f1 = nornir_imageregistration.EnsureNumpyArray(t1.FixedBoundingBox.Corners)
        f2 = nornir_imageregistration.EnsureNumpyArray(t2.FixedBoundingBox.Corners)
        xp.testing.assert_allclose(f1, f2, atol=__transform_tolerance)
