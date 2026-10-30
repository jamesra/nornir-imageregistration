"""
Host and GPU ``OneWayRBFWithLinearCorrection`` must agree on the rigid path.

The two twins reach that result differently and, after measurement, deliberately
so. The host class returns ``self._rigid_transform.Transform(Points)`` as soon as
``UseRigidTransform`` is set. The GPU twin instead zeroes the RBF weight sums and
evaluates the linear terms inline.

That is not a divergence in output. ``UseRigidTransform`` is set precisely when
the RBF deviation weights are ~0, which means the transform *is* its linear part,
so both routes compute the same affine map -- measured agreement 1e-6 to 8e-6
across translation, 15 and 90 degree rotations, scaling, and rotation plus shift.
The reported ``(1,N)``/``(N,)`` broadcast mismatch does not occur either, because
``cp.vstack`` promotes to ``(3,N)``.

Making the GPU twin delegate as well was tried and reverted: it is 3.6x slower at
1,000,000 points (0.755 -> 2.756 ms), because delegating misses the rigid
transform's ``angle == 0`` fast path over a ~1e-17 angle residue and then pays a
device round-trip per call.

So these tests pin the property that matters -- the two twins agree, and the GPU
twin stays on-device -- without freezing either implementation strategy.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.transforms.one_way_rbftransform import (
    OneWayRBFWithLinearCorrection,
    OneWayRBFWithLinearCorrection_GPUComponent,
)

cp = pytest.importorskip('cupy')

pytestmark = pytest.mark.skipif(
    cp.cuda.runtime.getDeviceCount() == 0, reason='requires a CUDA device')

GRID = np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 0.0], [100.0, 100.0],
                 [50.0, 50.0], [0.0, 50.0], [50.0, 0.0]])

# Includes points outside the control hull, where extrapolation behaviour differs
# most between an affine evaluation and a delegated rigid transform.
PROBE = np.array([[10.0, 20.0], [55.0, 65.0], [90.0, 5.0], [0.0, 0.0],
                  [33.0, 77.0], [-15.0, 130.0]])


def _rotate(points: np.ndarray, degrees: float,
            center: tuple[float, float] = (50.0, 50.0)) -> np.ndarray:
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    origin = np.asarray(center)
    return (points - origin) @ rotation.T + origin


RIGID_CASES = {
    'translation': GRID + np.array([11.0, 7.0]),
    'rotation_15': _rotate(GRID, 15.0),
    'rotation_90': _rotate(GRID, 90.0),
    'scale': GRID * 1.5,
    'rotation_and_shift': _rotate(GRID, 30.0) + np.array([13.0, -6.0]),
}


def _build(target: np.ndarray):
    host = OneWayRBFWithLinearCorrection(GRID, target)
    device = OneWayRBFWithLinearCorrection_GPUComponent(GRID, target)
    host.PrecomputeWeights()
    device.PrecomputeWeights()
    return host, device


@pytest.mark.parametrize('name', sorted(RIGID_CASES))
def test_gpu_matches_host_on_the_rigid_path(name):
    """Both twins map points identically once the fit is rigid."""
    host, device = _build(RIGID_CASES[name])

    assert host.UseRigidTransform, 'expected the host rigid shortcut'
    assert device.UseRigidTransform, 'expected the device rigid shortcut'

    host_out = np.asarray(host.Transform(PROBE))
    device_out = nornir_imageregistration.EnsureNumpyArray(device.Transform(PROBE))

    np.testing.assert_allclose(device_out, host_out, atol=1e-4)


@pytest.mark.parametrize('name', sorted(RIGID_CASES))
def test_gpu_matches_the_rigid_transform_object(name):
    """The GPU twin's inline evaluation equals delegating to the rigid transform.

    This is the substance of the parity claim: the inline path is not an
    approximation of a different map, it is the same map.
    """
    _host, device = _build(RIGID_CASES[name])

    inline_out = nornir_imageregistration.EnsureNumpyArray(device.Transform(PROBE))
    delegated_out = nornir_imageregistration.EnsureNumpyArray(
        device._rigid_transform.Transform(cp.asarray(PROBE)))

    np.testing.assert_allclose(inline_out, delegated_out, atol=1e-4)


def test_gpu_result_stays_on_device():
    """The rigid path must not quietly move points to the host."""
    _host, device = _build(RIGID_CASES['rotation_and_shift'])

    result = device.Transform(cp.asarray(PROBE))

    assert cp.get_array_module(result) is cp, 'result left the device'


def test_non_rigid_fit_still_uses_the_rbf_path():
    """A genuinely warped fit must not take the rigid path in either twin."""
    warped_target = GRID.copy()
    warped_target[4] += np.array([25.0, -18.0])  # move the centre point only

    host, device = _build(warped_target)

    assert not host.UseRigidTransform
    assert not device.UseRigidTransform

    host_out = np.asarray(host.Transform(PROBE))
    device_out = nornir_imageregistration.EnsureNumpyArray(device.Transform(PROBE))

    np.testing.assert_allclose(device_out, host_out, atol=1e-3)


def test_vstack_shapes_do_not_mismatch():
    """The reported (1,N)/(N,) broadcast failure does not occur."""
    num_points = 5
    weight_sums = cp.zeros((1, num_points))
    linear_term = cp.arange(num_points, dtype=cp.float64)

    stacked = cp.vstack((weight_sums, linear_term, linear_term))

    assert stacked.shape == (3, num_points)


def test_rigid_transform_angle_is_not_exactly_zero():
    """Documents why delegating is slow.

    The rigid transform derived for a pure translation carries a tiny angle
    residue from the fit, which defeats ``Rigid.Transform``'s ``angle == 0``
    fast path and sends every call through the homogeneous matmul.
    """
    _host, device = _build(RIGID_CASES['translation'])

    angle = float(device._rigid_transform.angle)
    assert angle != 0.0, 'residue is gone; the delegated fast path may now apply'
    assert abs(angle) < 1e-12, f'unexpectedly large angle {angle}'
