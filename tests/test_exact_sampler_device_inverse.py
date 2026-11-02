"""The exact-ROI sampler maps cells through an on-device mesh when NORNIR_REFINE_GPU_TRANSFORM is on.

STOS refine inverse-maps every pixel of every cell through the current mesh. On the host
that is SciPy's barycentric lookup, about 33% of the refine main thread on srv (RC3
Grid16, 2026-11-02). Under CuPy the flag (default on since 2026-11) maps them with
cupyx ``LinearNDInterpolator`` instead: the same piecewise-linear map with no RBF, NaN
outside the mesh. cupyx triangulates on its own, so coordinates are close to, not
bit-identical with, SciPy's.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.refine_shared.runtime_config import RefineRuntimeConfig
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback

_SIZE = 256
_CELL = np.array((48, 48))


def _jittered_mesh(seed: int = 5) -> MeshWithRBFFallback:
    """A warped mesh with jittered control points, so neither Qhull nor cupyx sees a degenerate lattice."""
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(np.linspace(0, _SIZE - 1, 7), np.linspace(0, _SIZE - 1, 7), indexing='ij')
    source = np.stack((ys.ravel(), xs.ravel()), axis=1) + rng.uniform(-4, 4, size=(49, 2))
    target = source.copy()
    target[:, 1] = source[:, 1] * (0.9 + 0.2 * source[:, 0] / (_SIZE - 1))
    target[:, 0] += 6.0 * np.sin(source[:, 1] / 40.0)
    return MeshWithRBFFallback(np.hstack((target, source)))


def _env(value: str | None):
    env = {k: v for k, v in os.environ.items() if k != 'NORNIR_REFINE_GPU_TRANSFORM'}
    if value is not None:
        env['NORNIR_REFINE_GPU_TRANSFORM'] = value
    return mock.patch.dict(os.environ, env, clear=True)


class TestGpuTransformDefault(unittest.TestCase):

    def test_on_when_unset_off_only_when_falsey(self) -> None:
        for value, expected in ((None, True), ('', True), ('1', True), ('0', False), ('false', False)):
            with self.subTest(value=value), _env(value):
                self.assertEqual(RefineRuntimeConfig.from_env().gpu_transform, expected)

    def test_numpy_runs_keep_the_host_mesh(self) -> None:
        with _env(None):
            self.assertIsNone(ldc._device_inverse_interpolator(_jittered_mesh(), np))


@unittest.skipUnless(nornir_imageregistration.HasCupy(), 'CuPy not available')
class TestDeviceExactSampler(unittest.TestCase):

    def setUp(self) -> None:
        self._previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        if not nornir_imageregistration.UsingCupy():
            self.skipTest('CuPy runtime probe failed (no usable GPU)')

    def tearDown(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(self._previous)

    def _stack(self, flag: str, transform: MeshWithRBFFallback) -> np.ndarray:
        import cupy as cp
        rng = np.random.default_rng(3)
        source_image = cp.asarray(rng.random((_SIZE, _SIZE)).astype(np.float32))
        # Interior cells plus one straddling the mesh edge, where samples read NaN.
        target_points = np.array([[80.0, 90.0], [128.0, 128.0], [170.0, 150.0], [20.0, 240.0]])
        with _env(flag):
            stack, _ = ldc.BuildExactMovingROIsBatched(
                transform=transform, source_image=source_image, source_image_stats=None,
                target_points=target_points, alignment_area=_CELL, xp=cp)
        return nornir_imageregistration.EnsureNumpyArray(stack)

    def test_device_map_engages_and_matches_the_host_map(self) -> None:
        transform = _jittered_mesh()
        with _env(None):
            import cupy as cp
            self.assertIsNotNone(ldc._device_inverse_interpolator(transform, cp))

        host = self._stack('0', transform)
        device = self._stack('1', transform)

        self.assertEqual(host.shape, device.shape)
        both = np.isfinite(host) & np.isfinite(device)
        disagree = np.isfinite(host) != np.isfinite(device)
        # Only pixels on the mesh boundary may flip between inside and outside.
        self.assertLess(disagree.mean(), 0.01)
        np.testing.assert_allclose(device[both], host[both], atol=1e-3)

    def test_mesh_edited_in_place_is_rebuilt(self) -> None:
        import cupy as cp
        transform = _jittered_mesh()
        with _env(None):
            first = ldc._device_inverse_interpolator(transform, cp)
            self.assertIs(ldc._device_inverse_interpolator(transform, cp), first)
            transform.AddPoints(np.array([[130.0, 131.0, 128.0, 129.0]]))
            self.assertIsNot(ldc._device_inverse_interpolator(transform, cp), first)


if __name__ == '__main__':
    unittest.main()
