"""Ensure* helpers preserve input array backend unless explicitly forced."""

import unittest

import numpy as np

import nornir_imageregistration


@unittest.skipUnless(nornir_imageregistration.HasCupy(), 'CuPy required')
class TestEnsureArrayBackendPreservation(unittest.TestCase):
    """Neutral Ensure* helpers must not upgrade/downgrade based on global computation lib."""

    def setUp(self) -> None:
        import cupy as cp
        self.cp = cp

    def test_ensure_array_respects_input_not_global(self) -> None:
        host = np.array([1.0, 2.0], dtype=np.float32)
        device = self.cp.asarray([1.0, 2.0], dtype=np.float32)

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.assertIsInstance(nornir_imageregistration.EnsureArray(host), np.ndarray)
        self.assertIsInstance(nornir_imageregistration.EnsureArray(device), self.cp.ndarray)

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.assertIsInstance(nornir_imageregistration.EnsureArray(host), np.ndarray)
        self.assertIsInstance(nornir_imageregistration.EnsureArray(device), self.cp.ndarray)

    def test_ensure_points_1d_and_2d_respect_input(self) -> None:
        host_2d = np.array([[1.0, 2.0]], dtype=np.float32)
        device_2d = self.cp.asarray([[1.0, 2.0]], dtype=np.float32)

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.assertIsInstance(nornir_imageregistration.EnsurePointsAre2DArray(host_2d), np.ndarray)
        self.assertIsInstance(nornir_imageregistration.EnsurePointsAre2DArray(device_2d), self.cp.ndarray)
        self.assertIsInstance(
            nornir_imageregistration.EnsurePointsAre1DArray(host_2d.ravel()), np.ndarray,
        )
        self.assertIsInstance(
            nornir_imageregistration.EnsurePointsAre1DArray(device_2d.ravel()), self.cp.ndarray,
        )

    def test_ensure_points_4xn_respects_input(self) -> None:
        host = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        device = self.cp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.assertIsInstance(nornir_imageregistration.EnsurePointsAre4xN_Array(host), np.ndarray)
        self.assertIsInstance(nornir_imageregistration.EnsurePointsAre4xN_Array(device), self.cp.ndarray)

    def test_sequences_become_numpy_even_when_cupy_active(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        seq = (1.0, 2.0)
        self.assertIsInstance(nornir_imageregistration.EnsureArray(seq), np.ndarray)
        self.assertIsInstance(nornir_imageregistration.EnsurePointsAre1DArray(seq), np.ndarray)

    def test_explicit_flavors_force_backend(self) -> None:
        host = np.array([1.0, 2.0], dtype=np.float32)
        device = self.cp.asarray([1.0, 2.0], dtype=np.float32)

        self.assertIsInstance(nornir_imageregistration.EnsureNumpyArray(device), np.ndarray)
        self.assertIsInstance(nornir_imageregistration.EnsureCupyArray(host), self.cp.ndarray)


if __name__ == '__main__':
    unittest.main()
