"""``is_alignable_cell`` must reach one host sync, not four, on device arrays.

The host path reads three device scalars in a row -- ``amin == amax``,
``amax == 0``, then ``float(std)`` -- and each comparison drags a 0-d array back
across the bus. ``cell_intensity_std`` adds more, because ``count_nonzero`` and
the ``arr[valid]`` boolean gather both need the element count on the host.

The batched path had already dropped those syncs; the serial gate had not, and it
runs once per cell, twice per measurement.

The device path now keeps every reduction as a device scalar and combines them
into a single boolean. The std is computed arithmetically instead of by gathering
the valid elements, since a gather needs a size only the host would know.

Verdicts must be identical to the host path, so most of these tests assert
device/host agreement rather than hard-coded booleans.
"""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.refine_shared.cell_validity import (
    cell_intensity_std, is_alignable_cell)
from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None

_HAS_CUPY = nornir_imageregistration.HasCupy() and cp is not None


def _cells() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    host = rng.random((64, 64))
    return {
        'textured': host,
        'constant': np.full((64, 64), 0.5),
        'zeros': np.zeros((64, 64)),
        'ones': np.ones((64, 64)),
        'low_contrast': host * 1e-6,
        'just_above_floor': host * 4e-3,
        'with_nan': np.where(rng.random((64, 64)) > 0.5, host, np.nan),
        'all_nan': np.full((64, 64), np.nan),
        'single_finite': np.where(np.arange(4096).reshape(64, 64) == 0, 1.0, np.nan),
        'two_finite': np.where(np.arange(4096).reshape(64, 64) < 2, 1.0, np.nan),
        'negative': -host,
        'single_hot_pixel': np.where(np.arange(4096).reshape(64, 64) == 0, 1.0, 0.0),
    }


class TestHostPathUnchanged(unittest.TestCase):
    """The numpy branch still short-circuits; only the device branch changed."""

    def test_constant_and_zero_cells_rejected(self):
        self.assertFalse(is_alignable_cell(np.full((64, 64), 0.5)))
        self.assertFalse(is_alignable_cell(np.zeros((64, 64))))

    def test_textured_cell_accepted(self):
        self.assertTrue(is_alignable_cell(np.random.default_rng(0).random((64, 64))))

    def test_empty_and_none_rejected(self):
        self.assertFalse(is_alignable_cell(np.zeros((0, 0))))
        self.assertFalse(is_alignable_cell(None))

    def test_min_std_override_is_honoured(self):
        cell = np.random.default_rng(0).random((64, 64)) * 1e-2
        self.assertTrue(is_alignable_cell(cell, min_std=1e-6))
        self.assertFalse(is_alignable_cell(cell, min_std=1.0))

    def test_zero_threshold_skips_the_std_gate(self):
        cell = np.random.default_rng(0).random((64, 64)) * 1e-12
        self.assertTrue(is_alignable_cell(cell, min_std=0.0))


@unittest.skipUnless(_HAS_CUPY, 'requires CuPy')
class TestDeviceMatchesHost(unittest.TestCase):
    """A backend switch must never change a verdict."""

    def setUp(self) -> None:
        self._previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)
        get_runtime_config(refresh=True)

    def tearDown(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(self._previous)

    def test_unmasked_verdicts_agree(self):
        for name, host in _cells().items():
            with self.subTest(cell=name):
                self.assertEqual(is_alignable_cell(cp.asarray(host)),
                                 is_alignable_cell(host))

    def test_masked_verdicts_agree(self):
        mask = np.random.default_rng(1).random((64, 64)) > 0.1
        for name, host in _cells().items():
            with self.subTest(cell=name):
                self.assertEqual(
                    is_alignable_cell(cp.asarray(host), mask=cp.asarray(mask)),
                    is_alignable_cell(host, mask=mask))

    def test_verdicts_agree_across_thresholds(self):
        host = np.random.default_rng(0).random((64, 64)) * 1e-2
        device = cp.asarray(host)
        for threshold in (0.0, 1e-9, 1e-3, 2.8e-3, 3.0e-3, 1e-2, 1.0):
            with self.subTest(threshold=threshold):
                self.assertEqual(is_alignable_cell(device, min_std=threshold),
                                 is_alignable_cell(host, min_std=threshold))

    def test_fully_masked_cell_is_rejected(self):
        host = np.random.default_rng(0).random((64, 64))
        mask = np.zeros((64, 64), dtype=bool)

        self.assertFalse(is_alignable_cell(cp.asarray(host), mask=cp.asarray(mask)))
        self.assertFalse(is_alignable_cell(host, mask=mask))

    def test_device_std_matches_host_std(self):
        """The arithmetic std must agree with the gather-based one it replaced."""
        for name, host in _cells().items():
            with self.subTest(cell=name):
                expected = cell_intensity_std(host)
                actual = cell_intensity_std(cp.asarray(host))
                if np.isnan(expected):
                    self.assertTrue(np.isnan(actual))
                else:
                    self.assertAlmostEqual(float(actual), float(expected), places=10)

    def test_returns_a_python_bool(self):
        result = is_alignable_cell(cp.asarray(np.random.default_rng(0).random((64, 64))))

        self.assertIsInstance(result, bool)


@unittest.skipUnless(_HAS_CUPY, 'requires CuPy')
class TestSingleHostSync(unittest.TestCase):
    """Counts device-to-host transfers rather than trusting wall time."""

    def setUp(self) -> None:
        self._previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)
        get_runtime_config(refresh=True)

    def tearDown(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(self._previous)

    def _count_syncs(self, **kwargs) -> int:
        """Count ndarray.get calls, which is how a device scalar reaches the host."""
        calls = []
        original = cp.ndarray.get

        def counting_get(self, *args, **kw):
            calls.append(1)
            return original(self, *args, **kw)

        cell = cp.asarray(np.random.default_rng(0).random((64, 64)))
        device_kwargs = {k: cp.asarray(v) if v is not None else None
                         for k, v in kwargs.items()}
        is_alignable_cell(cell, **device_kwargs)  # warm the config cache

        cp.ndarray.get = counting_get
        try:
            is_alignable_cell(cell, **device_kwargs)
        finally:
            cp.ndarray.get = original
        return len(calls)

    def test_unmasked_reaches_one_sync(self):
        self.assertLessEqual(self._count_syncs(), 1)

    def test_masked_reaches_one_sync(self):
        mask = np.random.default_rng(1).random((64, 64)) > 0.1
        self.assertLessEqual(self._count_syncs(mask=mask), 1)


if __name__ == '__main__':
    unittest.main()
