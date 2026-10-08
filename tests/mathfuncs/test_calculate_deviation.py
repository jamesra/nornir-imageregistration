"""Direct tests for mathfuncs.calculate_deviation (percentile cross-product curve)."""

from __future__ import annotations

import unittest

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

import nornir_imageregistration
from nornir_imageregistration.mathfuncs.calculate_deviation import calculate_deviation

_PERCENTILES = np.linspace(0, 100, 101, dtype=np.float64)


def _assert_result_shape(result: np.ndarray, *, start_index: int) -> None:
    expected_rows = 101 - start_index
    np.testing.assert_array_equal(result.shape, (expected_rows, 2))
    np.testing.assert_allclose(result[:, 0], _PERCENTILES[start_index:])


class TestCalculateDeviationNumpy(unittest.TestCase):
    """NumPy-path behavior of calculate_deviation."""

    def test_above_index_none_matches_zero(self) -> None:
        values = np.exp(np.linspace(0.0, 3.0, 256, dtype=np.float64))
        full = calculate_deviation(values, above_index=None)
        from_zero = calculate_deviation(values, above_index=0)
        np.testing.assert_allclose(full, from_zero)

    def test_above_index_truncates_percentile_axis(self) -> None:
        values = np.sort(np.random.default_rng(0).normal(size=400))
        start = 37
        result = calculate_deviation(values, above_index=start)
        _assert_result_shape(result, start_index=start)

    def test_endpoints_have_zero_cross_product(self) -> None:
        values = np.linspace(1.0, 50.0, 512, dtype=np.float64) ** 1.7
        result = calculate_deviation(values)
        _assert_result_shape(result, start_index=0)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(result[-1, 1], 0.0, atol=1e-5)

    def test_subset_endpoints_have_zero_cross_product(self) -> None:
        values = np.exp(np.linspace(0.0, 4.0, 800, dtype=np.float64))
        start = 25
        result = calculate_deviation(values, above_index=start)
        _assert_result_shape(result, start_index=start)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(result[-1, 1], 0.0, atol=1e-4)

    @given(
        values=hnp.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=32, max_value=512),
            elements=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        ),
        start=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=40, deadline=None)
    def test_endpoints_zero_and_percentile_column(
        self, values: np.ndarray, start: int
    ) -> None:
        result = calculate_deviation(values, above_index=start)
        _assert_result_shape(result, start_index=start)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(result[-1, 1], 0.0, atol=1e-4)


@unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy required")
class TestCalculateDeviationCupy(unittest.TestCase):
    """CuPy inputs follow the same numeric curve as NumPy."""

    def setUp(self) -> None:
        import cupy as cp

        self.cp = cp

    def test_matches_numpy_on_device_input(self) -> None:
        host = np.exp(np.linspace(0.0, 2.5, 600, dtype=np.float64))
        start = 12
        expected = calculate_deviation(host, above_index=start)
        device = self.cp.asarray(host)
        actual_host = calculate_deviation(device, above_index=start)
        self.assertIsInstance(actual_host, self.cp.ndarray)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(actual_host),
            expected,
            rtol=1e-5,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
