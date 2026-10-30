"""Unit tests for mathfuncs.calculate_deviation (signed cross product along percentile curves)."""

from __future__ import annotations

import unittest

import hypothesis
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np

import nornir_imageregistration
from nornir_imageregistration.mathfuncs.calculate_deviation import calculate_deviation


def _reference_calculate_deviation(
    values: np.ndarray, above_index: int | None = None
) -> np.ndarray:
    """Host NumPy reference matching calculate_deviation geometry."""
    p = np.linspace(0, 100, 101)
    start_index = 0 if above_index is None else above_index
    percentile_values = np.percentile(values, p, method="linear")
    percentile_value_subset = percentile_values[start_index:]
    percentile_subset = p[start_index:]
    min_value = percentile_value_subset[0]
    max_value = percentile_value_subset[-1]

    min_vectors = np.column_stack(
        (
            percentile_value_subset - min_value,
            percentile_subset - percentile_subset[0],
            np.zeros(len(percentile_subset)),
        )
    )
    max_vectors = np.column_stack(
        (
            max_value - percentile_value_subset,
            percentile_subset - percentile_subset[-1],
            np.zeros(len(percentile_subset)),
        )
    )
    cross_products = np.cross(min_vectors, max_vectors)
    return np.vstack((percentile_subset, cross_products[:, 2])).T


class TestCalculateDeviation(unittest.TestCase):
    """Signed cross-product helper used by estimate_cutoff inflection selection."""

    @staticmethod
    def _percentile_curve_from_records(records: np.ndarray) -> np.ndarray:
        percentiles = np.linspace(0, 100, 101)
        return np.percentile(records, percentiles, method="linear")

    def test_none_above_index_matches_zero(self) -> None:
        records = np.linspace(0.0, 1.0, 250) ** 2
        curve = self._percentile_curve_from_records(records)
        without = calculate_deviation(curve, above_index=None)
        with_zero = calculate_deviation(curve, above_index=0)
        np.testing.assert_allclose(without, with_zero, rtol=0, atol=0)

    def test_endpoints_have_zero_cross_product(self) -> None:
        records = np.sort(np.random.default_rng(0).normal(size=400))
        curve = self._percentile_curve_from_records(records)
        result = calculate_deviation(curve, above_index=15)
        self.assertEqual(result.shape, (101 - 15, 2))
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(result[-1, 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(result[:, 0], np.linspace(15, 100, 101 - 15))

    @hypothesis.example(above_index=20, records=np.linspace(0.0, 1.0, 101) ** 2)
    @hypothesis.example(above_index=0, records=np.linspace(-3.0, 7.0, 180))
    @hypothesis.given(
        st.integers(min_value=0, max_value=90),
        hnp.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=32, max_value=512),
            elements=st.floats(
                min_value=-1e4,
                max_value=1e4,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
    )
    @hypothesis.settings(max_examples=40, deadline=None)
    def test_matches_numpy_reference(self, above_index: int, records: np.ndarray) -> None:
        hypothesis.assume(np.ptp(records) > 1e-6)
        curve = self._percentile_curve_from_records(records)
        actual = calculate_deviation(curve, above_index=above_index)
        expected = _reference_calculate_deviation(curve, above_index=above_index)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(actual),
            expected,
            rtol=1e-5,
            atol=1e-4,
        )

    def test_convex_curve_interior_below_chord_is_negative(self) -> None:
        """Quadratic percentile curve yields negative signed cross product at the interior."""
        percentiles = np.linspace(0, 100, 101)
        curve = percentiles**2
        z = calculate_deviation(curve)[50, 1]
        self.assertLess(float(z), 0.0)


@unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy required")
class TestCalculateDeviationCupy(unittest.TestCase):
    def setUp(self) -> None:
        import cupy as cp

        self.cp = cp

    def test_cupy_matches_numpy(self) -> None:
        records_host = np.sort(np.random.default_rng(1).uniform(-50.0, 50.0, size=300))
        percentiles = np.linspace(0, 100, 101)
        curve_host = np.percentile(records_host, percentiles, method="linear")
        curve_device = self.cp.asarray(curve_host)

        host_out = calculate_deviation(curve_host, above_index=25)
        device_out = calculate_deviation(curve_device, above_index=25)

        self.assertIsInstance(device_out, self.cp.ndarray)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(device_out),
            nornir_imageregistration.EnsureNumpyArray(host_out),
            rtol=1e-5,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
