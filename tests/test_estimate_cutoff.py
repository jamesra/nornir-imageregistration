"""Unit tests for histogram cutoff estimation (estimate_cutoff, linear_percentile_curve)."""

from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import nornir_imageregistration
from nornir_imageregistration.mathfuncs.cutoff_types import (
    CutoffMethod,
    EstimateCutoffResult,
)
from nornir_imageregistration.mathfuncs.estimate_cutoff import (
    estimate_cutoff,
    linear_percentile_curve,
)

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None


class TestLinearPercentileCurve(unittest.TestCase):
    """linear_percentile_curve matches NumPy linear percentiles."""

    def test_empty_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No values"):
            linear_percentile_curve(np.array([]), np.array([50.0]))

    def test_known_sorted_values(self) -> None:
        records = np.array([1.0, 2.0, 3.0, 4.0])
        percentiles = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        expected = np.percentile(records, percentiles, method="linear")
        actual = linear_percentile_curve(records, percentiles)
        np.testing.assert_allclose(actual, expected)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=64),
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        ),
        n_q=st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=40, deadline=None)
    def test_matches_numpy_percentile(self, data: np.ndarray, n_q: int) -> None:
        percentiles = np.linspace(0.0, 100.0, n_q)
        expected = np.percentile(data, percentiles, method="linear")
        actual = linear_percentile_curve(data, percentiles)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_cupy_matches_numpy(self) -> None:
        assert cp is not None
        rng = np.random.default_rng(3)
        host = rng.random(512).astype(np.float64)
        q = np.linspace(0.0, 100.0, 51)
        expected = np.percentile(host, q, method="linear")
        actual = nornir_imageregistration.EnsureNumpyArray(
            linear_percentile_curve(cp.asarray(host), q)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class TestEstimateCutoff(unittest.TestCase):
    """estimate_cutoff validation and method branches."""

    def test_polyfit_degree_below_one_raises(self) -> None:
        records = np.linspace(0.1, 1.0, 40)
        with self.assertRaisesRegex(ValueError, "Polyfit degree"):
            estimate_cutoff(records, polyfit_degree=0)

    def test_unknown_method_raises(self) -> None:
        records = np.linspace(0.1, 1.0, 40)
        bogus = cast(CutoffMethod, object())
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            estimate_cutoff(records, method=bogus)

    def test_no_inflection_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No inflection points"):
            estimate_cutoff(np.zeros(50))

    def test_step_distribution_all_methods(self) -> None:
        """A sharp step yields a cutoff in the high tail for each method."""
        records = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)]).astype(np.float64)
        results: dict[CutoffMethod, EstimateCutoffResult] = {}
        for method in CutoffMethod:
            results[method] = estimate_cutoff(records, method=method)
        for method, result in results.items():
            with self.subTest(method=method.name):
                self.assertGreater(result.cutoff_value, 0.5)
                self.assertIsInstance(result.cutoff_percentile_index, int)
                if method == CutoffMethod.Raw:
                    self.assertIsNone(result.y_fit)
                else:
                    self.assertIsNotNone(result.y_fit)

    def test_average_blends_raw_and_polyfit_cutoffs(self) -> None:
        """Average uses the polyfit inflection anchor and means raw/poly cutoff values."""
        records = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)]).astype(np.float64)
        raw = estimate_cutoff(records, method=CutoffMethod.Raw)
        poly = estimate_cutoff(records, method=CutoffMethod.Polyfit)
        avg = estimate_cutoff(records, method=CutoffMethod.Average)
        self.assertAlmostEqual(
            avg.cutoff_value,
            (raw.cutoff_value + poly.cutoff_value) / 2.0,
        )
        self.assertEqual(avg.highest_inflection_point, poly.highest_inflection_point)
        self.assertIsNotNone(avg.y_fit)

    def test_precomputed_percentile_curve(self) -> None:
        records = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)]).astype(np.float64)
        percentiles = np.linspace(0.0, 100.0, 101)
        curve = linear_percentile_curve(records, percentiles)
        direct = estimate_cutoff(records, method=CutoffMethod.Raw)
        precomputed = estimate_cutoff(
            records,
            percentiles=percentiles,
            precomputed_percentile_values=curve,
            method=CutoffMethod.Raw,
        )
        self.assertAlmostEqual(precomputed.cutoff_value, direct.cutoff_value)
        self.assertEqual(precomputed.cutoff_percentile_index, direct.cutoff_percentile_index)


if __name__ == "__main__":
    unittest.main()
