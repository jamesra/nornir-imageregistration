"""Direct tests for percentile-scale inflection point detection."""

import unittest

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from nornir_imageregistration.mathfuncs.cutoff_types import InflectionPointsResult
from nornir_imageregistration.mathfuncs.find_inflection_points import (
    find_inflection_points,
)


class TestFindInflectionPointsExamples(unittest.TestCase):
    def test_cubic_has_inflection_near_origin(self) -> None:
        x = np.linspace(-5.0, 5.0, 101)
        y = x**3
        result = find_inflection_points(x, y)
        self.assertIsInstance(result, InflectionPointsResult)
        self.assertEqual(len(result.indices), 1)
        self.assertAlmostEqual(float(result.values[0]), 0.0, delta=0.15)

    def test_percentile_sigmoid_inflection_near_midpoint(self) -> None:
        percentile = np.linspace(0.0, 100.0, 101)
        y = 1.0 / (1.0 + np.exp(-(percentile - 50.0) / 5.0))
        result = find_inflection_points(percentile, y)
        self.assertGreaterEqual(len(result.indices), 1)
        self.assertTrue(np.all(result.values >= 45.0))
        self.assertTrue(np.all(result.values <= 55.0))
        np.testing.assert_array_equal(result.values, percentile[result.indices])

    def test_monotonic_log_returns_no_inflections(self) -> None:
        percentile = np.linspace(0.0, 100.0, 101)
        y = np.log(percentile + 1.0)
        result = find_inflection_points(percentile, y)
        self.assertEqual(len(result.indices), 0)
        self.assertEqual(len(result.values), 0)

    def test_short_arrays_return_empty_result(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 8.0])
        result = find_inflection_points(x, y)
        self.assertEqual(len(result.indices), 0)
        self.assertEqual(len(result.values), 0)

    def test_named_tuple_unpacking_matches_fields(self) -> None:
        percentile = np.linspace(0.0, 100.0, 101)
        y = (percentile - 50.0) ** 3 / 1000.0
        indices, values = find_inflection_points(percentile, y)
        result = find_inflection_points(percentile, y)
        np.testing.assert_array_equal(indices, result.indices)
        np.testing.assert_array_equal(values, result.values)


class TestFindInflectionPointsProperties(unittest.TestCase):
    @settings(max_examples=40, deadline=None)
    @given(n=st.integers(min_value=4, max_value=120))
    @example(n=101)
    def test_result_values_match_x_at_indices(self, n: int) -> None:
        x = np.linspace(0.0, 100.0, n)
        y = (x - 50.0) ** 3
        result = find_inflection_points(x, y)
        self.assertEqual(len(result.indices), len(result.values))
        if len(result.indices) > 0:
            self.assertTrue(np.all(result.indices >= 0))
            self.assertTrue(np.all(result.indices < n))
            np.testing.assert_array_equal(result.values, x[result.indices])
