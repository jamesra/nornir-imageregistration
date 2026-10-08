"""Tests for nornir_imageregistration.mathfuncs.ema.EMA."""

from __future__ import annotations

import math
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from nornir_imageregistration.mathfuncs.ema import EMA


class TestEMAConstruction(unittest.TestCase):
    def test_num_samples_below_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            EMA(0)
        with self.assertRaises(ValueError):
            EMA(-3)


class TestEMABeforeSamples(unittest.TestCase):
    def test_has_samples_false_until_add(self) -> None:
        ema = EMA(3)
        self.assertFalse(ema.has_samples)
        with self.assertRaises(ValueError):
            _ = ema.ema_value


class TestEMAUpdates(unittest.TestCase):
    def test_first_sample_ema_equals_value(self) -> None:
        ema = EMA(5, smooth=2.0)
        ema.add(10.0)
        self.assertTrue(ema.has_samples)
        self.assertEqual(ema.ema_value, 10.0)

    def test_exponential_blend_matches_formula(self) -> None:
        smooth = 2.0
        ema = EMA(5, smooth=smooth)
        ema.add(10.0)
        first = ema.ema_value
        ema.add(20.0)
        scalar = smooth / (1 + 2)
        expected = 20.0 * scalar + first * (1.0 - scalar)
        self.assertAlmostEqual(ema.ema_value, expected)

    def test_reset_then_add_recomputes_from_reset_value(self) -> None:
        ema = EMA(4, smooth=2.0)
        ema.add(1.0)
        ema.add(2.0)
        ema.reset(100.0)
        self.assertEqual(ema.ema_value, 100.0)
        ema.add(200.0)
        scalar = 2.0 / (1 + 4)
        expected = 200.0 * scalar + 100.0 * (1.0 - scalar)
        self.assertAlmostEqual(ema.ema_value, expected)

    def test_sample_count_caps_at_num_samples(self) -> None:
        ema = EMA(2, smooth=2.0)
        ema.add(1.0)
        ema.add(2.0)
        after_two = ema.ema_value
        ema.add(3.0)
        scalar = 2.0 / (1 + 2)
        expected = 3.0 * scalar + after_two * (1.0 - scalar)
        self.assertAlmostEqual(ema.ema_value, expected)
        ema.add(4.0)
        scalar = 2.0 / (1 + 2)
        expected = 4.0 * scalar + expected * (1.0 - scalar)
        self.assertAlmostEqual(ema.ema_value, expected)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=15,
    ),
    st.integers(min_value=1, max_value=6),
    st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_ema_value_stays_within_seen_range(
    values: list[float], window: int, smooth: float
) -> None:
    ema = EMA(window, smooth=smooth)
    seen: list[float] = []
    for v in values:
        seen.append(v)
        ema.add(v)
        lo, hi = min(seen), max(seen)
        assert lo - 1e-6 <= ema.ema_value <= hi + 1e-6
        assert math.isfinite(ema.ema_value)


if __name__ == "__main__":
    unittest.main()
