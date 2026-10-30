"""Unit tests for mathfuncs.ema.EMA."""

from __future__ import annotations

import math
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from nornir_imageregistration.mathfuncs.ema import EMA


def _reference_ema_after_adds(values: list[float], *, window: int, smooth: float) -> float:
    """Independent step-through of EMA.add() / ema_value for oracle checks."""
    collected = 0
    last_ema_value: float | None = None
    current_value: float | None = None
    current_ema_value: float | None = None

    def ema_value() -> float:
        nonlocal current_ema_value
        if collected == 0:
            raise ValueError("no samples")
        if current_ema_value is None:
            scalar = smooth / (1 + collected)
            current_ema_value = (current_value or 0) * scalar
            current_ema_value += (last_ema_value or 0) * (1.0 - scalar)
        return current_ema_value

    for value in values:
        last_ema_value = ema_value() if collected > 0 else value
        collected = collected + 1 if collected < window else window
        current_value = value
        current_ema_value = None

    return ema_value()


class TestEMAConstruction(unittest.TestCase):
    """Construction guards."""

    def test_rejects_non_positive_window(self) -> None:
        with self.assertRaises(ValueError):
            EMA(0)
        with self.assertRaises(ValueError):
            EMA(-3)


class TestEMABeforeFirstSample(unittest.TestCase):
    """State before add()."""

    def test_has_samples_false_until_add(self) -> None:
        ema = EMA(3, smooth=2)
        self.assertFalse(ema.has_samples)

    def test_ema_value_raises_until_add(self) -> None:
        ema = EMA(3, smooth=2)
        with self.assertRaises(ValueError):
            _ = ema.ema_value


class TestEMAAddSequence(unittest.TestCase):
    """Deterministic add() behavior."""

    def test_single_sample_equals_value(self) -> None:
        ema = EMA(5, smooth=2)
        ema.add(7.5)
        self.assertTrue(ema.has_samples)
        self.assertEqual(ema.ema_value, 7.5)

    def test_two_sample_smooth_two(self) -> None:
        ema = EMA(5, smooth=2)
        ema.add(10.0)
        ema.add(20.0)
        scalar = 2.0 / 3.0
        expected = 20.0 * scalar + 10.0 * (1.0 - scalar)
        self.assertAlmostEqual(ema.ema_value, expected, places=12)

    def test_sample_count_caps_at_window(self) -> None:
        ema = EMA(2, smooth=2)
        ema.add(0.0)
        ema.add(0.0)
        self.assertAlmostEqual(ema.ema_value, 0.0, places=12)
        ema.add(1.0)
        self.assertAlmostEqual(ema.ema_value, 2.0 / 3.0, places=12)


class TestEMAReset(unittest.TestCase):
    """reset() seeds a full window."""

    def test_reset_makes_ema_readable_at_value(self) -> None:
        ema = EMA(4, smooth=2)
        ema.reset(42.0)
        self.assertTrue(ema.has_samples)
        self.assertEqual(ema.ema_value, 42.0)

    def test_reset_after_adds(self) -> None:
        ema = EMA(3, smooth=2)
        ema.add(1.0)
        ema.add(2.0)
        ema.reset(-5.0)
        self.assertEqual(ema.ema_value, -5.0)
        ema.add(10.0)
        scalar = 2.0 / (1.0 + 3.0)
        expected = 10.0 * scalar + (-5.0) * (1.0 - scalar)
        self.assertAlmostEqual(ema.ema_value, expected, places=12)


class TestEMAHypothesis(unittest.TestCase):
    """Property checks on generated inputs."""

    @given(
        window=st.integers(min_value=1, max_value=32),
        smooth=st.floats(min_value=0.25, max_value=8.0, allow_nan=False, allow_infinity=False),
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=80, deadline=None)
    def test_first_add_ema_equals_value(self, window: int, smooth: float, value: float) -> None:
        ema = EMA(window, smooth)
        ema.add(value)
        self.assertTrue(ema.has_samples)
        self.assertTrue(math.isclose(ema.ema_value, value, rel_tol=0, abs_tol=1e-9))

    @given(
        values=st.lists(
            st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=12,
        ),
        window=st.integers(min_value=1, max_value=16),
        smooth=st.floats(min_value=0.5, max_value=4.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=60, deadline=None)
    def test_add_sequence_matches_reference(self, values: list[float], window: int, smooth: float) -> None:
        ema = EMA(window, smooth)
        for v in values:
            ema.add(v)
        expected = _reference_ema_after_adds(values, window=window, smooth=smooth)
        self.assertTrue(math.isclose(ema.ema_value, expected, rel_tol=1e-9, abs_tol=1e-9))
