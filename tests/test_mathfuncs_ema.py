import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from nornir_imageregistration.mathfuncs.ema import EMA


class TestEMAConstruction(unittest.TestCase):
    def test_rejects_non_positive_window(self) -> None:
        with self.assertRaises(ValueError):
            EMA(0)
        with self.assertRaises(ValueError):
            EMA(-1)

    def test_has_samples_false_until_add(self) -> None:
        ema = EMA(3)
        self.assertFalse(ema.has_samples)
        with self.assertRaises(ValueError):
            _ = ema.ema_value


class TestEMASmoothing(unittest.TestCase):
    def test_first_sample_equals_value(self) -> None:
        ema = EMA(10, smooth=2)
        ema.add(42.0)
        self.assertTrue(ema.has_samples)
        self.assertEqual(ema.ema_value, 42.0)

    def test_known_sequence_smooth_two(self) -> None:
        ema = EMA(100, smooth=2)
        ema.add(10.0)
        self.assertAlmostEqual(ema.ema_value, 10.0)
        ema.add(20.0)
        self.assertAlmostEqual(ema.ema_value, 50.0 / 3.0)
        ema.add(30.0)
        self.assertAlmostEqual(ema.ema_value, 70.0 / 3.0)

    def test_reset_sets_average_immediately(self) -> None:
        ema = EMA(5, smooth=2)
        ema.add(1.0)
        ema.add(2.0)
        ema.reset(99.0)
        self.assertEqual(ema.ema_value, 99.0)
        ema.add(100.0)
        scalar = 2.0 / (1.0 + 5)
        expected = 100.0 * scalar + 99.0 * (1.0 - scalar)
        self.assertAlmostEqual(ema.ema_value, expected)


class TestEMASampleCap(unittest.TestCase):
    def test_sample_count_saturates_at_window(self) -> None:
        """After num_samples adds, further adds use the capped divisor, not a larger one."""
        values = (10.0, 20.0, 30.0, 40.0)
        capped = EMA(3, smooth=2)
        for value in values:
            capped.add(value)

        uncapped = EMA(10, smooth=2)
        for value in values:
            uncapped.add(value)

        self.assertNotAlmostEqual(capped.ema_value, uncapped.ema_value)
        self.assertAlmostEqual(capped.ema_value, 95.0 / 3.0)


class TestEMAProperties(unittest.TestCase):
    @settings(max_examples=100)
    @given(
        constant=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        num_samples=st.integers(min_value=1, max_value=8),
    )
    def test_constant_stream_stays_at_constant(self, constant: float, num_samples: int) -> None:
        ema = EMA(num_samples, smooth=2)
        for _ in range(num_samples + 3):
            ema.add(constant)
        self.assertAlmostEqual(ema.ema_value, constant)


if __name__ == "__main__":
    unittest.main()
