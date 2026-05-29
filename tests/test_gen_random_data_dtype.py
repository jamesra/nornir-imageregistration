"""Smoke tests for dtype range promotion in GenRandomData and padding."""
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.core._core import promote_dtype_for_value_range


class TestPromoteDtypeForValueRange(unittest.TestCase):
    def test_float16_range_ok(self) -> None:
        dt = promote_dtype_for_value_range(np.float16, -1.0, 1.0)
        self.assertEqual(dt, np.dtype(np.float16))

    def test_promotes_past_float16_max(self) -> None:
        dt = promote_dtype_for_value_range(np.float16, -70000.0, 1.0)
        self.assertEqual(dt, np.dtype(np.float32))

    def test_gen_random_data_respects_promotion(self) -> None:
        a = nornir_imageregistration.GenRandomData(
            8, 8, 0.0, 1.0, -70000.0, 1.0, dtype=np.float16
        )
        self.assertEqual(a.dtype, np.dtype(np.float32))


class TestPadPhaseCorrelationDtype(unittest.TestCase):
    def test_border_matches_padded_dtype(self) -> None:
        p = np.zeros((8, 8), dtype=np.float32)
        p[2:6, 2:6] = 1.0
        out = nornir_imageregistration.pad_image_for_phase_correlation(
            p, power_of_two=True
        )
        self.assertEqual(out.dtype, np.dtype(np.float32))


if __name__ == "__main__":
    unittest.main()
