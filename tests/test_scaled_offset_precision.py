"""Regression for #181: sub-pixel peak offset keeps float64 CoM precision."""
from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.phasecorrelation import _scaled_offset_from_center_of_mass


class TestScaledOffsetPrecision(unittest.TestCase):
    def test_large_frame_matches_float64_not_float32(self) -> None:
        """#181: float32 drops ~4e-4 px at 16k; production path must keep float64."""
        shape = (16384, 16384)
        com = (8192.123456789, 8192.987654321)
        expected_f64 = (
            shape[0] / 2.0 - com[0],
            shape[1] / 2.0 - com[1],
        )
        lost_f32 = tuple(
            float(x)
            for x in (
                np.asarray(shape, dtype=np.float32) / np.float32(2.0)
                - np.asarray(com, dtype=np.float32)
            )
        )
        self.assertGreater(abs(expected_f64[0] - lost_f32[0]), 1e-4)

        got = _scaled_offset_from_center_of_mass(shape, com, np)
        np.testing.assert_allclose(got, expected_f64, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
