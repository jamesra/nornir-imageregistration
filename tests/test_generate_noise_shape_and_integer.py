"""ImageStats.GenerateNoise shape dispatch and integer dtypes (#248)."""
from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.image_stats import ImageStats


class TestGenerateNoiseShapeAndInteger(unittest.TestCase):
    def setUp(self) -> None:
        # Non-trivial range so clip and rint are exercised (not a constant image).
        image = (np.arange(256, dtype=np.float32).reshape(16, 16) / 255.0)
        self.stats = ImageStats.CalcStats(image)

    def test_tuple_shape_matches_ndarray_shape(self) -> None:
        from_tuple = self.stats.GenerateNoise((8, 8), dtype=np.float32, xp=np)
        from_array = self.stats.GenerateNoise(np.asarray((8, 8)), dtype=np.float32, xp=np)
        self.assertEqual(from_tuple.shape, (8, 8))
        self.assertEqual(from_array.shape, (8, 8))
        self.assertEqual(from_tuple.dtype, np.float32)

    def test_list_shape_and_one_d_sequence(self) -> None:
        grid = self.stats.GenerateNoise([4, 5], dtype=np.float32, xp=np)
        self.assertEqual(grid.shape, (4, 5))
        vector = self.stats.GenerateNoise((12,), dtype=np.float32, xp=np)
        self.assertEqual(vector.shape, (12,))

    def test_integer_dtypes_stay_in_bounds(self) -> None:
        # Scale stats into a uint-friendly range for the integer casts.
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        stats = ImageStats.CalcStats(image.astype(np.float32))
        for dtype in (np.uint8, np.uint16, np.int32):
            with self.subTest(dtype=np.dtype(dtype).name):
                out = stats.GenerateNoise((16, 16), dtype=dtype, xp=np)
                self.assertEqual(out.dtype, np.dtype(dtype))
                self.assertEqual(out.shape, (16, 16))
                self.assertGreaterEqual(int(out.min()), int(np.ceil(stats.min)) - 1)
                self.assertLessEqual(int(out.max()), int(np.floor(stats.max)) + 1)
                self.assertGreaterEqual(float(out.min()), stats.min - 0.5)
                self.assertLessEqual(float(out.max()), stats.max + 0.5)

    def test_float_still_cast_then_clip(self) -> None:
        out = self.stats.GenerateNoise((32, 32), dtype=np.float16, xp=np)
        self.assertEqual(out.dtype, np.float16)
        self.assertGreaterEqual(float(out.min()), self.stats.min)
        self.assertLessEqual(float(out.max()), self.stats.max)


if __name__ == '__main__':
    unittest.main()
