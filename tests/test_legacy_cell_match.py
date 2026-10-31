"""Legacy ir-refine-grid vertex estimator (refine_shared.legacy_cell_match) through the mosaic pass path."""
import unittest

import numpy as np
import scipy.ndimage

import nornir_imageregistration.local_distortion_correction as ldc


def _texture(seed: int = 3, size: int = 700) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tile = scipy.ndimage.gaussian_filter(rng.random((size, size)), 1.0).astype(np.float32)
    return (tile - tile.min()) / (tile.max() - tile.min())


class TestLegacyCellMatch(unittest.TestCase):

    def _measure(self, fixed: np.ndarray, moving: np.ndarray, starts: np.ndarray):
        pair = ldc._PairCells(0, 0, np.arange(len(starts)), fixed, starts, moving, starts)
        return ldc._correlate_pair_cells([pair], np.array([128, 128]), 0.25, legacy_match=True)

    def test_recovers_content_shift_with_the_mosaic_sign(self):
        tile = _texture()
        starts = np.array([[200, 200], [300, 350], [250, 150]], dtype=np.int64)
        for dy, dx in ((0, 0), (3, -5), (-7, 2), (12, 9), (-20, -15)):
            shifts, ok = self._measure(tile, np.roll(tile, (dy, dx), axis=(0, 1)), starts)
            self.assertTrue(np.all(ok == 1.0))
            # Vertex shift is minus the content displacement, as the single-peak path reports.
            np.testing.assert_allclose(shifts, np.tile([-dy, -dx], (3, 1)), atol=0.25)

    def test_matches_single_peak_estimator_on_clear_cells(self):
        tile = _texture(seed=5)
        starts = np.array([[150, 150], [400, 220]], dtype=np.int64)
        moving = np.roll(tile, (6, -4), axis=(0, 1))
        legacy, _ = self._measure(tile, moving, starts)
        pair = ldc._PairCells(0, 0, np.arange(2), tile, starts, moving, starts)
        current, _ = ldc._correlate_pair_cells([pair], np.array([128, 128]), 0.25, legacy_match=False)
        np.testing.assert_allclose(legacy, current, atol=0.25)

    def test_flat_cells_are_not_measured(self):
        flat = np.full((700, 700), 0.5, dtype=np.float32)
        _, ok = self._measure(flat, flat, np.array([[200, 200]], dtype=np.int64))
        self.assertEqual(ok.tolist(), [0.0])


if __name__ == '__main__':
    unittest.main()
