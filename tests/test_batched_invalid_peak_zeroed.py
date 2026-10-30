"""An invalid batched cell must not report an offset.

``batched_find_offset`` zeroed ``weights`` and ``peak_ratios`` for cells it judged
invalid but returned ``peaks`` straight from ``batched_find_peak``.  For a cell
with no usable signal that argmax is the centre of the correlation surface, so a
32x32 cell reported an offset of ``(16, 16)`` -- half the cell, in both axes --
alongside a weight of 0.

Serial ``find_offset`` blanks the entire correlation image when its max is
non-finite (``correlation_image[...] = 0``) and consequently reports ``(0, 0)``.

Both production call sites in ``local_distortion_correction`` skip on
``weight <= 0``, so nothing was mis-registered in practice.  This is about the
offset not carrying a plausible number that the weight declares meaningless,
since the offset is the primary return value.
"""
from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.batched_phase_correlation import batched_find_offset
from nornir_imageregistration.phasecorrelation import find_offset

_N = 32
_SHAPE = np.asarray([_N, _N], dtype=np.int64)


def _batched(cell_a: np.ndarray, cell_b: np.ndarray):
    peaks, weights, ratios = batched_find_offset(
        cell_a[None].copy(), cell_b[None].copy(), _SHAPE, min_overlap=0.25)
    return np.asarray(peaks[0], dtype=np.float64), float(weights[0]), float(ratios[0])


def _serial(cell_a: np.ndarray, cell_b: np.ndarray):
    record = find_offset(cell_a.copy(), cell_b.copy(), min_overlap=0.25, max_overlap=1.0,
                         target_shape=_SHAPE, source_shape=_SHAPE, fft_required=True)
    return np.asarray(record.peak, dtype=np.float64), float(record.weight)


def _nan_cell(seed: int = 2) -> np.ndarray:
    cell = np.random.default_rng(seed).random((_N, _N))
    cell[5, 5] = np.nan
    return cell


class TestInvalidCellsReportNoOffset(unittest.TestCase):

    def test_nan_cell_reports_zero_offset(self):
        peak, weight, _ = _batched(_nan_cell(), np.random.default_rng(3).random((_N, _N)))

        self.assertEqual(weight, 0.0)
        np.testing.assert_array_equal(peak, np.zeros(2))

    def test_constant_cell_reports_zero_offset(self):
        cell = np.full((_N, _N), 0.5)

        peak, weight, _ = _batched(cell, cell)

        self.assertEqual(weight, 0.0)
        np.testing.assert_array_equal(peak, np.zeros(2))

    def test_all_zero_cell_reports_zero_offset(self):
        cell = np.zeros((_N, _N))

        peak, weight, _ = _batched(cell, cell)

        self.assertEqual(weight, 0.0)
        np.testing.assert_array_equal(peak, np.zeros(2))

    def test_offset_is_never_the_surface_centre(self):
        """The specific failure mode: half-cell offsets that look like real shifts."""
        for label, cell_a, cell_b in (
                ('nan', _nan_cell(), np.random.default_rng(3).random((_N, _N))),
                ('constant', np.full((_N, _N), 0.5), np.full((_N, _N), 0.5)),
                ('zero', np.zeros((_N, _N)), np.zeros((_N, _N))),
        ):
            with self.subTest(cell=label):
                peak, weight, _ = _batched(cell_a, cell_b)

                self.assertEqual(weight, 0.0, 'premise: this cell must be judged invalid')
                self.assertFalse(np.allclose(peak, [_N // 2, _N // 2]),
                                 f'offset {peak} is the correlation surface centre')


class TestSerialParity(unittest.TestCase):
    """Serial blanks the surface; batched must report the same offset it does."""

    def test_degenerate_offsets_match_serial(self):
        cases = {
            'nan': (_nan_cell(), np.random.default_rng(3).random((_N, _N))),
            'constant': (np.full((_N, _N), 0.5), np.full((_N, _N), 0.5)),
            'all_zero': (np.zeros((_N, _N)), np.zeros((_N, _N))),
        }
        for label, (cell_a, cell_b) in cases.items():
            with self.subTest(cell=label):
                serial_peak, serial_weight = _serial(cell_a, cell_b)
                batched_peak, batched_weight, _ = _batched(cell_a, cell_b)

                self.assertEqual(serial_weight, 0.0)
                self.assertEqual(batched_weight, 0.0)
                np.testing.assert_allclose(batched_peak, serial_peak, atol=1e-9)


class TestValidCellsUnaffected(unittest.TestCase):
    """The masking must not touch cells that produced a real measurement."""

    def test_healthy_offset_matches_serial(self):
        cell = np.random.default_rng(2).random((_N, _N))
        shifted = np.roll(cell, 3, axis=0)

        serial_peak, serial_weight = _serial(cell, shifted)
        batched_peak, batched_weight, _ = _batched(cell, shifted)

        self.assertGreater(batched_weight, 0.0)
        self.assertAlmostEqual(batched_weight, serial_weight, places=6)
        np.testing.assert_allclose(batched_peak, serial_peak, atol=1e-9)

    def test_mixed_batch_keeps_the_valid_offset(self):
        """Masking is per-cell, so a good cell beside a bad one is untouched."""
        good = np.random.default_rng(2).random((_N, _N))
        shifted = np.roll(good, 3, axis=0)
        bad = np.full((_N, _N), 0.5)

        alone_peak, _, _ = _batched(good, shifted)
        peaks, weights, _ = batched_find_offset(
            np.stack([bad, good]), np.stack([bad, shifted]), _SHAPE, min_overlap=0.25)
        peaks = np.asarray(peaks, dtype=np.float64)

        self.assertEqual(float(weights[0]), 0.0)
        np.testing.assert_array_equal(peaks[0], np.zeros(2))
        self.assertGreater(float(weights[1]), 0.0)
        np.testing.assert_allclose(peaks[1], alone_peak, atol=1e-9)

    def test_nonzero_offsets_survive_for_a_whole_healthy_batch(self):
        rng = np.random.default_rng(9)
        cells = [rng.random((_N, _N)) for _ in range(4)]
        shifts = (1, 2, 3, 4)
        fixed = np.stack(cells)
        moving = np.stack([np.roll(c, s, axis=0) for c, s in zip(cells, shifts)])

        peaks, weights, _ = batched_find_offset(fixed, moving, _SHAPE, min_overlap=0.25)

        self.assertTrue(all(float(w) > 0 for w in weights))
        for peak in np.asarray(peaks, dtype=np.float64):
            self.assertFalse(np.allclose(peak, np.zeros(2)),
                             'a healthy cell had its offset zeroed')


if __name__ == '__main__':
    unittest.main()
