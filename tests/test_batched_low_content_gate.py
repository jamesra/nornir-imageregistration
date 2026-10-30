"""The batched degeneracy gate must match serial ``is_alignable_cell``.

``batched_find_offset`` rejected only cells with zero span or an all-zero
maximum.  Serial ``is_alignable_cell`` additionally rejects cells whose intensity
std falls below ``NORNIR_REFINE_LOW_CONTENT_STD_MIN`` (default 1e-3), so
low-contrast cells that serial declined were measured in batched mode.

That matters because each cell is normalized by its own span, which amplifies
micro-contrast up to full range.  A cell with std 3e-5 produced a confident
weight of ~1.77 and a garbage sub-pixel peak instead of the zero weight serial
reports.

The gate uses a range bound to stay cheap: for ``n`` samples with range
``span``, std >= span / sqrt(2n), so a large span proves the floor is met and the
std reduction is skipped. The exact reduction runs only when some cell is
ambiguous.
"""
from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.batched_phase_correlation import batched_find_offset
from nornir_imageregistration.refine_shared.cell_measurement import (
    measure_translation_cell, measure_translation_cells_batched)
from nornir_imageregistration.refine_shared.cell_validity import (
    DEFAULT_LOW_CONTENT_STD_MIN, cell_intensity_std, is_alignable_cell)

_N = 32
_SHAPE = np.asarray([_N, _N], dtype=np.int64)


def _low_contrast(seed: int = 11) -> np.ndarray:
    """Nonzero span, but std roughly 30x below the default floor."""
    return np.random.default_rng(seed).random((_N, _N)) * 1e-4


def _healthy(seed: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).random((_N, _N))


def _batched_weight(cell_a: np.ndarray, cell_b: np.ndarray) -> float:
    _, weights, _ = measure_translation_cells_batched(
        cell_a[None].copy(), cell_b[None].copy(), _SHAPE)
    return float(weights[0])


class TestLowContrastCellsAreRejected(unittest.TestCase):

    def test_fixture_is_actually_below_the_floor(self):
        """Guard the premise, so the test cannot silently stop testing anything."""
        cell = _low_contrast()

        self.assertGreater(float(cell.max() - cell.min()), 0.0,
                           'span must be nonzero or the old gate would catch it')
        self.assertLess(cell_intensity_std(cell), DEFAULT_LOW_CONTENT_STD_MIN)
        self.assertFalse(is_alignable_cell(cell))

    def test_batched_rejects_what_serial_rejects(self):
        cell_a, cell_b = _low_contrast(11), _low_contrast(12)

        serial = measure_translation_cell(cell_a.copy(), cell_b.copy(), _SHAPE)

        self.assertEqual(serial.weight, 0.0)
        self.assertEqual(_batched_weight(cell_a, cell_b), 0.0)

    def test_peak_ratio_is_also_zeroed(self):
        cell_a, cell_b = _low_contrast(11), _low_contrast(12)

        _, _, ratios = measure_translation_cells_batched(
            cell_a[None].copy(), cell_b[None].copy(), _SHAPE)

        self.assertEqual(float(ratios[0]), 0.0)

    def test_one_bad_cell_does_not_taint_its_neighbours(self):
        """The gate is per-cell, so a healthy cell in the same batch is unaffected."""
        healthy = _healthy()
        shifted = np.roll(healthy, 3, axis=0)
        stack_fixed = np.stack([_low_contrast(11), healthy])
        stack_moving = np.stack([_low_contrast(12), shifted])

        _, weights, _ = measure_translation_cells_batched(stack_fixed, stack_moving, _SHAPE)

        self.assertEqual(float(weights[0]), 0.0)
        self.assertGreater(float(weights[1]), 0.0)

    def test_rejection_matches_serial_across_contrast_levels(self):
        for scale in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            with self.subTest(scale=scale):
                rng = np.random.default_rng(7)
                cell_a = rng.random((_N, _N)) * scale
                cell_b = np.roll(cell_a, 2, axis=0)

                serial = measure_translation_cell(cell_a.copy(), cell_b.copy(), _SHAPE)
                batched = _batched_weight(cell_a, cell_b)

                self.assertEqual(serial.weight <= 0.0, batched <= 0.0,
                                 f'serial {serial.weight} vs batched {batched}')


class TestHealthyCellsUnaffected(unittest.TestCase):
    """The gate must not disturb the measurements that already worked."""

    def test_weight_and_peak_match_serial_exactly(self):
        healthy = _healthy()
        shifted = np.roll(healthy, 3, axis=0)

        serial = measure_translation_cell(healthy.copy(), shifted.copy(), _SHAPE)
        peaks, weights, _ = measure_translation_cells_batched(
            healthy[None].copy(), shifted[None].copy(), _SHAPE)

        self.assertAlmostEqual(float(weights[0]), float(serial.weight), places=9)
        np.testing.assert_allclose(np.asarray(peaks[0]), np.asarray(serial.peak), atol=1e-9)

    def test_modest_contrast_just_above_the_floor_still_measures(self):
        rng = np.random.default_rng(3)
        cell = rng.random((_N, _N)) * 0.01 + 0.5
        self.assertGreater(cell_intensity_std(cell), DEFAULT_LOW_CONTENT_STD_MIN)

        weight = _batched_weight(cell, np.roll(cell, 2, axis=0))

        self.assertGreater(weight, 0.0)

    def test_big_span_with_tiny_std_is_not_wrongly_accepted_or_rejected(self):
        """A single hot pixel gives a full span; serial decides on std, so must batched.

        This is the case the range-bound fast path must not get wrong.
        """
        cell = np.full((_N, _N), 0.5)
        cell[0, 0] = 1.0

        serial = measure_translation_cell(cell.copy(), cell.copy(), _SHAPE)
        batched = _batched_weight(cell, cell)

        self.assertEqual(serial.weight <= 0.0, batched <= 0.0)


class TestConstantAndZeroCellsStillRejected(unittest.TestCase):
    """Pre-existing rejections must survive the change."""

    def test_constant(self):
        cell = np.full((_N, _N), 0.5)
        self.assertEqual(_batched_weight(cell, cell), 0.0)

    def test_all_zero(self):
        cell = np.zeros((_N, _N))
        self.assertEqual(_batched_weight(cell, cell), 0.0)

    def test_nan_cell(self):
        """NaN already failed the span test; it must keep failing."""
        cell = _healthy().copy()
        cell[0, 0] = np.nan
        self.assertEqual(_batched_weight(cell, cell), 0.0)


class TestMinStdParameter(unittest.TestCase):
    """The threshold is overridable, matching is_alignable_cell(min_std=...)."""

    def test_zero_threshold_disables_the_std_floor(self):
        cell_a, cell_b = _low_contrast(11), _low_contrast(12)

        _, weights, _ = batched_find_offset(
            cell_a[None].copy(), cell_b[None].copy(), _SHAPE, min_std=0.0)

        self.assertGreater(float(weights[0]), 0.0,
                           'with the floor disabled the old behaviour should return')

    def test_high_threshold_rejects_healthy_cells(self):
        healthy = _healthy()

        _, weights, _ = batched_find_offset(
            healthy[None].copy(), np.roll(healthy, 3, axis=0)[None].copy(),
            _SHAPE, min_std=10.0)

        self.assertEqual(float(weights[0]), 0.0)

    def test_threshold_agrees_with_is_alignable_cell(self):
        rng = np.random.default_rng(21)
        cell = rng.random((_N, _N)) * 0.02
        std = cell_intensity_std(cell)

        for threshold in (std * 0.5, std * 2.0):
            with self.subTest(threshold=threshold):
                expected = is_alignable_cell(cell, min_std=threshold)
                _, weights, _ = batched_find_offset(
                    cell[None].copy(), np.roll(cell, 2, axis=0)[None].copy(),
                    _SHAPE, min_std=threshold)

                self.assertEqual(float(weights[0]) > 0.0, expected)


if __name__ == '__main__':
    unittest.main()
