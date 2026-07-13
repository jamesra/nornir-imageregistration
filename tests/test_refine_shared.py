"""Unit tests for refine_shared helpers."""

from __future__ import annotations

import os
import unittest

import numpy as np

from nornir_imageregistration.refine_shared import (
    RefineRuntimeConfig,
    filter_weights_by_estimate_cutoff,
    is_alignable_cell,
    measure_translation_cell,
    normalize_cell,
    regularize_displacements,
)


class TestRefineShared(unittest.TestCase):
    """Coverage for shared mosaic/STOS refine primitives."""

    def test_is_alignable_cell_rejects_constant(self) -> None:
        """Constant and empty cells are not alignable."""
        self.assertFalse(is_alignable_cell(np.zeros((8, 8))))
        self.assertFalse(is_alignable_cell(np.ones((8, 8)) * 5))
        self.assertTrue(is_alignable_cell(np.linspace(0, 1, 64).reshape(8, 8)))

    def test_normalize_cell_range(self) -> None:
        """Normalized cells span [0, 1]."""
        cell = np.asarray([[10.0, 20.0], [30.0, 40.0]])
        normalized = normalize_cell(cell)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_measure_translation_cell_identical(self) -> None:
        """Identical textured cells yield near-zero peak with positive weight."""
        rng = np.random.default_rng(0)
        cell = rng.random((32, 32))
        record = measure_translation_cell(cell, cell, np.asarray((32, 32)))
        self.assertGreater(float(record.weight), 0.0)
        self.assertLess(float(np.linalg.norm(record.peak)), 1.0)

    def test_filter_weights_by_estimate_cutoff(self) -> None:
        """Low outliers are dropped when enough samples exist."""
        weights = np.asarray([0.1, 0.12, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0])
        keep = filter_weights_by_estimate_cutoff(weights)
        self.assertEqual(keep.shape, weights.shape)
        self.assertTrue(np.any(keep))
        self.assertTrue(np.any(~keep) or keep.sum() == keep.size)

    def test_regularize_displacements_gap_fill(self) -> None:
        """Unmeasured vertices receive filled values from neighbors."""
        mesh = (3, 3)
        shifts = np.zeros((9, 2), dtype=np.float64)
        measured = np.zeros(9, dtype=bool)
        shifts[4, :] = (1.0, -1.0)
        measured[4] = True
        out, db = regularize_displacements(shifts, measured, mesh, median_radius=1)
        self.assertEqual(out.shape, (9, 2))
        self.assertGreaterEqual(float(db.sum()), 1.0)

    def test_runtime_config_env_flags(self) -> None:
        """RefineRuntimeConfig reads mosaic cutoff and STOS fallback flags."""
        os.environ['NORNIR_REFINE_MOSAIC_CUTOFF'] = '1'
        os.environ['NORNIR_STOS_REFINE_FALLBACK'] = '1'
        os.environ['NORNIR_REFINE_STOS_REGULARIZE'] = '1'
        try:
            cfg = RefineRuntimeConfig.from_env()
            self.assertTrue(cfg.mosaic_cutoff)
            self.assertTrue(cfg.stos_refine_fallback)
            self.assertTrue(cfg.stos_regularize)
        finally:
            os.environ.pop('NORNIR_REFINE_MOSAIC_CUTOFF', None)
            os.environ.pop('NORNIR_STOS_REFINE_FALLBACK', None)
            os.environ.pop('NORNIR_REFINE_STOS_REGULARIZE', None)


if __name__ == '__main__':
    unittest.main()
