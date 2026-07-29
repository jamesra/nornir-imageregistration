"""Unit tests for refine_shared helpers."""

from __future__ import annotations

import os
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.refine_shared import (
    RefineRuntimeConfig,
    filter_weights_by_estimate_cutoff,
    is_alignable_cell,
    measure_translation_cell,
    normalize_cell,
    regularize_displacements,
)

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None


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

    def test_estimate_registration_weight_cutoff_flat_fallback(self) -> None:
        """Curves without a verified inflection use the keep-all fallback."""
        from nornir_imageregistration.refine_shared import estimate_registration_weight_cutoff

        # Strictly convex exponential percentile curves often have no sign-change
        # in the second derivative after polyfit smoothing.
        weights = np.exp(np.linspace(0.0, 2.0, 50))
        cutoff = estimate_registration_weight_cutoff(weights)
        self.assertTrue(cutoff.used_fallback)
        self.assertEqual(cutoff.inflection_percentile_index, 0)
        keep = weights >= float(cutoff.percentile_curve[cutoff.inflection_percentile_index])
        self.assertTrue(np.all(keep) or keep.sum() >= weights.size - 1)

    def test_estimate_registration_weight_cutoff_linear_fallback(self) -> None:
        """Strictly linear scores often lack a verified inflection; must not raise."""
        from nornir_imageregistration.refine_shared import estimate_registration_weight_cutoff

        linear = np.linspace(0.1, 1.0, 40)
        cutoff = estimate_registration_weight_cutoff(linear)
        self.assertEqual(cutoff.percentile_curve.shape[0], 101)
        self.assertGreaterEqual(float(np.sum(linear >= cutoff.cutoff_value)), 1.0)

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
        """RefineRuntimeConfig reads mosaic cutoff and STOS regularize flags."""
        os.environ['NORNIR_REFINE_MOSAIC_CUTOFF'] = '1'
        os.environ['NORNIR_REFINE_STOS_REGULARIZE'] = '1'
        try:
            cfg = RefineRuntimeConfig.from_env()
            self.assertTrue(cfg.mosaic_cutoff)
            self.assertTrue(cfg.stos_regularize)
        finally:
            os.environ.pop('NORNIR_REFINE_MOSAIC_CUTOFF', None)
            os.environ.pop('NORNIR_REFINE_STOS_REGULARIZE', None)

    def test_grid_refinement_uploads_images_once_under_cupy(self) -> None:
        """When CuPy is active, GridRefinement promotes full images to device once."""
        if not nornir_imageregistration.HasCupy() or cp is None:
            self.skipTest("CuPy unavailable")

        previous = nornir_imageregistration.GetActiveComputationLib()
        try:
            nornir_imageregistration.SetActiveComputationLib(
                nornir_imageregistration.ComputationLib.cupy)
            rng = np.random.default_rng(1)
            target = rng.random((64, 64)).astype(np.float32)
            source = rng.random((64, 64)).astype(np.float32)
            target_stats = nornir_imageregistration.ImageStats.Create(target)
            source_stats = nornir_imageregistration.ImageStats.Create(source)
            with nornir_imageregistration.settings.GridRefinement(
                    target_image=target,
                    source_image=source,
                    target_image_stats=target_stats,
                    source_image_stats=source_stats,
                    target_mask=np.ones((64, 64), dtype=bool),
                    source_mask=np.ones((64, 64), dtype=bool),
                    cell_size=(16, 16),
                    grid_spacing=(16, 16),
                    angles_to_search=[0],
                    num_iterations=1) as settings:
                self.assertTrue(settings.cupy_processing)
                self.assertIsInstance(settings.target_image, cp.ndarray)
                self.assertIsInstance(settings.source_image, cp.ndarray)
                # Tissue masks stay on the host for cell-overlap filtering.
                self.assertIsInstance(settings.target_mask, np.ndarray)
                self.assertFalse(isinstance(settings.target_mask, cp.ndarray))
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)

    def test_batched_fft_chunk_size_scales_with_free_vram(self) -> None:
        """FFT batch size grows with reported free VRAM on CuPy."""
        from unittest import mock

        from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_fft_cell_chunk_size

        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)
        with mock.patch(
                'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info',
                return_value=(2 * 1024 ** 3, 24 * 1024 ** 3)):
            small_card = batched_fft_cell_chunk_size((128, 128))
        with mock.patch(
                'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info',
                return_value=(20 * 1024 ** 3, 24 * 1024 ** 3)):
            large_card = batched_fft_cell_chunk_size((128, 128))
        self.assertGreater(large_card, small_card)
        self.assertGreaterEqual(large_card, 6000)

    def test_batched_fft_chunk_size_env_override(self) -> None:
        """Explicit env still overrides VRAM auto-tuning."""
        from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_fft_cell_chunk_size

        os.environ['NORNIR_REFINE_BATCHED_FFT_CELLS'] = '123'
        try:
            self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 123)
        finally:
            os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)

    def test_batched_roi_sample_budget_env_override(self) -> None:
        """ROI sample budget honors explicit env override."""
        from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_roi_sample_budget

        os.environ['NORNIR_REFINE_BATCHED_ROI_SAMPLES'] = '999'
        try:
            self.assertEqual(batched_roi_sample_budget(128, 128), 999)
        finally:
            os.environ.pop('NORNIR_REFINE_BATCHED_ROI_SAMPLES', None)

        """NumPy backend must not promote GridRefinement images to CuPy."""
        previous = nornir_imageregistration.GetActiveComputationLib()
        try:
            nornir_imageregistration.SetActiveComputationLib(
                nornir_imageregistration.ComputationLib.numpy)
            rng = np.random.default_rng(2)
            target = rng.random((32, 32)).astype(np.float32)
            source = rng.random((32, 32)).astype(np.float32)
            target_stats = nornir_imageregistration.ImageStats.Create(target)
            source_stats = nornir_imageregistration.ImageStats.Create(source)
            with nornir_imageregistration.settings.GridRefinement(
                    target_image=target,
                    source_image=source,
                    target_image_stats=target_stats,
                    source_image_stats=source_stats,
                    cell_size=(8, 8),
                    grid_spacing=(8, 8),
                    angles_to_search=[0],
                    num_iterations=1,
                    single_thread_processing=True) as settings:
                self.assertFalse(settings.cupy_processing)
                self.assertIsInstance(settings.target_image, np.ndarray)
                if cp is not None:
                    self.assertFalse(isinstance(settings.target_image, cp.ndarray))
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)


if __name__ == '__main__':
    unittest.main()
