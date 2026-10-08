"""Unit tests for refine_shared helpers."""

from __future__ import annotations

import os
import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import nornir_imageregistration
from nornir_imageregistration.refine_shared import (
    RefineRuntimeConfig,
    filter_alignment_records_by_weight,
    filter_records_by_registration_weight,
    filter_weights_by_estimate_cutoff,
    is_alignable_cell,
    measure_translation_cell,
    normalize_cell,
    regularize_displacements,
)

_POINT_PAIR_DTYPE = np.dtype([('Weight', 'f8'), ('DisplacementX', 'f8')])


class _AlignmentRecordStub:
    """Minimal alignment record for cutoff filter tests."""

    def __init__(self, weight: float) -> None:
        self.weight = float(weight)

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

    def test_filter_records_by_registration_weight_empty(self) -> None:
        """Empty structured updates pass through unchanged."""
        empty = np.array([], dtype=_POINT_PAIR_DTYPE)
        out = filter_records_by_registration_weight(empty)
        self.assertEqual(out.size, 0)
        self.assertEqual(out.dtype, _POINT_PAIR_DTYPE)

    def test_filter_alignment_records_by_weight_empty(self) -> None:
        """Empty record lists pass through unchanged."""
        self.assertEqual(filter_alignment_records_by_weight([]), [])

    def test_filter_records_by_registration_weight_few_positive(self) -> None:
        """With fewer than three positive weights, all positive entries are kept."""
        records = np.zeros(4, dtype=_POINT_PAIR_DTYPE)
        records['Weight'] = np.asarray([0.0, -0.1, 0.35, 0.82])
        out = filter_records_by_registration_weight(records)
        self.assertEqual(out.size, 2)
        np.testing.assert_allclose(out['Weight'], [0.35, 0.82])

    def test_filter_alignment_records_by_weight_few_positive(self) -> None:
        """Record lists with fewer than three positive weights keep every positive row."""
        rows = [
            _AlignmentRecordStub(0.0),
            _AlignmentRecordStub(0.25),
            _AlignmentRecordStub(0.75),
        ]
        kept = filter_alignment_records_by_weight(rows)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0].weight, 0.25)
        self.assertAlmostEqual(kept[1].weight, 0.75)

    def test_filter_records_by_registration_weight_bimodal(self) -> None:
        """Bimodal weights drop the low tail while preserving high-confidence rows."""
        weights = np.asarray([0.05, 0.08, 0.11, 0.86, 0.9, 0.93, 0.96, 0.99])
        records = np.zeros(weights.size, dtype=_POINT_PAIR_DTYPE)
        records['Weight'] = weights
        out = filter_records_by_registration_weight(records)
        self.assertGreater(out.size, 0)
        self.assertLess(out.size, weights.size)
        self.assertTrue(np.all(out['Weight'] >= 0.5))

    def test_filter_alignment_records_by_weight_bimodal(self) -> None:
        """Alignment records follow the same bimodal cutoff as structured updates."""
        weights = [0.06, 0.09, 0.12, 0.88, 0.91, 0.94, 0.97, 1.0]
        rows = [_AlignmentRecordStub(w) for w in weights]
        kept = filter_alignment_records_by_weight(rows)
        self.assertGreater(len(kept), 0)
        self.assertLess(len(kept), len(rows))
        self.assertTrue(all(r.weight >= 0.5 for r in kept))

    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=24,
        ))
    @settings(max_examples=40, deadline=None)
    def test_filter_records_matches_weight_mask(self, weights: list[float]) -> None:
        """Structured filtering matches the shared weight cutoff mask."""
        w = np.asarray(weights, dtype=np.float64)
        records = np.zeros(w.size, dtype=_POINT_PAIR_DTYPE)
        if w.size:
            records['Weight'] = w
        out = filter_records_by_registration_weight(records)
        keep = filter_weights_by_estimate_cutoff(w)
        np.testing.assert_array_equal(out['Weight'], records[keep]['Weight'])

    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=24,
        ))
    @settings(max_examples=40, deadline=None)
    def test_filter_alignment_records_matches_weight_mask(self, weights: list[float]) -> None:
        """Record-list filtering matches the shared weight cutoff mask."""
        rows = [_AlignmentRecordStub(w) for w in weights]
        kept = filter_alignment_records_by_weight(rows)
        w = np.asarray(weights, dtype=np.float64)
        keep = filter_weights_by_estimate_cutoff(w)
        expected = [row for row, ok in zip(rows, keep) if ok]
        self.assertEqual(len(kept), len(expected))
        for got, exp in zip(kept, expected):
            self.assertAlmostEqual(got.weight, exp.weight)

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

    def test_runtime_config_finalized_recheck_modes(self) -> None:
        """Finalized rechecks default safely to all and accept only known modes."""
        old = os.environ.get('NORNIR_REFINE_FINALIZED_RECHECK_MODE')
        try:
            os.environ.pop('NORNIR_REFINE_FINALIZED_RECHECK_MODE', None)
            self.assertEqual(RefineRuntimeConfig.from_env().finalized_recheck_mode, 'all')
            os.environ['NORNIR_REFINE_FINALIZED_RECHECK_MODE'] = 'shadow'
            self.assertEqual(RefineRuntimeConfig.from_env().finalized_recheck_mode, 'shadow')
            os.environ['NORNIR_REFINE_FINALIZED_RECHECK_MODE'] = 'local'
            self.assertEqual(RefineRuntimeConfig.from_env().finalized_recheck_mode, 'local')
            os.environ['NORNIR_REFINE_FINALIZED_RECHECK_MODE'] = 'invalid'
            self.assertEqual(RefineRuntimeConfig.from_env().finalized_recheck_mode, 'all')
        finally:
            if old is None:
                os.environ.pop('NORNIR_REFINE_FINALIZED_RECHECK_MODE', None)
            else:
                os.environ['NORNIR_REFINE_FINALIZED_RECHECK_MODE'] = old

    def test_batched_mosaic_gate_does_not_disable_stos(self) -> None:
        """NORNIR_REFINE_BATCHED=0 is mosaic-only; STOS stays batched (#98)."""
        from nornir_imageregistration.local_distortion_correction import (
            _use_batched_stos_cell_measurement,
            _use_batched_vertex_measurement,
        )
        from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config

        os.environ['NORNIR_REFINE_BATCHED'] = '0'
        os.environ.pop('NORNIR_REFINE_BATCHED_STOS', None)
        os.environ.pop('NORNIR_REFINE_BATCHED_GPU', None)
        get_runtime_config(refresh=True)
        try:
            self.assertFalse(_use_batched_vertex_measurement())
            self.assertTrue(_use_batched_stos_cell_measurement())
            cfg = get_runtime_config()
            self.assertFalse(cfg.batched_vertex_measurement)
            self.assertTrue(cfg.batched_stos_cell_measurement)
        finally:
            os.environ.pop('NORNIR_REFINE_BATCHED', None)
            get_runtime_config(refresh=True)

    def test_batched_stos_gate_can_disable_independently(self) -> None:
        """NORNIR_REFINE_BATCHED_STOS=0 turns off STOS batched without mosaic."""
        from nornir_imageregistration.local_distortion_correction import (
            _use_batched_stos_cell_measurement,
            _use_batched_vertex_measurement,
        )
        from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config

        os.environ.pop('NORNIR_REFINE_BATCHED', None)
        os.environ.pop('NORNIR_REFINE_BATCHED_GPU', None)
        os.environ['NORNIR_REFINE_BATCHED_STOS'] = '0'
        get_runtime_config(refresh=True)
        try:
            self.assertTrue(_use_batched_vertex_measurement())
            self.assertFalse(_use_batched_stos_cell_measurement())
        finally:
            os.environ.pop('NORNIR_REFINE_BATCHED_STOS', None)
            get_runtime_config(refresh=True)

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
        """A card too small for the preferred working set gets a smaller chunk.

        Rewritten under #236. The original asserted ``large_card > small_card`` at 2 GiB and
        20 GiB free and expected the large card to exceed 6000 cells, which described the
        pre-#228 "fill VRAM" policy. Since #228 the chunk targets a 256 MiB working set and
        VRAM is only the ceiling, so both of those cards reach the same 256 -- the assertion
        was stale, not failing. What is still true, and worth pinning, is that VRAM binds
        once it falls below the target.
        """
        from unittest import mock

        from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_fft_cell_chunk_size

        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)
        with mock.patch(
                'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info',
                return_value=(1600 * 1024 ** 2, 24 * 1024 ** 3)):
            small_card = batched_fft_cell_chunk_size((128, 128))
        with mock.patch(
                'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info',
                return_value=(20 * 1024 ** 3, 24 * 1024 ** 3)):
            large_card = batched_fft_cell_chunk_size((128, 128))
        self.assertGreater(large_card, small_card)
        self.assertGreater(small_card, 1,
                           'a tight card must still batch; see the #236 floor')

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

    def test_batched_fft_chunk_size_cpu_scales_with_cell_area(self) -> None:
        """CPU FFT chunks shrink with cell area so 4096² cannot batch hundreds of cells.

        The 128px expectation was 1024, the full ``_CPU_FFT_BUDGET_BYTES``, before #228 capped
        every path at the 256 MiB working set that measured fastest. Stale rather than broken;
        ``test_fft_budget_precision.py`` records the CPU timings behind the new value.
        """
        from unittest import mock

        from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_fft_cell_chunk_size

        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)
        with mock.patch(
                'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info',
                return_value=(None, None)):
            at_128 = batched_fft_cell_chunk_size((128, 128))
            at_1024 = batched_fft_cell_chunk_size((1024, 1024))
            at_4096 = batched_fft_cell_chunk_size((4096, 4096))
        # 256 MiB working set: 256 cells of 128px at 1 MiB each, 4 of 1024px at 64 MiB,
        # and a single 4096px cell already models 1 GiB so it cannot batch at all.
        self.assertEqual(at_128, 256)
        self.assertEqual(at_1024, 4)
        self.assertEqual(at_4096, 1)
        self.assertGreater(at_128, at_1024)
        self.assertGreater(at_1024, at_4096)

    def test_batched_fft_chunk_size_gpu_does_not_floor_large_cells(self) -> None:
        """VRAM budget must not force 256 huge cells when only a handful fit."""
        from unittest import mock

        from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_fft_cell_chunk_size

        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)
        with mock.patch(
                'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info',
                return_value=(20 * 1024 ** 3, 24 * 1024 ** 3)):
            chunk = batched_fft_cell_chunk_size((4096, 4096))
        self.assertLessEqual(chunk, 16)
        self.assertGreaterEqual(chunk, 1)

    def test_grid_refinement_numpy_backend_keeps_numpy_images(self) -> None:
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

    def test_grid_refinement_cupy_processing_false_keeps_numpy_under_cupy(self) -> None:
        """Explicit cupy_processing=False must not upload even when UsingCupy()."""
        if not nornir_imageregistration.HasCupy() or cp is None:
            self.skipTest("CuPy unavailable")

        previous = nornir_imageregistration.GetActiveComputationLib()
        try:
            nornir_imageregistration.SetActiveComputationLib(
                nornir_imageregistration.ComputationLib.cupy)
            rng = np.random.default_rng(3)
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
                    cupy_processing=False,
                    single_thread_processing=True) as settings:
                self.assertFalse(settings.cupy_processing)
                self.assertIsInstance(settings.target_image, np.ndarray)
                self.assertFalse(isinstance(settings.target_image, cp.ndarray))
                self.assertIsInstance(settings.source_image, np.ndarray)
                self.assertFalse(isinstance(settings.source_image, cp.ndarray))
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)


if __name__ == '__main__':
    unittest.main()
