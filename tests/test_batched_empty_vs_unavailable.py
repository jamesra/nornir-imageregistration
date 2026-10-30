"""An empty batched result is an answer, not a signal that batching failed.

``_attempt_align_points_translation_batched`` used to end with
``return records if len(records) > 0 else None``, and returned ``None`` when its
alignability gate rejected every cell.  The caller reads ``None`` as "batched
unavailable" and re-measures the entire grid with ``AttemptAlignPoint``, a
different peak finder.

So whether the control points came from the batched or the serial measurement
depended on whether the batched result happened to come back empty.  With the
weights forced to zero on textured cells, the pre-fix caller discarded the empty
batched answer and returned four serial records with weight ~52 instead.

``None`` is now reserved for "the batched path could not run": ROI extraction
failed for too many cells, cell shapes disagree, or fewer than three cells
survive.  Those still fall back, because the serial pass reaches the same records.
"""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc

_CELL = np.array((64, 64), dtype=np.int64)
_IDENTITY_OFFSET = np.array((0.0, 0.0))


def _grid_points(count: int = 4) -> np.ndarray:
    """Points spaced so their cells sit well inside a 256x256 image."""
    return np.array([(64.0, 64.0), (96.0, 64.0), (64.0, 96.0), (96.0, 96.0)])[:count]


def _settings(image: np.ndarray):
    return nornir_imageregistration.settings.GridRefinement.CreateWithUnproccessedImages(
        target_image=image.copy(),
        source_image=image.copy(),
        cell_size=_CELL,
        single_thread_processing=True)


def _identity():
    return nornir_imageregistration.transforms.RigidTranslation(
        target_offset=_IDENTITY_OFFSET.copy())


def _textured() -> np.ndarray:
    return np.random.default_rng(3).random((256, 256)).astype(np.float32)


def _blank() -> np.ndarray:
    return np.full((256, 256), 0.5, dtype=np.float32)


def _low_contrast() -> np.ndarray:
    return (np.random.default_rng(3).random((256, 256)) * 1e-4).astype(np.float32)


def _run_batched(image: np.ndarray, count: int = 4):
    points = _grid_points(count)
    return ldc._attempt_align_points_translation_batched(
        keys=[(i, 0) for i in range(len(points))],
        source_points=points.copy(),
        target_points=points,
        rigid_transforms=[_identity()] * len(points),
        settings=_settings(image))


class _SerialSpyCase(unittest.TestCase):
    """Counts serial ``AttemptAlignPoint`` calls made through the caller."""

    def _refine_counting_serial_calls(self, image: np.ndarray, count: int = 4):
        points = _grid_points(count)
        calls: list[int] = []
        real = ldc.AttemptAlignPoint

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        ldc.AttemptAlignPoint = spy
        try:
            records = ldc._RefinePointsForTwoImages(
                _identity(),
                [(i, 0) for i in range(len(points))],
                points.copy(),
                points,
                _settings(image))
        finally:
            ldc.AttemptAlignPoint = real
        return records, len(calls)


class TestPreconditions(_SerialSpyCase):
    """Without these the caller-level tests below would be vacuous."""

    def test_batched_path_is_enabled(self):
        self.assertTrue(ldc._use_batched_vertex_measurement())

    def test_default_angles_are_translation_only(self):
        settings = _settings(_textured())
        self.assertTrue(ldc._angles_are_translation_only(settings.angles_to_search))

    def test_textured_grid_really_does_align(self):
        """The serial path must be capable of producing records on this fixture."""
        records = _run_batched(_textured())

        self.assertIsNotNone(records)
        assert records is not None
        self.assertEqual(len(records), 4)


class TestGateRejectsEveryCell(unittest.TestCase):
    """The alignability gate is a decision, so it returns an empty list."""

    def test_blank_image_returns_empty_list_not_none(self):
        records = _run_batched(_blank())

        self.assertIsNotNone(records, 'None would send the caller to a different peak finder')
        self.assertEqual(records, [])

    def test_low_contrast_image_returns_empty_list_not_none(self):
        records = _run_batched(_low_contrast())

        self.assertIsNotNone(records)
        self.assertEqual(records, [])


class TestAllPeaksUnusable(unittest.TestCase):
    """Measured, but every peak unusable, is also a decision."""

    def _measure_returning_zero_weights(self):
        real = ldc.measure_translation_cells_batched

        def zero_weights(fixed, moving, cell_shape, **kwargs):
            peaks, weights, ratios = real(fixed, moving, cell_shape, **kwargs)
            xp = nornir_imageregistration.cp.get_array_module(weights)
            return peaks, xp.zeros_like(weights), xp.zeros_like(ratios)

        return real, zero_weights

    def test_zero_weights_on_textured_cells_returns_empty_list(self):
        real, patched = self._measure_returning_zero_weights()
        ldc.measure_translation_cells_batched = patched
        try:
            records = _run_batched(_textured())
        finally:
            ldc.measure_translation_cells_batched = real

        self.assertIsNotNone(records)
        self.assertEqual(records, [])

    def test_nan_peaks_on_textured_cells_returns_empty_list(self):
        real = ldc.measure_translation_cells_batched

        def nan_peaks(fixed, moving, cell_shape, **kwargs):
            peaks, weights, ratios = real(fixed, moving, cell_shape, **kwargs)
            xp = nornir_imageregistration.cp.get_array_module(peaks)
            return xp.full_like(peaks, xp.nan), weights, ratios

        ldc.measure_translation_cells_batched = nan_peaks
        try:
            records = _run_batched(_textured())
        finally:
            ldc.measure_translation_cells_batched = real

        self.assertIsNotNone(records)
        self.assertEqual(records, [])


class TestCallerHonoursEmptyResult(_SerialSpyCase):
    """The caller must not re-measure a grid the batched path already answered."""

    def test_blank_grid_does_not_reach_the_serial_peak_finder(self):
        records, serial_calls = self._refine_counting_serial_calls(_blank())

        self.assertEqual(serial_calls, 0, 'the batched path already rejected every cell')
        self.assertEqual(len(records), 0)

    def test_low_contrast_grid_does_not_reach_the_serial_peak_finder(self):
        records, serial_calls = self._refine_counting_serial_calls(_low_contrast())

        self.assertEqual(serial_calls, 0)
        self.assertEqual(len(records), 0)

    def test_empty_batched_result_is_not_replaced_by_serial_records(self):
        """The divergence itself: serial can align these cells, batched said no.

        Pre-fix this returned four serial records with weight ~52 in place of the
        empty batched answer.
        """
        real = ldc.measure_translation_cells_batched

        def zero_weights(fixed, moving, cell_shape, **kwargs):
            peaks, weights, ratios = real(fixed, moving, cell_shape, **kwargs)
            xp = nornir_imageregistration.cp.get_array_module(weights)
            return peaks, xp.zeros_like(weights), xp.zeros_like(ratios)

        ldc.measure_translation_cells_batched = zero_weights
        try:
            records, serial_calls = self._refine_counting_serial_calls(_textured())
        finally:
            ldc.measure_translation_cells_batched = real

        self.assertEqual(serial_calls, 0)
        self.assertEqual(len(records), 0)


class TestUnavailableStillFallsBack(_SerialSpyCase):
    """``None`` must keep meaning "could not run", or batching becomes mandatory."""

    def test_too_few_cells_to_batch_falls_back_to_serial(self):
        records = _run_batched(_textured(), count=2)

        self.assertIsNone(records, 'fewer than three cells cannot be batched')

    def test_caller_measures_serially_when_batched_is_unavailable(self):
        points = _grid_points(2)
        _records, serial_calls = self._refine_counting_serial_calls(_textured(), count=2)

        self.assertEqual(serial_calls, len(points))

    def test_caller_measures_serially_when_batched_returns_none(self):
        real = ldc._attempt_align_points_translation_batched
        ldc._attempt_align_points_translation_batched = lambda **kwargs: None
        try:
            _records, serial_calls = self._refine_counting_serial_calls(_textured())
        finally:
            ldc._attempt_align_points_translation_batched = real

        self.assertEqual(serial_calls, 4)

    def test_caller_returns_batched_records_when_present(self):
        _records, serial_calls = self._refine_counting_serial_calls(_textured())

        self.assertEqual(serial_calls, 0, 'batched succeeded, so serial must not run')


if __name__ == '__main__':
    unittest.main()
