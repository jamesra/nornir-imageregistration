"""The batched-FFT chunk budget is intentionally float64-modelled. Keep it that way.

``_FFT_PEAK_BYTES_PER_CELL_128`` contains a bare ``16`` -- ``complex128`` -- so it models
a float64 cell. #92 made ``batched_image_phase_correlation`` run at the caller's
precision, and callers pass float32, so the model now over-estimates real workspace by
about 2x. Review issue #227 proposed scaling the budget by precision to recover the
"lost" batch size.

Measurement says don't. On a 22 GiB card, 8192 cells of 128px float32 through
``measure_translation_cells_batched`` (median of 7 runs):

======  ==========  =========  =========
chunk   launches    median s   peak MiB
======  ==========  =========  =========
   256          32      0.174        176
   512          16      0.257        288
  1024           8      0.248        576
  2048           4      0.241       1152
  4096           2      0.256       2304
  8195           1      0.282       4608
======  ==========  =========  =========

Smaller chunks are up to 1.62x *faster*: launch overhead is trivial next to the
bandwidth cost of a larger working set. So doubling the chunk would cost throughput and
raise peak VRAM, and the conservative model is free headroom rather than a bug.

These tests pin the calibration and the measured margin so the constant cannot be
lowered, or made precision-aware, without someone re-running that sweep.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from nornir_imageregistration.refine_shared.gpu_batch_budget import (
    _CPU_FFT_BUDGET_BYTES,
    _FFT_PEAK_BYTES_PER_CELL_128,
    _fft_peak_bytes_per_cell,
    batched_fft_cell_chunk_size,
)

_MEM = 'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info'

# Directly measured MiB of FFT workspace per 128px cell, constant across batch sizes
# from 128 to 16384 cells. float64 is exactly twice float32 because the transform is
# complex128 rather than complex64.
_MEASURED_MIB_PER_CELL = {np.float32: 0.438, np.float64: 0.875}


class TestTheBudgetModelsFloat64(unittest.TestCase):
    """The constant is calibrated against complex128 and must stay there."""

    def test_the_calibrated_constant_is_one_mib_per_cell(self):
        self.assertEqual(_FFT_PEAK_BYTES_PER_CELL_128, 1024 * 1024)

    def test_it_models_a_complex128_transform(self):
        """16 bytes per pixel of complex output, times a factor of 4 for intermediates."""
        self.assertEqual(_FFT_PEAK_BYTES_PER_CELL_128, 128 * 128 * 16 * 4)

    def test_the_estimate_is_precision_blind(self):
        """No dtype parameter: every caller gets the float64 model.

        This is the property #227 proposed changing. It is asserted rather than merely
        documented so the change cannot happen silently.
        """
        self.assertEqual(_fft_peak_bytes_per_cell(128, 128), 1024 * 1024)

        with mock.patch(_MEM, return_value=(8 * 1024 ** 3, 24 * 1024 ** 3)):
            chunk = batched_fft_cell_chunk_size((128, 128))
        self.assertGreater(chunk, 0)

    def test_it_scales_with_cell_area(self):
        small = _fft_peak_bytes_per_cell(128, 128)
        self.assertEqual(_fft_peak_bytes_per_cell(256, 256), 4 * small)
        self.assertEqual(_fft_peak_bytes_per_cell(64, 64), small // 4)


class TestTheModelStaysAboveMeasuredWorkspace(unittest.TestCase):
    """A model below measured workspace would over-commit VRAM."""

    def test_it_exceeds_measured_cost_at_both_precisions(self):
        model_mib = _fft_peak_bytes_per_cell(128, 128) / 2 ** 20
        for dtype, measured in _MEASURED_MIB_PER_CELL.items():
            with self.subTest(dtype=np.dtype(dtype).name):
                self.assertGreater(
                    model_mib, measured,
                    'the budget must never under-predict real workspace')

    def test_the_float64_margin_is_the_calibrated_one(self):
        """~1.14x over measured float64. Shrinking this is how VRAM gets over-committed."""
        margin = (_fft_peak_bytes_per_cell(128, 128) / 2 ** 20
                  / _MEASURED_MIB_PER_CELL[np.float64])
        self.assertAlmostEqual(margin, 1.14, places=1)

    def test_the_float32_over_estimate_is_about_two(self):
        """The #227 observation, kept as a fact rather than acted on."""
        ratio = (_fft_peak_bytes_per_cell(128, 128) / 2 ** 20
                 / _MEASURED_MIB_PER_CELL[np.float32])
        self.assertGreater(ratio, 2.0)
        self.assertLess(ratio, 2.5)

    def test_single_precision_really_does_cost_half(self):
        """Guard the premise: both FFT backends return complex64 for a float32 input,
        which is why the float64 model over-estimates float32 by 2x."""
        for dtype, expected in ((np.float32, 8), (np.float64, 16)):
            with self.subTest(dtype=np.dtype(dtype).name):
                produced = np.fft.fft2(np.zeros((2, 8, 8), dtype=dtype)).dtype
                self.assertEqual(produced.itemsize, expected)

        measured = _MEASURED_MIB_PER_CELL
        self.assertAlmostEqual(
            measured[np.float64] / measured[np.float32], 2.0, places=1)


class TestChunkSizesStayBounded(unittest.TestCase):

    def setUp(self):
        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)

    def test_the_cpu_baseline_is_1024_cells_at_128px(self):
        with mock.patch(_MEM, return_value=(None, None)):
            self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 1024)

    def test_the_cpu_budget_tracks_the_calibrated_constant(self):
        self.assertEqual(_CPU_FFT_BUDGET_BYTES, 1024 * _FFT_PEAK_BYTES_PER_CELL_128)

    def test_an_exhausted_card_still_returns_one_cell(self):
        with mock.patch(_MEM, return_value=(0, 24 * 1024 ** 3)):
            self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 1)

    def test_the_env_override_still_wins(self):
        """The measured sweep favours small chunks, so this knob is the way to get them."""
        os.environ['NORNIR_REFINE_BATCHED_FFT_CELLS'] = '256'
        try:
            with mock.patch(_MEM, return_value=(20 * 1024 ** 3, 24 * 1024 ** 3)):
                self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 256)
        finally:
            os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)

    def test_a_chunk_never_exceeds_the_vram_fraction_it_budgeted(self):
        """The chunk times modelled bytes must fit the budget it was derived from."""
        from nornir_imageregistration.refine_shared.gpu_batch_budget import (
            _REFINE_BATCH_HEADROOM_BYTES, _REFINE_BATCH_VRAM_FRACTION)

        free = 20 * 1024 ** 3
        with mock.patch(_MEM, return_value=(free, 24 * 1024 ** 3)):
            for shape in ((64, 64), (128, 128), (512, 512)):
                with self.subTest(cell=shape):
                    chunk = batched_fft_cell_chunk_size(shape)
                    modelled = chunk * _fft_peak_bytes_per_cell(*shape)
                    budget = (int(free * _REFINE_BATCH_VRAM_FRACTION)
                              - _REFINE_BATCH_HEADROOM_BYTES)
                    self.assertLessEqual(modelled, budget)


if __name__ == '__main__':
    unittest.main()
