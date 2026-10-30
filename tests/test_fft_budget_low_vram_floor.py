"""A low-VRAM card must not silently serialise the batched FFT (review #236).

`batched_fft_cell_chunk_size` subtracts a 512 MiB headroom reserve from 40% of free VRAM, so
below roughly 1.28 GiB free the budget went non-positive and the function returned a chunk of
**1** -- one CUDA launch per cell. Measured on 4096 cells of 128px, float32, median of 3, with
results byte-identical at every chunk size:

    chunk   1 (4096 launches) 9.4480s  142.50x
    chunk  32 ( 128 launches) 0.3133s    4.73x
    chunk  64 (  64 launches) 0.1585s    2.39x
    chunk 128 (  32 launches) 0.0790s    1.19x
    chunk 256 (  16 launches) 0.0663s    1.00x

Two things follow, and both are asserted below. The cliff costs 142x, and the floor's size
matters -- a floor of 32 or 64 would still concede 4.7x or 2.4x. The floor is therefore a
byte target (half the preferred working set), not a cell count, for the reason #228 records:
a cell count does not transfer across cell sizes.

Returning 1 remains correct for exactly one case -- a single cell genuinely does not fit --
which is a different question from the reserve being exhausted.
"""

from __future__ import annotations

import logging
import os
import unittest
from unittest import mock

from nornir_imageregistration.refine_shared.gpu_batch_budget import (
    _FFT_MIN_WORKSPACE_BYTES,
    _FFT_PREFERRED_WORKSPACE_BYTES,
    _MAX_FFT_CELL_CHUNK,
    _REFINE_BATCH_HEADROOM_BYTES,
    _REFINE_BATCH_VRAM_FRACTION,
    _fft_peak_bytes_per_cell,
    batched_fft_cell_chunk_size,
)

_MEM = 'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info'
_MODULE = 'nornir_imageregistration.refine_shared.gpu_batch_budget'

_TOTAL = 24 * 1024 ** 3

# Free bytes at which 40% of free exactly equals the headroom reserve. Below this the old
# code returned 1.
_CLIFF_BYTES = int(_REFINE_BATCH_HEADROOM_BYTES / _REFINE_BATCH_VRAM_FRACTION)

# Measured optima from #228: cell size -> best chunk.
_MEASURED_OPTIMA = {64: 1024, 128: 256, 256: 64}


def _reset_throttle() -> None:
    """The warning is throttled to once a minute, which would hide it from later tests."""
    import nornir_imageregistration.refine_shared.gpu_batch_budget as mod
    mod._last_low_vram_warning = 0.0


class LowVramFloorTestBase(unittest.TestCase):

    def setUp(self):
        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)
        _reset_throttle()


class TestTheCliffIsGone(LowVramFloorTestBase):

    def test_below_the_cliff_the_chunk_is_no_longer_one(self):
        # 1 GiB free: 40% is 410 MiB, under the 512 MiB reserve, so the budget is negative.
        with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
            chunk = batched_fft_cell_chunk_size((128, 128))
        self.assertGreater(chunk, 1,
                           'an exhausted headroom reserve must not serialise the batch')
        self.assertEqual(chunk, _FFT_MIN_WORKSPACE_BYTES // _fft_peak_bytes_per_cell(128, 128))

    def test_the_floor_is_within_a_small_factor_of_the_measured_optimum(self):
        """Half the working set was measured at 1.19x the optimum; 32 or 64 would be 4.7x/2.4x."""
        with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
            chunk = batched_fft_cell_chunk_size((128, 128))
        optimum = _MEASURED_OPTIMA[128]
        self.assertGreaterEqual(chunk, optimum // 2,
                                f'floor {chunk} concedes more than 2x against the measured '
                                f'optimum {optimum}')

    def test_no_free_vram_level_causes_a_collapse(self):
        """Sweeping free memory down, the chunk must never fall off a cliff.

        This is the defect stated as a property: the old code stepped 256 -> 1 crossing
        1.28 GiB, a 256x drop from one megabyte of free memory to the next.
        """
        levels = [8 * 1024 ** 3, 4 * 1024 ** 3, 2 * 1024 ** 3,
                  _CLIFF_BYTES + 1024 ** 2, _CLIFF_BYTES - 1024 ** 2,
                  1024 ** 3, 768 * 1024 ** 2, 512 * 1024 ** 2, 384 * 1024 ** 2]
        chunks = []
        for free in levels:
            _reset_throttle()
            with mock.patch(_MEM, return_value=(free, _TOTAL)):
                chunks.append(batched_fft_cell_chunk_size((128, 128)))

        for (free_a, chunk_a), (free_b, chunk_b) in zip(
                zip(levels, chunks), zip(levels[1:], chunks[1:])):
            with self.subTest(free_mib=free_b // 1024 ** 2):
                self.assertGreater(chunk_b, 0)
                self.assertLessEqual(
                    chunk_a, chunk_b * 4,
                    f'chunk fell from {chunk_a} to {chunk_b} between '
                    f'{free_a // 1024 ** 2} and {free_b // 1024 ** 2} MiB free')

    def test_crossing_the_cliff_boundary_is_smooth(self):
        with mock.patch(_MEM, return_value=(_CLIFF_BYTES + 32 * 1024 ** 2, _TOTAL)):
            above = batched_fft_cell_chunk_size((128, 128))
        _reset_throttle()
        with mock.patch(_MEM, return_value=(_CLIFF_BYTES - 32 * 1024 ** 2, _TOTAL)):
            below = batched_fft_cell_chunk_size((128, 128))
        self.assertGreater(below, 1)
        self.assertLessEqual(above, below * 4,
                             f'{above} above the boundary vs {below} below it')


class TestOneCellIsStillCorrectWhenItIsTrue(LowVramFloorTestBase):

    def test_a_card_with_nothing_free_returns_one(self):
        with mock.patch(_MEM, return_value=(0, _TOTAL)):
            self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 1)

    def test_a_cell_larger_than_the_whole_fallback_returns_one(self):
        # 4096px cells model 1 GiB each, so 40% of 256 MiB free cannot hold one.
        with mock.patch(_MEM, return_value=(256 * 1024 ** 2, _TOTAL)):
            self.assertEqual(batched_fft_cell_chunk_size((4096, 4096)), 1)


class TestTheFallbackStaysWithinItsMeans(LowVramFloorTestBase):
    """The reserve is gone in this path, but the 40%-of-free bound is not."""

    def test_it_never_claims_more_than_the_vram_fraction(self):
        for free_mib in (64, 128, 256, 512, 768, 1024, 1200):
            with self.subTest(free_mib=free_mib):
                _reset_throttle()
                free = free_mib * 1024 ** 2
                with mock.patch(_MEM, return_value=(free, _TOTAL)):
                    chunk = batched_fft_cell_chunk_size((128, 128))
                claimed = chunk * _fft_peak_bytes_per_cell(128, 128)
                allowed = max(int(free * _REFINE_BATCH_VRAM_FRACTION),
                              _fft_peak_bytes_per_cell(128, 128))
                self.assertLessEqual(
                    claimed, allowed,
                    f'{claimed / 1024 ** 2:.0f} MiB claimed from {free_mib} MiB free, '
                    f'above the {_REFINE_BATCH_VRAM_FRACTION:.0%} bound')

    def test_the_floor_never_exceeds_the_preferred_target(self):
        self.assertLessEqual(_FFT_MIN_WORKSPACE_BYTES, _FFT_PREFERRED_WORKSPACE_BYTES)

    def test_the_floor_respects_the_hard_cap(self):
        for cell_px in (8, 16, 32):
            with self.subTest(cell_px=cell_px):
                _reset_throttle()
                with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                    self.assertLessEqual(
                        batched_fft_cell_chunk_size((cell_px, cell_px)),
                        _MAX_FFT_CELL_CHUNK)


class TestTheFloorIsAByteTargetNotACellCount(LowVramFloorTestBase):
    """#228's lesson: a cell count does not transfer across cell sizes."""

    def test_the_fallback_chunk_scales_inversely_with_cell_area(self):
        chunks = {}
        for cell_px in (64, 128, 256):
            _reset_throttle()
            with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                chunks[cell_px] = batched_fft_cell_chunk_size((cell_px, cell_px))

        self.assertGreater(chunks[64], chunks[128])
        self.assertGreater(chunks[128], chunks[256])

    def test_the_fallback_lands_at_half_the_measured_optimum_at_every_cell_size(self):
        for cell_px, optimum in _MEASURED_OPTIMA.items():
            with self.subTest(cell_px=cell_px):
                _reset_throttle()
                with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                    chunk = batched_fft_cell_chunk_size((cell_px, cell_px))
                self.assertEqual(chunk, optimum // 2)


class TestItSaysSoOutLoud(LowVramFloorTestBase):
    """The old behaviour was correct-but-142x-slower with no warning, which is the worst kind."""

    def test_the_exhausted_budget_is_warned_about(self):
        with self.assertLogs(_MODULE, level=logging.WARNING) as captured:
            with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                batched_fft_cell_chunk_size((128, 128))
        joined = '\n'.join(captured.output)
        self.assertIn('Free VRAM', joined)
        self.assertIn('NORNIR_REFINE_BATCHED_FFT_CELLS', joined,
                      'the warning should name the override that works around it')

    def test_the_warning_reports_the_actual_free_memory(self):
        with self.assertLogs(_MODULE, level=logging.WARNING) as captured:
            with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                batched_fft_cell_chunk_size((128, 128))
        self.assertIn('1024 MiB', '\n'.join(captured.output))

    def test_the_warning_is_throttled(self):
        """One call per refine step, so an unthrottled warning is thousands of lines."""
        with self.assertLogs(_MODULE, level=logging.WARNING) as captured:
            with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                for _ in range(50):
                    batched_fft_cell_chunk_size((128, 128))
        self.assertEqual(len(captured.output), 1,
                         f'{len(captured.output)} warnings from 50 calls')

    def test_a_healthy_card_is_silent(self):
        logger = logging.getLogger(_MODULE)
        with mock.patch.object(logger, 'warning') as warn:
            with mock.patch(_MEM, return_value=(20 * 1024 ** 3, _TOTAL)):
                batched_fft_cell_chunk_size((128, 128))
        warn.assert_not_called()


class TestTheOverrideStillWins(LowVramFloorTestBase):

    def test_the_env_override_beats_the_floor(self):
        os.environ['NORNIR_REFINE_BATCHED_FFT_CELLS'] = '7'
        try:
            with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 7)
        finally:
            os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)


if __name__ == '__main__':
    unittest.main()
