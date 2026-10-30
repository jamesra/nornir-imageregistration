"""The batched-FFT chunk target is a working-set size, not a cell count.

Review #228 measured a spread favouring smaller chunks and proposed a cell-count ceiling.
Sweeping chunk size at three cell sizes shows why a cell count is the wrong knob. On an
RTX 4500 Ada, 8192 cells (2048 at 256px), float32, median of 7, with the results verified
identical at every chunk size:

| cell size | best chunk | best time | worst time | spread | cell input at best | peak at best |
|---|---|---|---|---|---|---|
| 64px | 1024 | 0.0353 s | 0.3001 s | 8.50x | 16 MiB | 145 MiB |
| 128px | 256 | 0.1250 s | 0.2400 s | 2.32x | 16 MiB | 145 MiB |
| 256px | 64 | 0.1326 s | 0.2614 s | 1.97x | 16 MiB | 145 MiB |

The best *cell count* moves by 16x, while the best *working set* is identical every time.
So the optimum is a cache/bandwidth effect, and a byte target is the form with any chance
of transferring to other hardware. Expressed in the units `_fft_peak_bytes_per_cell`
already models, all three optima are 256 MiB, which is how the constant is written.

Two things this corrects in #228:

  - "monotonically favouring smaller chunks" was an artifact of sweeping only down to 256
    at 128px, which is the optimum. There is an interior optimum, and undershooting hurts
    much more than overshooting: at 64px, chunk 64 is **8.5x** slower than chunk 1024.
  - the `NORNIR_REFINE_BATCHED_FFT_CELLS=256` workaround it suggested is right only at
    128px. It is 4x too large at 256px cells and 4x too small at 64px.

The CPU path is nearly flat by comparison (1.05-1.09x across the same sweep, favouring
small), so it is compute-bound and the target simply caps it.

The VRAM budget is deliberately retained as the upper bound rather than replaced, so the
target can only ever *lower* the chunk. That matters because the constant is calibrated on
one GPU: a wrong target costs some throughput, whereas overriding the VRAM ceiling could
cost an allocation failure.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from nornir_imageregistration.refine_shared.gpu_batch_budget import (
    _FFT_PREFERRED_WORKSPACE_BYTES,
    _MAX_FFT_CELL_CHUNK,
    _fft_peak_bytes_per_cell,
    batched_fft_cell_chunk_size,
)

_MEM = 'nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info'

# Measured optima: cell size -> best chunk.
_MEASURED_OPTIMA = {64: 1024, 128: 256, 256: 64}

# A card with plenty free, so the preferred target is what binds rather than VRAM.
_ROOMY = (20 * 1024 ** 3, 24 * 1024 ** 3)


class TestTheTargetIsScaleInvariant(unittest.TestCase):
    """One byte target must reproduce every measured optimum, across a 16x cell spread."""

    def setUp(self):
        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)

    def test_it_reproduces_the_measured_optimum_at_each_cell_size(self):
        with mock.patch(_MEM, return_value=_ROOMY):
            for cell_px, expected in _MEASURED_OPTIMA.items():
                with self.subTest(cell_px=cell_px):
                    self.assertEqual(batched_fft_cell_chunk_size((cell_px, cell_px)),
                                     expected,
                                     f'{cell_px}px cells measured fastest at {expected} '
                                     'cells per launch')

    def test_the_optima_are_one_constant_in_modelled_bytes(self):
        # The point of the whole exercise: a single number explains all three rows.
        for cell_px, chunk in _MEASURED_OPTIMA.items():
            with self.subTest(cell_px=cell_px):
                self.assertEqual(chunk * _fft_peak_bytes_per_cell(cell_px, cell_px),
                                 _FFT_PREFERRED_WORKSPACE_BYTES)

    def test_a_cell_count_ceiling_could_not_do_this(self):
        # Guard the reasoning, not just the result: any fixed cell count is wrong for two
        # of the three measured sizes, which is why #228's suggested 256 is not the fix.
        for candidate in _MEASURED_OPTIMA.values():
            wrong = [px for px, best in _MEASURED_OPTIMA.items() if best != candidate]
            self.assertEqual(len(wrong), 2,
                             f'a ceiling of {candidate} cells mismatches {wrong}')


class TestTheVramCeilingStillWins(unittest.TestCase):
    """The target lowers the chunk; it must never raise it past what fits."""

    def setUp(self):
        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)

    def test_a_tight_card_is_still_the_binding_constraint(self):
        # 1 GiB free: 40% minus 512 MiB headroom leaves well under the preferred target.
        with mock.patch(_MEM, return_value=(1024 ** 3, 24 * 1024 ** 3)):
            tight = batched_fft_cell_chunk_size((128, 128))
        with mock.patch(_MEM, return_value=_ROOMY):
            roomy = batched_fft_cell_chunk_size((128, 128))
        self.assertLess(tight, roomy,
                        'a nearly full card must still get a smaller chunk than a roomy '
                        'one; the target is a ceiling, not a floor')

    def test_an_exhausted_card_still_returns_one_cell(self):
        with mock.patch(_MEM, return_value=(0, 24 * 1024 ** 3)):
            self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 1)

    def test_the_target_never_exceeds_the_hard_cap(self):
        with mock.patch(_MEM, return_value=_ROOMY):
            for cell_px in (8, 16, 32, 64, 128, 256, 512):
                with self.subTest(cell_px=cell_px):
                    self.assertLessEqual(
                        batched_fft_cell_chunk_size((cell_px, cell_px)),
                        _MAX_FFT_CELL_CHUNK)

    def test_the_chunk_stays_within_the_target(self):
        with mock.patch(_MEM, return_value=_ROOMY):
            for cell_px in (64, 128, 256, 512):
                with self.subTest(cell_px=cell_px):
                    chunk = batched_fft_cell_chunk_size((cell_px, cell_px))
                    modelled = chunk * _fft_peak_bytes_per_cell(cell_px, cell_px)
                    self.assertLessEqual(modelled, _FFT_PREFERRED_WORKSPACE_BYTES)

    def test_the_env_override_still_beats_the_target(self):
        os.environ['NORNIR_REFINE_BATCHED_FFT_CELLS'] = '4096'
        try:
            with mock.patch(_MEM, return_value=_ROOMY):
                self.assertEqual(batched_fft_cell_chunk_size((128, 128)), 4096,
                                 'the override is how another card gets re-tuned without '
                                 'a code change, so it must win over the target')
        finally:
            os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)


class TestTheOldPolicyTookTheSlowRow(unittest.TestCase):
    """Show what changed, so the regression is visible if the target is removed."""

    def setUp(self):
        os.environ.pop('NORNIR_REFINE_BATCHED_FFT_CELLS', None)

    def test_the_target_now_binds_before_vram_on_a_roomy_card(self):
        with mock.patch(_MEM, return_value=_ROOMY):
            chunk = batched_fft_cell_chunk_size((128, 128))
        vram_bytes = int(_ROOMY[0] * 0.40) - 512 * 1024 ** 2
        unbounded = vram_bytes // _fft_peak_bytes_per_cell(128, 128)
        self.assertLess(chunk, min(unbounded, _MAX_FFT_CELL_CHUNK),
                        'on a roomy card the old policy filled to the VRAM limit, which '
                        'measured 1.9x slower at 128px; the target must bind first')

    def test_the_old_policy_landed_in_the_slow_region(self):
        # On a roomy card VRAM, not the hard cap, was what bound the old chunk: 7680 cells
        # of 128px here, matching the ~8195 the original #227 sweep reported. That row
        # measured 0.2400s against the 0.1250s the 256 this now returns.
        vram_bytes = int(_ROOMY[0] * 0.40) - 512 * 1024 ** 2
        old_chunk = min(vram_bytes // _fft_peak_bytes_per_cell(128, 128),
                        _MAX_FFT_CELL_CHUNK)
        self.assertGreater(old_chunk, 4096,
                           'the old default sat in the measured slow region; if it drops '
                           'below the sweep range this comparison no longer applies')
        with mock.patch(_MEM, return_value=_ROOMY):
            self.assertLess(batched_fft_cell_chunk_size((128, 128)), old_chunk)


if __name__ == '__main__':
    unittest.main()
