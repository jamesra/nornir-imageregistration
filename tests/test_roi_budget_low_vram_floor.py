"""Low-VRAM ROI sample budget must not serialise map_coordinates (#240).

Measured CuPy map_coordinates (order=3, prefiltered once, median of 3):

    cell  64px: chunk 1 = 12.4x vs best (~12.5 MiB sample bytes)
    cell 128px: chunk 1 =  2.8x vs best (~12.5 MiB)
    cell 256px: chunk 1 =  1.4x vs best (~25 MiB)

Preferred working set is 12 MiB (64/128 optima); min floor is half of that.
"""

from __future__ import annotations

import logging
import os
import unittest
from unittest import mock

from nornir_imageregistration.refine_shared.gpu_batch_budget import (
    _MAX_ROI_SAMPLE_BUDGET,
    _REFINE_BATCH_HEADROOM_BYTES,
    _REFINE_BATCH_VRAM_FRACTION,
    _ROI_BYTES_PER_SAMPLE,
    _ROI_MIN_WORKSPACE_BYTES,
    _ROI_PREFERRED_WORKSPACE_BYTES,
    batched_roi_sample_budget,
)

_MEM = "nornir_imageregistration.refine_shared.gpu_batch_budget.cuda_memory_info"
_MODULE = "nornir_imageregistration.refine_shared.gpu_batch_budget"
_TOTAL = 24 * 1024 ** 3
_CLIFF_BYTES = int(_REFINE_BATCH_HEADROOM_BYTES / _REFINE_BATCH_VRAM_FRACTION)


def _reset_throttle() -> None:
    import nornir_imageregistration.refine_shared.gpu_batch_budget as mod
    mod._last_low_vram_warning = 0.0


def _chunk_cells(sample_budget: int, cell: int) -> int:
    return max(1, int(sample_budget) // (cell * cell))


class TestRoiLowVramFloor(unittest.TestCase):

    def setUp(self) -> None:
        os.environ.pop("NORNIR_REFINE_BATCHED_ROI_SAMPLES", None)
        _reset_throttle()

    def test_below_cliff_chunk_is_not_one(self) -> None:
        with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
            budget = batched_roi_sample_budget(128, 128)
        self.assertGreater(_chunk_cells(budget, 128), 1)
        self.assertEqual(budget, _ROI_MIN_WORKSPACE_BYTES // _ROI_BYTES_PER_SAMPLE)

    def test_healthy_path_uses_preferred_working_set(self) -> None:
        with mock.patch(_MEM, return_value=(20 * 1024 ** 3, _TOTAL)):
            budget = batched_roi_sample_budget(128, 128)
        self.assertEqual(budget, _ROI_PREFERRED_WORKSPACE_BYTES // _ROI_BYTES_PER_SAMPLE)
        self.assertEqual(_chunk_cells(budget, 128), 64)

    def test_floor_is_half_preferred(self) -> None:
        self.assertEqual(_ROI_MIN_WORKSPACE_BYTES * 2, _ROI_PREFERRED_WORKSPACE_BYTES)

    def test_floor_matches_half_optima_at_64_and_128(self) -> None:
        """6 MiB floor -> 128 cells @64px and 32 cells @128px (half measured bests)."""
        with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
            self.assertEqual(_chunk_cells(batched_roi_sample_budget(64, 64), 64), 128)
            self.assertEqual(_chunk_cells(batched_roi_sample_budget(128, 128), 128), 32)

    def test_no_free_vram_level_causes_collapse(self) -> None:
        levels = [8 * 1024 ** 3, 4 * 1024 ** 3, 2 * 1024 ** 3,
                  _CLIFF_BYTES + 1024 ** 2, _CLIFF_BYTES - 1024 ** 2,
                  1024 ** 3, 768 * 1024 ** 2, 512 * 1024 ** 2, 384 * 1024 ** 2]
        chunks = []
        for free in levels:
            _reset_throttle()
            with mock.patch(_MEM, return_value=(free, _TOTAL)):
                chunks.append(_chunk_cells(batched_roi_sample_budget(128, 128), 128))
        for (free_a, chunk_a), (free_b, chunk_b) in zip(
                zip(levels, chunks), zip(levels[1:], chunks[1:])):
            with self.subTest(free_mib=free_b // 1024 ** 2):
                self.assertGreater(chunk_b, 0)
                self.assertLessEqual(
                    chunk_a, chunk_b * 4,
                    f"chunk fell from {chunk_a} to {chunk_b} between "
                    f"{free_a // 1024 ** 2} and {free_b // 1024 ** 2} MiB free")

    def test_zero_free_returns_one_cell(self) -> None:
        with mock.patch(_MEM, return_value=(0, _TOTAL)):
            self.assertEqual(
                batched_roi_sample_budget(128, 128), 128 * 128)

    def test_warns_when_floor_applies(self) -> None:
        with self.assertLogs(_MODULE, level=logging.WARNING) as captured:
            with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                batched_roi_sample_budget(128, 128)
        joined = "\n".join(captured.output)
        self.assertIn("Free VRAM", joined)
        self.assertIn("NORNIR_REFINE_BATCHED_ROI_SAMPLES", joined)

    def test_env_override_wins(self) -> None:
        os.environ["NORNIR_REFINE_BATCHED_ROI_SAMPLES"] = "999"
        try:
            with mock.patch(_MEM, return_value=(1024 ** 3, _TOTAL)):
                self.assertEqual(batched_roi_sample_budget(128, 128), 999)
        finally:
            os.environ.pop("NORNIR_REFINE_BATCHED_ROI_SAMPLES", None)

    def test_floor_respects_hard_cap(self) -> None:
        self.assertLessEqual(_ROI_MIN_WORKSPACE_BYTES, _MAX_ROI_SAMPLE_BUDGET * _ROI_BYTES_PER_SAMPLE)


if __name__ == "__main__":
    unittest.main()
