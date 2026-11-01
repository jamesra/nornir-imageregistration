"""A translation-only STOS refine measures every cell in the batched path.

Grid refine measures all of a pass's cells in one batched FFT. Cells that miss the
batched path go one at a time through ``AttemptAlignPoint``, which calls StosBrute's
``SliceToSliceRigidRegistration`` and so inherits its defaults: a log-polar angle
estimate and a scale search, about 1-2 s per cell on CuPy against milliseconds
batched. Late passes often have only one or two unlocked cells, and the batched path
used to refuse fewer than three, so those went serial: on srv (2026-11-01) RC3 Grid32
refine ran at 6.7 s per pair, and RC3 1503-1502 spent 4.3 s of 9.0 s on two cells.

This pins the invariant on a real multi-pass refine: no cell reaches the serial peak
finder, and the batched path is asked at least once per pass.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from nornir_imageregistration import local_distortion_correction as ldc

_FIXTURE = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'idoc_690_691', 'StosBrute16',
    '690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos')


class TestRefineCellsAllBatched(unittest.TestCase):

    def test_no_cell_is_measured_serially(self):
        serial_calls: list[int] = []
        batched_cell_counts: list[int] = []
        real_serial = ldc.AttemptAlignPoint
        real_batched = ldc._attempt_align_points_translation_batched

        def serial_spy(*args, **kwargs):
            serial_calls.append(1)
            return real_serial(*args, **kwargs)

        def batched_spy(*args, **kwargs):
            batched_cell_counts.append(len(kwargs['keys']))
            return real_batched(*args, **kwargs)

        ldc.AttemptAlignPoint = serial_spy
        ldc._attempt_align_points_translation_batched = batched_spy
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                ldc.RefineStosFile(
                    InputStos=_FIXTURE,
                    OutputStosPath=os.path.join(output_dir, 'Final.stos'),
                    num_iterations=6,
                    cell_size=(256, 256),
                    grid_spacing=(192, 192))
        finally:
            ldc.AttemptAlignPoint = real_serial
            ldc._attempt_align_points_translation_batched = real_batched

        self.assertTrue(batched_cell_counts, 'refine never reached the batched path')
        self.assertEqual(
            len(serial_calls), 0,
            f'{len(serial_calls)} cells measured one at a time; batched calls had '
            f'{batched_cell_counts} cells')


if __name__ == '__main__':
    unittest.main()
