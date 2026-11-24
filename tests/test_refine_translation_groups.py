"""Batched translation cells are measured in groups when the whole grid would not fit in device memory.

RC3-v2 324-325 Grid16 (2026-11-24): a doubled 512 px cell size made one pass's stacks reach 21.5 GB on CuPy
and the pair failed. Groups must return the same records as one call, in the same order.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nornir_imageregistration import local_distortion_correction as ldc

_FIXTURE = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'idoc_690_691', 'StosBrute16',
    '690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos')


def _refine(chunk: int | None, calls: list[int]) -> str:
    real = ldc._attempt_align_points_translation_batched

    def spy(*args, **kwargs):
        calls.append(len(kwargs['keys']))
        return real(*args, **kwargs)

    with mock.patch.object(ldc, '_attempt_align_points_translation_batched', spy), \
            mock.patch.object(ldc, 'batched_translation_cell_chunk_size', lambda _shape: chunk), \
            tempfile.TemporaryDirectory() as output_dir:
        out = os.path.join(output_dir, 'Final.stos')
        ldc.RefineStosFile(InputStos=_FIXTURE, OutputStosPath=out, num_iterations=3,
                           cell_size=(128, 128), grid_spacing=(96, 96))
        with open(out) as handle:
            return handle.read()


class TestRefineTranslationGroups(unittest.TestCase):

    def test_groups_match_one_call(self) -> None:
        whole_calls: list[int] = []
        whole = _refine(None, whole_calls)
        grouped_calls: list[int] = []
        grouped = _refine(3, grouped_calls)
        self.assertTrue(any(n > 3 for n in whole_calls), f'fixture too small to group: {whole_calls}')
        self.assertTrue(grouped_calls and max(grouped_calls) <= 3, grouped_calls)
        self.assertEqual(whole, grouped)


if __name__ == '__main__':
    unittest.main()
