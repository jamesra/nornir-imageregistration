"""RefineStosFile must not run a full ``gc.collect`` per pair.

After a STOS refine a full collection found nothing to free, but its cost grows
with every live object in the process, and buildmanager holds the whole volume
model: 0.65 s a pair with 3M live objects (RC3/RC2 Grid32, CuPy, 2026-11-01).
The CuPy memory pool is still released.
"""
from __future__ import annotations

import gc
import os
import tempfile
import unittest
from unittest import mock

from nornir_imageregistration import local_distortion_correction as ldc

_FIXTURE = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'idoc_690_691', 'StosBrute16',
    '690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos')


class TestRefineStosNoFullGc(unittest.TestCase):

    def test_refine_stos_file_does_not_collect_garbage(self):
        with tempfile.TemporaryDirectory() as output_dir, \
                mock.patch.object(gc, 'collect', wraps=gc.collect) as collect:
            ldc.RefineStosFile(
                InputStos=_FIXTURE,
                OutputStosPath=os.path.join(output_dir, 'out.stos'),
                num_iterations=2, cell_size=(256, 256), grid_spacing=(192, 192))

        self.assertEqual(collect.call_count, 0)

    def test_release_still_collects_by_default(self):
        with mock.patch.object(gc, 'collect') as collect:
            ldc._release_refinement_worker_memory()
            ldc._release_refinement_worker_memory(collect_garbage=False)
            ldc._release_refinement_worker_memory(reclaim_caches=False)

        self.assertEqual(collect.call_count, 1)


if __name__ == '__main__':
    unittest.main()
