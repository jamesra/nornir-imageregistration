"""Unit tests for scaled-transform cache and prefetch env in assemble_tiles."""
import os
import tempfile
import unittest

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.mosaic_tileset
from nornir_imageregistration.transforms.rigid import Rigid


class TestScaledTransformCache(unittest.TestCase):

    def test_cache_reuses_scaled_transform_within_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            import numpy as np
            path = os.path.join(tmp, '0.png')
            nornir_imageregistration.SaveImage(path, np.ones((8, 8), dtype=np.float32), bpp=8)
            tileset = nornir_imageregistration.mosaic_tileset.Create(
                [Rigid(target_offset=(0.0, 0.0))], [path], image_to_source_space_scale=4.0)
            tile = next(iter(tileset.values()))
            source_scale = 0.125
            target_scale = 0.125
            with at._scaled_transform_cache_scope():
                first = at._get_scaled_transform_for_tile(tile, source_scale, target_scale)
                second = at._get_scaled_transform_for_tile(tile, source_scale, target_scale)
            self.assertIs(first, second)

    def test_prefetch_default_on_for_cupy(self):
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not available")
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        env = os.environ.pop('NORNIR_ASSEMBLE_PREFETCH', None)
        try:
            self.assertTrue(at._assemble_prefetch_enabled())
        finally:
            if env is not None:
                os.environ['NORNIR_ASSEMBLE_PREFETCH'] = env

    def test_prefetch_disabled_by_env(self):
        os.environ['NORNIR_ASSEMBLE_PREFETCH'] = '0'
        try:
            self.assertFalse(at._assemble_prefetch_enabled())
        finally:
            os.environ.pop('NORNIR_ASSEMBLE_PREFETCH', None)

    def test_grid_extrapolate_default_off(self):
        env = os.environ.pop('NORNIR_ASSEMBLE_GRID_EXTRAPOLATE', None)
        try:
            self.assertFalse(at._assemble_grid_extrapolate())
        finally:
            if env is not None:
                os.environ['NORNIR_ASSEMBLE_GRID_EXTRAPOLATE'] = env

    def test_inverse_scipy_default_cupy_extrapolate_still_cupy(self):
        from nornir_imageregistration.transforms import gridtransform as gt
        for k in ('NORNIR_ASSEMBLE_INVERSE_SCIPY', 'NORNIR_ASSEMBLE_GRID_EXTRAPOLATE'):
            os.environ.pop(k, None)
        try:
            self.assertFalse(gt._assemble_inverse_use_scipy())
            os.environ['NORNIR_ASSEMBLE_INVERSE_SCIPY'] = '1'
            self.assertTrue(gt._assemble_inverse_use_scipy())
            os.environ.pop('NORNIR_ASSEMBLE_INVERSE_SCIPY', None)
            os.environ['NORNIR_ASSEMBLE_GRID_EXTRAPOLATE'] = '1'
            self.assertFalse(gt._assemble_inverse_use_scipy())
        finally:
            for k in ('NORNIR_ASSEMBLE_INVERSE_SCIPY', 'NORNIR_ASSEMBLE_GRID_EXTRAPOLATE'):
                os.environ.pop(k, None)

    def test_distance_warp_order_default_cubic(self):
        from nornir_imageregistration import assemble
        env = os.environ.pop('NORNIR_ASSEMBLE_DISTANCE_WARP_ORDER', None)
        try:
            self.assertIsNone(assemble._assemble_distance_warp_order())
            os.environ['NORNIR_ASSEMBLE_DISTANCE_WARP_ORDER'] = '0'
            self.assertEqual(assemble._assemble_distance_warp_order(), 0)
            os.environ['NORNIR_ASSEMBLE_DISTANCE_WARP_ORDER'] = '1'
            self.assertEqual(assemble._assemble_distance_warp_order(), 1)
        finally:
            if env is not None:
                os.environ['NORNIR_ASSEMBLE_DISTANCE_WARP_ORDER'] = env
            else:
                os.environ.pop('NORNIR_ASSEMBLE_DISTANCE_WARP_ORDER', None)


if __name__ == '__main__':
    unittest.main()
