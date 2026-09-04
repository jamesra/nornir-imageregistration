"""Parallel assemble must composite in tile submission order (#241).

Equal distance (z) at an interior seam keeps the first writer
(``CompositeImageWithZBuffer`` uses strict ``>``). Completing warps out of
order therefore changes which tile wins the seam, which is what made
``TilesToImageParallel`` disagree with serial by ~0.14 on float16 fixtures.

Fail-before: on IDoc 004, unfixed Parallel differs on 33 pixels at max 0.142;
with the fix, serial and parallel are bit-identical on that region.
"""

from __future__ import annotations

import glob
import os
import tempfile
import unittest

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as assemble_tiles
from nornir_imageregistration.mosaic import Mosaic
from nornir_imageregistration.transforms.rigid import Rigid


class TestEqualZKeepsFirstWriter(unittest.TestCase):
    """Pin the blend rule that makes composite order load-bearing."""

    def test_equal_distance_keeps_earlier_tile(self):
        canvas = np.zeros((4, 4), dtype=np.float32)
        zbuf = np.full((4, 4), np.finfo(np.float32).max, dtype=np.float32)
        a = np.full((4, 4), 0.25, dtype=np.float32)
        b = np.full((4, 4), 0.75, dtype=np.float32)
        za = np.full((4, 4), 10.0, dtype=np.float32)
        zb = np.full((4, 4), 10.0, dtype=np.float32)

        assemble_tiles.CompositeImageWithZBuffer(canvas, zbuf, a, za, (0, 0))
        assemble_tiles.CompositeImageWithZBuffer(canvas, zbuf, b, zb, (0, 0))
        np.testing.assert_array_equal(canvas, a)

    def test_reversed_order_selects_the_other_tile(self):
        canvas = np.zeros((4, 4), dtype=np.float32)
        zbuf = np.full((4, 4), np.finfo(np.float32).max, dtype=np.float32)
        a = np.full((4, 4), 0.25, dtype=np.float32)
        b = np.full((4, 4), 0.75, dtype=np.float32)
        z = np.full((4, 4), 10.0, dtype=np.float32)

        assemble_tiles.CompositeImageWithZBuffer(canvas, zbuf, b, z, (0, 0))
        assemble_tiles.CompositeImageWithZBuffer(canvas, zbuf, a, z.copy(), (0, 0))
        np.testing.assert_array_equal(canvas, b)


def _build_overlapping_row(tmp_dir: str, n_tiles: int = 4, tile_shape=(64, 64)):
    stride = float(tile_shape[1]) * 0.75
    transforms = []
    image_paths = []
    rng = np.random.default_rng(seed=241)
    for i in range(n_tiles):
        img = rng.random(tile_shape).astype(np.float32)
        png_path = os.path.join(tmp_dir, f'{i}.png')
        nornir_imageregistration.SaveImage(png_path, img, bpp=8)
        image_paths.append(png_path)
        transforms.append(Rigid(target_offset=(0.0, i * stride)))
    return nornir_imageregistration.mosaic_tileset.Create(
        transforms, image_paths, image_to_source_space_scale=1.0
    )


class TestParallelMatchesSerialSeams(unittest.TestCase):
    """After #241, parallel must match serial at overlapping seams."""

    def test_four_tile_row_bit_close(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tileset = _build_overlapping_row(tmp_dir)
            serial, serial_mask = assemble_tiles.TilesToImage(tileset)
            parallel, parallel_mask = assemble_tiles.TilesToImageParallel(tileset)

        self.assertIsNotNone(serial)
        self.assertIsNotNone(parallel)
        np.testing.assert_array_equal(serial_mask, parallel_mask)
        s = np.asarray(serial, dtype=np.float64)
        p = np.asarray(parallel, dtype=np.float64)
        delta = np.abs(s - p)
        self.assertLessEqual(float(delta.max()), 1e-3,
                             f'serial vs parallel max |delta|={delta.max():.6g} '
                             f'({int(np.count_nonzero(delta))} differing pixels)')
        self.assertLessEqual(float(delta.mean()), 1e-5)

    def test_idoc_004_region_matches_serial(self):
        """Regression for the measured #241 seam (fails against completion-order Parallel)."""
        input_root = os.environ.get('TESTINPUTPATH')
        if not input_root:
            self.skipTest('TESTINPUTPATH not set')
        base = os.path.join(input_root, 'Transforms', 'mosaics', 'IDOC1')
        mosaic_files = glob.glob(os.path.join(base, '*.mosaic'))
        tiles_dir = os.path.join(base, 'Leveled', 'TilePyramid', '004')
        if not mosaic_files or not os.path.isdir(tiles_dir):
            self.skipTest('IDOC1 004 fixture not available')

        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        downsample = 4.0
        mosaic_obj = Mosaic.LoadFromMosaicFile(mosaic_files[0])
        tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
            mosaic_obj, tiles_dir, downsample)
        tileset.TranslateToZeroOrigin()
        min_y, min_x, _, _ = tileset.TargetBoundingBox
        scaled = np.array([min_y + 512, min_x + 1024, min_y + 1024, min_x + 2048])
        fixed = nornir_imageregistration.Rectangle(scaled * downsample)

        serial, serial_mask = assemble_tiles.TilesToImage(
            tileset, TargetRegion=fixed, target_space_scale=1.0 / downsample)
        parallel, parallel_mask = assemble_tiles.TilesToImageParallel(
            tileset, TargetRegion=fixed, target_space_scale=1.0 / downsample)

        np.testing.assert_array_equal(serial_mask, parallel_mask)
        delta = np.abs(np.asarray(serial, dtype=np.float64) -
                       np.asarray(parallel, dtype=np.float64))
        differing = int(np.count_nonzero(delta))
        max_abs = float(delta.max()) if delta.size else 0.0
        # Unfixed Parallel: differing=33, max≈0.142. Fixed: both 0.
        self.assertEqual(differing, 0,
                         f'serial vs parallel still disagree on {differing} pixels '
                         f'(max |delta|={max_abs:.6g})')
        self.assertEqual(max_abs, 0.0)


if __name__ == '__main__':
    unittest.main()
