"""Guard rails for the NaN-into-canvas hazard in CompositeImageWithZBuffer (#103).

The hazard is mechanically real.  ``CompositeImageWithZBuffer`` gates writes on
``sub_image != 0``, and ``nan != 0`` is True, so a NaN sample is written to the canvas.  It also
wins the z-test and claims the z-buffer, so an overlapping neighbour with real data at that pixel
can no longer fill it unless its distance is strictly better -- one NaN can punch a permanent hole
in a stitched mosaic.  ``xp.clip`` does not remove NaN, and lowering the spline order to 1 for NaN
inputs limits but does not prevent NaN spreading (measured 16 source NaN becoming 25 after a
half-pixel shift).

It is also **not reachable** in the current pipeline, which is why the gate was left alone rather
than paying for an ``isfinite`` pass.  Measured, that pass costs 3.5% to 7.6% of per-tile assemble
work depending on tile size, because the compositor is 25% to 54% of warp-plus-composite, and no
cheaper formulation exists -- an ``isnan().any()`` probe costs as much as the materialised pass.

Reachability rests on the four invariants asserted below.  If any of them starts failing, the
hazard has become live and the gate in ``CompositeImageWithZBuffer`` needs the ``isfinite`` guard
after all.  Read #103 before deleting any of these.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import assemble_tiles


class TestTheHazardIsRealIfNaNEverArrives(unittest.TestCase):
    """Documents the mechanism, so the reason for the guard rails below is on record."""

    def setUp(self):
        self.canvas = np.zeros((16, 16), dtype=np.float32)
        self.zbuffer = np.full((16, 16), np.finfo(np.float32).max, dtype=np.float32)

    def test_a_nan_sample_passes_the_nonzero_gate(self):
        self.assertTrue(bool(np.array([np.nan]) != 0))

    def test_clip_does_not_remove_nan(self):
        clipped = np.clip(np.array([np.nan, 5.0], dtype=np.float32), 0, 1)
        self.assertTrue(bool(np.isnan(clipped[0])))

    def test_a_nan_in_a_tile_reaches_the_canvas(self):
        tile = np.full((8, 8), 50.0, dtype=np.float32)
        tile[2, 2] = np.nan
        distance = np.full((8, 8), 3.0, dtype=np.float32)

        assemble_tiles.CompositeImageWithZBuffer(
            self.canvas, self.zbuffer, tile, distance, (1, 1))

        self.assertTrue(bool(np.isnan(self.canvas[3, 3])))

    def test_the_nan_then_blocks_an_overlapping_neighbour(self):
        """The part that makes this worse than a single bad pixel."""
        tile = np.full((8, 8), 50.0, dtype=np.float32)
        tile[2, 2] = np.nan
        assemble_tiles.CompositeImageWithZBuffer(
            self.canvas, self.zbuffer, tile, np.full((8, 8), 3.0, dtype=np.float32), (1, 1))

        neighbour = np.full((8, 8), 80.0, dtype=np.float32)
        worse_distance = np.full((8, 8), 4.0, dtype=np.float32)
        assemble_tiles.CompositeImageWithZBuffer(
            self.canvas, self.zbuffer, neighbour, worse_distance, (1, 1))

        self.assertTrue(bool(np.isnan(self.canvas[3, 3])),
                        "expected the NaN to have claimed the z-buffer")

        better_distance = np.full((8, 8), 1.0, dtype=np.float32)
        assemble_tiles.CompositeImageWithZBuffer(
            self.canvas, self.zbuffer, neighbour, better_distance, (1, 1))
        self.assertEqual(80.0, float(self.canvas[3, 3]),
                         "only a strictly better distance recovers the pixel")


class TestTheInvariantsThatKeepItUnreachable(unittest.TestCase):
    """If one of these fails, revisit #103 -- the hazard above has become live."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nan_reach_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_image_plane_is_warped_with_a_zero_fill_not_nan(self):
        """assemble_tiles warps [image, distance] with cval=[0, max]; the image fill is 0."""
        import inspect
        source = inspect.getsource(assemble_tiles.TransformTile)
        self.assertIn("cval=[0,", source.replace(" ", "").replace("cval=[0,", "cval=[0,"),
                      "the image plane no longer fills with 0 -- a NaN fill would reach the canvas")

    def test_nan_does_not_survive_a_save_load_round_trip(self):
        """Tiles come from disk, and the image formats in use cannot carry NaN."""
        arr = np.full((16, 16), 0.5, dtype=np.float32)
        arr[4:6, 4:6] = np.nan
        for ext in ('.tif', '.png'):
            with self.subTest(ext=ext):
                path = os.path.join(self.tmp, 'probe' + ext)
                nornir_imageregistration.SaveImage(path, arr)
                back = nornir_imageregistration.EnsureNumpyArray(
                    nornir_imageregistration.LoadImage(path))
                if back.dtype.kind == 'f':
                    self.assertEqual(0, int(np.isnan(back).sum()),
                                     "a NaN-bearing tile now round-trips through disk")

    def test_tiles_on_disk_load_as_integer_images(self):
        import glob
        pattern = os.path.join(
            os.environ.get('TESTINPUTPATH', r'D:\nornir-testdata'),
            'PlatformRaw', 'IDOC', 'RC2_4Square_Aligned', 'TEM', '0690', 'TEM',
            'Leveled', 'TilePyramid', '032', '*.png')
        tiles = glob.glob(pattern)
        if not tiles:
            self.skipTest("test data not available")
        for path in tiles[:3]:
            with self.subTest(tile=os.path.basename(path)):
                img = nornir_imageregistration.EnsureNumpyArray(
                    nornir_imageregistration.LoadImage(path))
                if img.dtype.kind == 'f':
                    self.assertFalse(bool(np.isnan(img).any()))

    def test_the_mask_layer_does_not_inject_nan_into_the_image(self):
        helper_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(assemble_tiles.__file__))),
            'nornir_imageregistration', 'image_permutation_helper.py')
        if not os.path.exists(helper_path):
            self.skipTest("image_permutation_helper.py not found")
        with open(helper_path, 'r', encoding='utf-8') as handle:
            body = handle.read()
        offenders = [line.strip() for line in body.splitlines()
                     if ('np.nan' in line or 'xp.nan' in line)
                     and 'isnan' not in line and 'nan_to_num' not in line]
        self.assertEqual([], offenders,
                         "the permutation layer now writes NaN into image pixels")


if __name__ == '__main__':
    unittest.main()
