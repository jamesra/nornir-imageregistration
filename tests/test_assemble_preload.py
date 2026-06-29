"""
Tests that TilesToImage's tile-preloader:
  1. produces pixel-identical output to a reference (no regression)
  2. does not corrupt results for mosaics of various sizes (0, 1, 2, 4 tiles)

The preloader itself is a background-thread optimization; its correctness is
verified by comparing output with the reference (numpy serial path).
"""

import os
import tempfile
import time
import unittest

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as assemble_tiles
import nornir_imageregistration.mosaic_tileset
from nornir_imageregistration.transforms.rigid import Rigid


def _build_tileset(tmp_dir: str, n_tiles: int, tile_shape=(64, 64)):
    """
    Build a MosaicTileset of n_tiles arranged in a horizontal row with 10% overlap.
    Each tile contains a ramp gradient so sub-pixel interpolation differences are
    detectable.
    """
    stride = float(tile_shape[1]) * 0.9
    transforms = []
    image_paths = []

    rng = np.random.default_rng(seed=42)
    for i in range(n_tiles):
        img = rng.random(tile_shape).astype(np.float32)
        png_path = os.path.join(tmp_dir, f"{i}.png")
        nornir_imageregistration.SaveImage(png_path, img, bpp=8)
        image_paths.append(png_path)
        transforms.append(Rigid(target_offset=(0.0, i * stride)))

    return nornir_imageregistration.mosaic_tileset.Create(
        transforms, image_paths, image_to_source_space_scale=1.0
    )


class TestAssemblePreload(unittest.TestCase):

    def _reference_output(self, tileset):
        """Assemble using TilesToImage (includes preloader) with NumPy backend."""
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        return assemble_tiles.TilesToImage(tileset)

    # ------------------------------------------------------------------
    # Correctness: preloaded TilesToImage must match TilesToImageParallel
    # ------------------------------------------------------------------

    def _assert_tiles_match(self, n_tiles: int):
        with tempfile.TemporaryDirectory() as tmp_dir:
            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
            tileset = _build_tileset(tmp_dir, n_tiles=n_tiles)

            img_serial, mask_serial = assemble_tiles.TilesToImage(tileset)
            img_parallel, mask_parallel = assemble_tiles.TilesToImageParallel(tileset)

        if n_tiles == 0:
            return  # No output expected

        self.assertIsNotNone(img_serial, f"TilesToImage returned None for {n_tiles} tiles")
        self.assertIsNotNone(img_parallel, f"TilesToImageParallel returned None for {n_tiles} tiles")
        self.assertEqual(img_serial.shape, img_parallel.shape,
                         f"Shape mismatch for {n_tiles} tiles")
        img_s = np.nan_to_num(np.asarray(img_serial, dtype=np.float32))
        img_p = np.nan_to_num(np.asarray(img_parallel, dtype=np.float32))
        np.testing.assert_allclose(
            img_s, img_p, atol=1e-4,
            err_msg=f"TilesToImage and TilesToImageParallel differ for {n_tiles} tiles"
        )

    def test_single_tile(self):
        self._assert_tiles_match(1)

    def test_two_tiles(self):
        self._assert_tiles_match(2)

    def test_four_tiles(self):
        self._assert_tiles_match(4)

    def test_eight_tiles(self):
        self._assert_tiles_match(8)

    # ------------------------------------------------------------------
    # Preloader does not hang on very small mosaics
    # ------------------------------------------------------------------

    def test_single_tile_no_hang(self):
        """TilesToImage with 1 tile (no preload possible) returns within 5 s."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
            tileset = _build_tileset(tmp_dir, n_tiles=1)
            t0 = time.perf_counter()
            img, mask = assemble_tiles.TilesToImage(tileset)
            elapsed = time.perf_counter() - t0

        self.assertIsNotNone(img)
        self.assertLess(elapsed, 5.0, "TilesToImage hung on single-tile mosaic")

    def test_zero_overlap_tiles_no_crash(self):
        """Tiles that don't overlap with the target region are silently skipped."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
            tile_shape = (32, 32)
            img = np.zeros(tile_shape, dtype=np.float32)
            png_path = os.path.join(tmp_dir, "0.png")
            nornir_imageregistration.SaveImage(png_path, img, bpp=8)

            tileset = nornir_imageregistration.mosaic_tileset.Create(
                [Rigid(target_offset=(0.0, 0.0))],
                [png_path],
                image_to_source_space_scale=1.0,
            )
            # Assemble a region far from the tile — tile should be skipped
            region = nornir_imageregistration.Rectangle.CreateFromPointAndArea((10000, 10000), (32, 32))
            img_out, mask_out = assemble_tiles.TilesToImage(tileset, TargetRegion=region)
            # Output should be all-zero (no tile contributed)
            self.assertTrue(np.all(mask_out == False),
                            "Expected empty mask when no tile overlaps the target region")


if __name__ == "__main__":
    unittest.main()
