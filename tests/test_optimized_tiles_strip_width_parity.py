"""Optimized tiles must not depend on the strip width GenerateOptimizedTiles assembles in.

Regression (RC2 1240, 2026-11-05): the warp cropped each source tile to the region it fed before the cubic spline
prefilter, so the same cell assembled in a 1-column and a 6-column strip differed by up to 80 DN. The strip width
follows free memory (EstimateMaxTempImageArea), so output depended on the machine.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.transforms.rigid import Rigid


class TestOptimizedTilesStripWidthParity(unittest.TestCase):
  """Same tiles for any strip width."""

  def setUp(self) -> None:
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)

  def _overlapping_tileset(self, tmp_dir: str):
    """A 2 x 3 grid of textured 96 x 96 tiles overlapping by ~20 px at fractional offsets."""
    rng = np.random.default_rng(1240)
    paths, transforms = [], []
    for row in range(2):
      for col in range(3):
        path = os.path.join(tmp_dir, f"{row * 3 + col}.png")
        texture = rng.random((96, 96)).astype(np.float32) * 0.6 + 0.2
        nornir_imageregistration.SaveImage(path, texture, bpp=8)
        paths.append(path)
        transforms.append(Rigid(target_offset=(row * 76.37, col * 75.61)))
    tileset = nornir_imageregistration.mosaic_tileset.Create(transforms, paths, image_to_source_space_scale=1.0)
    tileset.TranslateToZeroOrigin()
    return tileset

  def _tiles(self, tileset, columns_per_strip: int) -> dict[tuple[int, int], np.ndarray]:
    tile_dims = np.asarray((32, 32), dtype=np.int64)
    grid = nornir_imageregistration.TileGridShape(tileset.TargetBoundingBox.shape, tile_size=tile_dims)
    area = int(grid[0] * tile_dims[0] * tile_dims[1] * columns_per_strip)
    return {(row, col): np.asarray(nornir_imageregistration.EnsureNumpyArray(image), dtype=np.float32)
            for row, col, image in tileset.GenerateOptimizedTiles(tile_dims=tile_dims, max_temp_image_area=area,
                                                                  target_space_scale=1.0)}

  def test_one_column_and_whole_mosaic_strips_give_identical_tiles(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._overlapping_tileset(tmp_dir)
      narrow = self._tiles(tileset, columns_per_strip=1)
      wide = self._tiles(tileset, columns_per_strip=100)
      self.assertEqual(set(narrow), set(wide))
      self.assertGreater(len(narrow), 20)
      for key in narrow:
        np.testing.assert_array_equal(narrow[key], wide[key], err_msg=f"tile {key} differs with strip width")


if __name__ == '__main__':
  unittest.main()
