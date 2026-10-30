"""Tests for skipping empty optimized tileset cells and column strips."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.transforms.rigid import Rigid


def _save_test_tile(path: str, shape: tuple[int, int], value: float = 128.0) -> None:
  """Write a small grayscale PNG tile for mosaic assembly tests."""
  img = np.full(shape, value, dtype=np.float32)
  nornir_imageregistration.SaveImage(path, img, bpp=8)


class TestGenerateOptimizedTilesEmpty(unittest.TestCase):
  """Verify empty optimized tiles are not yielded or assembled."""

  def setUp(self) -> None:
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)

  def _tileset_small_mosaic(self, tmp_dir: str):
    """One 48×48 tile at origin."""
    tile_path = os.path.join(tmp_dir, "0.png")
    _save_test_tile(tile_path, (48, 48))
    return nornir_imageregistration.mosaic_tileset.Create(
        [Rigid(target_offset=(0.0, 0.0))],
        [tile_path],
        image_to_source_space_scale=1.0,
    )

  def test_skips_cells_without_assemble_coverage(self) -> None:
    """Cells with no True pixels in the assemble mask are not yielded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._tileset_small_mosaic(tmp_dir)
      tile_dims = np.asarray((32, 32), dtype=np.int64)
      tileset.TranslateToZeroOrigin()

      grid_dims = nornir_imageregistration.TileGridShape(
          tileset.TargetBoundingBox.shape, tile_size=tile_dims)
      expected_cells = int(np.prod(grid_dims))
      self.assertEqual(expected_cells, 4)

      original_assemble = tileset.AssembleImage

      def sparse_mask_assemble(FixedRegion=None, target_space_scale=None):
        image, mask = original_assemble(
            FixedRegion=FixedRegion, target_space_scale=target_space_scale)
        mask_host = np.asarray(nornir_imageregistration.EnsureNumpyArray(mask), dtype=bool)
        mask_host[:, :] = False
        mask_host[:32, :32] = True
        return image, mask_host

      with mock.patch.object(tileset, 'AssembleImage', side_effect=sparse_mask_assemble):
        yielded = list(tileset.GenerateOptimizedTiles(
            tile_dims=tile_dims,
            max_temp_image_area=int(np.prod(grid_dims * tile_dims)),
            target_space_scale=1.0))

      self.assertEqual(len(yielded), 1)
      self.assertEqual(yielded[0][0], 0)
      self.assertEqual(yielded[0][1], 0)
      self.assertLess(len(yielded), expected_cells)

  def test_skips_column_strips_with_no_source_overlap(self) -> None:
    """Strips with no intersecting source tiles do not call AssembleImage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._tileset_small_mosaic(tmp_dir)
      tile_dims = np.asarray((32, 32), dtype=np.int64)
      tileset.TranslateToZeroOrigin()
      tileset._target_space_bounding_box = (
          nornir_imageregistration.Rectangle.CreateFromPointAndArea(
              (0, 0), np.asarray((48, 32 * 4), dtype=np.int64)))

      original_intersections = tileset.TargetSpaceIntersections
      assemble_calls = 0
      original_assemble = tileset.AssembleImage

      def intersections(rect):
        if rect.BottomLeft[1] >= 64:
          return []
        return original_intersections(rect)

      def counting_assemble(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return original_assemble(*args, **kwargs)

      with mock.patch.object(tileset, 'TargetSpaceIntersections', side_effect=intersections):
        with mock.patch.object(tileset, 'AssembleImage', side_effect=counting_assemble):
          yielded = list(tileset.GenerateOptimizedTiles(
              tile_dims=tile_dims,
              max_temp_image_area=32 * 32 * 2,
              target_space_scale=1.0))

      self.assertEqual(assemble_calls, 2)
      self.assertGreater(len(yielded), 0)

  def test_grid_dims_unchanged_for_full_logical_grid(self) -> None:
    """CalculateGridDimensions reports the full mosaic grid independent of sparse yields."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._tileset_small_mosaic(tmp_dir)
      tile_dims = np.asarray((32, 32), dtype=np.int64)
      tileset.TranslateToZeroOrigin()

      grid_dims = tileset.CalculateGridDimensions(tile_dims, expected_scale=1.0)
      self.assertEqual(tuple(int(x) for x in grid_dims), (2, 2))

      original_assemble = tileset.AssembleImage

      def sparse_mask_assemble(FixedRegion=None, target_space_scale=None):
        image, mask = original_assemble(
            FixedRegion=FixedRegion, target_space_scale=target_space_scale)
        mask_host = np.asarray(nornir_imageregistration.EnsureNumpyArray(mask), dtype=bool)
        mask_host[:, :] = False
        mask_host[:32, :32] = True
        return image, mask_host

      with mock.patch.object(tileset, 'AssembleImage', side_effect=sparse_mask_assemble):
        yielded = list(tileset.GenerateOptimizedTiles(
            tile_dims=tile_dims,
            max_temp_image_area=int(np.prod(grid_dims * tile_dims)),
            target_space_scale=1.0))

      self.assertEqual(int(np.prod(grid_dims)), 4)
      self.assertEqual(len(yielded), 1)


if __name__ == '__main__':
  unittest.main()
