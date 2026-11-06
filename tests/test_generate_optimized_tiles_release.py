"""Tests that GenerateOptimizedTiles releases a tile's decoded image after the last strip that covers it."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.transforms.rigid import Rigid


def _save_test_tile(path: str, shape: tuple[int, int], value: float) -> None:
  """Write a small grayscale PNG tile."""
  nornir_imageregistration.SaveImage(path, np.full(shape, value, dtype=np.float32), bpp=8)


class TestGenerateOptimizedTilesRelease(unittest.TestCase):
  """Tile.Image caches for the life of the tile; a long mosaic must not keep every tile resident."""

  def setUp(self) -> None:
    nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

  def _row_of_tiles(self, tmp_dir: str, count: int = 6):
    """``count`` 32x32 tiles side by side, no overlap."""
    paths = []
    transforms = []
    for i in range(count):
      path = os.path.join(tmp_dir, f"{i}.png")
      _save_test_tile(path, (32, 32), 40.0 + 20.0 * i)
      paths.append(path)
      transforms.append(Rigid(target_offset=(0.0, 32.0 * i)))
    tileset = nornir_imageregistration.mosaic_tileset.Create(transforms, paths, image_to_source_space_scale=1.0)
    tileset.TranslateToZeroOrigin()
    return tileset

  def _generate(self, tileset, prefetch_strips=0, wait_for_tile=None):
    """
    Run GenerateOptimizedTiles one tile column per strip as the CuPy backend does (one strip at a time, tiles decoded
    in this process), recording which tiles hold an image after each strip's tiles are loaded.

    The strip assembly is replaced by a stand-in that loads ``Image`` of the tiles covering the strip, as the threaded
    CuPy assembly does; the CPU path decodes in worker processes and never caches in this process.
    """
    resident_at_strip: list[list[bool]] = []
    prefetched_before_first_strip: list[bool] = []
    tiles = [tileset[key] for key in sorted(tileset)]

    def loading_assemble(FixedRegion=None, target_space_scale=None):
      if wait_for_tile is not None and not resident_at_strip:
        # The first strip's assembly: give the prefetcher time to reach the tile it should be loading ahead
        deadline = time.monotonic() + 5.0
        while tiles[wait_for_tile]._image is None and time.monotonic() < deadline:
          time.sleep(0.01)
        prefetched_before_first_strip.append(tiles[wait_for_tile]._image is not None)
      for tile in tileset.TargetSpaceIntersections(FixedRegion):
        _ = tile.Image
      resident_at_strip.append([t._image is not None for t in tiles])
      shape = (int(FixedRegion.Height), int(FixedRegion.Width))
      return np.full(shape, 0.5, dtype=np.float32), np.ones(shape, dtype=bool)

    tileset.AssembleImage = loading_assemble
    with mock.patch.object(nornir_imageregistration, 'GetActiveComputationLib',
                           return_value=nornir_imageregistration.ComputationLib.cupy):
      yielded = list(tileset.GenerateOptimizedTiles(tile_dims=(32, 32), max_temp_image_area=32 * 32,
                                                    target_space_scale=1.0,
                                                    prefetch_strips=prefetch_strips))
    self.prefetched_before_first_strip = prefetched_before_first_strip
    return yielded, resident_at_strip, tiles

  def test_tile_image_is_released_after_its_last_strip(self) -> None:
    """Strip N holds only the tiles that cover strips N-1 to N+1 (a tile touching a strip's edge covers it)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      yielded, resident_at_strip, tiles = self._generate(self._row_of_tiles(tmp_dir))
      self.assertEqual(len(yielded), 6)
      self.assertEqual(len(resident_at_strip), 6)
      for strip_number, resident in enumerate(resident_at_strip):
        held = {i for i, is_held in enumerate(resident) if is_held}
        self.assertLessEqual(held, {strip_number - 1, strip_number, strip_number + 1},
                             f"strip {strip_number} holds images of tiles {sorted(held)}")
      self.assertFalse(any(t._image is not None for t in tiles))

  def test_prefetch_loads_tiles_ahead_and_stays_bounded(self) -> None:
    """With prefetch_strips=2 a tile three strips on is decoded before strip 0 assembles, and none beyond the window."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      yielded, resident_at_strip, tiles = self._generate(self._row_of_tiles(tmp_dir, count=10), prefetch_strips=2,
                                                         wait_for_tile=3)
      self.assertEqual(len(yielded), 10)
      self.assertEqual(self.prefetched_before_first_strip, [True])
      for strip_number, resident in enumerate(resident_at_strip):
        held = {i for i, is_held in enumerate(resident) if is_held}
        # Tiles N-1..N+1 cover strip N; the assembly window plus prefetch reaches strip N+3 and its tiles N+2..N+4
        self.assertLessEqual(held, set(range(strip_number - 1, strip_number + 5)),
                             f"strip {strip_number} holds images of tiles {sorted(held)}")
      self.assertFalse(any(t._image is not None for t in tiles))

  def test_closing_the_generator_early_releases_prefetched_images(self) -> None:
    """Abandoning the run after one strip leaves no decoded images behind."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._row_of_tiles(tmp_dir, count=10)
      tiles = [tileset[key] for key in sorted(tileset)]

      def loading_assemble(FixedRegion=None, target_space_scale=None):
        for tile in tileset.TargetSpaceIntersections(FixedRegion):
          _ = tile.Image
        shape = (int(FixedRegion.Height), int(FixedRegion.Width))
        return np.full(shape, 0.5, dtype=np.float32), np.ones(shape, dtype=bool)

      tileset.AssembleImage = loading_assemble
      with mock.patch.object(nornir_imageregistration, 'GetActiveComputationLib',
                             return_value=nornir_imageregistration.ComputationLib.cupy):
        generator = tileset.GenerateOptimizedTiles(tile_dims=(32, 32), max_temp_image_area=32 * 32,
                                                   target_space_scale=1.0, prefetch_strips=2)
        next(generator)
        generator.close()
      self.assertFalse(any(t._image is not None for t in tiles))

  def test_release_does_not_change_the_output(self) -> None:
    """Pixels are identical whether or not images are released between strips."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      def generate():
        tileset = self._row_of_tiles(tmp_dir)
        return list(tileset.GenerateOptimizedTiles(tile_dims=(32, 32), max_temp_image_area=32 * 32,
                                                   target_space_scale=1.0))

      with_release = generate()
      original = nornir_imageregistration.tile.Tile.ReleaseLoadedImage
      nornir_imageregistration.tile.Tile.ReleaseLoadedImage = lambda self: None
      try:
        without_release = generate()
      finally:
        nornir_imageregistration.tile.Tile.ReleaseLoadedImage = original
      self.assertEqual(len(with_release), len(without_release))
      for a, b in zip(with_release, without_release):
        np.testing.assert_array_equal(a[2], b[2])

  def test_in_memory_tile_keeps_its_image(self) -> None:
    """A tile made from an array has no path to reload from, so it is never released."""
    image = np.full((32, 32), 0.5, dtype=np.float32)
    tile = nornir_imageregistration.tile.Tile(Rigid(target_offset=(0.0, 0.0)), image, 1.0, 0)
    tile.ReleaseLoadedImage()
    self.assertIs(tile._image, image)


if __name__ == '__main__':
  unittest.main()
