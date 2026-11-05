"""GenerateOptimizedTiles assembles column strips in a bounded window and stops when its consumer stops.

Regression for RC2 section 1240 (2026-11-05): every strip was submitted up front, so finished strips stayed alive in
their futures until the generator ended (memory grew by ~1.5 GB per strip), and an early exit waited for the whole
section to assemble before the error surfaced.
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from unittest import mock

import nornir_imageregistration
import numpy as np
from nornir_imageregistration.transforms.rigid import Rigid


def _save_test_tile(path: str, shape: tuple[int, int], value: float = 128.0) -> None:
  """Write a small grayscale PNG tile for mosaic assembly tests."""
  img = np.full(shape, value, dtype=np.float32)
  nornir_imageregistration.SaveImage(path, img, bpp=8)


class TestGenerateOptimizedTilesWindow(unittest.TestCase):
  """Strip assembly is bounded and cancelled on early exit."""

  def setUp(self) -> None:
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)

  def _wide_tileset(self, tmp_dir: str, columns: int):
    """A row of 32x32 tiles side by side, so each 32-px column strip has one source tile."""
    paths, transforms = [], []
    for i in range(columns):
      path = os.path.join(tmp_dir, f"{i}.png")
      _save_test_tile(path, (32, 32), value=10.0 + i)
      paths.append(path)
      transforms.append(Rigid(target_offset=(0.0, 32.0 * i)))
    tileset = nornir_imageregistration.mosaic_tileset.Create(transforms, paths, image_to_source_space_scale=1.0)
    tileset.TranslateToZeroOrigin()
    return tileset

  def _counting_assemble(self, tileset):
    """Wrap AssembleImage to record calls and the peak number running at once."""
    original = tileset.AssembleImage
    state = {"calls": 0, "running": 0, "peak": 0}
    lock = threading.Lock()

    def wrapped(FixedRegion=None, target_space_scale=None):
      with lock:
        state["calls"] += 1
        state["running"] += 1
        state["peak"] = max(state["peak"], state["running"])
      try:
        return original(FixedRegion=FixedRegion, target_space_scale=target_space_scale)
      finally:
        with lock:
          state["running"] -= 1

    return wrapped, state

  def test_all_strips_yield_in_column_order(self) -> None:
    """Every strip is assembled once and tiles come out column by column."""
    columns = 8
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._wide_tileset(tmp_dir, columns)
      tile_dims = np.asarray((32, 32), dtype=np.int64)
      wrapped, state = self._counting_assemble(tileset)
      with mock.patch.object(tileset, 'AssembleImage', side_effect=wrapped):
        yielded = list(tileset.GenerateOptimizedTiles(tile_dims=tile_dims, max_temp_image_area=32 * 32,
                                                      target_space_scale=1.0))
      self.assertEqual(state["calls"], columns)
      self.assertEqual([entry[1] for entry in yielded], list(range(columns)))
      self.assertLessEqual(state["peak"], 2)

  def test_closing_the_generator_does_not_assemble_the_rest(self) -> None:
    """Stopping after the first tile assembles at most the strips already in flight."""
    columns = 12
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._wide_tileset(tmp_dir, columns)
      tile_dims = np.asarray((32, 32), dtype=np.int64)
      wrapped, state = self._counting_assemble(tileset)
      with mock.patch.object(tileset, 'AssembleImage', side_effect=wrapped):
        generator = tileset.GenerateOptimizedTiles(tile_dims=tile_dims, max_temp_image_area=32 * 32,
                                                   target_space_scale=1.0)
        next(generator)
        generator.close()
      # strip_workers is 2 on NumPy: the strip being sliced plus at most two more were submitted
      self.assertLessEqual(state["calls"], 3)

  def test_consumer_error_surfaces_without_assembling_the_rest(self) -> None:
    """An exception in the consumer propagates promptly instead of waiting for every strip."""
    columns = 12
    with tempfile.TemporaryDirectory() as tmp_dir:
      tileset = self._wide_tileset(tmp_dir, columns)
      tile_dims = np.asarray((32, 32), dtype=np.int64)
      wrapped, state = self._counting_assemble(tileset)
      with mock.patch.object(tileset, 'AssembleImage', side_effect=wrapped), self.assertRaises(RuntimeError):
        for _ in tileset.GenerateOptimizedTiles(tile_dims=tile_dims, max_temp_image_area=32 * 32,
                                                target_space_scale=1.0):
          raise RuntimeError("save failed")
      self.assertLessEqual(state["calls"], 3)


if __name__ == '__main__':
  unittest.main()
