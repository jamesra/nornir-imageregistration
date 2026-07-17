"""Unit tests for ImageToTilesGenerator coverage filtering."""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration


class TestImageToTilesGeneratorCoverage(unittest.TestCase):
  """Verify optional coverage_mask skips empty tile cells."""

  def test_coverage_mask_skips_empty_cells(self) -> None:
    """Only tiles with any True mask pixels are yielded."""
    image = np.arange(64, dtype=np.float32).reshape(4, 16)
    mask = np.zeros((4, 16), dtype=bool)
    mask[:2, :8] = True
    tile_size = np.asarray((2, 8), dtype=np.int64)
    grid_shape = np.asarray((2, 2), dtype=np.int64)

    yielded = list(nornir_imageregistration.ImageToTilesGenerator(
        source_image=image,
        tile_size=tile_size,
        grid_shape=grid_shape,
        coverage_mask=mask))

    self.assertEqual(len(yielded), 1)
    self.assertEqual((yielded[0][0], yielded[0][1]), (0, 0))


if __name__ == '__main__':
  unittest.main()
