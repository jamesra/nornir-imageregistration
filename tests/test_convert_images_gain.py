"""ConvertImagesInDict divides a tile by its gain before the MinMax stretch."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import nornir_imageregistration as nir


class TestConvertImagesWithGain(unittest.TestCase):

    def _convert(self, gains) -> float:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'in.png')
            Image.fromarray(np.full((16, 16), 1000, dtype=np.uint16)).save(source)
            target = os.path.join(tmp, 'out.png')
            nir.ConvertImagesInDict({source: target}, InputBpp=16, OutputBpp=16, MinMax=[0, 4000], SourceGains=gains and {source: gains})
            with Image.open(target) as im:
                return float(np.asarray(im).mean())

    def test_no_gain_leaves_the_stretch_unchanged(self) -> None:
        self.assertAlmostEqual(self._convert(None), self._convert(1.0), places=3)

    def test_a_gain_below_one_brightens_the_tile(self) -> None:
        self.assertAlmostEqual(self._convert(0.5) / self._convert(None), 2.0, places=2)

    def test_the_result_is_clipped_to_the_range(self) -> None:
        self.assertLessEqual(self._convert(0.01), 65535)


if __name__ == '__main__':
    unittest.main()
