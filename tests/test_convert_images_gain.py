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


class TestGainFirstEightBitScaleIsFixed(unittest.TestCase):
    """RC3 516: gain-first legacy tiles were rescaled by each tile's own maximum on the way to 8 bits (checkerboard)."""

    def _convert(self, raw: np.ndarray, gain: float) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'in.png')
            Image.fromarray(raw.astype(np.uint16)).save(source)
            target = os.path.join(tmp, 'out.png')
            nir.ConvertImagesInDict({source: target}, InputBpp=16, OutputBpp=8, MinMax=[4000, 28000],
                                    SourceGains={source: gain})
            with Image.open(target) as im:
                return np.asarray(im).astype(float)

    def test_the_same_level_maps_to_the_same_value_whatever_the_tile_maximum(self) -> None:
        """Two tiles whose pixels sit at raw/gain = 16000, one also holding a very bright pixel: same 8-bit value."""
        plain = np.full((16, 16), 16000, dtype=np.uint16)
        with_bright = plain.copy()
        with_bright[0, 0] = 60000  # saturates the stretch in this tile only
        a, b = self._convert(plain, 1.0), self._convert(with_bright, 1.0)
        self.assertEqual(a[8, 8], b[8, 8])
        self.assertAlmostEqual(a[8, 8], 255 * (16000 - 4000) / 24000, delta=1.0)

    def test_the_gain_is_applied_before_the_fixed_scale(self) -> None:
        raw = np.full((16, 16), 16000, dtype=np.uint16)
        self.assertAlmostEqual(self._convert(raw, 0.5)[8, 8], 255.0, delta=1.0)  # 16000 / 0.5 = 32000, past the 28000 maximum: clipped
        self.assertAlmostEqual(self._convert(raw, 2.0)[8, 8], 255 * (8000 - 4000) / 24000, delta=1.0)
