"""Crushed tiles are gain-corrected while they are assembled, and the Viking pyramid keeps the crush chunk.

TileGains records a gain per tile in its ``nornir_crush`` chunk (codes untouched). Assembly loads crushed tiles
as ``code / max_code`` and applies that gain there, so assembled images and optimized tiles are codes of the
section quanta with gain 1; nothing is inflated to counts (James, 2026-11-05).
"""
import os
import tempfile
import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import noise_crush, tile_gains, tileset_functions
from nornir_imageregistration.noise_crush import CrushParams
from PIL import Image

_PARAMS = CrushParams(k=2.5, step=12, sigma=4.8, offset=-30, bits=8, source_dtype='int16')


def _corrected_codes(codes: np.ndarray, params: CrushParams, gain: float) -> np.ndarray:
    """Gain-corrected counts, ``(code * step + step / 2 + offset) / gain``, as codes of the same quanta."""
    counts = (codes.astype(np.float64) * params.step + params.step / 2.0 + params.offset) / gain
    return (counts - params.offset) / params.step - 0.5


class TestCrushedGain(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
        self.path = os.path.join(self.dir, 'tile.png')
        noise_crush.save_crushed_png(self.path, self.codes, _PARAMS)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self, crush_gain: bool) -> np.ndarray:
        image = nornir_imageregistration.LoadImage(self.path, dtype=np.float64, value_scale=_PARAMS.max_code,
                                                   crush_gain=crush_gain, backend='numpy')
        return np.asarray(image) * _PARAMS.max_code

    def test_no_gain_loads_the_codes(self) -> None:
        np.testing.assert_allclose(self._load(crush_gain=True), self.codes)

    def test_gain_is_applied_in_code_units(self) -> None:
        self.assertTrue(tile_gains.stamp_gain_png(self.path, 1.25))

        np.testing.assert_allclose(self._load(crush_gain=True), _corrected_codes(self.codes, _PARAMS, 1.25),
                                   atol=1e-4)
        np.testing.assert_allclose(self._load(crush_gain=False), self.codes)

    def test_gain_of_one_and_missing_chunk_change_nothing(self) -> None:
        self.assertIsNone(noise_crush.unit_gain_correction(None))
        self.assertIsNone(noise_crush.unit_gain_correction(_PARAMS.to_json()))
        tile_gains.stamp_gain_png(self.path, 1.0)
        np.testing.assert_allclose(self._load(crush_gain=True), self.codes)

    def test_pyramid_tile_keeps_the_crush_chunk(self) -> None:
        names = []
        for i in range(4):
            name = os.path.join(self.dir, f'q{i}.png')
            noise_crush.save_crushed_png(name, self.codes, _PARAMS)
            names.append(name)
        output = os.path.join(self.dir, 'out.png')

        tileset_functions.CreateOneTilesetTileWithPillow((16, 16), *names, OutputFileFullPath=output)

        self.assertEqual(noise_crush.read_crush_params(output), _PARAMS)
        with Image.open(output) as im:
            self.assertEqual(im.mode, 'L')

    def test_pyramid_tile_never_exceeds_the_top_code(self) -> None:
        params = CrushParams(k=2.5, step=12, sigma=4.8, offset=0, bits=7, source_dtype='uint16')
        edges = np.zeros((16, 16), dtype=np.uint8)
        edges[:, ::2] = params.max_code  # Lanczos rings past the top code at every edge
        names = []
        for i in range(4):
            name = os.path.join(self.dir, f'e{i}.png')
            noise_crush.save_crushed_png(name, edges, params)
            names.append(name)
        output = os.path.join(self.dir, 'out.png')

        tileset_functions.CreateOneTilesetTileWithPillow((16, 16), *names, OutputFileFullPath=output)

        with Image.open(output) as im:
            self.assertLessEqual(int(np.asarray(im).max()), params.max_code)
        self.assertEqual(noise_crush.read_crush_params(output), params)

    def test_uncrushed_pyramid_tile_has_no_chunk(self) -> None:
        names = []
        for i in range(4):
            name = os.path.join(self.dir, f'u{i}.png')
            Image.fromarray(self.codes).save(name)
            names.append(name)
        output = os.path.join(self.dir, 'out.png')

        tileset_functions.CreateOneTilesetTileWithPillow((16, 16), *names, OutputFileFullPath=output)

        self.assertIsNone(noise_crush.read_crush_params(output))


if __name__ == '__main__':
    unittest.main()
