"""Crushed tiles keep their codes through loading, Shrink and assembly, and carry the crush metadata.

Assembly loaded each integer tile as float divided by that tile's own maximum, then saved
x255. On RC1_Crush (codes 0-109, median 21) assembled Raw8 images came out at median 167,
with a different gain per tile (Build2 RC1, 2026-11-04). Pyramid levels written by Shrink
also dropped the ``nornir_crush`` chunk.
"""
import os
import tempfile
import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import noise_crush
from nornir_imageregistration.files.mosaicfile import MosaicFile
from nornir_imageregistration.noise_crush import CrushParams
from PIL import Image

_PARAMS = CrushParams(k=2.5, step=12, sigma=4.8, offset=0, bits=8, source_dtype='uint16')
_TILE = 128
_ORIGINS = ((0, 0), (_TILE, 0), (0, _TILE), (_TILE, _TILE))  # abutting: no overlap, so codes are exact


def _write_tiles(tile_dir: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Crushed tiles with different maxima, so per-tile normalisation would give each a different gain."""
    tiles = {}
    for i, top in enumerate((109, 60, 90, 40)):
        codes = np.clip(rng.gamma(2.0, top / 10.0, (_TILE, _TILE)), 0, top).astype(np.uint8)
        codes[0, 0] = top
        name = f'{i:03d}.png'
        noise_crush.save_crushed_png(os.path.join(tile_dir, name), codes, _PARAMS)
        tiles[name] = codes
    return tiles


def _mosaic(path: str, names: list[str]) -> None:
    mosaic = MosaicFile()
    for name, (x0, y0) in zip(names, _ORIGINS):
        mosaic.ImageToTransformString[name] = (
            f'GridTransform_double_2_2 vp 8 {x0} {y0} {x0 + _TILE} {y0} {x0} {y0 + _TILE} {x0 + _TILE} {y0 + _TILE} '
            f'fp 7 0 1 1 0 0 {_TILE} {_TILE}')
    mosaic.Save(path)


class TestCrushedAssembly(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.tile_dir = os.path.join(self.dir, '001')
        os.makedirs(self.tile_dir)
        self.tiles = _write_tiles(self.tile_dir, np.random.default_rng(3))
        self.mosaic_path = os.path.join(self.dir, 't.mosaic')
        _mosaic(self.mosaic_path, sorted(self.tiles))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_value_scale_load_keeps_a_shared_unit(self) -> None:
        path = os.path.join(self.tile_dir, '003.png')
        fixed = nornir_imageregistration.LoadImage(path, dtype=np.float32, value_scale=255, backend='numpy')
        np.testing.assert_allclose(fixed * 255, self.tiles['003.png'], atol=1e-3)
        stretched = nornir_imageregistration.LoadImage(path, dtype=np.float32, backend='numpy')
        self.assertAlmostEqual(float(stretched.max()), 1.0)  # unchanged default: per-image maximum

    def test_assembled_codes_match_the_tiles(self) -> None:
        mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(self.mosaic_path)
        tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
            mosaic, self.tile_dir, image_to_source_space_scale=1, tile_value_scale=_PARAMS.max_code)
        image, _mask = tileset.AssembleImage(FixedRegion=None, target_space_scale=1.0)
        codes = noise_crush.codes_from_unit_image(image, _PARAMS)
        self.assertEqual(codes.dtype, np.uint8)
        names = sorted(self.tiles)
        for name, (x0, y0) in zip(names, _ORIGINS):
            # Interior only: the outermost row/column can be interpolated against the background.
            got = codes[y0 + 2:y0 + _TILE - 2, x0 + 2:x0 + _TILE - 2]
            expected = self.tiles[name][2:-2, 2:-2]
            self.assertLessEqual(int(np.abs(got.astype(int) - expected).max()), 1, name)
            self.assertGreater(float(np.mean(got == expected)), 0.99, name)

    def test_shrink_keeps_the_crush_chunk(self) -> None:
        source = os.path.join(self.tile_dir, '000.png')
        target = os.path.join(self.dir, 'half.png')
        nornir_imageregistration.Shrink(source, target, 0.5)
        self.assertEqual(noise_crush.read_crush_params(target), _PARAMS)
        with Image.open(target) as im:
            self.assertLessEqual(int(np.asarray(im).max()), 109)

    def test_stamp_adds_the_chunk_without_changing_pixels(self) -> None:
        path = os.path.join(self.dir, 'plain.png')
        codes = self.tiles['001.png']
        Image.fromarray(codes).save(path)
        self.assertIsNone(noise_crush.read_crush_params(path))
        self.assertTrue(noise_crush.stamp_crush_png(path, _PARAMS))
        self.assertEqual(noise_crush.read_crush_params(path), _PARAMS)
        with Image.open(path) as im:
            np.testing.assert_array_equal(np.asarray(im), codes)
        self.assertFalse(noise_crush.stamp_crush_png(path, _PARAMS))  # idempotent

    def test_stamp_rejects_non_png(self) -> None:
        path = os.path.join(self.dir, 'x.png')
        with open(path, 'wb') as f:
            f.write(b'not a png')
        with self.assertRaises(ValueError):
            noise_crush.stamp_crush_png(path, _PARAMS)


if __name__ == '__main__':
    unittest.main()
