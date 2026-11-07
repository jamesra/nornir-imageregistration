"""Gain-first crush: a tile is divided by its brightness gain before it is quantized; the gain is only a diagnostic."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from nornir_imageregistration import noise_crush, tile_gains
from nornir_imageregistration.noise_crush import CrushParams

PARAMS = CrushParams(k=2.5, step=40, sigma=16.0, offset=0, bits=8, source_dtype='uint16')


def _scene(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Texture whose contrast is several steps wide: bright counts around 4000 with a 600-count swing."""
    rng = np.random.default_rng(3)
    smooth = np.kron(rng.normal(size=(shape[0] // 8, shape[1] // 8)), np.ones((8, 8)))
    return 4000 + 300 * smooth


def _capture(scene: np.ndarray, gain: float, seed: int) -> np.ndarray:
    """What a tile dimmed by *gain* records: texture scaled by the gain, shot noise by about its square root."""
    rng = np.random.default_rng(seed)
    sigma = 16.0 * np.sqrt(gain)
    return np.clip(scene * gain + rng.normal(0, sigma, scene.shape), 0, 65535).astype(np.uint16)


class TestCrushWithGain(unittest.TestCase):

    def test_unit_gain_is_the_plain_crush(self) -> None:
        raw = _capture(_scene(), 1.0, 1)
        np.testing.assert_array_equal(noise_crush.crush(raw, PARAMS, 1.0), noise_crush.crush(raw, PARAMS))

    def test_codes_are_those_of_the_gain_corrected_counts(self) -> None:
        raw = _capture(_scene(), 0.25, 1)
        expected = np.floor(raw.astype(np.float64) / 0.25 / PARAMS.step).astype(np.uint8)
        np.testing.assert_array_equal(noise_crush.crush(raw, PARAMS, 0.25), expected)

    def test_a_dim_tile_keeps_its_texture(self) -> None:
        scene = _scene()
        dim = _capture(scene, 0.15, 2)
        as_captured = np.unique(noise_crush.crush(dim, PARAMS)).size
        gain_first = np.unique(noise_crush.crush(dim, PARAMS, 0.15)).size
        self.assertLess(as_captured, 12)
        self.assertGreater(gain_first, 2 * as_captured)

    def test_signed_offset_is_applied_after_the_gain(self) -> None:
        params = CrushParams(k=2.5, step=40, sigma=16.0, offset=-200, bits=8, source_dtype='int16')
        raw = np.array([[-50, 100, 1000]], dtype=np.int16)
        np.testing.assert_array_equal(noise_crush.crush(raw, params, 0.5),
                                      np.floor((raw / 0.5 + 200) / 40).clip(0).astype(np.uint8))


class TestImportGainChunk(unittest.TestCase):

    def test_chunk_records_the_gain_and_still_reads_as_params(self) -> None:
        text = noise_crush.crush_chunk_text(PARAMS, 0.37, True)
        values = json.loads(text)
        self.assertEqual(values[noise_crush.IMPORT_GAIN_KEY], 0.37)
        self.assertTrue(values[noise_crush.IMPORT_GAIN_CORRECTED_KEY])
        self.assertEqual(CrushParams.from_json(text), PARAMS)

    def test_chunk_without_a_gain_is_the_plain_params(self) -> None:
        self.assertEqual(noise_crush.crush_chunk_text(PARAMS), PARAMS.to_json())

    def test_assembly_does_not_correct_with_the_import_gain(self) -> None:
        """The codes already are gain-corrected: ``unit_gain_correction`` reads only TileGains' ``gain`` key."""
        self.assertIsNone(noise_crush.unit_gain_correction(noise_crush.crush_chunk_text(PARAMS, 0.2, True)))

    def test_tile_gains_stamp_keeps_the_import_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 't.png')
            codes = noise_crush.crush(_capture(_scene(), 0.5, 4), PARAMS, 0.5)
            noise_crush.save_crushed_png(path, codes, PARAMS, noise_crush.crush_chunk_text(PARAMS, 0.5, True))
            self.assertTrue(tile_gains.stamp_gain_png(path, 1.25))
            with Image.open(path) as im:
                values = json.loads(im.text[noise_crush.PNG_TEXT_KEY])
            self.assertEqual(values[noise_crush.IMPORT_GAIN_KEY], 0.5)
            self.assertEqual(values['gain'], 1.25)

    def test_crush_tile_to_file_records_applied_and_uncorrected_gains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 's.tif')
            Image.fromarray(_capture(_scene(), 0.5, 5)).save(source)
            corrected = os.path.join(tmp, 'a.png')
            flagged = os.path.join(tmp, 'b.png')
            noise_crush.crush_tile_to_file(source, corrected, PARAMS, False, 0.5, 0.5)
            noise_crush.crush_tile_to_file(source, flagged, PARAMS, False, 1.0, 0.02)
            with Image.open(corrected) as a, Image.open(flagged) as b:
                first, second = (json.loads(im.text[noise_crush.PNG_TEXT_KEY]) for im in (a, b))
            self.assertEqual((first['import_gain'], first['import_gain_corrected']), (0.5, True))
            self.assertEqual((second['import_gain'], second['import_gain_corrected']), (0.02, False))
            np.testing.assert_array_equal(noise_crush.load_raw_tile(corrected),
                                          noise_crush.crush(noise_crush.load_raw_tile(source), PARAMS, 0.5))


if __name__ == '__main__':
    unittest.main()
