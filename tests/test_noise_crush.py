"""Tests for noise-aware quantization (noise_crush): sigma estimate, codes, PNG metadata, flip."""
import os
import tempfile
import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import noise_crush
from nornir_imageregistration.noise_crush import CrushParams
from PIL import Image


class TestEstimateNoiseSigma(unittest.TestCase):

    def test_gaussian_noise(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.normal(1000, 10, (1024, 1024))
        self.assertAlmostEqual(noise_crush.estimate_noise_sigma(image), 10.0, delta=0.3)

    def test_smooth_structure_does_not_inflate_sigma(self) -> None:
        """A ramp and a broad blob are almost invisible to the finest Haar subband."""
        rng = np.random.default_rng(1)
        y, x = np.mgrid[0:1024, 0:1024]
        structure = 20.0 * x + 5000.0 * np.exp(-((x - 512) ** 2 + (y - 512) ** 2) / 2e5)
        image = structure + rng.normal(0, 10, structure.shape)
        self.assertAlmostEqual(noise_crush.estimate_noise_sigma(image), 10.0, delta=0.5)

    def test_poisson_noise(self) -> None:
        rng = np.random.default_rng(2)
        image = rng.poisson(1000, (1024, 1024)).astype(np.uint16)
        self.assertAlmostEqual(noise_crush.estimate_noise_sigma(image), np.sqrt(1000), delta=0.05 * np.sqrt(1000))

    def test_center_crop_bounds_work_on_large_tiles(self) -> None:
        rng = np.random.default_rng(3)
        image = rng.normal(0, 5, (4081, 3001))
        self.assertAlmostEqual(noise_crush.estimate_noise_sigma(image, crop=512), 5.0, delta=0.3)

    def test_section_sigma_is_median(self) -> None:
        self.assertEqual(noise_crush.section_sigma([10.0, 11.0, 500.0]), 11.0)
        with self.assertRaises(ValueError):
            noise_crush.section_sigma([])

    def test_sample_indices_spread_and_cap(self) -> None:
        self.assertEqual(noise_crush.sample_indices(5, 16), [0, 1, 2, 3, 4])
        picked = noise_crush.sample_indices(100, 16)
        self.assertEqual(len(picked), 16)
        self.assertEqual((picked[0], picked[-1]), (0, 99))


class TestCrushParams(unittest.TestCase):

    def test_step_from_k_sigma(self) -> None:
        params = noise_crush.make_crush_params(316.5, 32754, np.uint16, k=2.5)
        self.assertEqual(params.step, 791)
        self.assertEqual((params.offset, params.bits, params.storage_dtype), (0, 8, np.dtype(np.uint8)))

    def test_bits_switch_to_uint16_past_255_codes(self) -> None:
        self.assertEqual(noise_crush.make_crush_params(4.0, 255 * 10 + 9, np.uint16, k=2.5).bits, 8)
        params = noise_crush.make_crush_params(4.0, 256 * 10, np.uint16, k=2.5)
        self.assertEqual((params.bits, params.storage_dtype), (9, np.dtype(np.uint16)))

    def test_signed_data_offsets_by_section_minimum(self) -> None:
        params = noise_crush.make_crush_params(10.0, 5000, np.int16, k=2.0, section_min=-123.4)
        self.assertEqual(params.offset, -124)
        self.assertEqual(noise_crush.make_crush_params(10.0, 5000, np.uint16, section_min=-123.4).offset, 0)

    def test_rejects_bad_sigma(self) -> None:
        for sigma in (0.0, -1.0, float('nan')):
            with self.assertRaises(ValueError):
                noise_crush.make_crush_params(sigma, 1000, np.uint16)

    def test_json_and_attribute_round_trip(self) -> None:
        params = noise_crush.make_crush_params(316.53076352853964, 32754, np.uint16)
        self.assertEqual(CrushParams.from_json(params.to_json()), params)
        self.assertEqual(CrushParams.from_attributes(params.to_attributes()), params)
        self.assertIsNone(CrushParams.from_attributes({'MinIntensityCutoff': '1'}))

    def test_same_quantization_ignores_sigma_digits(self) -> None:
        a = CrushParams(k=2.5, step=791, sigma=316.5, offset=0, bits=8, source_dtype='uint16')
        b = CrushParams(k=2.5, step=791, sigma=316.49, offset=0, bits=8, source_dtype='uint16')
        c = CrushParams(k=3.0, step=950, sigma=316.5, offset=0, bits=8, source_dtype='uint16')
        self.assertTrue(a.same_quantization(b))
        self.assertFalse(a.same_quantization(c))
        self.assertFalse(a.same_quantization(None))


class TestCrushCodes(unittest.TestCase):

    def test_uncrush_error_within_half_step(self) -> None:
        rng = np.random.default_rng(4)
        raw = rng.integers(0, 30000, (256, 256), dtype=np.uint16)
        params = noise_crush.make_crush_params(300.0, 30000, np.uint16)
        codes = noise_crush.crush(raw, params)
        self.assertEqual(codes.dtype, np.uint8)
        error = noise_crush.uncrush(codes, params) - raw
        self.assertLessEqual(float(np.abs(error).max()), params.step / 2.0)

    def test_out_of_range_values_clip(self) -> None:
        params = CrushParams(k=2.5, step=10, sigma=4.0, offset=-50, bits=8, source_dtype='int16')
        raw = np.array([[-200, -50, 0, 30000]], dtype=np.int16)
        self.assertEqual(noise_crush.crush(raw, params).tolist(), [[0, 0, 5, 255]])


class TestCrushedPng(unittest.TestCase):

    def test_png_round_trip_uint8_and_uint16(self) -> None:
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            for bits in (8, 12):
                params = CrushParams(k=2.5, step=3, sigma=1.2, offset=0, bits=bits, source_dtype='uint16')
                codes = rng.integers(0, params.max_code + 1, (64, 48)).astype(params.storage_dtype)
                path = os.path.join(tmp, f'{bits}.png')
                noise_crush.save_crushed_png(path, codes, params)
                np.testing.assert_array_equal(noise_crush.load_raw_tile(path), codes)
                self.assertEqual(noise_crush.read_crush_params(path), params)

    def test_read_params_of_plain_png_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'plain.png')
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
            self.assertIsNone(noise_crush.read_crush_params(path))

    def test_tile_file_flip_matches_convert_images_flip(self) -> None:
        """crush_tile_to_file(flip=True) must mirror rows exactly as ConvertImagesInDict's Flip does."""
        rng = np.random.default_rng(6)
        raw = rng.integers(0, 4000, (40, 30), dtype=np.uint16)
        raw[0, :] = 3999  # make the orientation unambiguous
        params = noise_crush.make_crush_params(5.0, 4000, np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'source.tif')
            Image.fromarray(raw).save(source)
            target = os.path.join(tmp, 'sub', 'target.png')
            noise_crush.crush_tile_to_file(source, target, params, flip=True)
            converted = nornir_imageregistration.core._ConvertSingleImage(raw, Flip=True)
            np.testing.assert_array_equal(noise_crush.load_raw_tile(target), noise_crush.crush(converted, params))


if __name__ == '__main__':
    unittest.main()
