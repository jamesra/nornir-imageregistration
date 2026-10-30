"""Tests for Pillow mode bit-suffix parsing and extrema dtype estimation in pillow_helpers."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from nornir_imageregistration.pillow_helpers import (
    _try_estimate_dtype_from_extrema,
    _try_read_bpp_from_pillow_mode,
    dtype_for_pillow_image,
    get_image_file_dtype,
    load_image_array,
)


class TestTryReadBppFromPillowMode(unittest.TestCase):
    def test_mode_string_with_bit_suffix(self) -> None:
        self.assertEqual(_try_read_bpp_from_pillow_mode("I;16"), 16)

    def test_image_with_bit_suffix_mode(self) -> None:
        im = mock.Mock(mode="I;16")
        self.assertEqual(_try_read_bpp_from_pillow_mode(im), 16)

    def test_plain_integer_mode_without_suffix(self) -> None:
        self.assertIsNone(_try_read_bpp_from_pillow_mode("I"))

    def test_non_numeric_bit_suffix_returns_none(self) -> None:
        self.assertIsNone(_try_read_bpp_from_pillow_mode("I;notanint"))

    def test_unexpected_parse_error_propagates(self) -> None:
        with (
            mock.patch(
                "nornir_imageregistration.pillow_helpers.int",
                side_effect=RuntimeError("unexpected"),
            ),
            self.assertRaises(RuntimeError),
        ):
            _try_read_bpp_from_pillow_mode("I;16")


class TestTryEstimateDtypeFromExtrema(unittest.TestCase):
    def test_i_mode_uint8_range(self) -> None:
        arr = np.array([[0, 200]], dtype=np.int32)
        im = Image.fromarray(arr, mode="I")
        self.assertIs(_try_estimate_dtype_from_extrema(im), np.uint8)

    def test_i_mode_negative_min_in_uint8_range_raises(self) -> None:
        im = mock.Mock(mode="I")
        im.getextrema.return_value = (-1, 100)
        with self.assertRaises(ValueError):
            _try_estimate_dtype_from_extrema(im)

    def test_i_mode_uint16_unsigned(self) -> None:
        im = mock.Mock(mode="I")
        im.getextrema.return_value = (0, 40000)
        self.assertIs(_try_estimate_dtype_from_extrema(im), np.uint16)

    def test_i_mode_int16_signed(self) -> None:
        im = mock.Mock(mode="I")
        im.getextrema.return_value = (-100, 40000)
        self.assertIs(_try_estimate_dtype_from_extrema(im), np.int16)

    def test_i_mode_int32_signed(self) -> None:
        im = mock.Mock(mode="I")
        im.getextrema.return_value = (-1, (1 << 20))
        self.assertIs(_try_estimate_dtype_from_extrema(im), np.int32)

    def test_i_mode_int64_when_extrema_exceed_uint32(self) -> None:
        im = mock.Mock(mode="I")
        im.getextrema.return_value = (0, 1 << 32)
        self.assertIs(_try_estimate_dtype_from_extrema(im), np.int64)

    def test_i_mode_uint32_unsigned(self) -> None:
        im = mock.Mock(mode="I")
        im.getextrema.return_value = (0, (1 << 20))
        self.assertIs(_try_estimate_dtype_from_extrema(im), np.uint32)

    def test_f_mode_returns_float32(self) -> None:
        im = mock.Mock(mode="F")
        im.getextrema.return_value = (0.0, 1.0)
        self.assertIs(_try_estimate_dtype_from_extrema(im), np.float32)

    def test_invalid_mode_prefix_raises(self) -> None:
        with self.assertRaises(ValueError):
            _try_estimate_dtype_from_extrema("RGB")


class TestDtypeForPillowImage(unittest.TestCase):
    def test_i_mode_without_bpp_uses_extrema(self) -> None:
        arr = np.array([[10, 50]], dtype=np.int32)
        im = Image.fromarray(arr, mode="I")
        self.assertIs(dtype_for_pillow_image(im), np.uint8)

    def test_i16_mode_uses_suffix(self) -> None:
        im = Image.new("I;16", (2, 2))
        self.assertEqual(dtype_for_pillow_image(im), np.uint16)

    def test_common_uint8_modes(self) -> None:
        cases: list[tuple[str, Image.Image]] = [
            ("L", Image.new("L", (1, 1), 128)),
            ("RGB", Image.new("RGB", (1, 1), (1, 2, 3))),
            ("RGBA", Image.new("RGBA", (1, 1), (1, 2, 3, 255))),
        ]
        for _label, im in cases:
            with self.subTest(mode=im.mode):
                self.assertIs(dtype_for_pillow_image(im), np.uint8)

    def test_binary_mode_is_bool(self) -> None:
        im = Image.new("1", (1, 1), 0)
        self.assertIs(dtype_for_pillow_image(im), bool)

    def test_palette_mode_is_uint8(self) -> None:
        im = Image.new("P", (2, 2))
        im.putpalette([i % 256 for i in range(768)])
        self.assertIs(dtype_for_pillow_image(im), np.uint8)

    def test_i_suffix_selects_width_without_extrema(self) -> None:
        for mode, expected in (
            ("I;8", np.uint8),
            ("I;1", bool),
            ("I;32", np.uint32),
        ):
            with self.subTest(mode=mode):
                im = mock.Mock(mode=mode)
                self.assertIs(dtype_for_pillow_image(im), expected)

    def test_float_mode_default_and_suffix(self) -> None:
        arr = np.array([[0.25, 0.75]], dtype=np.float32)
        im = Image.fromarray(arr, mode="F")
        self.assertIs(dtype_for_pillow_image(im), np.float32)
        im16 = mock.Mock(mode="F;16")
        self.assertIs(dtype_for_pillow_image(im16), np.float16)

    def test_unexpected_mode_raises(self) -> None:
        im = mock.Mock(mode="XY")
        with self.assertRaises(ValueError) as ctx:
            dtype_for_pillow_image(im)
        self.assertIn("Unexpected pillow image mode", str(ctx.exception))


class TestPillowHelpersFilePaths(unittest.TestCase):
    def test_get_image_file_dtype_reads_mode_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gray.png"
            Image.new("L", (2, 2), 42).save(path)
            self.assertIs(get_image_file_dtype(str(path)), np.uint8)

    def test_load_image_array_returns_pixel_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            Image.new("RGB", (1, 2), (10, 20, 30)).save(path)
            arr = load_image_array(str(path))
            self.assertEqual(arr.dtype, np.uint8)
            self.assertEqual(arr.ndim, 3)
            self.assertEqual(arr.shape[-1], 3)
            self.assertTrue((arr == np.array([10, 20, 30], dtype=np.uint8)).all())


if __name__ == "__main__":
    unittest.main()
