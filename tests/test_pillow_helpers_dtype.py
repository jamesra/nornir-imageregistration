"""Tests for Pillow mode bit-suffix parsing and extrema dtype estimation in pillow_helpers."""
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from nornir_imageregistration.pillow_helpers import (
    _try_estimate_dtype_from_extrema,
    _try_read_bpp_from_pillow_mode,
    dtype_for_pillow_image,
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


if __name__ == "__main__":
    unittest.main()
