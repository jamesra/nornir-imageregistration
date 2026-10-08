"""Tests for Pillow mode bit-suffix parsing in pillow_helpers."""
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from nornir_imageregistration.pillow_helpers import (
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


class TestDtypeForPillowImageWithBitSuffix(unittest.TestCase):
    def test_i16_mode_uses_suffix(self) -> None:
        im = Image.new("I;16", (2, 2))
        self.assertEqual(dtype_for_pillow_image(im), np.uint16)


if __name__ == "__main__":
    unittest.main()
