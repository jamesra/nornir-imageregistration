"""Integer images scaled to 0..1 floats by ImageParamToImageArray / ImageParamToNumpyImageArray."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import nornir_imageregistration
import numpy as np
from nornir_imageregistration.core import _core

_SLAB = "nornir_imageregistration.core._core._INT_TO_FLOAT_SLAB_ELEMENTS"
_CONVERTERS = (nornir_imageregistration.ImageParamToImageArray,
               nornir_imageregistration.ImageParamToNumpyImageArray)


def _reference(image: np.ndarray, dtype: type) -> np.ndarray:
    """Scale in float64 and round once to *dtype*, returned as float64 for comparison."""
    return (image.astype(np.float64) / np.iinfo(image.dtype).max).astype(dtype).astype(np.float64)


class TestIntImageToUnitFloat(unittest.TestCase):
    def test_16bit_and_wider_into_float16_do_not_overflow(self) -> None:
        """The integer maximum (65535 for uint16) does not fit float16, whose limit is 65504."""
        for int_dtype in (np.uint16, np.int32, np.uint32):
            top = np.iinfo(int_dtype).max
            image = np.array([[0, top // 3], [top // 2, top]], dtype=int_dtype)
            for convert in _CONVERTERS:
                with np.errstate(over="raise", invalid="raise"):
                    scaled = convert(image, dtype=np.float16)
                self.assertEqual(scaled.dtype, np.float16, (convert.__name__, int_dtype))
                np.testing.assert_allclose(scaled.astype(np.float64), _reference(image, np.float16),
                                           rtol=1e-3, err_msg=f"{convert.__name__} {int_dtype}")
                self.assertEqual(float(scaled.max()), 1.0)

    def test_8bit_into_float16_is_unchanged(self) -> None:
        """8-bit images keep the original cast-then-divide result, bit for bit."""
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        for convert in _CONVERTERS:
            np.testing.assert_array_equal(convert(image, dtype=np.float16),
                                          image.astype(np.float16) / 255)

    def test_16bit_into_float32_is_unchanged(self) -> None:
        image = np.arange(0, 65536, 257, dtype=np.uint16).reshape(16, 16)
        for convert in _CONVERTERS:
            np.testing.assert_array_equal(convert(image, dtype=np.float32),
                                          image.astype(np.float32) / 65535)

    def test_rows_are_scaled_in_slabs(self) -> None:
        """A slab smaller than the image still fills every row."""
        image = (np.arange(30, dtype=np.uint16) * 2000).reshape(10, 3)
        with patch(_SLAB, 7):
            scaled = _core._int_image_to_unit_float(image, np.float16)
        np.testing.assert_allclose(scaled.astype(np.float64), _reference(image, np.float16), rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
