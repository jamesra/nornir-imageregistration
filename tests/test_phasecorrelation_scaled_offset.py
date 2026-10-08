"""Hypothesis coverage for fftshift parity in peak offset from center of mass."""
from __future__ import annotations

import unittest

import hypothesis
import hypothesis.strategies as st
import numpy as np

from nornir_imageregistration.phasecorrelation import _scaled_offset_from_center_of_mass

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None


def _cupy_available() -> bool:
    if cp is None:
        return False
    if getattr(cp, "__name__", "") == "nornir_imageregistration.cupy_thunk":
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _reference_scaled_offset(
    image_shape: tuple[int, int], peak_center_of_mass: tuple[float, float]
) -> tuple[float, float]:
    """Integer half-index matches ``numpy.fft.fftshift`` DC placement (#238)."""
    cy = image_shape[0] // 2
    cx = image_shape[1] // 2
    return (
        float(cy) - float(peak_center_of_mass[0]),
        float(cx) - float(peak_center_of_mass[1]),
    )


def _odd_dimension(min_side: int, max_side: int) -> st.SearchStrategy[int]:
    return st.integers(min_value=min_side, max_value=max_side).map(lambda n: 2 * n + 1)


@st.composite
def fft_frame_shape_and_com(draw: st.DrawFn) -> tuple[tuple[int, int], tuple[float, float]]:
    height = draw(st.integers(min_value=3, max_value=257))
    width = draw(st.integers(min_value=3, max_value=257))
    com_y = draw(
        st.floats(
            min_value=0.0,
            max_value=float(height - 1),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    com_x = draw(
        st.floats(
            min_value=0.0,
            max_value=float(width - 1),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return (height, width), (com_y, com_x)


@st.composite
def odd_fft_frame_shape_and_com(draw: st.DrawFn) -> tuple[tuple[int, int], tuple[float, float]]:
    height = draw(_odd_dimension(1, 128))
    width = draw(_odd_dimension(1, 128))
    com_y = draw(
        st.floats(
            min_value=0.0,
            max_value=float(height - 1),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    com_x = draw(
        st.floats(
            min_value=0.0,
            max_value=float(width - 1),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return (height, width), (com_y, com_x)


class TestScaledOffsetFromCenterOfMass(unittest.TestCase):
    @hypothesis.settings(deadline=None)
    @hypothesis.given(fft_frame_shape_and_com())
    def test_numpy_matches_integer_fftshift_center(
        self, shape_and_com: tuple[tuple[int, int], tuple[float, float]]
    ) -> None:
        image_shape, peak_com = shape_and_com
        expected = _reference_scaled_offset(image_shape, peak_com)
        result = _scaled_offset_from_center_of_mass(image_shape, peak_com, np)
        self.assertEqual(result, expected)

    @hypothesis.settings(deadline=None)
    @hypothesis.given(fft_frame_shape_and_com())
    def test_cupy_matches_numpy(
        self, shape_and_com: tuple[tuple[int, int], tuple[float, float]]
    ) -> None:
        if not _cupy_available():
            self.skipTest("CuPy not available")
        image_shape, peak_com = shape_and_com
        np_result = _scaled_offset_from_center_of_mass(image_shape, peak_com, np)
        cp_result = _scaled_offset_from_center_of_mass(image_shape, peak_com, cp)
        self.assertEqual(cp_result, np_result)

    @hypothesis.example(((31, 31), (15.0, 15.0)))
    @hypothesis.example(((33, 17), (16.5, 8.25)))
    @hypothesis.settings(deadline=None)
    @hypothesis.given(odd_fft_frame_shape_and_com())
    def test_odd_frames_reject_float_half_center_bias(
        self, shape_and_com: tuple[tuple[int, int], tuple[float, float]]
    ) -> None:
        image_shape, peak_com = shape_and_com
        result = _scaled_offset_from_center_of_mass(image_shape, peak_com, np)
        float_half = (
            (image_shape[0] / 2.0) - peak_com[0],
            (image_shape[1] / 2.0) - peak_com[1],
        )
        self.assertNotEqual(result, float_half)


if __name__ == "__main__":
    unittest.main()
