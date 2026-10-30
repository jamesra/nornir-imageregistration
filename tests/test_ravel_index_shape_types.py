"""ravel_index takes the shape as a tuple (``image.shape``) and matches ``numpy.ravel_multi_index``.

Callers pass shape tuples: ``assemble`` (``mask.shape`` / ``outputImage.shape``) and
``index_with_array`` (``image.shape``). ``index_with_array`` once passed ``xp.asarray(image.shape)``,
which ``cupy.ravel_multi_index`` rejects with TypeError, so CuPy images are covered below.
"""

import unittest

import nornir_imageregistration
import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st
from nornir_imageregistration.core._core import index_with_array
from numpy.typing import NDArray

try:
    import cupy

    _HAVE_CUPY = bool(cupy.cuda.is_available())
except Exception:  # noqa: BLE001 - any CuPy import/driver failure means no GPU check
    _HAVE_CUPY = False


@st.composite
def _shape_and_coords(draw: st.DrawFn) -> tuple[tuple[int, ...], NDArray[np.int64]]:
    """Draw a 2D or 3D shape (sizes up to real tile/mosaic magnitudes) and in-bounds coordinates."""
    ndim = draw(st.integers(min_value=2, max_value=3))
    max_dim = 70000 if ndim == 2 else 600
    # A 3D shape with a leading 1 takes ravel_index's shp[0] == 1 shortcut (returns column 1), which
    # only matches ravel_multi_index for 2D shapes; no caller passes such a 3D shape, so it is excluded.
    first_min = 1 if ndim == 2 else 2
    shape = (draw(st.integers(min_value=first_min, max_value=max_dim)),) + tuple(
        draw(st.integers(min_value=1, max_value=max_dim)) for _ in range(ndim - 1))
    n = draw(st.integers(min_value=1, max_value=20))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    coords = np.stack([rng.integers(0, dim, size=n) for dim in shape], axis=1).astype(np.int64)
    return shape, coords


class TestRavelIndexShapeTypes(unittest.TestCase):
    """Tuple shapes give results that match numpy.ravel_multi_index on NumPy and CuPy."""

    @given(_shape_and_coords())
    @example(((1, 9), np.array([[0, 0], [0, 8], [0, 4]], dtype=np.int64)))
    @example(((7, 1), np.array([[0, 0], [6, 0]], dtype=np.int64)))
    @example(((65536, 65536), np.array([[65535, 65535], [1, 0]], dtype=np.int64)))
    @settings(deadline=None, max_examples=200)
    def test_tuple_shape_matches_reference(
            self, case: tuple[tuple[int, ...], NDArray[np.int64]]) -> None:
        """ravel_index(coords, shape) equals ravel_multi_index for a tuple shape."""
        shape, coords = case
        expected = np.ravel_multi_index(tuple(coords.T), shape)

        np.testing.assert_array_equal(nornir_imageregistration.ravel_index(coords, shape), expected)

    def test_index_with_array_reads_pixels(self) -> None:
        """index_with_array returns image values at (row, col) coords."""
        image = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
        coords = np.array([[0, 0], [4, 6], [2, 3], [1, 6]], dtype=np.int64)
        np.testing.assert_array_equal(index_with_array(image, coords), image[coords[:, 0], coords[:, 1]])

    @unittest.skipUnless(_HAVE_CUPY, "CuPy with a CUDA device is not available")
    def test_cupy_coords_tuple_shape_match_numpy(self) -> None:
        """CuPy coords with a tuple shape give the same flat indices as the NumPy path."""
        shape = (300, 517)
        coords = np.array([[0, 0], [299, 516], [123, 45]], dtype=np.int64)
        expected = nornir_imageregistration.ravel_index(coords, shape)
        result = nornir_imageregistration.ravel_index(cupy.asarray(coords), shape)
        np.testing.assert_array_equal(cupy.asnumpy(result), expected)

    @unittest.skipUnless(_HAVE_CUPY, "CuPy with a CUDA device is not available")
    def test_index_with_array_cupy_image(self) -> None:
        """index_with_array on a CuPy image and CuPy coords stays on the GPU and matches NumPy.

        Regression: the shape used to be passed as ``cupy.asarray(image.shape)``, which made
        ``cupy.ravel_multi_index`` raise TypeError (reachable from
        ``GridDivisionBase.ApplyTargetImageMask`` / ``ApplySourceImageMask`` with a CuPy mask).
        """
        host_image = np.arange(300 * 517, dtype=np.float32).reshape(300, 517)
        host_coords = np.array([[0, 0], [299, 516], [123, 45], [7, 400]], dtype=np.int64)
        result = index_with_array(cupy.asarray(host_image), cupy.asarray(host_coords))
        self.assertIsInstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), host_image[host_coords[:, 0], host_coords[:, 1]])

    @unittest.skipUnless(_HAVE_CUPY, "CuPy with a CUDA device is not available")
    def test_index_with_array_cupy_bool_mask(self) -> None:
        """A CuPy boolean mask (the grid_subdivision use) returns the mask values at the coords."""
        host_mask = np.zeros((64, 96), dtype=bool)
        host_mask[10:20, 30:50] = True
        host_coords = np.array([[15, 40], [0, 0], [19, 49], [20, 50]], dtype=np.int64)
        result = index_with_array(cupy.asarray(host_mask), cupy.asarray(host_coords))
        np.testing.assert_array_equal(cupy.asnumpy(result), [True, False, True, False])


if __name__ == '__main__':
    unittest.main()
