"""
Tests that TilesToImageThreaded with CuPy (GPU) backend:
  1. completes without deadlock
  2. produces pixel-identical output to TilesToImage (serial path)

These tests create a small synthetic mosaic so no test-data files are needed.
"""

import os
import tempfile
import unittest

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as assemble_tiles
import nornir_imageregistration.mosaic_tileset
import nornir_imageregistration.transforms.factory as factory
from nornir_imageregistration.transforms.rigid import Rigid


def _build_synthetic_mosaic(tmp_dir: str, n_tiles: int, tile_shape=(64, 64)) -> nornir_imageregistration.mosaic_tileset.MosaicTileset:
    """
    Build a MosaicTileset of n_tiles arranged in a horizontal row.
    Each tile contains a unique constant value so compositing can be verified.
    """
    transforms = []
    image_paths = []
    stride = float(tile_shape[1]) * 0.9  # 10% overlap

    for i in range(n_tiles):
        img = np.full(tile_shape, fill_value=float(i + 1) / n_tiles, dtype=np.float32)
        png_path = os.path.join(tmp_dir, f"{i}.png")
        nornir_imageregistration.SaveImage(png_path, img, bpp=8)
        image_paths.append(png_path)

        t = Rigid(target_offset=(0.0, i * stride))
        transforms.append(t)

    tileset = nornir_imageregistration.mosaic_tileset.Create(
        transforms, image_paths, image_to_source_space_scale=1.0
    )
    return tileset


class TestAssembleGpuThreaded(unittest.TestCase):

    _EDGE_CROP = 2  # pixels to strip from each image edge before comparison

    def _run_both_paths(self, n_tiles: int, backend: nornir_imageregistration.ComputationLib):
        """Assemble the same mosaic with serial and parallel paths; return both results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nornir_imageregistration.SetActiveComputationLib(backend)
            tileset = _build_synthetic_mosaic(tmp_dir, n_tiles=n_tiles)

            # Serial path
            img_serial, mask_serial = assemble_tiles.TilesToImage(tileset)

            # Thread-parallel GPU path (production default when CuPy is active)
            img_parallel, mask_parallel = assemble_tiles.TilesToImageThreaded(tileset)

        return img_serial, mask_serial, img_parallel, mask_parallel

    def _compare_images(self, img_s, img_p, msg=""):
        """Compare two images ignoring up to _EDGE_CROP pixels at each border."""
        img_s = np.nan_to_num(np.asarray(img_s, dtype=np.float32))
        img_p = np.nan_to_num(np.asarray(img_p, dtype=np.float32))
        c = self._EDGE_CROP
        h = min(img_s.shape[0], img_p.shape[0]) - c
        w = min(img_s.shape[1], img_p.shape[1]) - c
        np.testing.assert_allclose(img_s[c:h, c:w], img_p[c:h, c:w], atol=1e-3,
                                   err_msg=msg or "serial and parallel outputs differ")

    def test_serial_and_parallel_numpy_identical(self):
        """CPU serial and parallel produce the same output (interior pixels)."""
        img_s, mask_s, img_p, mask_p = self._run_both_paths(
            n_tiles=4, backend=nornir_imageregistration.ComputationLib.numpy
        )
        self.assertIsNotNone(img_s)
        self.assertIsNotNone(img_p)
        self._compare_images(img_s, img_p, "CPU serial and parallel outputs differ")

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_serial_and_parallel_cupy_no_deadlock(self):
        """GPU parallel path completes without deadlock."""
        img_s, mask_s, img_p, mask_p = self._run_both_paths(
            n_tiles=8, backend=nornir_imageregistration.ComputationLib.cupy
        )
        self.assertIsNotNone(img_s)
        self.assertIsNotNone(img_p)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_serial_and_parallel_cupy_pixel_identical(self):
        """GPU serial and parallel paths produce the same output (interior pixels)."""
        img_s, mask_s, img_p, mask_p = self._run_both_paths(
            n_tiles=4, backend=nornir_imageregistration.ComputationLib.cupy
        )
        self._compare_images(img_s, img_p, "GPU serial and parallel outputs differ")

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_cupy_parallel_consistent_with_numpy_serial(self):
        """GPU parallel output is close (within bilinear rounding) to CPU serial output.

        CPU and GPU use the same bilinear interpolation but may differ by up to 1 pixel
        at image edges due to floating-point rounding in coordinate transforms. Crop
        a shared interior region for comparison.
        """
        _EDGE_CROP = 2  # pixels to strip from each edge before comparison
        with tempfile.TemporaryDirectory() as tmp_dir:
            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
            tileset = _build_synthetic_mosaic(tmp_dir, n_tiles=4)
            img_cpu, _ = assemble_tiles.TilesToImage(tileset)

            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
            tileset2 = _build_synthetic_mosaic(tmp_dir, n_tiles=4)
            img_gpu, _ = assemble_tiles.TilesToImageThreaded(tileset2)

        img_cpu = np.nan_to_num(np.asarray(img_cpu, dtype=np.float32))
        img_gpu = np.nan_to_num(np.asarray(img_gpu, dtype=np.float32))
        # Crop to common interior, skipping edge pixels that legitimately differ
        h = min(img_cpu.shape[0], img_gpu.shape[0]) - _EDGE_CROP
        w = min(img_cpu.shape[1], img_gpu.shape[1]) - _EDGE_CROP
        np.testing.assert_allclose(
            img_cpu[_EDGE_CROP:h, _EDGE_CROP:w],
            img_gpu[_EDGE_CROP:h, _EDGE_CROP:w],
            atol=2e-2,
            err_msg="CPU serial and GPU parallel outputs differ beyond tolerance (interior crop)"
        )


if __name__ == "__main__":
    unittest.main()
