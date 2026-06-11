import os
import importlib
import tempfile
import unittest

import numpy as np

import nornir_imageregistration


def _blob_filter_module():
    return importlib.import_module("nornir_imageregistration.blob_filter")


def _cp_for_get_array_module():
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import nornir_imageregistration.cupy_thunk as cp
    except ImportError:
        import nornir_imageregistration.cupy_thunk as cp
    return cp


class TestBlobFilter(unittest.TestCase):

    def _sample_image(self) -> np.ndarray:
        image = np.zeros((64, 64), dtype=np.float32)
        image[8:56, 8:56] = 0.5
        image[24:40, 24:40] = 0.95
        image += np.linspace(0.0, 0.2, num=image.shape[1], dtype=np.float32)[None, :]
        image[::2, ::2] += 0.15
        return np.clip(image, 0.0, 1.0)

    def test_blob_filter_numpy_backend(self):
        cp = _cp_for_get_array_module()
        image = self._sample_image()
        output, diagnostics = _blob_filter_module().BlobFilter(
            image,
            radius=2,
            median_radius=1,
            max_value=3.0,
            return_diagnostics=True)

        self.assertEqual(output.shape, image.shape)
        self.assertIs(cp.get_array_module(output), np)
        self.assertEqual(diagnostics.backend, "numpy")
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_blob_filter_respects_mask(self):
        image = self._sample_image()
        mask = np.ones_like(image, dtype=bool)
        mask[10:20, 10:20] = False
        output, diagnostics = _blob_filter_module().BlobFilter(
            image,
            radius=2,
            median_radius=1,
            max_value=3.0,
            mask=mask,
            return_diagnostics=True)

        self.assertEqual(int(np.count_nonzero(output[10:20, 10:20])), 0)
        self.assertGreater(diagnostics.masked_pixel_count, 0)

    @unittest.skipIf(not nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_blob_filter_cupy_matches_numpy(self):
        import cupy as cp

        image_np = self._sample_image()
        image_cp = cp.asarray(image_np)
        mask_np = np.ones_like(image_np, dtype=bool)
        mask_np[0:4, 0:4] = False
        mask_cp = cp.asarray(mask_np)

        out_cp, diag_cp = _blob_filter_module().BlobFilter(
            image_cp,
            radius=2,
            median_radius=1,
            max_value=3.0,
            mask=mask_cp,
            return_diagnostics=True)
        out_np, _diag_np = _blob_filter_module().BlobFilter(
            image_np,
            radius=2,
            median_radius=1,
            max_value=3.0,
            mask=mask_np,
            return_diagnostics=True)

        self.assertIs(cp.get_array_module(out_cp), cp)
        self.assertEqual(diag_cp.backend, "cupy")
        np.testing.assert_allclose(cp.asnumpy(out_cp), out_np, atol=5e-3, rtol=5e-3)

    @unittest.skipIf(not nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_blob_filter_cupy_large_masked_image(self):
        import cupy as cp

        size = 1024
        image_np = np.zeros((size, size), dtype=np.float32)
        image_np[64:size - 64, 64:size - 64] = 0.5
        image_np[256:768, 256:768] = 0.85
        image_np += np.linspace(0.0, 0.15, num=size, dtype=np.float32)[None, :]
        image_np = np.clip(image_np, 0.0, 1.0)

        mask_np = np.ones((size, size), dtype=bool)
        mask_np[0:32, :] = False
        mask_np[:, 0:32] = False
        mask_np[512:544, 512:544] = False

        image_cp = cp.asarray(image_np)
        mask_cp = cp.asarray(mask_np)

        output, diagnostics = _blob_filter_module().BlobFilter(
            image_cp,
            radius=9,
            median_radius=7,
            max_value=3.0,
            mask=mask_cp,
            return_diagnostics=True)

        self.assertEqual(output.shape, image_np.shape)
        self.assertIs(cp.get_array_module(output), cp)
        self.assertEqual(diagnostics.backend, "cupy")
        self.assertFalse(diagnostics.used_numpy_fallback)
        self.assertGreaterEqual(float(cp.asnumpy(output).min()), 0.0)
        self.assertLessEqual(float(cp.asnumpy(output).max()), 1.0)
        self.assertEqual(int(cp.asnumpy(output)[512:544, 512:544].sum()), 0)

    def test_blob_filter_image_file(self):
        image = self._sample_image()
        mask = np.ones_like(image, dtype=bool)
        mask[:5, :5] = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            load_path = os.path.join(tmp_dir, "input.png")
            save_path = os.path.join(tmp_dir, "output.png")
            mask_path = os.path.join(tmp_dir, "mask.png")
            nornir_imageregistration.SaveImage(load_path, image, bpp=8)
            nornir_imageregistration.SaveImage(mask_path, mask.astype(np.uint8) * 255, bpp=8)

            diagnostics = _blob_filter_module().BlobFilterImageFile(
                load_path,
                save_path,
                radius=2,
                median_radius=1,
                max_value=3.0,
                mask_path=mask_path,
                return_diagnostics=True)

            self.assertTrue(os.path.exists(save_path))
            self.assertIn(diagnostics.backend, {"numpy", "cupy"})


if __name__ == "__main__":
    unittest.main()
