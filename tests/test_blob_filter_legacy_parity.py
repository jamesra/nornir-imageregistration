import os
import importlib
import subprocess
import tempfile
import unittest

import numpy as np

import nornir_imageregistration


def _blob_filter_module():
    return importlib.import_module("nornir_imageregistration.blob_filter")


class TestBlobFilterLegacyParity(unittest.TestCase):
    """
    Optional parity harness for comparing Python blob output to legacy ir-blob.

    Required environment:
      - NORNIR_LEGACY_IR_BLOB: path to legacy ir-blob executable
      - NORNIR_BLOB_FIXTURE_IMAGE: path to fixture input image

    Optional:
      - NORNIR_BLOB_FIXTURE_MASK: path to fixture mask image
      - NORNIR_BLOB_RADIUS (default 9)
      - NORNIR_BLOB_MEDIAN (default 7)
      - NORNIR_BLOB_MAX (default 3.0)
      - NORNIR_BLOB_PARITY_MAE_MAX (default 0.01)
      - NORNIR_BLOB_PARITY_P99_MAX (default 1.0)
    """

    def test_python_matches_legacy_fixture(self):
        legacy_exe = os.environ.get("NORNIR_LEGACY_IR_BLOB")
        fixture_image = os.environ.get("NORNIR_BLOB_FIXTURE_IMAGE")
        fixture_mask = os.environ.get("NORNIR_BLOB_FIXTURE_MASK")
        if not legacy_exe or not fixture_image:
            self.skipTest("Legacy ir-blob parity env vars are not configured")

        if not os.path.exists(legacy_exe):
            self.skipTest(f"Legacy ir-blob executable not found: {legacy_exe}")
        if not os.path.exists(fixture_image):
            self.skipTest(f"Fixture image not found: {fixture_image}")
        if fixture_mask and not os.path.exists(fixture_mask):
            self.skipTest(f"Fixture mask not found: {fixture_mask}")

        radius = int(os.environ.get("NORNIR_BLOB_RADIUS", "9"))
        median = int(os.environ.get("NORNIR_BLOB_MEDIAN", "7"))
        max_value = float(os.environ.get("NORNIR_BLOB_MAX", "3.0"))
        mae_max = float(os.environ.get("NORNIR_BLOB_PARITY_MAE_MAX", "0.01"))
        p99_max = float(os.environ.get("NORNIR_BLOB_PARITY_P99_MAX", "1.0"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_out = os.path.join(tmp_dir, "legacy_blob.png")
            python_out = os.path.join(tmp_dir, "python_blob.png")

            cmd = [
                legacy_exe,
                "-load", fixture_image,
                "-save", legacy_out,
                "-r", str(radius),
                "-median", str(median),
                "-max", str(max_value),
                "-sh", "1"
            ]
            if fixture_mask:
                cmd.extend(["-mask", fixture_mask])

            completed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                completed.returncode, 0,
                msg=f"Legacy ir-blob failed.\ncmd={' '.join(cmd)}\nstdout={completed.stdout}\nstderr={completed.stderr}")
            self.assertTrue(os.path.exists(legacy_out), "Legacy ir-blob did not create output image")

            _blob_filter_module().BlobFilterImageFile(
                fixture_image,
                python_out,
                radius=radius,
                median_radius=median,
                max_value=max_value,
                mask_path=fixture_mask)
            self.assertTrue(os.path.exists(python_out), "Python blob did not create output image")

            legacy_img = nornir_imageregistration.LoadImage(legacy_out, dtype=np.float32, backend="numpy")
            python_img = nornir_imageregistration.LoadImage(python_out, dtype=np.float32, backend="numpy")
            self.assertEqual(legacy_img.shape, python_img.shape)

            abs_diff = np.abs(legacy_img - python_img)
            mae = float(np.mean(abs_diff))
            p99 = float(np.percentile(abs_diff, 99))
            self.assertLessEqual(mae, mae_max, f"MAE {mae} exceeds threshold {mae_max}")
            self.assertLessEqual(p99, p99_max, f"P99 abs diff {p99} exceeds threshold {p99_max}")


if __name__ == "__main__":
    unittest.main()
