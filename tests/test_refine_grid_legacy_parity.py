import os
import subprocess
import tempfile
import unittest

import numpy as np

import nornir_imageregistration


class TestRefineGridLegacyParity(unittest.TestCase):
    """
    Optional parity harness comparing Python RefineGridMosaic to legacy ir-refine-grid.

    Required environment:
      - NORNIR_LEGACY_IR_REFINE_GRID: path to legacy ir-refine-grid executable
      - NORNIR_REFINE_GRID_INPUT_MOSAIC: path to translated input .mosaic
      - NORNIR_REFINE_GRID_IMAGE_DIR: tile image directory for -image_dir

    Optional:
      - NORNIR_REFINE_GRID_SP (default 4)
      - NORNIR_REFINE_GRID_ITERATIONS (default 5)
      - NORNIR_REFINE_GRID_CELL (default 96)
      - NORNIR_REFINE_GRID_MESH (default 8)
      - NORNIR_REFINE_GRID_THRESHOLD (default 0.5)
      - NORNIR_REFINE_GRID_TARGET_DELTA_MAX (default 2.0): max mean target delta (pixels)
    """

    def test_python_matches_legacy_fixture(self):
        legacy_exe = os.environ.get("NORNIR_LEGACY_IR_REFINE_GRID")
        input_mosaic = os.environ.get("NORNIR_REFINE_GRID_INPUT_MOSAIC")
        image_dir = os.environ.get("NORNIR_REFINE_GRID_IMAGE_DIR")
        if not legacy_exe or not input_mosaic or not image_dir:
            self.skipTest("Legacy ir-refine-grid parity env vars are not configured")

        if not os.path.exists(legacy_exe):
            self.skipTest(f"Legacy ir-refine-grid executable not found: {legacy_exe}")
        if not os.path.exists(input_mosaic):
            self.skipTest(f"Input mosaic not found: {input_mosaic}")
        if not os.path.isdir(image_dir):
            self.skipTest(f"Image directory not found: {image_dir}")

        spacing = int(os.environ.get("NORNIR_REFINE_GRID_SP", "4"))
        iterations = int(os.environ.get("NORNIR_REFINE_GRID_ITERATIONS", "5"))
        cell_size = int(os.environ.get("NORNIR_REFINE_GRID_CELL", "96"))
        mesh_size = int(os.environ.get("NORNIR_REFINE_GRID_MESH", "8"))
        threshold = float(os.environ.get("NORNIR_REFINE_GRID_THRESHOLD", "0.5"))
        delta_max = float(os.environ.get("NORNIR_REFINE_GRID_TARGET_DELTA_MAX", "2.0"))

        image_scale = 1.0 / float(spacing)

        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_out = os.path.join(tmp_dir, "legacy_grid.mosaic")
            python_out = os.path.join(tmp_dir, "python_grid.mosaic")

            cmd = [
                legacy_exe,
                "-load", input_mosaic,
                "-save", legacy_out,
                "-image_dir", image_dir,
                "-sp", str(spacing),
                "-it", str(iterations),
                "-cell", str(cell_size),
                "-mesh", str(mesh_size),
                "-threshold", str(threshold),
                "-sh", "1",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                completed.returncode, 0,
                msg=(
                    f"Legacy ir-refine-grid failed.\n"
                    f"cmd={' '.join(cmd)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
                ))
            self.assertTrue(os.path.exists(legacy_out), "Legacy ir-refine-grid did not create output mosaic")

            refined_mosaic = nornir_imageregistration.RefineGridMosaic(
                input_mosaic,
                image_dir,
                iterations=iterations,
                cell_size=cell_size,
                mesh_shape=(mesh_size, mesh_size),
                displacement_threshold=threshold,
                imageScale=image_scale)
            refined_mosaic.SaveToMosaicFile(python_out)
            self.assertTrue(os.path.exists(python_out), "Python RefineGridMosaic did not create output mosaic")

            legacy_mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(legacy_out)
            python_mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(python_out)
            self.assertEqual(set(legacy_mosaic.ImageToTransform.keys()), set(python_mosaic.ImageToTransform.keys()))

            deltas: list[float] = []
            for image_key in legacy_mosaic.ImageToTransform.keys():
                legacy_transform = legacy_mosaic.ImageToTransform[image_key]
                python_transform = python_mosaic.ImageToTransform[image_key]
                if not isinstance(legacy_transform, nornir_imageregistration.transforms.IGridTransform):
                    continue
                if not isinstance(python_transform, nornir_imageregistration.transforms.IGridTransform):
                    self.fail(f"Python output for {image_key} is not a grid transform")

                legacy_targets = nornir_imageregistration.EnsureNumpyArray(
                    legacy_transform.TargetPoints, dtype=np.float64)
                python_targets = nornir_imageregistration.EnsureNumpyArray(
                    python_transform.TargetPoints, dtype=np.float64)
                self.assertEqual(legacy_targets.shape, python_targets.shape)
                deltas.append(float(np.mean(np.linalg.norm(legacy_targets - python_targets, axis=1))))

            self.assertGreater(len(deltas), 0, "No comparable grid transforms found in legacy output")
            mean_delta = float(np.mean(deltas))
            self.assertLessEqual(
                mean_delta,
                delta_max,
                f"Mean per-tile target delta {mean_delta} exceeds threshold {delta_max}")


if __name__ == "__main__":
    unittest.main()
