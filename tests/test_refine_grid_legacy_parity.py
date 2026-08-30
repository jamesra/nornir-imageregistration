import os
import subprocess
import tempfile
import unittest

import numpy as np

import nornir_imageregistration

# Windows returns these NTSTATUS values as the process exit code when the loader
# cannot start the image at all, before main() runs. ir-refine-grid is a VS2010
# x64 build that imports VCOMP100.DLL (the VS2010 OpenMP runtime), which is not
# part of a stock Windows install, so this is the common failure on a machine
# that has the executable but not the matching redistributable.
_STATUS_DLL_NOT_FOUND = 0xC0000135
_STATUS_ENTRYPOINT_NOT_FOUND = 0xC0000139


def _loader_failure_status(returncode: int) -> int | None:
    """Return the NTSTATUS if *returncode* is a loader failure, else None.

    subprocess reports these as either the unsigned NTSTATUS or its signed
    two's-complement equivalent depending on the platform and Python version.
    """
    for status in (_STATUS_DLL_NOT_FOUND, _STATUS_ENTRYPOINT_NOT_FOUND):
        if returncode in (status, status - (1 << 32)):
            return status
    return None


class TestLoaderFailureDetection(unittest.TestCase):
    """The harness must not report a missing runtime as a parity failure."""

    def test_dll_not_found_is_recognized_signed_and_unsigned(self):
        self.assertEqual(_loader_failure_status(0xC0000135), _STATUS_DLL_NOT_FOUND)
        self.assertEqual(_loader_failure_status(3221225781), _STATUS_DLL_NOT_FOUND)
        self.assertEqual(_loader_failure_status(-1073741515), _STATUS_DLL_NOT_FOUND)

    def test_entrypoint_not_found_is_recognized(self):
        self.assertEqual(_loader_failure_status(0xC0000139), _STATUS_ENTRYPOINT_NOT_FOUND)
        self.assertEqual(_loader_failure_status(-1073741511), _STATUS_ENTRYPOINT_NOT_FOUND)

    def test_ordinary_exit_codes_are_not_loader_failures(self):
        for code in (0, 1, 2, -1, 255):
            with self.subTest(code=code):
                self.assertIsNone(_loader_failure_status(code))


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
                # ir-refine-grid's -mesh takes rows *and* columns; passing one value
                # makes it consume the following flag and fail with "bad -mesh cols".
                "-mesh", str(mesh_size), str(mesh_size),
                # The legacy tool spells this -displacement_threshold; -threshold is
                # rejected as an unknown option.
                "-displacement_threshold", str(threshold),
                "-sh", "1",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)

            loader_status = _loader_failure_status(completed.returncode)
            if loader_status is not None:
                self.skipTest(
                    f"Legacy ir-refine-grid could not be started by the OS loader "
                    f"(NTSTATUS {loader_status:#010x}); a dependency DLL is missing. "
                    f"{legacy_exe} is a VS2010 x64 build and imports VCOMP100.DLL, "
                    f"which requires the Visual C++ 2010 x64 redistributable. "
                    f"This is an environment gap, not a parity result.")

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
