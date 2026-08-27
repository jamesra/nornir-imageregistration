"""Tests for relative and absolute path handling in STOS files."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from hypothesis import example, given, settings, strategies as st

import numpy as np

import nornir_imageregistration.core as core
from nornir_imageregistration.files import stosfile
from nornir_imageregistration.files.stosfile import (
    StosFile,
    _can_express_relative,
    _looks_like_windows_absolute,
    _normalize_stos_path,
    _path_for_stos_file,
    _path_from_stos_file,
    paths_refer_to_same_file,
    stos_transform_maps_onto_control_image,
    transform_text_contains_nonfinite,
)

# Minimal valid rigid transform for a 4x4 image pair.
_MIN_TRANSFORM = (
    "FixedCenterOfRotationAffineTransform_double_2_2 vp 8 1 0 0 1 0 0 1 1 fp 2 2 2"
)


def _write_tiny_png(path: str) -> None:
    """Create a minimal PNG file for STOS dimension probing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    core.SaveImage(path, np.zeros((4, 4), dtype=np.uint8))


def _build_idoc_like_layout(root: str) -> dict[str, str]:
    """Layout mimicking IDoc StosBrute16 next to section blob images."""
    stos_dir = os.path.join(root, "TEM", "StosBrute16")
    os.makedirs(stos_dir, exist_ok=True)
    control_image = os.path.join(root, "TEM", "0691", "TEM", "Blob", "Images", "016", "0691_TEM_Blob.png")
    mapped_image = os.path.join(root, "TEM", "0692", "TEM", "Blob", "Images", "016", "0692_TEM_Blob.png")
    control_mask = os.path.join(root, "TEM", "0691", "TEM", "Blob", "Masks", "016", "0691_TEM_Blob_mask.png")
    mapped_mask = os.path.join(root, "TEM", "0692", "TEM", "Blob", "Masks", "016", "0692_TEM_Blob_mask.png")
    stos_path = os.path.join(stos_dir, "0692-0691_ctrl-TEM_Blob_map-TEM_Blob_16.stos")

    for path in (control_image, mapped_image, control_mask, mapped_mask):
        _write_tiny_png(path)

    return {
        "stos_dir": stos_dir,
        "stos_path": stos_path,
        "control_image": control_image,
        "mapped_image": mapped_image,
        "control_mask": control_mask,
        "mapped_mask": mapped_mask,
    }


class TestStosPathHelpers(unittest.TestCase):
    """Unit tests for STOS path helper functions."""

    def test_normalize_stos_path_uses_forward_slashes(self) -> None:
        normalized = _normalize_stos_path(os.path.join("a", "b", "c.png"))
        self.assertNotIn("\\", normalized)
        self.assertEqual(normalized, "a/b/c.png")

    def test_path_from_stos_file_relative(self) -> None:
        # Use a real absolute base so Windows and POSIX agree on isabs().
        with tempfile.TemporaryDirectory() as root:
            stos_dir = os.path.join(root, "TEM", "StosBrute16")
            stored = "../../0691/TEM/Blob/Images/016/0691_TEM_Blob.png"
            resolved = _path_from_stos_file(stored, stos_dir)
            self.assertTrue(os.path.isabs(resolved))
            self.assertTrue(
                resolved.endswith(
                    os.path.join("0691", "TEM", "Blob", "Images", "016", "0691_TEM_Blob.png")
                )
            )
            # ../../ from TEM/StosBrute16 lands at root/0691/...
            self.assertEqual(
                os.path.normpath(resolved),
                os.path.normpath(
                    os.path.join(root, "0691", "TEM", "Blob", "Images", "016", "0691_TEM_Blob.png")
                ),
            )

    def test_path_from_stos_file_absolute(self) -> None:
        absolute = os.path.abspath(os.path.join(tempfile.gettempdir(), "abs.png"))
        self.assertEqual(
            _path_from_stos_file(absolute, os.path.join(tempfile.gettempdir(), "any", "stos")),
            os.path.normpath(absolute),
        )

    def test_looks_like_windows_absolute(self) -> None:
        self.assertTrue(_looks_like_windows_absolute(r"Y:\Volumes\RC2\TEM\a.png"))
        self.assertTrue(_looks_like_windows_absolute("Y:/Volumes/RC2/TEM/a.png"))
        self.assertTrue(_looks_like_windows_absolute(r"\\server\share\a.png"))
        self.assertFalse(_looks_like_windows_absolute("../../TEM/a.png"))
        self.assertFalse(_looks_like_windows_absolute("/storage4/RC2/TEM/a.png"))

    def test_path_from_stos_file_rebases_windows_absolute(self) -> None:
        """Legacy Windows abs paths map onto the volume that holds the .stos file."""
        with tempfile.TemporaryDirectory() as root:
            volume = os.path.join(root, "RC2")
            image = os.path.join(
                volume, "TEM", "0999", "TEM", "Leveled", "Images", "032", "0999_TEM_Leveled.png")
            stos_dir = os.path.join(volume, "TEM", "Grid32")
            os.makedirs(os.path.dirname(image), exist_ok=True)
            os.makedirs(stos_dir, exist_ok=True)
            _write_tiny_png(image)

            stored = r"Y:\Volumes\RC2\TEM\0999\TEM\Leveled\Images\032\0999_TEM_Leveled.png"
            resolved = _path_from_stos_file(stored, stos_dir)
            self.assertEqual(os.path.normpath(resolved), os.path.normpath(image))
            self.assertTrue(os.path.isfile(resolved))
            self.assertNotIn("Y:", resolved)
            self.assertNotIn("Volumes", resolved.split(os.sep))

    def test_path_from_stos_file_does_not_log_unchanged_windows_path(self) -> None:
        """Same-machine Windows paths that already exist should not log a rebase."""
        with tempfile.TemporaryDirectory() as root:
            image = os.path.join(root, "TEM", "1383", "TEM", "Leveled", "Images", "032", "1383_TEM_Leveled.png")
            stos_dir = os.path.join(root, "TEM", "Grid32")
            os.makedirs(os.path.dirname(image), exist_ok=True)
            os.makedirs(stos_dir, exist_ok=True)
            _write_tiny_png(image)
            stored = os.path.abspath(image)

            with mock.patch.object(stosfile._logger, "info") as info:
                resolved = _path_from_stos_file(stored, stos_dir)

            self.assertEqual(os.path.normpath(resolved), os.path.normpath(image))
            for call in info.call_args_list:
                message = call.args[0] if call.args else ""
                self.assertNotIn("Rebased Windows STOS path", message)

    def test_path_from_stos_file_logs_when_windows_path_changes(self) -> None:
        """Log only when rebase maps a Windows path onto a different location."""
        with tempfile.TemporaryDirectory() as root:
            volume = os.path.join(root, "RC2")
            image = os.path.join(
                volume, "TEM", "0999", "TEM", "Leveled", "Images", "032", "0999_TEM_Leveled.png")
            stos_dir = os.path.join(volume, "TEM", "Grid32")
            os.makedirs(os.path.dirname(image), exist_ok=True)
            os.makedirs(stos_dir, exist_ok=True)
            _write_tiny_png(image)
            stored = r"Y:\Volumes\RC2\TEM\0999\TEM\Leveled\Images\032\0999_TEM_Leveled.png"

            with mock.patch.object(stosfile._logger, "info") as info:
                resolved = _path_from_stos_file(stored, stos_dir)

            self.assertEqual(os.path.normpath(resolved), os.path.normpath(image))
            rebase_calls = [
                call for call in info.call_args_list
                if call.args and "Rebased Windows STOS path" in call.args[0]
            ]
            self.assertEqual(len(rebase_calls), 1)

    def test_path_from_stos_file_rebases_flattened_desktop_export(self) -> None:
        """Desktop copies named 1042_TEM_32_Leveled.png map onto the volume pyramid file."""
        with tempfile.TemporaryDirectory() as root:
            volume = os.path.join(root, "RC2")
            image = os.path.join(
                volume, "TEM", "1042", "TEM", "Leveled", "Images", "032", "1042_TEM_Leveled.png")
            stos_dir = os.path.join(volume, "TEM", "Grid32", "Manual")
            os.makedirs(os.path.dirname(image), exist_ok=True)
            os.makedirs(stos_dir, exist_ok=True)
            _write_tiny_png(image)

            stored = r"C:\Users\u0490822\Desktop\RC2_LocalEnhanced\1042_TEM_32_Leveled.png"
            resolved = _path_from_stos_file(stored, stos_dir)
            self.assertEqual(os.path.normpath(resolved), os.path.normpath(image))
            self.assertTrue(os.path.isfile(resolved))

    def test_stos_load_rebases_flattened_desktop_export_lines(self) -> None:
        """Manual .stos files from a local enhanced folder load volume images."""
        with tempfile.TemporaryDirectory() as root:
            volume = os.path.join(root, "RC2")
            stos_dir = os.path.join(volume, "TEM", "Grid32", "Manual")
            os.makedirs(stos_dir, exist_ok=True)
            control = os.path.join(
                volume, "TEM", "1042", "TEM", "Leveled", "Images", "032", "1042_TEM_Leveled.png")
            mapped = os.path.join(
                volume, "TEM", "1044", "TEM", "Leveled", "Images", "032", "1044_TEM_Leveled.png")
            for path in (control, mapped):
                _write_tiny_png(path)

            stos_path = os.path.join(
                stos_dir, "1044-1042_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
            with open(stos_path, "w", encoding="utf-8") as handle:
                handle.write(r"C:\Users\u0490822\Desktop\RC2_LocalEnhanced\1042_TEM_32_Leveled.png" + "\n")
                handle.write(r"C:\Users\u0490822\Desktop\RC2_LocalEnhanced\1044_TEM_32_Leveled.png" + "\n")
                handle.write("0\n0\n")
                handle.write("1 1 4 4\n1 1 4 4\n")
                handle.write(f"{_MIN_TRANSFORM}\n")

            loaded = StosFile.Load(stos_path)
            self.assertEqual(os.path.normpath(loaded.ControlImageFullPath), os.path.normpath(control))
            self.assertEqual(os.path.normpath(loaded.MappedImageFullPath), os.path.normpath(mapped))

    def test_path_from_stos_file_rebases_windows_absolute_when_missing(self) -> None:
        """Rebase by shared volume folder name even if the image is not on disk yet."""
        with tempfile.TemporaryDirectory() as root:
            volume = os.path.join(root, "RC2")
            stos_dir = os.path.join(volume, "TEM", "Grid32", "Manual")
            os.makedirs(stos_dir, exist_ok=True)
            expected = os.path.join(
                volume, "TEM", "0385", "TEM", "Leveled", "Images", "032", "0385_TEM_Leveled.png")
            stored = r"Y:\Volumes\RC2\TEM\0385\TEM\Leveled\Images\032\0385_TEM_Leveled.png"
            resolved = _path_from_stos_file(stored, stos_dir)
            self.assertEqual(os.path.normpath(resolved), os.path.normpath(expected))
            self.assertFalse(resolved.startswith(stos_dir))

    def test_stos_load_rebases_legacy_windows_image_lines(self) -> None:
        """StosFile.Load resolves Windows absolute image/mask lines onto the volume."""
        with tempfile.TemporaryDirectory() as root:
            volume = os.path.join(root, "RC2")
            stos_dir = os.path.join(volume, "TEM", "Grid32")
            os.makedirs(stos_dir, exist_ok=True)
            control = os.path.join(
                volume, "TEM", "0999", "TEM", "Leveled", "Images", "032", "0999_TEM_Leveled.png")
            mapped = os.path.join(
                volume, "TEM", "1000", "TEM", "Leveled", "Images", "032", "1000_TEM_Leveled.png")
            control_mask = os.path.join(
                volume, "TEM", "0999", "TEM", "Mask", "Images", "032", "0999_TEM_Mask.png")
            mapped_mask = os.path.join(
                volume, "TEM", "1000", "TEM", "Mask", "Images", "032", "1000_TEM_Mask.png")
            for path in (control, mapped, control_mask, mapped_mask):
                _write_tiny_png(path)

            stos_path = os.path.join(stos_dir, "1000-999_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
            with open(stos_path, "w", encoding="utf-8") as handle:
                handle.write(r"Y:\Volumes\RC2\TEM\0999\TEM\Leveled\Images\032\0999_TEM_Leveled.png" + "\n")
                handle.write(r"Y:\Volumes\RC2\TEM\1000\TEM\Leveled\Images\032\1000_TEM_Leveled.png" + "\n")
                handle.write("0\n0\n")
                handle.write("1 1 4 4\n1 1 4 4\n")
                handle.write(f"{_MIN_TRANSFORM}\n")
                handle.write("two_user_supplied_masks:\n")
                handle.write(r"Y:\Volumes\RC2\TEM\0999\TEM\Mask\Images\032\0999_TEM_Mask.png" + "\n")
                handle.write(r"Y:\Volumes\RC2\TEM\1000\TEM\Mask\Images\032\1000_TEM_Mask.png" + "\n")

            loaded = StosFile.Load(stos_path)
            self.assertEqual(os.path.normpath(loaded.ControlImageFullPath), os.path.normpath(control))
            self.assertEqual(os.path.normpath(loaded.MappedImageFullPath), os.path.normpath(mapped))
            self.assertEqual(
                os.path.normpath(loaded.ControlMaskFullPath),  # type: ignore[arg-type]
                os.path.normpath(control_mask))
            self.assertEqual(
                os.path.normpath(loaded.MappedMaskFullPath),  # type: ignore[arg-type]
                os.path.normpath(mapped_mask))


class TestStosFileRelativePaths(unittest.TestCase):
    """Integration tests for StosFile Load/Save path behavior."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        self.layout = _build_idoc_like_layout(self.root)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_stos(self) -> StosFile:
        stos = StosFile()
        stos.ControlImageFullPath = self.layout["control_image"]
        stos.MappedImageFullPath = self.layout["mapped_image"]
        stos.ControlMaskFullPath = self.layout["control_mask"]
        stos.MappedMaskFullPath = self.layout["mapped_mask"]
        stos.ControlImageDim = [1.0, 1.0, 4, 4]
        stos.MappedImageDim = [1.0, 1.0, 4, 4]
        stos.Transform = _MIN_TRANSFORM
        return stos

    def test_save_writes_relative_paths_when_possible(self) -> None:
        stos = self._make_stos()
        stos.Save(self.layout["stos_path"], relative_paths=True)

        with open(self.layout["stos_path"], "r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle.readlines()]

        self.assertFalse(os.path.isabs(lines[0].replace("/", os.sep)))
        self.assertFalse(os.path.isabs(lines[1].replace("/", os.sep)))
        self.assertNotIn("\\", lines[0])
        self.assertNotIn("\\", lines[1])
        self.assertIn("..", lines[0])
        self.assertIn("..", lines[1])

    def test_round_trip_relative_paths(self) -> None:
        stos = self._make_stos()
        stos.Save(self.layout["stos_path"], relative_paths=True)

        loaded = StosFile.Load(self.layout["stos_path"])
        self.assertEqual(os.path.normpath(loaded.ControlImageFullPath),
                         os.path.normpath(self.layout["control_image"]))
        self.assertEqual(os.path.normpath(loaded.MappedImageFullPath),
                         os.path.normpath(self.layout["mapped_image"]))
        self.assertEqual(os.path.normpath(loaded.ControlMaskFullPath),  # type: ignore[arg-type]
                         os.path.normpath(self.layout["control_mask"]))
        self.assertEqual(os.path.normpath(loaded.MappedMaskFullPath),  # type: ignore[arg-type]
                         os.path.normpath(self.layout["mapped_mask"]))

    def test_legacy_absolute_paths_load(self) -> None:
        stos_path = os.path.join(self.layout["stos_dir"], "legacy_absolute.stos")
        with open(stos_path, "w", encoding="utf-8") as handle:
            handle.write(f"{self.layout['control_image']}\n")
            handle.write(f"{self.layout['mapped_image']}\n")
            handle.write("0\n0\n")
            handle.write("1 1 4 4\n1 1 4 4\n")
            handle.write(f"{_MIN_TRANSFORM}\n")

        loaded = StosFile.Load(stos_path)
        self.assertEqual(os.path.normpath(loaded.ControlImageFullPath),
                         os.path.normpath(self.layout["control_image"]))

    def test_save_falls_back_to_absolute_when_relative_impossible(self) -> None:
        stos = self._make_stos()
        with mock.patch.object(stosfile, "_can_express_relative", return_value=False):
            stos.Save(self.layout["stos_path"], relative_paths=True)

        with open(self.layout["stos_path"], "r", encoding="utf-8") as handle:
            control_line = handle.readline().strip()

        self.assertTrue(os.path.isabs(control_line.replace("/", os.sep)))

    def test_mixed_relative_and_absolute_on_save(self) -> None:
        stos = self._make_stos()

        def _can_express(full_path: str, stos_dir: str) -> bool:
            return full_path == self.layout["control_image"]

        with mock.patch.object(stosfile, "_can_express_relative", side_effect=_can_express):
            stos.Save(self.layout["stos_path"], relative_paths=True)

        with open(self.layout["stos_path"], "r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle.readlines()]

        self.assertIn("..", lines[0])
        self.assertTrue(os.path.isabs(lines[1].replace("/", os.sep)))

        loaded = StosFile.Load(self.layout["stos_path"])
        self.assertEqual(os.path.normpath(loaded.ControlImageFullPath),
                         os.path.normpath(self.layout["control_image"]))
        self.assertEqual(os.path.normpath(loaded.MappedImageFullPath),
                         os.path.normpath(self.layout["mapped_image"]))

    def test_mask_lines_use_relative_when_possible(self) -> None:
        stos = self._make_stos()
        stos.Save(self.layout["stos_path"], relative_paths=True)

        with open(self.layout["stos_path"], "r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle.readlines()]

        self.assertEqual(lines[7], "two_user_supplied_masks:")
        self.assertIn("..", lines[8])
        self.assertIn("..", lines[9])
        self.assertNotIn("\\", lines[8])

    def test_convert_paths_to_relative_and_absolute(self) -> None:
        stos = self._make_stos()
        stos.ConvertPathsToRelative(self.layout["stos_dir"])
        self.assertIn("..", stos.ControlImageFullPath)
        stos.ConvertPathsToAbsolute(self.layout["stos_dir"])
        self.assertEqual(os.path.normpath(stos.ControlImageFullPath),
                         os.path.normpath(self.layout["control_image"]))

    def test_path_for_stos_file_relative(self) -> None:
        stored = _path_for_stos_file(self.layout["control_image"], self.layout["stos_dir"])
        self.assertIn("..", stored)
        resolved = _path_from_stos_file(stored, self.layout["stos_dir"])
        self.assertEqual(os.path.normpath(resolved), os.path.normpath(self.layout["control_image"]))

    def test_paths_refer_to_same_file(self) -> None:
        self.assertTrue(paths_refer_to_same_file(None, None))
        self.assertFalse(paths_refer_to_same_file(self.layout["control_image"], None))
        absolute = os.path.abspath(self.layout["control_image"])
        self.assertTrue(paths_refer_to_same_file(absolute, self.layout["control_image"]))
        self.assertFalse(
            paths_refer_to_same_file(self.layout["control_image"], self.layout["mapped_image"]))


class TestStosTransformPlausibility(unittest.TestCase):
    """Reject GridTransforms whose control points are off the recorded image."""

    def test_identity_grid_is_plausible(self) -> None:
        stos = StosFile()
        stos.ControlImageDim = [1.0, 1.0, 64.0, 64.0]
        stos.MappedImageDim = [1.0, 1.0, 64.0, 64.0]
        stos.Transform = (
            "FixedCenterOfRotationAffineTransform_double_2_2 vp 8 1 0 0 1 0 0 1 1 fp 2 32 32")
        self.assertTrue(stos_transform_maps_onto_control_image(stos))

    def test_huge_grid_control_points_are_implausible(self) -> None:
        stos = StosFile()
        stos.ControlImageDim = [1.0, 1.0, 3637.0, 3506.0]
        stos.MappedImageDim = [1.0, 1.0, 3495.0, 3496.0]
        stos.Transform = (
            "GridTransform_double_2_2 vp 8 18836250 264763264 27859404 391586080 "
            "36882556 518408864 45905708 645231680 fp 7 0 1 1 0 0 3495 3496")
        self.assertFalse(stos_transform_maps_onto_control_image(stos))

    def test_inset_mesh_is_plausible_without_image_corners(self) -> None:
        """Tissue meshes do not cover the image rectangle; corners must not reject them."""
        stos = StosFile()
        stos.ControlImageDim = [1.0, 1.0, 64.0, 64.0]
        stos.MappedImageDim = [1.0, 1.0, 64.0, 64.0]
        stos.Transform = (
            "MeshTransform_double_2_2 vp 12 "
            "0.3 0.3 20 20 0.7 0.3 45 20 0.5 0.7 32 45 "
            "fp 8 0 16 16 0 0 64 64 3")
        self.assertTrue(stos_transform_maps_onto_control_image(stos))

    @given(width=st.integers(min_value=8, max_value=512), height=st.integers(min_value=8, max_value=512))
    @example(width=64, height=64)
    @settings(max_examples=25, deadline=None)
    def test_identity_affine_is_plausible_for_any_image_size(self, width: int, height: int) -> None:
        stos = StosFile()
        stos.ControlImageDim = [1.0, 1.0, float(width), float(height)]
        stos.MappedImageDim = [1.0, 1.0, float(width), float(height)]
        stos.Transform = (
            "FixedCenterOfRotationAffineTransform_double_2_2 vp 8 1 0 0 1 0 0 1 1 "
            f"fp 2 {width / 2:g} {height / 2:g}")
        self.assertTrue(stos_transform_maps_onto_control_image(stos))


class TestStosNonfiniteTransform(unittest.TestCase):
    def test_detects_nan_and_inf_tokens(self) -> None:
        self.assertTrue(transform_text_contains_nonfinite("GridTransform_double_2_2 vp 4 nan nan nan nan"))
        self.assertTrue(transform_text_contains_nonfinite("vp 2 inf -inf"))
        self.assertFalse(transform_text_contains_nonfinite(_MIN_TRANSFORM))

    def test_save_refuses_nan_transform(self) -> None:
        stos = StosFile()
        stos.ControlImageFullPath = "control.png"
        stos.MappedImageFullPath = "mapped.png"
        stos.ControlImageDim = [1.0, 1.0, 4.0, 4.0]
        stos.MappedImageDim = [1.0, 1.0, 4.0, 4.0]
        stos.Transform = "GridTransform_double_2_2 vp 4 nan nan nan nan"
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = os.path.join(temp_dir, "bad.stos")
            with self.assertRaises(ValueError) as raised:
                stos.Save(out_path)
            self.assertIn("NaN/Inf", str(raised.exception))
            self.assertFalse(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
