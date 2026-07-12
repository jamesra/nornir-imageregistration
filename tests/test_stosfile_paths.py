"""Tests for relative and absolute path handling in STOS files."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration.core as core
from nornir_imageregistration.files import stosfile
from nornir_imageregistration.files.stosfile import (
    StosFile,
    _can_express_relative,
    _normalize_stos_path,
    _path_for_stos_file,
    _path_from_stos_file,
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


if __name__ == "__main__":
    unittest.main()
