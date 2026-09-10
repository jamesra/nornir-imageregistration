"""Unit tests for assemble.WriteStosPreviewImages overlay/diff/warped outputs."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import nornir_imageregistration
from nornir_imageregistration import assemble
from nornir_imageregistration.transforms.rigid import Rigid


class TestWriteStosPreviewImages(unittest.TestCase):
    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.outdir = self._tmpdir.name

        h, w = 32, 40
        # Distinct patterns so channel assignment is unambiguous after identity warp.
        self.control = np.zeros((h, w), dtype=np.float32)
        self.control[:, : w // 2] = 1.0
        self.mapped = np.zeros((h, w), dtype=np.float32)
        self.mapped[h // 2 :, :] = 1.0
        self.transform = Rigid(target_offset=(0.0, 0.0))

    def test_writes_overlay_diff_warped_with_pyre_channel_dodge(self) -> None:
        overlay_path = os.path.join(self.outdir, "overlay.png")
        diff_path = os.path.join(self.outdir, "diff.png")
        warped_path = os.path.join(self.outdir, "warped.png")

        warped = assemble.WriteStosPreviewImages(
            self.transform,
            overlay_path=overlay_path,
            diff_path=diff_path,
            warped_path=warped_path,
            fixedImage=self.control,
            warpedImage=self.mapped,
        )

        self.assertTrue(os.path.exists(overlay_path))
        self.assertTrue(os.path.exists(diff_path))
        self.assertTrue(os.path.exists(warped_path))
        self.assertEqual(tuple(warped.shape), self.control.shape)

        overlay = np.asarray(Image.open(overlay_path))
        warped_img = np.asarray(Image.open(warped_path))
        diff_img = np.asarray(Image.open(diff_path))

        self.assertEqual(overlay.ndim, 3)
        self.assertEqual(overlay.shape[2], 3)
        # Mapped/source → R and B (magenta); control/target → G
        np.testing.assert_array_equal(overlay[:, :, 0], overlay[:, :, 2])
        np.testing.assert_allclose(overlay[:, :, 0].astype(np.float32), warped_img.astype(np.float32), atol=1)
        np.testing.assert_allclose(
            overlay[:, :, 1].astype(np.float32),
            nornir_imageregistration.image_to_uint8(self.control).astype(np.float32),
            atol=1,
        )

        expected_diff = np.abs(
            nornir_imageregistration.image_to_uint8(self.control).astype(np.float32)
            - warped_img.astype(np.float32)
        ).astype(np.uint8)
        np.testing.assert_allclose(diff_img.astype(np.float32), expected_diff.astype(np.float32), atol=1)

    def test_omitted_paths_are_skipped(self) -> None:
        warped_path = os.path.join(self.outdir, "warped_only.png")
        assemble.WriteStosPreviewImages(
            self.transform,
            warped_path=warped_path,
            fixedImage=self.control,
            warpedImage=self.mapped,
        )
        self.assertTrue(os.path.exists(warped_path))
        self.assertFalse(os.path.exists(os.path.join(self.outdir, "overlay.png")))
        self.assertFalse(os.path.exists(os.path.join(self.outdir, "diff.png")))


if __name__ == "__main__":
    unittest.main()
