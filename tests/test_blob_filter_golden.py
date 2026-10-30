"""Golden regression tests for legacy ir-blob parity."""

from __future__ import annotations

import importlib
import os
import unittest

import numpy as np
from PIL import Image

from nornir_imageregistration.core._core import image_to_uint8


def _blob_filter_module():
    return importlib.import_module("nornir_imageregistration.blob_filter")


_LEGACY_6872_DIR = (
    "/legacycode/code/BuildScript/Test/Data/PlatformRaw/PNG/6872"
)
_LEGACY_FIXTURE_BASENAME = "0001_LeveledShadingCorrectedgfp"
_LEGACY_GOLDEN_RADIUS = 3
_LEGACY_GOLDEN_MEDIAN = 5
_LEGACY_GOLDEN_MAX = 3.0
_GOLDEN_MAX_ABS_DIFF = 1


def _legacy_fixture_paths() -> tuple[str, str, str] | None:
    if not os.path.isdir(_LEGACY_6872_DIR):
        return None

    mosaic = os.path.join(
        _LEGACY_6872_DIR,
        f"{_LEGACY_FIXTURE_BASENAME}_mosaic_1.png",
    )
    mask = os.path.join(
        _LEGACY_6872_DIR,
        f"{_LEGACY_FIXTURE_BASENAME}_mask_1.png",
    )
    golden = os.path.join(
        _LEGACY_6872_DIR,
        f"{_LEGACY_FIXTURE_BASENAME}_blob_1.png",
    )
    if not all(os.path.exists(path) for path in (mosaic, mask, golden)):
        return None
    return mosaic, mask, golden


def _load_legacy_uint8(path: str) -> np.ndarray:
    with Image.open(path, "r") as im:
        return np.array(im, dtype=np.uint8)


class TestBlobFilterGolden(unittest.TestCase):
    """Compare Python blob output to legacy PNG fixtures when available."""

    def test_legacy_6872_full_image_matches_golden(self):
        paths = _legacy_fixture_paths()
        if paths is None:
            self.skipTest("Legacy 6872 ir-blob fixtures are not available")

        mosaic_path, mask_path, golden_path = paths
        blob_filter = _blob_filter_module()

        image = blob_filter._load_legacy_tile_image(mosaic_path)
        mask = blob_filter._load_legacy_tile_mask(mask_path)
        golden_u8 = _load_legacy_uint8(golden_path)

        output, diagnostics = blob_filter.BlobFilter(
            image,
            radius=_LEGACY_GOLDEN_RADIUS,
            median_radius=_LEGACY_GOLDEN_MEDIAN,
            max_value=_LEGACY_GOLDEN_MAX,
            mask=mask,
            return_diagnostics=True,
        )
        self.assertEqual(diagnostics.backend, "numpy")

        output_u8 = image_to_uint8(output)
        self.assertEqual(output_u8.shape, golden_u8.shape)

        abs_diff = np.abs(
            output_u8.astype(np.int16) - golden_u8.astype(np.int16)
        )
        self.assertLessEqual(
            int(abs_diff.max()),
            _GOLDEN_MAX_ABS_DIFF,
            msg=(
                "Legacy golden mismatch: max abs diff "
                f"{int(abs_diff.max())} > {_GOLDEN_MAX_ABS_DIFF}"
            ),
        )

    def test_legacy_6872_center_crop_matches_golden(self):
        """Fast crop check against output from a full-image legacy-parity run."""
        paths = _legacy_fixture_paths()
        if paths is None:
            self.skipTest("Legacy 6872 ir-blob fixtures are not available")

        mosaic_path, mask_path, golden_path = paths
        blob_filter = _blob_filter_module()

        image = blob_filter._load_legacy_tile_image(mosaic_path)
        mask = blob_filter._load_legacy_tile_mask(mask_path)
        golden_u8 = _load_legacy_uint8(golden_path)

        output, _diagnostics = blob_filter.BlobFilter(
            image,
            radius=_LEGACY_GOLDEN_RADIUS,
            median_radius=_LEGACY_GOLDEN_MEDIAN,
            max_value=_LEGACY_GOLDEN_MAX,
            mask=mask,
            return_diagnostics=True,
        )
        output_u8 = image_to_uint8(output)

        height, width = output_u8.shape
        crop = 256
        row0 = (height // 2) - crop
        row1 = (height // 2) + crop
        col0 = (width // 2) - crop
        col1 = (width // 2) + crop

        abs_diff = np.abs(
            output_u8[row0:row1, col0:col1].astype(np.int16)
            - golden_u8[row0:row1, col0:col1].astype(np.int16)
        )
        self.assertLessEqual(int(abs_diff.max()), _GOLDEN_MAX_ABS_DIFF)


if __name__ == "__main__":
    unittest.main()
