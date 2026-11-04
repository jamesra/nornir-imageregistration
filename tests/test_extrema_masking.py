"""Behavioural tests for extrema masking, which keeps translate from registering large saturated regions.

A large region of pure black or white (a fold, resin, a blank slide area) correlates perfectly with
any other such region, so translate masks continuous min/max regions above a size cutoff and fills
them with noise (``mask_extrema``). The existing tests only cover shapes and return types, and the
core.py cases are visual, so nothing asserted that large regions are masked, small saturated features
survive, or that translate masks extrema by default.
"""

from __future__ import annotations

import inspect
import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import arrange_mosaic

# Module-level private, so the leading double underscore is not mangled here.
_get_overlapping_image = getattr(arrange_mosaic, '_ArrangeMosaic__get_overlapping_image', None)
if _get_overlapping_image is None:
    _get_overlapping_image = getattr(arrange_mosaic, '__get_overlapping_image')

_SHAPE = (100, 100)


def _tissue_with_extrema(black_block: tuple[slice, slice] | None = None,
                         white_block: tuple[slice, slice] | None = None,
                         scattered_white_pixels: int = 0) -> np.ndarray:
    """Mid-grey texture in [0.3, 0.7], with optional saturated black/white blocks and isolated white pixels."""
    rng = np.random.default_rng(1234)
    image = rng.uniform(0.3, 0.7, _SHAPE).astype(np.float32)
    if black_block is not None:
        image[black_block] = 0.0
    if white_block is not None:
        image[white_block] = 1.0
    if scattered_white_pixels:
        # Spaced so that no two are adjacent: every one is its own single-pixel region.
        ys = np.arange(1, _SHAPE[0] - 1, 3)[:scattered_white_pixels]
        image[ys, ys] = 1.0
        # Ensure a black pixel exists so the minimum is not a texture value.
        image[0, 0] = 0.0
    return image


class TestCreateExtremaMaskSizeCutoff(unittest.TestCase):
    """Large continuous min/max regions are excluded; small ones are kept."""

    def test_a_large_white_region_is_masked(self) -> None:
        white = (slice(10, 40), slice(10, 40))  # 900 px = 9% of the image
        mask = nornir_imageregistration.CreateExtremaMask(_tissue_with_extrema(white_block=white, black_block=(slice(60, 61), slice(60, 61))),
                                                          size_cutoff=0.01)
        self.assertFalse(mask[white].any(), "A large saturated white region must be excluded (False)")

    def test_a_large_black_region_is_masked(self) -> None:
        black = (slice(50, 90), slice(50, 90))  # 1600 px
        mask = nornir_imageregistration.CreateExtremaMask(_tissue_with_extrema(black_block=black, white_block=(slice(5, 6), slice(5, 6))),
                                                          size_cutoff=0.01)
        self.assertFalse(mask[black].any(), "A large saturated black region must be excluded (False)")

    def test_tissue_outside_the_extrema_is_kept(self) -> None:
        white = (slice(10, 40), slice(10, 40))
        black = (slice(50, 90), slice(50, 90))
        mask = nornir_imageregistration.CreateExtremaMask(_tissue_with_extrema(white_block=white, black_block=black),
                                                          size_cutoff=0.01)
        keep = np.ones(_SHAPE, dtype=bool)
        keep[white] = False
        keep[black] = False
        self.assertTrue(mask[keep].all(), "Texture pixels must remain included (True)")

    def test_small_saturated_features_are_kept(self) -> None:
        # Single saturated pixels (dense stained cells in the real data) are below the cutoff.
        image = _tissue_with_extrema(scattered_white_pixels=20)
        mask = nornir_imageregistration.CreateExtremaMask(image, size_cutoff=0.01)
        self.assertTrue(mask.all(), "Regions smaller than the cutoff must not be masked")

    def test_the_last_region_in_scan_order_is_sized_like_the_others(self) -> None:
        """Regression: region sizes were summed over labels 0..nLabels-1, so the last region was always masked.

        With a black pixel at (0, 0) and a single white pixel at (50, 50) the white pixel is the last label;
        it is 1 px against a 100 px cutoff and must be kept. Any pixel position works, so check both orders.
        """
        for white in ((50, 50), (99, 99), (1, 1)):
            with self.subTest(white=white):
                image = np.random.default_rng(7).uniform(0.3, 0.7, _SHAPE).astype(np.float32)
                image[0, 0] = 0.0
                image[white] = 1.0
                mask = nornir_imageregistration.CreateExtremaMask(image, size_cutoff=0.01)
                self.assertTrue(mask.all())

    def test_a_large_last_region_is_still_masked(self) -> None:
        image = np.random.default_rng(8).uniform(0.3, 0.7, _SHAPE).astype(np.float32)
        image[0, 0] = 0.0
        image[60:90, 60:90] = 1.0  # last label in scan order, 900 px
        mask = nornir_imageregistration.CreateExtremaMask(image, size_cutoff=0.01)
        self.assertFalse(mask[60:90, 60:90].any())

    def test_a_cutoff_of_none_masks_every_extreme_pixel(self) -> None:
        image = _tissue_with_extrema(scattered_white_pixels=20)
        mask = nornir_imageregistration.CreateExtremaMask(image, size_cutoff=None)
        # size_cutoff=None returns the raw candidate mask: True marks the extrema.
        self.assertTrue(mask[image == image.max()].all())
        self.assertTrue(mask[image == image.min()].all())
        self.assertFalse(mask[(image != image.max()) & (image != image.min())].any())

    def test_an_absolute_pixel_cutoff_is_a_count_not_a_fraction(self) -> None:
        block = (slice(10, 20), slice(10, 20))  # 100 px
        image = _tissue_with_extrema(white_block=block, black_block=(slice(60, 61), slice(60, 61)))
        self.assertFalse(nornir_imageregistration.CreateExtremaMask(image, size_cutoff=50)[block].any(),
                         "A 100 px region exceeds an absolute cutoff of 50 px and must be masked")
        self.assertTrue(nornir_imageregistration.CreateExtremaMask(image, size_cutoff=500)[block].all(),
                        "A 100 px region is below an absolute cutoff of 500 px and must be kept")


class TestTranslateMasksExtremaByDefault(unittest.TestCase):
    """The translate path asks for extrema masking unless the settings turn it off."""

    def test_translate_settings_default_to_masking_extrema(self) -> None:
        self.assertTrue(nornir_imageregistration.settings.TranslateSettings().mask_extrema)

    def test_translate_settings_can_turn_it_off(self) -> None:
        self.assertFalse(nornir_imageregistration.settings.TranslateSettings(mask_extrema=False).mask_extrema)

    def test_tile_offset_search_defaults_to_masking_extrema(self) -> None:
        default = inspect.signature(arrange_mosaic._FindTileOffsets).parameters['mask_extrema'].default
        self.assertTrue(default)


class TestOverlapCropReplacesExtremaWithNoise(unittest.TestCase):
    """The crop translate correlates must not contain the saturated region it was told to mask."""

    def test_a_large_saturated_block_in_the_overlap_is_noise_filled(self) -> None:
        white = (slice(10, 40), slice(10, 40))
        image = _tissue_with_extrema(white_block=white, black_block=(slice(60, 61), slice(60, 61)))
        rect = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), _SHAPE)
        cropped, mask = _get_overlapping_image(image, rect, excess_scalar=1.0, mask_extrema=True,
                                               cval='random', dtype=np.float32)
        self.assertEqual(cropped.shape, mask.shape)
        self.assertFalse(mask[white].any(), "The saturated block must be marked excluded")
        self.assertLess(float(np.mean(cropped[white] == 1.0)), 0.05,
                        "The saturated block must be replaced by noise, not left at 1.0")

    def test_without_masking_the_saturated_block_is_left_alone(self) -> None:
        white = (slice(10, 40), slice(10, 40))
        image = _tissue_with_extrema(white_block=white, black_block=(slice(60, 61), slice(60, 61)))
        rect = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), _SHAPE)
        cropped = _get_overlapping_image(image, rect, excess_scalar=1.0, mask_extrema=False,
                                         cval='random', dtype=np.float32)
        self.assertTrue(np.all(cropped[white] == 1.0))


if __name__ == '__main__':
    unittest.main()
