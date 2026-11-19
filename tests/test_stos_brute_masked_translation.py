"""StosBrute's final translation comes from band-pass + masked NCC at the chosen angle and scale.

Masked registration stage 3a. Phase correlation needs masked pixels, extrema and rotation corners
filled with noise, and near a large masked area that noise can pull its peak. The masked translation
compares only pixels valid in both images and has the same frame and sign as ScoreOneAngle's peak.
"""
from __future__ import annotations

import unittest

import numpy as np
import scipy.ndimage

import nornir_imageregistration
from nornir_imageregistration import stos_brute


def _texture(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = scipy.ndimage.gaussian_filter(rng.random(shape), 3)
    return ((image - image.min()) / np.ptp(image)).astype(np.float32)


def _pair(angle: float, shift: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A 400 px target and a source showing the same tissue, rotated by -*angle* and shifted by *shift*."""
    world = _texture((700, 700), 1)
    target = world[150:550, 150:550]
    rotated = scipy.ndimage.rotate(world, angle, reshape=False, order=3)
    source = rotated[150 + shift[0]:550 + shift[0], 150 + shift[1]:550 + shift[1]].astype(np.float32)
    return target, source


class TestMaskedTranslationAtPose(unittest.TestCase):

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    def test_same_frame_and_sign_as_score_one_angle(self) -> None:
        for angle, shift in ((0.0, (23, -41)), (12.0, (-17, 30)), (-33.0, (40, 12))):
            with self.subTest(angle=angle):
                target, source = _pair(angle, shift)
                old = stos_brute.ScoreOneAngle(target, source, target.shape, source.shape, angle,
                                               target_image_prepadded=False, min_overlap=0.5)
                peak = stos_brute._masked_translation_at_pose(
                    target, np.ones(target.shape, bool), source, np.ones(source.shape, bool),
                    angle, 1.0, 0.5)
                self.assertIsNotNone(peak)
                np.testing.assert_allclose(peak, np.asarray(old.peak, float), atol=1.0)

    def test_masked_area_does_not_move_the_peak(self) -> None:
        """Whatever fills a masked area (here a bright block, or noise), the translation is the same."""
        target, source = _pair(8.0, (12, -20))
        target_mask = np.ones(target.shape, bool)
        target_mask[:, :170] = False
        source_mask = np.ones(source.shape, bool)
        peaks = []
        for fill in ('noise', 'bright'):
            filled = target.copy()
            filled[~target_mask] = (np.random.default_rng(5).random(int((~target_mask).sum()))
                                    if fill == 'noise' else 1.0)
            peaks.append(stos_brute._masked_translation_at_pose(filled, target_mask, source, source_mask,
                                                                8.0, 1.0, 0.25))
        clean = stos_brute._masked_translation_at_pose(target, np.ones(target.shape, bool), source, source_mask,
                                                       8.0, 1.0, 0.25)
        for peak in peaks:
            np.testing.assert_allclose(peak, clean, atol=0.5)

    def test_rotated_mask_matches_the_rotated_image_frame(self) -> None:
        image = _texture((123, 201), 2)
        mask = np.ones(image.shape, bool)
        for angle in (0.0, 7.0, -45.0, 90.0, 133.0):
            with self.subTest(angle=angle):
                rotated = stos_brute.rotate_image(image, angle, None)
                rotated_mask = stos_brute._rotate_mask(mask, angle)
                self.assertEqual(rotated.shape, rotated_mask.shape)
                if angle % 90:
                    self.assertFalse(rotated_mask[0, 0], "rotation corners are not valid")

    def test_scaled_mask_keeps_the_scaled_image_shape(self) -> None:
        image = _texture((157, 211), 3)
        for scale in ((0.5, None), (1.07, None), (0.93, 1.04)):
            with self.subTest(scale=scale):
                scaled = stos_brute._scale_registration_image(image, *scale)
                self.assertEqual(scaled.shape, stos_brute._scale_mask(np.ones(image.shape, bool), *scale).shape)

    def test_no_valid_overlap_returns_none(self) -> None:
        target, source = _pair(0.0, (0, 0))
        peak = stos_brute._masked_translation_at_pose(target, np.zeros(target.shape, bool), source,
                                                      np.ones(source.shape, bool), 0.0, 1.0, 0.5)
        self.assertIsNone(peak)


if __name__ == '__main__':
    unittest.main()
