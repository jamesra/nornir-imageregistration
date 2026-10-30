"""The image_stats=None recovery in stos_brute must recover instead of raising.

``rotate_image`` guarded against a missing ``image_stats`` but then called
``ImageStats.CalcStats(image_stats)`` -- passing the very ``None`` it had just
detected -- so the recovery raised ``ValueError`` instead of recovering.

``pad_and_rotate_image`` had a second, related hole.  ``rotate_image`` only
repairs its own local variable, so the caller's ``image_stats`` stayed ``None``
and ``pad_image_for_phase_correlation(image_median=image_stats.median, ...)``
raised ``AttributeError``.  With ``angle == 0`` the rotate call is skipped
entirely, so there was no guard on that path at all.

Stats are computed from the source image, whose distribution the fill noise has
to match.
"""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.stos_brute import pad_and_rotate_image, rotate_image


def _image(seed: int = 3, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    return (np.random.default_rng(seed).random(shape) * 200).astype(np.float32)


class TestRotateImageRecoversStats(unittest.TestCase):

    def test_none_stats_no_longer_raises(self):
        image = _image()

        rotated = rotate_image(image, 10.0, None)

        self.assertTrue(np.isfinite(rotated).all(),
                        'rotation fill noise should replace every NaN')

    def test_recovered_noise_matches_the_image_distribution(self):
        """The recovery must use the image, not some default."""
        image = _image()
        expected = nornir_imageregistration.ImageStats.CalcStats(image)

        rotated = rotate_image(image, 10.0, None)

        # Rotation adds noise-filled corners, so compare loosely against the source.
        self.assertAlmostEqual(float(rotated.mean()), float(expected.median), delta=25.0)
        self.assertAlmostEqual(float(rotated.std()), float(expected.std), delta=25.0)

    def test_zero_angle_returns_input_untouched(self):
        image = _image()

        self.assertIs(rotate_image(image, 0.0, None), image)

    def test_explicit_stats_still_work(self):
        image = _image()
        stats = nornir_imageregistration.ImageStats.CalcStats(image)

        rotated = rotate_image(image, 10.0, stats)

        self.assertTrue(np.isfinite(rotated).all())


class TestPadAndRotateRecoversStats(unittest.TestCase):
    """The padding path dereferences image_stats directly, so it needs its own guard."""

    def test_none_stats_with_rotation(self):
        image = _image()

        result = pad_and_rotate_image(image, 10.0, None, desired_shape=(96, 96))

        self.assertEqual(result.shape, (96, 96))
        self.assertTrue(np.isfinite(result).all())

    def test_none_stats_without_rotation(self):
        """angle == 0 skips rotate_image, so its guard cannot help here."""
        image = _image()

        result = pad_and_rotate_image(image, 0.0, None, desired_shape=(96, 96))

        self.assertEqual(result.shape, (96, 96))
        self.assertTrue(np.isfinite(result).all())

    def test_recovered_matches_explicit_stats(self):
        image = _image()
        stats = nornir_imageregistration.ImageStats.CalcStats(image)

        recovered = pad_and_rotate_image(image.copy(), 10.0, None, desired_shape=(96, 96))
        explicit = pad_and_rotate_image(image.copy(), 10.0, stats, desired_shape=(96, 96))

        self.assertEqual(recovered.shape, explicit.shape)
        # Both fill with random noise drawn from the same distribution, so compare
        # the distributions rather than the pixels.
        self.assertAlmostEqual(float(recovered.mean()), float(explicit.mean()), delta=15.0)
        self.assertAlmostEqual(float(recovered.std()), float(explicit.std()), delta=15.0)

    def test_explicit_stats_still_work(self):
        image = _image()
        stats = nornir_imageregistration.ImageStats.CalcStats(image)

        result = pad_and_rotate_image(image, 10.0, stats, desired_shape=(96, 96))

        self.assertEqual(result.shape, (96, 96))
        self.assertTrue(np.isfinite(result).all())

    def test_stats_are_computed_once(self):
        """Recovering in the caller avoids rotate_image recomputing the same stats."""
        image = _image()
        calls = []
        original = nornir_imageregistration.ImageStats.CalcStats

        def counting_calc_stats(img):
            calls.append(img)
            return original(img)

        nornir_imageregistration.ImageStats.CalcStats = counting_calc_stats  # type: ignore[assignment]
        try:
            pad_and_rotate_image(image, 10.0, None, desired_shape=(96, 96))
        finally:
            nornir_imageregistration.ImageStats.CalcStats = original  # type: ignore[assignment]

        self.assertEqual(len(calls), 1, f'expected a single stats computation, got {len(calls)}')

    def test_various_angles_and_shapes(self):
        image = _image()
        for angle in (0.0, 5.0, 45.0, -30.0, 90.0):
            with self.subTest(angle=angle):
                result = pad_and_rotate_image(image.copy(), angle, None, desired_shape=(128, 128))
                self.assertEqual(result.shape, (128, 128))
                self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
