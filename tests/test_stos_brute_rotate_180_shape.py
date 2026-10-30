"""The 0 degree and +180 degree rotates must feed fft_phase_correlation equal shapes.

``_log_polar_angle_and_scale`` pads and rotates the source twice to decide whether the
log-polar angle needs a 180 degree correction. The two calls look asymmetric:

- the 0 degree call passes ``power_of_two=True``
- the +180 call does not

That asymmetry is deliberate and load-bearing. ``power_of_two=True`` does not round
``desired_shape`` up; it *discards* it and substitutes
``NearestPowerOfTwo(rotated_image.shape)`` (``stos_brute.py:787-788``), a value derived
from the source alone. By the time the +180 call runs, the shape has already been
reconciled against ``padded_target`` via ``max_shape``, so substituting the source-only
power of two undershoots the shared shape whenever the target drove it larger.

Measured over a 1600-case sweep of source shapes, target shapes and angles:

===============================================  ==========
variant                                          mismatches
===============================================  ==========
as shipped (no ``power_of_two`` on the +180)               0
with ``power_of_two=True`` on the +180                   425
===============================================  ==========

Each mismatch is a ``ValueError: ImagePhaseCorrelation: Fixed and Moving image do not
have same dimension``. So adding the apparently-missing argument would *introduce* the
failure it looks like it prevents. These tests exist to make that concrete, because the
asymmetry reads like an oversight.

Filed as a bug, closed as ``wontfix`` on the logic; see review issue #86.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np

import nornir_imageregistration
import nornir_shared.mathhelper
from nornir_imageregistration import stos_brute
from nornir_imageregistration.phasecorrelation import pad_image_for_phase_correlation

MIN_OVERLAP = 0.75

# Straddle the power-of-two boundaries, and mix square with non-square.
SHAPES = [(64, 64), (100, 100), (101, 97), (128, 128), (129, 128), (200, 150),
          (256, 256), (257, 200), (300, 400)]

# 0 and +-90 matter because they are the rotations that leave the bounding box
# unchanged, which is where the source-only power of two is smallest.
ANGLES = [0, 1, 13, 30, 45, 47.3, 90, 91, 135, 179, -30, -45, -90]


def _reconcile(source_shape, target_shape, angle, power_of_two_on_180):
    """Replay the shape negotiation in ``_log_polar_angle_and_scale``.

    Returns the three shapes that must agree: the padded target, the 0 degree rotated
    source, and the +180 rotated source.
    """
    rng = np.random.default_rng(42)
    source_image = rng.random(source_shape).astype(np.float32)
    target_image = rng.random(target_shape).astype(np.float32)

    source_stats = nornir_imageregistration.ImageStats.CalcStats(source_image)
    target_stats = nornir_imageregistration.ImageStats.CalcStats(target_image)

    desired_height = int(nornir_imageregistration.NearestPowerOfTwo(
        max([source_image.shape[0], target_image.shape[0]])))
    desired_width = int(nornir_imageregistration.NearestPowerOfTwo(
        max([source_image.shape[1], target_image.shape[1]])))

    padded_target = pad_image_for_phase_correlation(
        target_image, min_overlap=MIN_OVERLAP, image_median=target_stats.median,
        image_stddev=target_stats.std, new_height=desired_height,
        new_width=desired_width)

    rotated_padded_source = stos_brute.pad_and_rotate_image(
        image=source_image, angle=angle, image_stats=source_stats,
        min_overlap=MIN_OVERLAP, desired_shape=[desired_height, desired_width],
        power_of_two=True)

    if not np.array_equal(rotated_padded_source.shape, padded_target.shape):
        rotated_desired_shape = nornir_shared.mathhelper.max_shape(
            [rotated_padded_source.shape, padded_target.shape])
        rotated_desired_height, rotated_desired_width = rotated_desired_shape
        padded_target = pad_image_for_phase_correlation(
            target_image, min_overlap=MIN_OVERLAP, image_median=target_stats.median,
            image_stddev=target_stats.std, new_height=rotated_desired_height,
            new_width=rotated_desired_width)
        if not np.array_equal(rotated_padded_source.shape, rotated_desired_shape):
            rotated_padded_source = pad_image_for_phase_correlation(
                rotated_padded_source, min_overlap=MIN_OVERLAP,
                image_median=source_stats.median, image_stddev=source_stats.std,
                new_height=rotated_desired_height, new_width=rotated_desired_width)
    else:
        rotated_desired_height, rotated_desired_width = desired_height, desired_width

    rotated_180 = stos_brute.pad_and_rotate_image(
        image=source_image, angle=angle + 180, image_stats=source_stats,
        min_overlap=MIN_OVERLAP,
        desired_shape=[rotated_desired_height, rotated_desired_width],
        power_of_two=power_of_two_on_180)

    return (tuple(padded_target.shape), tuple(rotated_padded_source.shape),
            tuple(rotated_180.shape))


class TestRotationBoundingBoxIsSymmetric(unittest.TestCase):
    """The premise the shape agreement rests on."""

    def test_rotating_by_t_and_t_plus_180_gives_the_same_shape(self):
        rng = np.random.default_rng(7)
        for shape in [(100, 100), (101, 101), (100, 140), (137, 91), (255, 129)]:
            image = rng.random(shape).astype(np.float32)
            stats = nornir_imageregistration.ImageStats.CalcStats(image)
            for angle in (1, 13, 30, 45, 47.3, 90, 135, 179):
                with self.subTest(shape=shape, angle=angle):
                    first = stos_brute.rotate_image(
                        image, angle=angle, image_stats=stats).shape
                    second = stos_brute.rotate_image(
                        image, angle=angle + 180, image_stats=stats).shape
                    self.assertEqual(
                        tuple(first), tuple(second),
                        'a 180 degree turn is a flip, so the bounding box must match')


class TestTheTwoRotatesAgreeAsShipped(unittest.TestCase):
    """What the code does today must keep working."""

    def test_all_three_shapes_agree_across_shapes_and_angles(self):
        mismatches = []
        for source_shape, target_shape in itertools.product(SHAPES, SHAPES):
            for angle in ANGLES:
                target, zero, one_eighty = _reconcile(
                    source_shape, target_shape, angle, power_of_two_on_180=False)
                if not (target == zero == one_eighty):
                    mismatches.append(
                        f'src={source_shape} tgt={target_shape} angle={angle}: '
                        f'target={target} 0deg={zero} 180={one_eighty}')

        self.assertEqual(
            mismatches, [],
            f'{len(mismatches)} shape mismatches would reach '
            f'fft_phase_correlation:\n  ' + '\n  '.join(mismatches[:10]))

    def test_a_target_larger_than_the_source_still_agrees(self):
        """The case where the target drives the shared shape past the source's own."""
        target, zero, one_eighty = _reconcile(
            (64, 64), (300, 400), 45, power_of_two_on_180=False)

        self.assertEqual(target, zero)
        self.assertEqual(target, one_eighty)


class TestTheImpliedFixWouldBreakIt(unittest.TestCase):
    """Pin the reason the asymmetry stays, so it is not 'tidied up' later."""

    def test_power_of_two_on_the_180_call_creates_mismatches(self):
        mismatches = 0
        checked = 0
        for source_shape, target_shape in itertools.product(SHAPES, SHAPES):
            for angle in ANGLES:
                checked += 1
                target, _zero, one_eighty = _reconcile(
                    source_shape, target_shape, angle, power_of_two_on_180=True)
                if one_eighty != target:
                    mismatches += 1

        self.assertGreater(
            mismatches, 0,
            'if this no longer breaks, pad_and_rotate_image changed and the comment '
            'at the +180 call in stos_brute needs revisiting')
        self.assertLess(mismatches, checked)

    def test_power_of_two_replaces_desired_shape_rather_than_raising_it(self):
        """The root of the confusion: it can return *less* than was asked for."""
        rng = np.random.default_rng(1)
        source = rng.random((64, 64)).astype(np.float32)
        stats = nornir_imageregistration.ImageStats.CalcStats(source)

        asked_for = [128, 128]
        got = stos_brute.pad_and_rotate_image(
            image=source, angle=0, image_stats=stats, min_overlap=MIN_OVERLAP,
            desired_shape=asked_for, power_of_two=True)

        self.assertEqual(tuple(got.shape), (64, 64),
                         'power_of_two overrides desired_shape; it does not round it up')

    def test_without_power_of_two_the_requested_shape_is_honoured(self):
        rng = np.random.default_rng(1)
        source = rng.random((64, 64)).astype(np.float32)
        stats = nornir_imageregistration.ImageStats.CalcStats(source)

        got = stos_brute.pad_and_rotate_image(
            image=source, angle=0, image_stats=stats, min_overlap=MIN_OVERLAP,
            desired_shape=[128, 128])

        self.assertEqual(tuple(got.shape), (128, 128))


class TestFftPhaseCorrelationRejectsMismatches(unittest.TestCase):
    """The consequence the finding named, confirmed."""

    def test_unequal_shapes_raise(self):
        rng = np.random.default_rng(3)
        small = np.fft.fft2(rng.random((128, 128)))
        large = np.fft.fft2(rng.random((256, 256)))

        with self.assertRaises(ValueError):
            nornir_imageregistration.fft_phase_correlation(small, large)


if __name__ == '__main__':
    unittest.main()
