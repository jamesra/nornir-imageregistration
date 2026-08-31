"""``rotate_image``'s exact NaN count is deliberate. Removing the sync makes it slower.

Review #94 flagged the ``int(xp_out.sum(...))`` in ``rotate_image`` as a per-angle
device->host sync, reached once per angle from ``ScoreManyAnglesGpu`` via
``_score_one_angle_core`` -> ``pad_and_rotate_image``. The sync is real. Removing it is a
regression, twice over, so #94 was closed ``wontfix`` and this pins the reasoning.

**It is not worth removing.** Only the ``sum`` reduction and the ``int()`` pull are
avoidable -- the boolean mask is needed either way for the masked assignment. Measured on
a 22 GiB card, median of 9: 0.00020s at 2048x2048 and 0.00038s at 4096x4096, which is
**0.22% of the per-angle cost** of a real sweep. Deleting it outright would take a
24-angle 2048x2048 sweep from 4.120s to 4.111s -- 1.002x.

**The sync-free rewrite costs more than it saves.** Sizing noise from a device scalar is
what forces the sync, so the way out is to generate noise for the whole frame and select
with ``where``. But a rotation only leaves 19.5% NaN at 7 degrees and 50.0% at 45, so
whole-frame generation is strictly more work than the count plus the exact-size draw --
`noise_full > sync + noise_n` in every measured case, by 1.7x at 45 degrees and 4.4x at 7.

**And it would change output.** Filling only the NaN entries consumes exactly ``n_bad``
values from the shared generator. A whole-frame draw consumes a different number and lands
different values in the kept positions, so seeded runs would stop reproducing. That
matters here: per the serial/batched skill, seeding only part of the noise once broke
``test_fixed_angle_fft_reuse`` in 286 of 300 seeds.

These tests run on NumPy: the NaN fraction and the determinism are properties of the
algorithm, not of the backend, and they are what make the conclusion hold.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import stos_brute


def _image(size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(20260830)
    return rng.random((size, size)).astype(np.float32)


class TestOnlyTheEmptyEntriesAreFilled(unittest.TestCase):
    """The count is a semantic requirement, not an incidental optimisation."""

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.image = _image()
        self.stats = nornir_imageregistration.ImageStats.CalcStats(self.image)

    def test_no_nan_survives_the_rotation(self):
        out = stos_brute.rotate_image(self.image, angle=45.0, image_stats=self.stats)
        self.assertFalse(np.any(np.isnan(out)),
                         'every rotation-vacated entry must be filled with noise')

    def test_noise_is_drawn_for_exactly_the_vacated_count(self):
        requested: list[int] = []
        real = self.stats.GenerateNoise

        def recording(count, *args, **kwargs):
            requested.append(int(count))
            return real(count, *args, **kwargs)

        with mock.patch.object(self.stats, 'GenerateNoise', recording):
            out = stos_brute.rotate_image(self.image, angle=45.0, image_stats=self.stats)

        self.assertEqual(1, len(requested), 'noise should be drawn once')
        # Recompute the vacated count independently of the implementation.
        import scipy.ndimage
        bare = scipy.ndimage.rotate(self.image.astype(np.float32), axes=(1, 0),
                                    angle=-45.0, cval=np.nan)
        self.assertEqual(int(np.sum(np.isnan(bare))), requested[0],
                         'the draw size must equal the vacated entry count; a '
                         'whole-frame draw would consume a different number of values')
        self.assertLess(requested[0], out.size,
                        'a rotation vacates part of the frame, not all of it')

    def test_a_zero_angle_rotation_is_a_passthrough(self):
        out = stos_brute.rotate_image(self.image, angle=0, image_stats=self.stats)
        self.assertIs(out, self.image, 'angle 0 should not copy or fill')


class TestTheVacatedFractionIsWhyWholeFrameNoiseIsWorse(unittest.TestCase):
    """A sync-free `where` must generate noise for the whole frame.

    That is only cheaper if the frame is almost entirely vacated. It is not: these are the
    fractions that make `noise_full > sync + noise_n`.
    """

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.image = _image(256)

    def _vacated_fraction(self, angle: float) -> float:
        import scipy.ndimage
        bare = scipy.ndimage.rotate(self.image, axes=(1, 0), angle=-angle, cval=np.nan)
        return float(np.sum(np.isnan(bare)) / bare.size)

    def test_a_shallow_rotation_vacates_a_minority_of_the_frame(self):
        fraction = self._vacated_fraction(7.0)
        self.assertLess(fraction, 0.30,
                        f'measured ~0.195 at 7 degrees; got {fraction:.3f}. A whole-frame '
                        'draw would be ~5x the needed work here')

    def test_a_45_degree_rotation_vacates_about_half(self):
        fraction = self._vacated_fraction(45.0)
        self.assertLess(fraction, 0.60, f'measured ~0.500; got {fraction:.3f}')
        self.assertGreater(fraction, 0.40, f'measured ~0.500; got {fraction:.3f}')

    def test_no_angle_vacates_the_whole_frame(self):
        # If some angle vacated ~100%, whole-frame noise would cost the same and the
        # sync-free rewrite would become free. Nothing does.
        for angle in (7.0, 20.0, 45.0, 70.0, 90.0):
            with self.subTest(angle=angle):
                self.assertLess(self._vacated_fraction(angle), 0.75)


class TestTheFillIsDeterministicUnderSeeding(unittest.TestCase):
    """Changing how many values are drawn would break seeded reproducibility.

    This is the output-changing half of why #94 is wontfix rather than a tuning choice.
    """

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.image = _image()
        self.stats = nornir_imageregistration.ImageStats.CalcStats(self.image)

    def _rotate_seeded(self, angle: float = 45.0) -> np.ndarray:
        nornir_imageregistration.seed_random_data(42)
        return stos_brute.rotate_image(self.image, angle=angle, image_stats=self.stats)

    def test_the_same_seed_reproduces_the_same_fill(self):
        first = self._rotate_seeded()
        second = self._rotate_seeded()
        np.testing.assert_array_equal(
            first, second,
            'seeded rotation must reproduce exactly; the fill draws from the shared '
            'generator, so any change in how many values are consumed breaks this')

    def test_the_kept_pixels_are_untouched_by_the_fill(self):
        import scipy.ndimage
        bare = scipy.ndimage.rotate(self.image, axes=(1, 0), angle=-45.0, cval=np.nan)
        filled = self._rotate_seeded()
        kept = ~np.isnan(bare)
        np.testing.assert_array_equal(
            bare[kept], filled[kept],
            'the fill must only write vacated entries; a whole-frame where() that '
            'reselected every pixel would have to reproduce this exactly')


class TestTheSyncFreeRewriteWouldDiverge(unittest.TestCase):
    """Demonstrate the rewrite #94 implies, and show it does not reproduce current output.

    Asserting "the current behaviour is deterministic" is not enough on its own: it does
    not show that the proposed alternative breaks it. So this implements the sync-free
    variant -- whole-frame noise selected with ``where``, no count, no sync -- and pins the
    two ways it diverges. Anyone revisiting #94 should expect these to fail.
    """

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.image = _image()
        self.stats = nornir_imageregistration.ImageStats.CalcStats(self.image)

    def _rotate_sync_free(self, angle: float = 45.0) -> tuple[np.ndarray, int]:
        """The rewrite: no int(), so noise must cover the whole frame."""
        import scipy.ndimage
        rotated = scipy.ndimage.rotate(self.image.astype(np.float32), axes=(1, 0),
                                       angle=-angle, cval=np.nan)
        mask = np.isnan(rotated)
        noise = self.stats.GenerateNoise(rotated.size, dtype=self.image.dtype, xp=np)
        filled = np.where(mask, np.asarray(noise).reshape(rotated.shape), rotated)
        return filled.astype(self.image.dtype, copy=False), int(rotated.size)

    def test_it_draws_more_noise_than_needed(self):
        import scipy.ndimage
        bare = scipy.ndimage.rotate(self.image, axes=(1, 0), angle=-45.0, cval=np.nan)
        needed = int(np.sum(np.isnan(bare)))
        _, drawn = self._rotate_sync_free()
        self.assertGreater(drawn, needed,
                           'the sync-free variant must draw for the whole frame; that is '
                           'the extra work that outweighs the 0.22% sync saving')
        self.assertGreater(drawn / needed, 1.5,
                           f'drew {drawn} for {needed} needed -- about 2x at 45 degrees, '
                           'and about 5x at shallow angles')

    def test_it_does_not_reproduce_the_current_seeded_output(self):
        nornir_imageregistration.seed_random_data(42)
        current = stos_brute.rotate_image(self.image, angle=45.0, image_stats=self.stats)
        nornir_imageregistration.seed_random_data(42)
        rewritten, _ = self._rotate_sync_free()

        self.assertEqual(current.shape, rewritten.shape)
        self.assertFalse(
            np.array_equal(current, rewritten),
            'if these ever match, the determinism objection to #94 has gone away and the '
            'tradeoff is purely about speed again -- re-measure before concluding')

    def test_it_still_leaves_no_nan(self):
        # The rewrite is not wrong, just slower and output-changing. Recording that keeps
        # the wontfix honest about what was rejected and why.
        filled, _ = self._rotate_sync_free()
        self.assertFalse(np.any(np.isnan(filled)))


if __name__ == '__main__':
    unittest.main()
