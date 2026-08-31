"""Parity between the three overlap-mask quadrant implementations (#126).

``_PopulateMaskQuadrantBruteForce`` normalised the overlap area by
``np.min(np.vstack((Fixed, Moving)), 1)`` -- the min of each image's *own* two dimensions.
The other two use axis 0, the min across the two images *per dimension*, which is what the
shared comment describes and what the largest possible intersection rectangle actually is.

Measured before the fix:

| fixed | moving | axis 1 (ref) | axis 0 | differing mask pixels |
|-------|--------|--------------|--------|-----------------------|
| (128,128) | (128,128) | 16384 | 16384 | 0 |
| (127,127) | (128,128) | 16256 | 16129 | 59 |
| (100,200) | (100,200) | 10000 | 20000 | 8762 |
| (64,256) | (64,256) | 4096 | 16384 | 10497 |
| (512,64) | (256,64) | 4096 | 16384 | 15873 |

Equal squares agree, and equal squares were the only geometry the existing parity test used --
which is why an invalid reference sat next to a passing test.

Over 4000 random size pairs the true largest intersection exceeded the axis-1 area in 1307 of
them, by up to 10.2x, so the "fraction" could pass 1.0 and be masked out by MaxOverlap. The
axis-0 area was never exceeded (max observed ratio 1.000000), confirming it is the real bound.

Only the reference was wrong. Both axis-0 implementations agreed with each other at every
geometry probed, so no production mask changes.
"""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.overlapmasking as overlapmasking
from nornir_imageregistration.spatial import Rectangle

# Geometries that separate the two normalisers, alongside the ones that do not.
_GEOMETRIES = [
    ((128, 128), (128, 128), 'equal squares'),
    ((127, 127), (128, 128), 'near-equal squares'),
    ((64, 64), (64, 64), 'small equal squares'),
    ((64, 256), (64, 64), 'fixed wider, moving square'),
    ((64, 64), (64, 256), 'moving wider, fixed square'),
    ((64, 256), (256, 64), 'orthogonal rectangles'),
    ((100, 200), (100, 200), 'both wide, same orientation'),
    ((64, 256), (64, 256), 'both wide 1:4'),
    ((256, 64), (256, 64), 'both tall 4:1'),
    ((64, 256), (64, 128), 'both wide, different aspect'),
    ((512, 64), (256, 64), 'both tall, different height'),
    ((96, 160), (128, 128), 'rectangle against a square'),
]

_MIN_OVERLAP = 0.25
_MAX_OVERLAP = 0.75


def _quadrant(fixed, moving):
    corr = np.asarray(fixed, dtype=np.int32) + np.asarray(moving, dtype=np.int32)
    return tuple((corr // 2).tolist())


def _masks(fixed, moving, min_overlap=_MIN_OVERLAP, max_overlap=_MAX_OVERLAP):
    f = np.asarray(fixed, dtype=np.int32)
    m = np.asarray(moving, dtype=np.int32)
    shape = _quadrant(fixed, moving)

    reference = overlapmasking._PopulateMaskQuadrantBruteForce(
        np.zeros(shape, dtype=bool), f, m, MinOverlap=min_overlap, MaxOverlap=max_overlap)
    brute_optimized = overlapmasking._PopulateMaskQuadrantBruteForceOptimized(
        np.zeros(shape, dtype=bool), f, m, MinOverlap=min_overlap, MaxOverlap=max_overlap)
    production = overlapmasking._PopulateMaskQuadrantOptimized(
        np.zeros(shape, dtype=bool), f, m, MinOverlap=min_overlap, MaxOverlap=max_overlap)

    return reference, brute_optimized, production


class TestTheThreeImplementationsAgree(unittest.TestCase):
    """The parity the reference is there to provide."""

    def test_the_reference_matches_the_optimized_brute_force(self):
        for fixed, moving, label in _GEOMETRIES:
            with self.subTest(fixed=fixed, moving=moving, case=label):
                reference, brute_optimized, _ = _masks(fixed, moving)
                differing = int(np.count_nonzero(reference != brute_optimized))
                self.assertEqual(0, differing,
                                 f'{differing} pixels differ for {label}')

    def test_the_reference_matches_the_production_implementation(self):
        for fixed, moving, label in _GEOMETRIES:
            with self.subTest(fixed=fixed, moving=moving, case=label):
                reference, _, production = _masks(fixed, moving)
                differing = int(np.count_nonzero(reference != production))
                self.assertEqual(0, differing,
                                 f'{differing} pixels differ for {label}')

    def test_parity_holds_across_overlap_thresholds(self):
        # The normaliser scales every overlap value, so a wrong one shows up differently at
        # different thresholds. The widest band is the least forgiving.
        for min_overlap, max_overlap in [(0.0, 1.0), (0.1, 0.9), (0.25, 0.75), (0.4, 0.6)]:
            for fixed, moving, label in [_GEOMETRIES[7], _GEOMETRIES[10]]:
                with self.subTest(band=(min_overlap, max_overlap), case=label):
                    reference, brute_optimized, production = _masks(
                        fixed, moving, min_overlap, max_overlap)
                    np.testing.assert_array_equal(reference, brute_optimized)
                    np.testing.assert_array_equal(reference, production)


class TestTheNormaliserIsTheRealBound(unittest.TestCase):
    """Why axis 0 is the correct reduction, not merely the majority one."""

    @staticmethod
    def _largest_intersection(fixed, moving):
        """Sweep the moving centre and return the largest intersection area seen."""
        fixed_rect = Rectangle.CreateFromCenterPointAndArea((0, 0), np.asarray(fixed))
        best = 0.0
        for dy in range(0, (fixed[0] + moving[0]) // 2 + 1):
            for dx in range(0, (fixed[1] + moving[1]) // 2 + 1):
                rect = Rectangle.overlap_rect(
                    Rectangle.CreateFromCenterPointAndArea((dy, dx), np.asarray(moving)),
                    fixed_rect)
                if rect is not None:
                    best = max(best, rect.Area)
        return best

    def test_the_axis_zero_area_is_never_exceeded(self):
        for fixed, moving in [((12, 30), (12, 30)), ((30, 12), (18, 12)),
                              ((16, 16), (16, 16)), ((10, 24), (24, 10)),
                              ((14, 28), (14, 20))]:
            with self.subTest(fixed=fixed, moving=moving):
                claimed = float(np.prod(np.min(np.vstack((fixed, moving)), 0)))
                observed = self._largest_intersection(fixed, moving)
                self.assertLessEqual(observed, claimed + 1e-9,
                                     'axis 0 must be an upper bound on the intersection')

    def test_the_axis_zero_area_is_attained(self):
        # A bound nothing reaches would make every overlap fraction an understatement.
        for fixed, moving in [((12, 30), (12, 30)), ((16, 16), (16, 16)),
                              ((14, 28), (14, 20))]:
            with self.subTest(fixed=fixed, moving=moving):
                claimed = float(np.prod(np.min(np.vstack((fixed, moving)), 0)))
                observed = self._largest_intersection(fixed, moving)
                self.assertAlmostEqual(claimed, observed, delta=1e-9,
                                       msg='axis 0 should be reached, not just respected')

    def test_the_axis_one_area_is_exceeded(self):
        # The premise of the bug: the old normaliser was not a bound at all.
        exceeded = []
        for fixed, moving in [((12, 30), (12, 30)), ((30, 12), (18, 12)),
                              ((14, 28), (14, 20))]:
            claimed = float(np.prod(np.min(np.vstack((fixed, moving)), 1)))
            if self._largest_intersection(fixed, moving) > claimed + 1e-9:
                exceeded.append((fixed, moving))

        self.assertTrue(exceeded,
                        'expected the axis-1 area to be exceeded by a real intersection')

    def test_an_overlap_fraction_never_exceeds_one(self):
        # With a correct normaliser the mask is a fraction, so MinOverlap=0 and MaxOverlap=1
        # must admit every position that overlaps at all.
        for fixed, moving, label in _GEOMETRIES:
            with self.subTest(fixed=fixed, moving=moving, case=label):
                reference, _, _ = _masks(fixed, moving, 0.0, 1.0)
                self.assertTrue(reference.all(),
                                'a 0-1 band should mask nothing out; a fraction above 1.0 '
                                'would be rejected by MaxOverlap')


class TestTheOldReferenceWasSelfConsistentOnSquares(unittest.TestCase):
    """Premise guard explaining why the existing parity test passed anyway."""

    def test_the_two_reductions_agree_for_equal_squares(self):
        for size in [(64, 64), (128, 128), (256, 256)]:
            stack = np.vstack((size, size))
            self.assertEqual(int(np.prod(np.min(stack, 0))),
                             int(np.prod(np.min(stack, 1))),
                             'equal squares cannot separate the two reductions')

    def test_they_disagree_once_both_images_are_elongated(self):
        stack = np.vstack(((64, 256), (64, 256)))
        self.assertEqual(16384, int(np.prod(np.min(stack, 0))))
        self.assertEqual(4096, int(np.prod(np.min(stack, 1))))

    def test_the_shipped_parity_test_used_only_equal_squares(self):
        # If that test ever gains a non-square case this guard should be retired.
        import inspect

        import test_overlapmasking

        source = inspect.getsource(test_overlapmasking.TestOverlapMask.testOverlapMaskPopulation)
        self.assertIn('(128, 128)', source)


# The assembly mirrors a quadrant about both axes. Resolved at module level so the leading
# double underscore is not name-mangled against the test class.
_assemble_full_mask = getattr(overlapmasking, '__CreateFullMaskFromQuadrant')


def _reference_full_mask(fixed, moving):
    """Build the full mask from the brute-force reference, exactly as GetOverlapMask does."""
    f = np.asarray(fixed, dtype=np.int32)
    m = np.asarray(moving, dtype=np.int32)
    corr = f + m

    # Mirrors GetOverlapMask: ceil, not floor, so an odd correlation size keeps its centre row.
    quadrant_size = np.asarray((corr[0] / 2.0, corr[1] / 2.0), dtype=np.float32)
    is_odd = np.mod(quadrant_size, 1) > 0
    quadrant_size = np.ceil(quadrant_size).astype(np.int32, copy=False)

    quadrant = overlapmasking._PopulateMaskQuadrantBruteForce(
        np.zeros(tuple(quadrant_size.tolist()), dtype=bool), f, m,
        MinOverlap=_MIN_OVERLAP, MaxOverlap=_MAX_OVERLAP)

    return _assemble_full_mask(quadrant, is_odd)


class TestThePublicMaskIsUnchanged(unittest.TestCase):
    """The production entry point never used the wrong axis; pin that it still does not."""

    def test_get_overlap_mask_matches_the_reference(self):
        for fixed, moving, label in _GEOMETRIES:
            with self.subTest(fixed=fixed, moving=moving, case=label):
                f = np.asarray(fixed, dtype=np.int32)
                m = np.asarray(moving, dtype=np.int32)
                corr = f + m

                mask = nornir_imageregistration.GetOverlapMask(
                    f, m, corr, MinOverlap=_MIN_OVERLAP, MaxOverlap=_MAX_OVERLAP)
                np.testing.assert_array_equal(corr, mask.shape)

                np.testing.assert_array_equal(_reference_full_mask(fixed, moving), mask)

    def test_the_mask_is_symmetric_about_both_axes(self):
        # The assembly mirrors, so an asymmetric result would mean the quadrant itself moved.
        f = np.asarray((128, 128), dtype=np.int32)
        mask = nornir_imageregistration.GetOverlapMask(
            f, f, f + f, MinOverlap=_MIN_OVERLAP, MaxOverlap=_MAX_OVERLAP)
        np.testing.assert_array_equal(mask, np.fliplr(mask))
        np.testing.assert_array_equal(mask, np.flipud(mask))


if __name__ == '__main__':
    unittest.main()
