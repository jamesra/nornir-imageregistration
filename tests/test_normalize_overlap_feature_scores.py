"""NormalizeOverlapFeatureScores must survive unscored and all-zero feature scores (#123).

Three failure modes were reachable:

* every score zero (an all-blank overlap set) divided by a zero max -> ZeroDivisionError
* a ``None`` score, which ``ScoreTileOverlaps`` explicitly allows, hit ``max()`` -> TypeError
* the parameter is annotated ``Iterable`` but consumed twice, so a generator silently produced
  no normalization at all

Only the attributes the function touches are stubbed; building a real ``TileOverlap`` needs a
pair of ``Tile`` objects with images, which would say nothing extra about this arithmetic.
"""

import unittest

import numpy as np

from nornir_imageregistration import arrange_mosaic


class _Overlap:
    """Stands in for TileOverlap's feature_scores / normalized_feature_scores pair."""

    def __init__(self, ID, feature_scores):
        self.ID = ID
        self.feature_scores = feature_scores
        self.normalized_feature_scores = None


def _normalize(pairs):
    overlaps = [_Overlap(i, scores) for i, scores in enumerate(pairs)]
    arrange_mosaic.NormalizeOverlapFeatureScores(overlaps)
    return [o.normalized_feature_scores for o in overlaps]


class TestTheOrdinaryCase(unittest.TestCase):
    """The arithmetic that already worked must be untouched."""

    def test_scores_are_divided_by_the_largest(self):
        result = _normalize([(0.2, 0.8), (0.4, 0.6)])
        self.assertEqual([(0.25, 1.0), (0.5, 0.7499999999999999)], result)

    def test_the_largest_score_normalizes_to_one(self):
        result = _normalize([(1.5, 3.0), (0.75, 2.25)])
        self.assertAlmostEqual(1.0, max(max(pair) for pair in result))

    def test_every_result_is_within_zero_and_one(self):
        result = _normalize([(0.1, 0.9), (0.5, 0.2), (0.33, 0.66)])
        for pair in result:
            for value in pair:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class TestAllScoresZero(unittest.TestCase):
    """An all-blank overlap set: max_score is 0."""

    def test_it_does_not_raise(self):
        """Before the fix this was ZeroDivisionError: division by zero."""
        _normalize([(0.0, 0.0), (0.0, 0.0)])

    def test_normalization_becomes_a_no_op(self):
        """Uniform 1.0: there is no relative information, so no overlap is preferred."""
        self.assertEqual([(1.0, 1.0), (1.0, 1.0)], _normalize([(0.0, 0.0), (0.0, 0.0)]))

    def test_a_single_zero_score_among_real_ones_is_still_zero(self):
        """Only the all-zero case is special; a lone zero must stay zero."""
        result = _normalize([(0.0, 0.5), (0.25, 0.5)])
        self.assertEqual(0.0, result[0][0])


class TestUnscoredEntries(unittest.TestCase):
    """None and nan both mean 'not scored yet' and must not be normalized."""

    def test_a_none_score_does_not_raise(self):
        """Before the fix: TypeError: '>' not supported between 'float' and 'NoneType'."""
        _normalize([(None, 0.5), (0.3, 0.7)])

    def test_a_none_score_is_passed_through(self):
        result = _normalize([(None, 0.5), (0.3, 0.7)])
        self.assertIsNone(result[0][0])

    def test_a_none_score_does_not_affect_the_maximum(self):
        """The maximum must come from the scored values only."""
        with_none = _normalize([(None, 0.5), (0.3, 0.7)])
        without = _normalize([(0.5, 0.5), (0.3, 0.7)])
        self.assertAlmostEqual(without[0][1], with_none[0][1])

    def test_a_nan_score_is_passed_through(self):
        result = _normalize([(np.nan, np.nan), (0.3, 0.7)])
        self.assertTrue(np.isnan(result[0][0]))
        self.assertTrue(np.isnan(result[0][1]))

    def test_a_nan_score_does_not_depend_on_ordering(self):
        """Python's max() with nan is order-dependent; the reduction must not be."""
        nan_first = _normalize([(np.nan, np.nan), (0.3, 0.7)])
        nan_last = _normalize([(0.3, 0.7), (np.nan, np.nan)])
        self.assertAlmostEqual(nan_first[1][1], nan_last[0][1])
        self.assertAlmostEqual(1.0, nan_last[0][1])

    def test_every_score_unscored_does_not_raise(self):
        result = _normalize([(None, None), (np.nan, np.nan)])
        self.assertEqual((None, None), result[0])
        self.assertTrue(all(np.isnan(v) for v in result[1]))


class TestTheIterableIsConsumedOnce(unittest.TestCase):
    """Both loops need the same elements."""

    def test_a_generator_is_still_normalized(self):
        """Before the fix the second loop saw an exhausted generator and did nothing."""
        overlaps = [_Overlap(0, (0.2, 0.8)), _Overlap(1, (0.4, 0.6))]
        arrange_mosaic.NormalizeOverlapFeatureScores(o for o in overlaps)
        for o in overlaps:
            with self.subTest(overlap=o.ID):
                self.assertIsNotNone(o.normalized_feature_scores)

    def test_a_generator_matches_the_list_result(self):
        overlaps = [_Overlap(0, (0.2, 0.8)), _Overlap(1, (0.4, 0.6))]
        arrange_mosaic.NormalizeOverlapFeatureScores(o for o in overlaps)
        from_generator = [o.normalized_feature_scores for o in overlaps]
        self.assertEqual(_normalize([(0.2, 0.8), (0.4, 0.6)]), from_generator)


class TestDegenerateInput(unittest.TestCase):

    def test_an_empty_sequence_does_not_raise(self):
        arrange_mosaic.NormalizeOverlapFeatureScores([])

    def test_a_single_overlap_normalizes_its_own_maximum_to_one(self):
        self.assertEqual([(0.5, 1.0)], _normalize([(0.4, 0.8)]))


class TestTheResultShape(unittest.TestCase):
    """Downstream reads min(normalized_feature_scores), so arity must hold."""

    def test_a_pair_is_returned_for_every_case(self):
        for label, pairs in (('ordinary', [(0.2, 0.8)]),
                             ('all zero', [(0.0, 0.0)]),
                             ('none', [(None, 0.5)]),
                             ('nan', [(np.nan, 0.5)])):
            with self.subTest(case=label):
                result = _normalize(pairs)
                self.assertEqual(2, len(result[0]))
                self.assertIsInstance(result[0], tuple)


if __name__ == '__main__':
    unittest.main()
