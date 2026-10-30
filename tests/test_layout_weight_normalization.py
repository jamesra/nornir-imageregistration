"""min/max_allowed_weight in the layout weight helpers (#130).

`NormalizeOffsetWeights` documented "Scale offset weights proportionally so they lie in
[min_allowed_weight, max_allowed_weight]", but assigned both parameters over `minWeight` /
`maxWeight` -- the *source* extrema read from the layout -- then mapped into [0, 1]. So they
named the range normalised *from*, not *to*, the opposite of both the docstring and
`ScaleOffsetWeightsByPopulationRank`, which uses them as the output range.

`TranslateTiles2` passes `config.min_offset_weight` / `max_offset_weight`, documented in
TranslateSettings as "the minimum weight we will allow an offset measurement between two tiles
to have". Measured on link weights 0.1-0.9 (10 directed values over 5 links):

| floor requested | before: output range | before: links at zero weight |
|---|---|---|
| None (default) | [0.0000, 1.0000] | 2 / 10 |
| 0.25 | [0.0000, 0.8667] | 2 / 10 |
| 0.50 | [0.0000, 0.8000] | 6 / 10 |
| 0.90 | [0.0000, 0.0000] | **10 / 10** |

So the floor was not merely ignored, it was inverted: raising it drove *more* weights to
exactly zero, and a floor of 0.9 zeroed every weight in the layout. A zero weight removes that
pair's pull from the relaxation entirely, so asking to trust only good offsets discarded all of
them.

After the fix each of those returns `[floor, 1.0]`, and both helpers agree exactly.

The default path is unchanged: with both None the output is still [0.0, 0.25, 0.5, 0.75, 1.0]
for those inputs, so only a configuration that sets these values behaves differently.

Two of the sibling helpers could not run at all -- `SetOffsetWeights` and
`ScaleOffsetWeightsByPopulationRank` assigned into `node.OffsetArray`, which hands back a
read-only copy, raising "ValueError: assignment destination is read-only". They are repaired
here through the `Weights` setter, because the parity this issue is about cannot be asserted
against a function that raises.
"""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration.layout as L

_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 0.9]


def _chain(weights=None):
    """A chain of nodes carrying `weights` on its links."""
    weights = _WEIGHTS if weights is None else weights
    layout = L.Layout()
    for i in range(len(weights) + 1):
        layout.CreateNode(i, np.asarray((0.0, i * 10.0)))
    for i, weight in enumerate(weights):
        layout.SetOffset(i, i + 1, np.asarray((0.0, 10.0)), weight)
    return layout


def _weights(layout):
    values = []
    for node in layout.nodes.values():
        if node.IsIsolated:
            continue
        values.extend(np.asarray(node.Weights).ravel().tolist())
    return np.asarray(values)


class TestTheConfiguredFloorIsApplied(unittest.TestCase):
    """The reported bug."""

    def test_no_weight_falls_below_the_floor(self):
        for floor in (0.1, 0.25, 0.5, 0.75, 0.9):
            with self.subTest(floor=floor):
                layout = _chain()
                L.NormalizeOffsetWeights(layout, min_allowed_weight=floor,
                                         max_allowed_weight=1.0)
                self.assertGreaterEqual(_weights(layout).min(), floor - 1e-9)

    def test_no_weight_exceeds_the_ceiling(self):
        for ceiling in (0.5, 0.8, 1.0, 2.0):
            with self.subTest(ceiling=ceiling):
                layout = _chain()
                L.NormalizeOffsetWeights(layout, min_allowed_weight=0.0,
                                         max_allowed_weight=ceiling)
                self.assertLessEqual(_weights(layout).max(), ceiling + 1e-9)

    def test_the_full_range_is_used(self):
        layout = _chain()
        L.NormalizeOffsetWeights(layout, min_allowed_weight=0.25, max_allowed_weight=1.0)
        weights = _weights(layout)
        # A floor that is respected but never reached would understate every weight.
        self.assertAlmostEqual(0.25, weights.min(), places=9)
        self.assertAlmostEqual(1.0, weights.max(), places=9)

    def test_a_high_floor_no_longer_zeroes_everything(self):
        # Measured at 10 of 10 weights at exactly zero before the fix.
        layout = _chain()
        L.NormalizeOffsetWeights(layout, min_allowed_weight=0.9, max_allowed_weight=1.0)
        weights = _weights(layout)
        self.assertEqual(0, int(np.count_nonzero(weights <= 1e-9)))
        self.assertAlmostEqual(0.9, weights.min(), places=9)

    def test_no_link_loses_its_vote(self):
        for floor in (0.25, 0.5, 0.9):
            with self.subTest(floor=floor):
                layout = _chain()
                L.NormalizeOffsetWeights(layout, min_allowed_weight=floor,
                                         max_allowed_weight=1.0)
                self.assertEqual(0, int(np.count_nonzero(_weights(layout) <= 1e-9)),
                                 'a zero weight drops that pair from the relaxation')

    def test_the_ordering_of_weights_is_preserved(self):
        layout = _chain()
        L.NormalizeOffsetWeights(layout, min_allowed_weight=0.25, max_allowed_weight=1.0)
        scaled = sorted(np.unique(_weights(layout)).tolist())
        self.assertEqual(len(_WEIGHTS), len(scaled),
                         'distinct inputs should stay distinct')
        self.assertEqual(scaled, sorted(scaled))


class TestTheTwoHelpersNowAgree(unittest.TestCase):
    """The parity this issue is filed under."""

    def test_they_produce_the_same_weights(self):
        for bounds in [(0.0, 1.0), (0.25, 1.0), (0.5, 0.9), (0.1, 0.2)]:
            with self.subTest(bounds=bounds):
                a = _chain()
                L.NormalizeOffsetWeights(a, min_allowed_weight=bounds[0],
                                         max_allowed_weight=bounds[1])
                b = _chain()
                L.ScaleOffsetWeightsByPopulationRank(b, min_allowed_weight=bounds[0],
                                                     max_allowed_weight=bounds[1])
                np.testing.assert_allclose(_weights(a), _weights(b), atol=1e-9)

    def test_they_agree_on_the_defaults(self):
        a = _chain()
        L.NormalizeOffsetWeights(a)
        b = _chain()
        L.ScaleOffsetWeightsByPopulationRank(b)
        np.testing.assert_allclose(_weights(a), _weights(b), atol=1e-9)

    def test_they_agree_when_every_weight_is_equal(self):
        a = _chain([0.4, 0.4, 0.4])
        L.NormalizeOffsetWeights(a, min_allowed_weight=0.25, max_allowed_weight=0.8)
        b = _chain([0.4, 0.4, 0.4])
        L.ScaleOffsetWeightsByPopulationRank(b, min_allowed_weight=0.25,
                                             max_allowed_weight=0.8)
        np.testing.assert_allclose(_weights(a), _weights(b), atol=1e-9)
        np.testing.assert_allclose(0.8, _weights(a), atol=1e-9)


class TestTheDefaultPathIsUnchanged(unittest.TestCase):
    """Only a configuration that sets these values should behave differently."""

    def test_the_default_output_is_zero_to_one(self):
        layout = _chain()
        L.NormalizeOffsetWeights(layout)
        np.testing.assert_allclose([0.0, 0.25, 0.5, 0.75, 1.0],
                                   np.unique(_weights(layout)), atol=1e-9)

    def test_passing_none_matches_passing_nothing(self):
        a = _chain()
        L.NormalizeOffsetWeights(a)
        b = _chain()
        L.NormalizeOffsetWeights(b, min_allowed_weight=None, max_allowed_weight=None)
        np.testing.assert_allclose(_weights(a), _weights(b), atol=1e-9)

    def test_zero_and_one_match_the_defaults(self):
        a = _chain()
        L.NormalizeOffsetWeights(a)
        b = _chain()
        L.NormalizeOffsetWeights(b, min_allowed_weight=0.0, max_allowed_weight=1.0)
        np.testing.assert_allclose(_weights(a), _weights(b), atol=1e-9)

    def test_equal_layout_weights_still_collapse_to_the_ceiling(self):
        layout = _chain([0.4, 0.4, 0.4])
        L.NormalizeOffsetWeights(layout)
        np.testing.assert_allclose(1.0, _weights(layout), atol=1e-9)


class TestEqualBoundsAreHonoured(unittest.TestCase):
    """TranslateSettings documents min == max as "treat all tiles equally"."""

    def test_equal_bounds_give_every_offset_the_same_weight(self):
        layout = _chain()
        L.NormalizeOffsetWeights(layout, min_allowed_weight=0.6, max_allowed_weight=0.6)
        np.testing.assert_allclose(0.6, _weights(layout), atol=1e-9)

    def test_it_does_not_raise_the_way_the_rank_helper_does(self):
        # The population-rank helper rejects min >= max. Equality is a documented option in
        # TranslateSettings, so this one accepts it.
        layout = _chain()
        L.NormalizeOffsetWeights(layout, min_allowed_weight=0.5, max_allowed_weight=0.5)

        with self.assertRaises(ValueError):
            L.ScaleOffsetWeightsByPopulationRank(_chain(), min_allowed_weight=0.5,
                                                 max_allowed_weight=0.5)

    def test_an_inverted_range_is_rejected(self):
        with self.assertRaises(ValueError):
            L.NormalizeOffsetWeights(_chain(), min_allowed_weight=0.9,
                                     max_allowed_weight=0.1)


class TestTheDeadHelpersCanRunAgain(unittest.TestCase):
    """Both raised ValueError on every call before this change."""

    def test_set_offset_weights_works(self):
        layout = _chain()
        L.SetOffsetWeights(layout, 0.5)
        np.testing.assert_allclose(0.5, _weights(layout), atol=1e-9)

    def test_scale_by_population_rank_works(self):
        layout = _chain()
        L.ScaleOffsetWeightsByPopulationRank(layout)
        self.assertEqual(len(_WEIGHTS) * 2, len(_weights(layout)))

    def test_the_offset_array_is_still_read_only(self):
        # The property's contract is unchanged; only the writers were wrong.
        layout = _chain()
        node = layout.nodes[0]
        with self.assertRaises(ValueError):
            node.OffsetArray[:, L.LayoutPosition.iOffsetWeight] = 0.5

    def test_set_offset_weights_leaves_isolated_nodes_alone(self):
        layout = _chain()
        layout.CreateNode(99, np.asarray((0.0, 999.0)))
        L.SetOffsetWeights(layout, 0.5)
        self.assertTrue(layout.nodes[99].IsIsolated)


class TestIsolatedNodesAreSkipped(unittest.TestCase):
    """Pruning leaves nodes with no offsets; they must not break the scaling."""

    def test_an_isolated_node_does_not_prevent_normalization(self):
        layout = _chain()
        layout.CreateNode(99, np.asarray((0.0, 999.0)))
        L.NormalizeOffsetWeights(layout, min_allowed_weight=0.25, max_allowed_weight=1.0)
        self.assertGreaterEqual(_weights(layout).min(), 0.25 - 1e-9)

    def test_an_isolated_node_stays_isolated(self):
        layout = _chain()
        layout.CreateNode(99, np.asarray((0.0, 999.0)))
        L.NormalizeOffsetWeights(layout)
        self.assertTrue(layout.nodes[99].IsIsolated)


if __name__ == '__main__':
    unittest.main()
