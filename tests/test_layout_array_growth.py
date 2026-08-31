"""Offset arrays must not be grown by repeated recopying (#133).

Two separate sites:

* ``OffsetsSortedByWeight`` accumulated rows with ``vstack`` onto a running array, recopying
  everything gathered so far on each of N iterations -- quadratic in the number of rows.
* ``SetOffset`` re-sorted a node's whole offset array on every insertion.

The sort in ``SetOffset`` is load-bearing and is *not* removed here: row order is the summation
order in ``WeightedNetTensionVector``, and float addition is not associative, so reordering rows
perturbs relaxation output.  What changes is that the sort is skipped when appending already
leaves the array sorted, which on a grid build is every insertion.

Both changes must be output-identical, so most of these tests are equivalence tests against an
independent reimplementation or an invariant, plus scaling guards.
"""

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.layout import Layout, LayoutPosition


def _grid(n, spacing=100.0, jitter=7.0, seed=42):
    """Offsets arrive in ascending ID order, the case the append fast path targets."""
    rng = np.random.default_rng(seed)
    layout = Layout()
    ids = {}
    for r in range(n):
        for c in range(n):
            ID = r * n + c
            ids[(r, c)] = ID
            layout.CreateNode(ID, np.asarray((r * spacing, c * spacing), dtype=np.float64))
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                offset = np.asarray((0.0, spacing)) + rng.normal(0, jitter, 2)
                layout.SetOffset(ids[(r, c)], ids[(r, c + 1)], offset,
                                 float(rng.uniform(0.3, 1.0)))
            if r + 1 < n:
                offset = np.asarray((spacing, 0.0)) + rng.normal(0, jitter, 2)
                layout.SetOffset(ids[(r, c)], ids[(r + 1, c)], offset,
                                 float(rng.uniform(0.3, 1.0)))
    return layout


def _random_layout(num_nodes, num_links, seed):
    """Links in arbitrary order, so offsets do NOT arrive with ascending IDs."""
    rng = np.random.default_rng(seed)
    layout = Layout()
    for ID in range(num_nodes):
        layout.CreateNode(ID, np.asarray(rng.normal(0, 500, 2), dtype=np.float64))
    seen = set()
    for _ in range(num_links):
        a, b = int(rng.integers(0, num_nodes)), int(rng.integers(0, num_nodes))
        if a == b or (min(a, b), max(a, b)) in seen:
            continue
        seen.add((min(a, b), max(a, b)))
        layout.SetOffset(a, b, np.asarray(rng.normal(0, 100, 2)),
                         float(rng.uniform(0.1, 1.0)))
    return layout


class TestTheOffsetArrayStaysSorted(unittest.TestCase):
    """The invariant the removed sort was there to maintain."""

    def test_ascending_inserts_stay_sorted(self):
        layout = _grid(6)
        for ID, node in layout.nodes.items():
            with self.subTest(node=ID):
                column = node._OffsetArray[:, LayoutPosition.iOffsetID]
                self.assertTrue(np.all(np.diff(column) > 0))

    def test_arbitrary_order_inserts_stay_sorted(self):
        """The fast path must not fire when the new ID does not belong at the end."""
        layout = _random_layout(60, 600, seed=11)
        checked = 0
        for ID, node in layout.nodes.items():
            if node.IsIsolated:
                continue
            checked += 1
            with self.subTest(node=ID):
                column = node._OffsetArray[:, LayoutPosition.iOffsetID]
                self.assertTrue(np.all(np.diff(column) > 0))
        self.assertGreater(checked, 40)

    def test_descending_inserts_stay_sorted(self):
        node = LayoutPosition(0, np.asarray((0.0, 0.0)))
        for ID in range(30, 0, -1):
            node.SetOffset(ID, np.asarray((float(ID), 0.0)), 0.5)
        column = node._OffsetArray[:, LayoutPosition.iOffsetID]
        self.assertTrue(np.all(np.diff(column) > 0))
        np.testing.assert_array_equal(np.arange(1, 31, dtype=np.float64), column)

    def test_reinserting_a_removed_offset_stays_sorted(self):
        """RemoveOffset then SetOffset can land in the middle, defeating the fast path."""
        node = LayoutPosition(0, np.asarray((0.0, 0.0)))
        for ID in (1, 5, 9, 14, 20):
            node.SetOffset(ID, np.asarray((float(ID), 0.0)), 0.5)
        node.RemoveOffset(9)
        node.SetOffset(9, np.asarray((99.0, 0.0)), 0.75)
        np.testing.assert_array_equal(
            np.asarray((1.0, 5.0, 9.0, 14.0, 20.0)),
            node._OffsetArray[:, LayoutPosition.iOffsetID])
        self.assertEqual(99.0, float(node.GetOffset(9)[0]))

    def test_an_update_does_not_reorder(self):
        node = LayoutPosition(0, np.asarray((0.0, 0.0)))
        for ID in (3, 7, 11):
            node.SetOffset(ID, np.asarray((float(ID), 0.0)), 0.5)
        node.SetOffset(7, np.asarray((123.0, 456.0)), 0.9)
        np.testing.assert_array_equal(np.asarray((3.0, 7.0, 11.0)),
                                      node._OffsetArray[:, LayoutPosition.iOffsetID])
        self.assertEqual(0.9, float(node.GetWeight(7)))

    def test_the_id_index_still_maps_correctly(self):
        """IDToIndex and ConnectedIDs are rebuilt from the array; they must agree with it."""
        for layout in (_grid(5), _random_layout(40, 300, seed=4)):
            for ID, node in layout.nodes.items():
                if node.IsIsolated:
                    continue
                with self.subTest(node=ID):
                    for row, connected_id in enumerate(node.ConnectedIDs):
                        self.assertEqual(row, node.IDToIndex[float(connected_id)])
                        self.assertEqual(float(connected_id),
                                         float(node._OffsetArray[row,
                                                                 LayoutPosition.iOffsetID]))


class TestSortedByWeightIsUnchanged(unittest.TestCase):
    """The concatenate-once rewrite must reproduce the accumulator's output exactly."""

    @staticmethod
    def _reference(layout):
        """The original shape: vstack onto an accumulator."""
        ret_array = np.empty((0, 5))
        for node in layout.nodes.values():
            if node.IsIsolated:
                continue
            offsets = node._OffsetArray
            iNewRows = offsets[:, 0] > node.ID
            if not np.any(iNewRows):
                continue
            new_column = np.ones((int(np.sum(iNewRows)), 1)) * node.ID
            ret_array = np.vstack((ret_array, np.hstack((new_column, offsets[iNewRows, :]))))
        iSorted = np.flipud(np.argsort(ret_array[:, 4], 0))
        return ret_array[iSorted, :]

    def test_it_matches_the_accumulator(self):
        for label, layout in (('grid4', _grid(4)), ('grid8', _grid(8)),
                              ('grid12', _grid(12)),
                              ('rand40', _random_layout(40, 300, seed=6)),
                              ('rand120', _random_layout(120, 2000, seed=7))):
            with self.subTest(layout=label):
                expected = self._reference(layout)
                actual = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
                self.assertEqual(expected.shape, actual.shape)
                np.testing.assert_array_equal(expected, actual)

    def test_the_shape_and_ordering_contract(self):
        layout = _grid(5)
        result = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        self.assertEqual(5, result.shape[1])
        self.assertEqual(2 * 5 * (5 - 1), result.shape[0])
        self.assertTrue(np.all(result[:, 0] < result[:, 1]))
        self.assertTrue(np.all(np.diff(result[:, 4]) <= 0))

    def test_every_link_is_reported_once(self):
        layout = _grid(6)
        result = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        pairs = [(int(row[0]), int(row[1])) for row in result]
        self.assertEqual(len(pairs), len(set(pairs)))
        expected = set()
        for ID, node in layout.nodes.items():
            for other in node.ConnectedIDs:
                expected.add((min(ID, int(other)), max(ID, int(other))))
        self.assertEqual(expected, set(pairs))

    def test_the_weights_match_the_nodes(self):
        layout = _random_layout(50, 400, seed=8)
        for row in nornir_imageregistration.layout.OffsetsSortedByWeight(layout):
            with self.subTest(link=(int(row[0]), int(row[1]))):
                self.assertEqual(float(layout.nodes[int(row[0])].GetWeight(int(row[1]))),
                                 float(row[4]))


class TestDegenerateLayouts(unittest.TestCase):
    """concatenate rejects an empty sequence where the accumulator tolerated it."""

    def test_an_empty_layout(self):
        result = nornir_imageregistration.layout.OffsetsSortedByWeight(Layout())
        self.assertEqual((0, 5), result.shape)

    def test_a_single_node_with_no_offsets(self):
        layout = Layout()
        layout.CreateNode(0, np.asarray((0.0, 0.0)))
        result = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        self.assertEqual((0, 5), result.shape)

    def test_only_isolated_nodes(self):
        layout = Layout()
        for ID in range(5):
            layout.CreateNode(ID, np.asarray((float(ID), 0.0)))
        result = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        self.assertEqual((0, 5), result.shape)

    def test_isolated_nodes_mixed_with_connected_ones(self):
        layout = _grid(3)
        expected = nornir_imageregistration.layout.OffsetsSortedByWeight(layout).shape[0]
        layout.CreateNode(999, np.asarray((0.0, 0.0)))
        result = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        self.assertEqual(expected, result.shape[0])
        self.assertNotIn(999, result[:, 0:2].astype(int).ravel().tolist())

    def test_a_single_link(self):
        layout = Layout()
        layout.CreateNode(0, np.asarray((0.0, 0.0)))
        layout.CreateNode(1, np.asarray((10.0, 0.0)))
        layout.SetOffset(0, 1, np.asarray((10.0, 0.0)), 0.8)
        result = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        self.assertEqual((1, 5), result.shape)
        self.assertEqual([0, 1], result[0, 0:2].astype(int).tolist())
        self.assertEqual(0.8, float(result[0, 4]))


class TestRelaxationIsUnaffected(unittest.TestCase):
    """Row order is the summation order, so any reordering would show up here."""

    def test_relaxation_is_deterministic(self):
        for label, factory in (('grid', lambda: _grid(8)),
                               ('random', lambda: _random_layout(60, 500, seed=9))):
            with self.subTest(layout=label):
                first, second = factory(), factory()
                for layout in (first, second):
                    nornir_imageregistration.layout.RelaxLayout(
                        layout, max_iter=15, max_tension_cutoff=1e-12, min_improvement=None)
                np.testing.assert_array_equal(first.GetPositions(), second.GetPositions())

    def test_relaxation_still_reduces_tension(self):
        layout = _random_layout(60, 500, seed=10)
        before = float(layout.MaxWeightedNetTensionMagnitude.Value)
        nornir_imageregistration.layout.RelaxLayout(layout, max_iter=25,
                                                    max_tension_cutoff=1e-12,
                                                    min_improvement=None)
        self.assertLess(float(layout.MaxWeightedNetTensionMagnitude.Value), before)


class TestTheGrowthIsNoLongerQuadratic(unittest.TestCase):
    """Scaling guard, so the accumulator cannot come back."""

    def test_no_vstack_accumulator_remains(self):
        import inspect
        source = inspect.getsource(nornir_imageregistration.layout.OffsetsSortedByWeight)
        self.assertIn('concatenate', source)
        self.assertNotIn('ret_array = np.vstack', source)

    def test_it_beats_the_accumulator_at_scale(self):
        """Compared against the accumulator in the same run, so no absolute timing threshold.

        The two are within noise of each other at small sizes -- the quadratic term only takes
        over once there are enough rows -- so this deliberately uses a large layout.
        """
        import time

        layout = _grid(72)

        def best_of(fn, reps=3):
            best = None
            for _ in range(reps):
                start = time.perf_counter()
                fn()
                elapsed = time.perf_counter() - start
                best = elapsed if best is None else min(best, elapsed)
            return best

        accumulator = best_of(lambda: TestSortedByWeightIsUnchanged._reference(layout))
        shipped = best_of(lambda: nornir_imageregistration.layout.OffsetsSortedByWeight(layout))
        self.assertLess(shipped, accumulator * 0.8,
                        f"accumulator {accumulator * 1000:.1f}ms vs "
                        f"shipped {shipped * 1000:.1f}ms -- expected a clear margin")


if __name__ == '__main__':
    unittest.main()
