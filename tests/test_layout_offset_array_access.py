"""Code inside layout.py must read the backing offset array, not the copying property (#132).

``LayoutPosition.OffsetArray`` hands back a defensive read-only *copy*.  That snapshot is worth
paying for at the package boundary, but seven call sites inside the module were paying it too --
four of them copying every row of a node's offsets to read a single ID scalar, and two of them
copying twice per node inside the same loop.

These tests pin the public contract (still a read-only copy) and the internal discipline (the
live paths make no property accesses at all), so the copy cannot creep back in.
"""

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.layout import Layout, LayoutPosition


def _grid(n, spacing=100.0, jitter=7.0, seed=42):
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


class _CountPropertyAccess:
    """Count reads of the public OffsetArray property within a block."""

    def __init__(self):
        self.count = 0

    def __enter__(self):
        self._original = LayoutPosition.OffsetArray
        counter = self
        getter = self._original.fget

        def counted(node):
            counter.count += 1
            return getter(node)

        LayoutPosition.OffsetArray = property(counted)
        return self

    def __exit__(self, *exc):
        LayoutPosition.OffsetArray = self._original


class TestThePublicContractIsUnchanged(unittest.TestCase):
    """External callers, including test_layout_weight_normalization, rely on all of this."""

    def setUp(self):
        self.node = max(_grid(4).nodes.values(), key=lambda n: n.NumConnections)

    def test_it_is_not_writeable(self):
        self.assertFalse(self.node.OffsetArray.flags.writeable)

    def test_assignment_raises(self):
        with self.assertRaises(ValueError):
            self.node.OffsetArray[:, LayoutPosition.iOffsetWeight] = 0.5

    def test_it_is_a_copy_not_a_view(self):
        """A snapshot, so a later mutation of the node does not change an array already handed out."""
        snapshot = self.node.OffsetArray
        self.assertIsNone(snapshot.base)
        before = snapshot.copy()
        self.node.Weights = 0.25
        np.testing.assert_array_equal(before, snapshot)

    def test_it_reports_the_same_values_as_the_backing_array(self):
        np.testing.assert_array_equal(self.node._OffsetArray, self.node.OffsetArray)

    def test_writing_the_snapshot_cannot_corrupt_the_node(self):
        snapshot = self.node.OffsetArray
        escaped = np.array(snapshot)
        escaped[:, LayoutPosition.iOffsetWeight] = 0.99
        self.assertFalse(np.all(self.node.Weights == 0.99))


class TestTheInternalPathsDoNotCopy(unittest.TestCase):
    """The perf guard. Each of these made one or two copies per node before."""

    def test_the_weight_extrema_scan_makes_no_copies(self):
        layout = _grid(6)
        with _CountPropertyAccess() as counter:
            layout.GetOffsetWeightExtrema()
        self.assertEqual(0, counter.count)

    def test_normalizing_weights_makes_no_copies(self):
        layout = _grid(6)
        with _CountPropertyAccess() as counter:
            nornir_imageregistration.layout.NormalizeOffsetWeights(layout)
        self.assertEqual(0, counter.count)

    def test_sorting_offsets_by_weight_makes_no_copies(self):
        layout = _grid(6)
        with _CountPropertyAccess() as counter:
            nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        self.assertEqual(0, counter.count)

    def test_building_a_layout_makes_no_copies(self):
        layout = _grid(6)
        with _CountPropertyAccess() as counter:
            nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(layout)
        self.assertEqual(0, counter.count)

    def test_the_tension_accessors_make_no_copies(self):
        layout = _grid(6)
        with _CountPropertyAccess() as counter:
            _ = layout.MaxTensionVectors
            _ = layout.MinTensionVectors
            _ = layout.MaxTensionMagnitude
            _ = layout.MinTensionMagnitude
        self.assertEqual(0, counter.count)

    def test_relaxation_makes_no_copies(self):
        """Already true before this change; asserted so it stays true."""
        layout = _grid(6)
        with _CountPropertyAccess() as counter:
            nornir_imageregistration.layout.RelaxLayout(layout, max_iter=5,
                                                        max_tension_cutoff=1e-12,
                                                        min_improvement=None)
        self.assertEqual(0, counter.count)


class TestTheResultsAreUnchanged(unittest.TestCase):
    """Reading the backing array must report exactly what the copy did."""

    def test_the_weight_extrema_match_a_direct_scan(self):
        layout = _grid(6)
        expected_min = min(float(np.min(node.Weights))
                           for node in layout.nodes.values() if not node.IsIsolated)
        expected_max = max(float(np.max(node.Weights))
                           for node in layout.nodes.values() if not node.IsIsolated)
        actual_min, actual_max = layout.GetOffsetWeightExtrema()
        self.assertEqual(expected_min, float(actual_min))
        self.assertEqual(expected_max, float(actual_max))

    def test_sorted_offsets_report_the_node_ids_and_weights(self):
        layout = _grid(5)
        sorted_offsets = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        self.assertEqual(5, sorted_offsets.shape[1])
        # One row per undirected link, and only A < B, so no duplicates.
        self.assertEqual(2 * 5 * (5 - 1), sorted_offsets.shape[0])
        self.assertTrue(np.all(sorted_offsets[:, 0] < sorted_offsets[:, 1]))
        # Descending by weight; _sort_array_on_column defaults to ascending=False.
        self.assertTrue(np.all(np.diff(sorted_offsets[:, 4]) <= 0))
        for row in sorted_offsets:
            node = layout.nodes[int(row[0])]
            with self.subTest(link=(int(row[0]), int(row[1]))):
                self.assertEqual(float(node.GetWeight(int(row[1]))), float(row[4]))

    def test_the_tension_accessors_report_a_connected_id(self):
        layout = _grid(5)
        for ID in sorted(layout.nodes):
            node = layout.nodes[ID]
            connected = layout.GetNodes(node.ConnectedIDs)
            connected_ids = set(int(x) for x in node.ConnectedIDs)
            with self.subTest(node=ID):
                for record in (node.MaxTensionVector(connected),
                               node.MinTensionVector(connected),
                               node.MaxTensionMagnitude(connected),
                               node.MinTensionMagnitude(connected)):
                    self.assertIn(int(record.ID), connected_ids)

    def test_the_offsets_are_not_mutated_by_reading_them(self):
        """The internal reads are read-only; hstack builds new storage."""
        layout = _grid(5)
        before = {ID: node._OffsetArray.copy() for ID, node in layout.nodes.items()}
        nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        layout.GetOffsetWeightExtrema()
        _ = layout.MaxTensionVectors
        for ID, node in layout.nodes.items():
            with self.subTest(node=ID):
                np.testing.assert_array_equal(before[ID], node._OffsetArray)

    def test_the_backing_array_stays_writeable(self):
        """Reads must not leave a write=False flag behind on the node's own storage."""
        layout = _grid(4)
        _ = [node.OffsetArray for node in layout.nodes.values()]
        nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        for ID, node in layout.nodes.items():
            with self.subTest(node=ID):
                self.assertTrue(node._OffsetArray.flags.writeable)
        # And a real write still lands.
        node = layout.nodes[0]
        node.Weights = 0.5
        self.assertTrue(np.all(node.Weights == 0.5))


if __name__ == '__main__':
    unittest.main()
