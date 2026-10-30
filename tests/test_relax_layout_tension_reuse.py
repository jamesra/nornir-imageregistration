"""RelaxLayout must not re-evaluate tension vectors it already has (#131).

Each pass used to evaluate every node's weighted net tension three times:

1. ``RelaxNodes`` first loop, to decide the order the movement loop visits nodes in
2. ``RelaxNodes`` movement loop, which must re-evaluate because it moves nodes as it goes
3. ``MaxWeightedNetTensionMagnitude``, to test convergence

Pass 3 and the next iteration's pass 1 compute the same values -- nothing between them moves a
node -- so pass 1 can reuse them.  Pass 2 is inherent to the sequential update and stays.

These tests pin both halves: that the reuse is *valid* (the state really is unchanged), and that
it is *taken* (the evaluation count actually dropped), plus that results are unchanged.
"""

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.layout import Layout


def _grid(n, spacing=100.0, jitter=7.0, seed=42):
    """An n x n 4-connected grid with perturbed offsets, so there is real tension to relax."""
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


class _CountEvaluations:
    """Count per-node weighted net tension evaluations inside a block."""

    def __init__(self):
        self.count = 0

    def __enter__(self):
        self._original = Layout.WeightedNetTensionVector
        counter = self

        def counted(layout_self, ID):
            counter.count += 1
            return counter._original(layout_self, ID)

        Layout.WeightedNetTensionVector = counted
        return self

    def __exit__(self, *exc):
        Layout.WeightedNetTensionVector = self._original


def _relax(layout, iterations):
    nornir_imageregistration.layout.RelaxLayout(layout, max_iter=iterations,
                                                max_tension_cutoff=1e-12,
                                                min_improvement=None)


class TestTheReuseIsValid(unittest.TestCase):
    """The premise: the layout state does not change between pass 3 and the next pass 1."""

    def test_evaluating_the_maximum_does_not_move_anything(self):
        layout = _grid(5)
        before = layout.GetPositions().copy()
        _ = layout.MaxWeightedNetTensionMagnitude
        np.testing.assert_array_equal(before, layout.GetPositions())

    def test_consecutive_evaluations_agree_exactly(self):
        layout = _grid(5)
        first = layout.WeightedNetTensionVectors()
        second = layout.WeightedNetTensionVectors()
        np.testing.assert_array_equal(first, second)

    def test_the_helper_agrees_with_the_public_property(self):
        layout = _grid(5)
        vectors, record = Layout._max_weighted_net_tension(layout)
        public = layout.MaxWeightedNetTensionMagnitude
        self.assertEqual(public.ID, record.ID)
        self.assertEqual(float(public.Value), float(record.Value))

    def test_the_id_index_matches_the_vectors(self):
        layout = _grid(4)
        vectors, _ = Layout._max_weighted_net_tension(layout)
        by_id = Layout._tension_vectors_by_id(vectors)
        self.assertEqual(len(layout.nodes), len(by_id))
        for ID, node in layout.nodes.items():
            if node.IsIsolated:
                continue
            with self.subTest(node=ID):
                np.testing.assert_array_equal(layout.WeightedNetTensionVector(ID), by_id[ID])


class TestTheResultIsUnchanged(unittest.TestCase):
    """Reuse must not perturb the answer, including the visit order it feeds."""

    def test_one_pass_matches_between_the_two_paths(self):
        recomputed = _grid(6)
        reused = _grid(6)
        Layout.RelaxNodes(recomputed, vector_scalar=1.0)
        vectors, _ = Layout._max_weighted_net_tension(reused)
        Layout.RelaxNodes(reused, vector_scalar=1.0,
                          node_tension_vectors=Layout._tension_vectors_by_id(vectors))
        np.testing.assert_array_equal(recomputed.GetPositions(), reused.GetPositions())

    def test_the_visit_order_is_identical(self):
        """The sort keys drive a sequential update, so any reordering would change positions."""
        recomputed = _grid(6)
        reused = _grid(6)
        order_a = Layout.RelaxNodes(recomputed, vector_scalar=1.0)
        vectors, _ = Layout._max_weighted_net_tension(reused)
        order_b = Layout.RelaxNodes(reused, vector_scalar=1.0,
                                    node_tension_vectors=Layout._tension_vectors_by_id(vectors))
        np.testing.assert_array_equal(order_a, order_b)

    def test_a_full_relaxation_is_deterministic(self):
        """Two runs of the shipped path agree exactly."""
        first = _grid(8)
        second = _grid(8)
        _relax(first, 20)
        _relax(second, 20)
        np.testing.assert_array_equal(first.GetPositions(), second.GetPositions())

    def test_relaxation_still_reduces_tension(self):
        for n in (4, 8):
            with self.subTest(grid=n):
                layout = _grid(n)
                before = float(layout.MaxWeightedNetTensionMagnitude.Value)
                _relax(layout, 25)
                after = float(layout.MaxWeightedNetTensionMagnitude.Value)
                self.assertLess(after, before)


class TestTheReuseIsTaken(unittest.TestCase):
    """Guards the optimization itself, so a refactor cannot silently undo it."""

    def test_two_evaluations_per_node_per_pass(self):
        """Was three: the sort pass no longer evaluates."""
        iterations = 5
        for n in (4, 6):
            with self.subTest(grid=n):
                layout = _grid(n)
                num_nodes = len(layout.nodes)
                with _CountEvaluations() as counter:
                    _relax(layout, iterations)
                # One extra full sweep before the loop establishes the starting tension.
                expected = num_nodes * iterations * 2 + num_nodes
                self.assertEqual(expected, counter.count)

    def test_the_sort_pass_does_not_evaluate_when_given_vectors(self):
        layout = _grid(5)
        vectors, _ = Layout._max_weighted_net_tension(layout)
        by_id = Layout._tension_vectors_by_id(vectors)
        connected = sum(1 for node in layout.nodes.values() if not node.IsIsolated)
        with _CountEvaluations() as counter:
            Layout.RelaxNodes(layout, vector_scalar=1.0, node_tension_vectors=by_id)
        self.assertEqual(connected, counter.count)

    def test_the_sort_pass_still_evaluates_when_not_given_vectors(self):
        layout = _grid(5)
        connected = sum(1 for node in layout.nodes.values() if not node.IsIsolated)
        with _CountEvaluations() as counter:
            Layout.RelaxNodes(layout, vector_scalar=1.0)
        self.assertEqual(connected * 2, counter.count)


class TestBackwardCompatibility(unittest.TestCase):
    """RelaxNodes is called directly by other tests with only vector_scalar."""

    def test_the_new_parameter_is_optional(self):
        layout = _grid(5)
        before = layout.GetPositions().copy()
        movement = Layout.RelaxNodes(layout, vector_scalar=1.0)
        self.assertEqual(2, movement.shape[1])
        self.assertFalse(np.array_equal(before, layout.GetPositions()))

    def test_isolated_nodes_are_still_excluded(self):
        """#128's guarantee must survive: one row per connected node only."""
        layout = _grid(4)
        layout.CreateNode(999, np.asarray((0.0, 0.0)))
        connected = sum(1 for node in layout.nodes.values() if not node.IsIsolated)
        vectors, _ = Layout._max_weighted_net_tension(layout)
        movement = Layout.RelaxNodes(layout, vector_scalar=1.0,
                                     node_tension_vectors=Layout._tension_vectors_by_id(vectors))
        self.assertEqual(connected, movement.shape[0])
        self.assertNotIn(999, movement[:, 0].astype(int).tolist())


if __name__ == '__main__':
    unittest.main()
