"""LayoutPosition.MaxTensionMagnitude must return the largest tension magnitude (#125).

It called MaxTensionVector -- which returns an ID_Value holding a single (2,) vector -- and then
summed that over axis 1, so every non-empty input raised AxisError.  Its sibling
MinTensionMagnitude is the same computation with argmin and has always worked; these tests pin
the pair to agreement.
"""

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.layout import Layout


def _star():
    """Node 0 linked to 1, 2 and 3 with tension magnitudes 0, 10 and 20 respectively."""
    layout = Layout()
    for ID, position in ((0, (0, 0)), (1, (10, 0)), (2, (0, 20)), (3, (-30, 0))):
        layout.CreateNode(ID, np.asarray(position, dtype=np.float64))
    for (a, b), offset in (((0, 1), (10, 0)), ((0, 2), (0, 10)), ((0, 3), (-10, 0))):
        layout.SetOffset(a, b, np.asarray(offset, dtype=np.float64))
    return layout


class TestTheMaximumIsReported(unittest.TestCase):
    """The reported failure, and the value it should have produced."""

    def setUp(self):
        self.layout = _star()
        self.node = self.layout.nodes[0]
        self.connected = self.layout.GetNodes(self.node.ConnectedIDs)

    def test_it_does_not_raise(self):
        """Before the fix this was AxisError: axis 1 is out of bounds for array of dimension 1."""
        self.node.MaxTensionMagnitude(self.connected)

    def test_it_returns_the_largest_magnitude(self):
        """Tensions are 0, 10 and 20; the largest is 20, against node 3."""
        result = self.node.MaxTensionMagnitude(self.connected)
        self.assertAlmostEqual(20.0, float(result.Value))
        self.assertEqual(3, int(result.ID))

    def test_it_agrees_with_the_tension_vectors_it_summarises(self):
        """Derive the answer independently from TensionVectors."""
        vectors = self.node.TensionVectors(self.connected)
        expected = float(np.max(np.sqrt(np.sum(vectors ** 2, 1))))
        self.assertAlmostEqual(expected, float(self.node.MaxTensionMagnitude(self.connected).Value))

    def test_it_agrees_with_max_tension_vector(self):
        """The magnitude form must summarise the same link the vector form picks."""
        vector_form = self.node.MaxTensionVector(self.connected)
        magnitude_form = self.node.MaxTensionMagnitude(self.connected)
        self.assertEqual(int(vector_form.ID), int(magnitude_form.ID))
        self.assertAlmostEqual(float(np.sqrt(np.sum(np.asarray(vector_form.Value) ** 2))),
                               float(magnitude_form.Value))


class TestItMirrorsItsWorkingSibling(unittest.TestCase):
    """MinTensionMagnitude was always correct; the pair must now be symmetric."""

    def setUp(self):
        self.layout = _star()
        self.node = self.layout.nodes[0]
        self.connected = self.layout.GetNodes(self.node.ConnectedIDs)

    def test_the_minimum_still_works(self):
        """Guard the sibling against collateral damage."""
        result = self.node.MinTensionMagnitude(self.connected)
        self.assertAlmostEqual(0.0, float(result.Value))
        self.assertEqual(1, int(result.ID))

    def test_the_maximum_is_not_below_the_minimum(self):
        for ID in sorted(self.layout.nodes.keys()):
            node = self.layout.nodes[ID]
            connected = self.layout.GetNodes(node.ConnectedIDs)
            if len(connected) == 0:
                continue
            with self.subTest(node=ID):
                self.assertGreaterEqual(float(node.MaxTensionMagnitude(connected).Value),
                                        float(node.MinTensionMagnitude(connected).Value))

    def test_both_return_the_same_shape_of_result(self):
        """Both are documented as (ID, magnitude); the magnitude must be scalar, not a vector."""
        for name in ('MaxTensionMagnitude', 'MinTensionMagnitude'):
            with self.subTest(method=name):
                value = getattr(self.node, name)(self.connected).Value
                self.assertEqual(0, np.asarray(value).ndim)

    def test_the_empty_guard_is_unchanged(self):
        for name in ('MaxTensionMagnitude', 'MinTensionMagnitude'):
            with self.subTest(method=name):
                result = getattr(self.node, name)([])
                self.assertIsNone(result.ID)
                self.assertEqual(0, result.Value)


class TestASingleLink(unittest.TestCase):
    """A node with one connection is the degenerate case for argmax."""

    def test_a_lone_link_reports_itself(self):
        layout = _star()
        for ID in (1, 2, 3):
            node = layout.nodes[ID]
            connected = layout.GetNodes(node.ConnectedIDs)
            with self.subTest(node=ID):
                self.assertEqual(1, len(connected))
                result = node.MaxTensionMagnitude(connected)
                self.assertEqual(0, int(result.ID))
                self.assertAlmostEqual(float(node.MinTensionMagnitude(connected).Value),
                                       float(result.Value))


class TestTheIdIsReadFromTheMatchingRow(unittest.TestCase):
    """magnitudes is indexed by position in connected_nodes, the ID by offset-array row (#255).

    All four accessors are covered, not just the magnitude pair.  Three of them read the ID with
    the connected_nodes index while a fourth had been corrected, so the family disagreed with
    itself: on a reversed sequence MaxTensionVector named node 1 and MinTensionVector named
    node 3, exactly swapping the two answers, while the magnitudes stayed right.  A plausible
    number attributed to the wrong link, with no error raised.
    """

    _MAXIMA = ('MaxTensionVector', 'MaxTensionMagnitude')
    _MINIMA = ('MinTensionVector', 'MinTensionMagnitude')

    def setUp(self):
        self.layout = _star()
        self.node = self.layout.nodes[0]
        self.connected = list(self.layout.GetNodes(self.node.ConnectedIDs))[::-1]
        magnitudes = np.sqrt(np.sum(self.node.TensionVectors(self.connected) ** 2, 1))
        self.i_max = int(magnitudes.argmax())
        self.i_min = int(magnitudes.argmin())

    def test_the_reordering_actually_permutes_the_rows(self):
        """Without this the tests below would pass trivially."""
        iRows = self.node.get_row_indices(self.connected)
        self.assertFalse(np.array_equal(iRows, np.arange(len(self.connected))))

    def test_a_reordered_sequence_still_names_the_node_holding_the_most_tension(self):
        expected = int(self.connected[self.i_max].ID)
        for name in self._MAXIMA:
            with self.subTest(method=name):
                self.assertEqual(expected, int(getattr(self.node, name)(self.connected).ID))

    def test_a_reordered_sequence_still_names_the_node_holding_the_least_tension(self):
        expected = int(self.connected[self.i_min].ID)
        for name in self._MINIMA:
            with self.subTest(method=name):
                self.assertEqual(expected, int(getattr(self.node, name)(self.connected).ID))

    def test_the_maximum_and_minimum_do_not_name_the_same_link(self):
        """The specific way the bug presented: the two answers traded places."""
        for maximum, minimum in zip(self._MAXIMA, self._MINIMA):
            with self.subTest(methods=(maximum, minimum)):
                self.assertNotEqual(int(getattr(self.node, maximum)(self.connected).ID),
                                    int(getattr(self.node, minimum)(self.connected).ID))

    def test_the_reported_id_does_not_depend_on_the_order_it_was_asked_in(self):
        """The invariant behind all of the above: ordering is a caller's convenience."""
        natural = list(self.layout.GetNodes(self.node.ConnectedIDs))
        for name in self._MAXIMA + self._MINIMA:
            with self.subTest(method=name):
                self.assertEqual(int(getattr(self.node, name)(natural).ID),
                                 int(getattr(self.node, name)(self.connected).ID))

    def test_the_magnitude_was_never_the_part_that_was_wrong(self):
        """Pins the diagnosis: only the ID moved, so a value-only test would have passed."""
        natural = list(self.layout.GetNodes(self.node.ConnectedIDs))
        for name in ('MaxTensionMagnitude', 'MinTensionMagnitude'):
            with self.subTest(method=name):
                self.assertAlmostEqual(float(getattr(self.node, name)(natural).Value),
                                       float(getattr(self.node, name)(self.connected).Value))


class TestTheVectorFormsAgreeWithTheMagnitudeForms(unittest.TestCase):
    """The four accessors share one helper; they must not disagree on which link is extreme."""

    def setUp(self):
        self.layout = _star()
        self.node = self.layout.nodes[0]

    def _orders(self):
        natural = list(self.layout.GetNodes(self.node.ConnectedIDs))
        yield 'natural', natural
        yield 'reversed', natural[::-1]

    def test_the_vector_magnitude_equals_the_reported_magnitude(self):
        for label, connected in self._orders():
            for vector_name, magnitude_name in (('MaxTensionVector', 'MaxTensionMagnitude'),
                                                ('MinTensionVector', 'MinTensionMagnitude')):
                with self.subTest(order=label, methods=(vector_name, magnitude_name)):
                    vector_form = getattr(self.node, vector_name)(connected)
                    magnitude_form = getattr(self.node, magnitude_name)(connected)
                    self.assertEqual(int(vector_form.ID), int(magnitude_form.ID))
                    self.assertAlmostEqual(
                        float(np.sqrt(np.sum(np.asarray(vector_form.Value) ** 2))),
                        float(magnitude_form.Value))

    def test_the_empty_guards_keep_their_distinct_sentinels(self):
        """The vector forms return a (2,) zero, the magnitude forms a scalar 0; both keep it."""
        for name in ('MaxTensionVector', 'MinTensionVector'):
            with self.subTest(method=name):
                result = getattr(self.node, name)([])
                self.assertIsNone(result.ID)
                np.testing.assert_array_equal(np.array((0, 0)), np.asarray(result.Value))
        for name in ('MaxTensionMagnitude', 'MinTensionMagnitude'):
            with self.subTest(method=name):
                result = getattr(self.node, name)([])
                self.assertIsNone(result.ID)
                self.assertEqual(0, result.Value)


class TestTheLayoutPropertyIsUnaffected(unittest.TestCase):
    """The live path never touched the broken method; confirm it is unchanged."""

    def test_the_layout_level_maximum(self):
        layout = _star()
        result = layout.MaxTensionMagnitude
        self.assertAlmostEqual(20.0, float(result.Value))
        self.assertEqual((0, 3), tuple(result.ID))

    def test_the_layout_level_minimum(self):
        layout = _star()
        result = layout.MinTensionMagnitude
        self.assertAlmostEqual(0.0, float(result.Value))
        self.assertEqual((0, 1), tuple(result.ID))

    def test_the_per_node_vector_tables_are_unchanged(self):
        """The #255 fix must not move the live output.

        Layout.MaxTensionVectors / MinTensionVectors pass GetNodes(ConnectedIDs), and
        ConnectedIDs comes back in offset-array order, so the two index spaces coincided and
        these tables were already correct.  Values captured from the pre-fix code.
        """
        layout = _star()

        np.testing.assert_allclose(
            np.array([[0.0, 3.0, -20.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0, -10.0],
                      [0.0, 3.0, 20.0, 0.0]]),
            layout.MaxTensionVectors)

        np.testing.assert_allclose(
            np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0, -10.0],
                      [0.0, 3.0, 20.0, 0.0]]),
            layout.MinTensionVectors)


if __name__ == '__main__':
    unittest.main()
