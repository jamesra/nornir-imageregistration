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
    """magnitudes is indexed by position in connected_nodes, the ID by offset-array row (#255)."""

    def test_a_reordered_connected_sequence_still_names_the_right_node(self):
        layout = _star()
        node = layout.nodes[0]
        connected = list(layout.GetNodes(node.ConnectedIDs))[::-1]
        vectors = node.TensionVectors(connected)
        i = int(np.sqrt(np.sum(vectors ** 2, 1)).argmax())
        self.assertEqual(int(connected[i].ID), int(node.MaxTensionMagnitude(connected).ID))

    def test_the_reordering_actually_permutes_the_rows(self):
        """Without this the test above would pass trivially."""
        layout = _star()
        node = layout.nodes[0]
        connected = list(layout.GetNodes(node.ConnectedIDs))[::-1]
        iRows = node.get_row_indices(connected)
        self.assertFalse(np.array_equal(iRows, np.arange(len(connected))))


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


if __name__ == '__main__':
    unittest.main()
