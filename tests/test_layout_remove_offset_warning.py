"""LayoutPosition.RemoveOffset's warning (#129).

The line read:

    Warning('Removing non-existent offset: {0}->{1}'.format(self.ID, ID))

`Warning(...)` constructs an exception instance and discards it. Probed: zero warnings recorded
for both the miss case and the success case. And it sat at the same indent as the `if`, so it
also ran when the removal succeeded -- meaning even a corrected `warnings.warn` in that position
would have reported success as a failure.

Before making it real, measured how often the miss branch is taken on real work, since a
warning that fires on every ordinary removal is noise rather than a fix:

| scenario | hits | misses |
|----------|------|--------|
| 4 overlapping tiles, 3 passes | 4 | 0 |
| 9 overlapping tiles | 6 | 0 |
| RemoveOverlap on a pair with nodes but no offset | 0 | 2 |

So it never fires on healthy mosaics. The two misses in the last row are one per direction,
because `Layout.RemoveOverlap` removes both.

Logging rather than `warnings.warn`: this is a runtime data condition, not API misuse, and
`warnings.warn` shows once per call site by default, which would hide repeats. The single
`warnings.warn` in this module is a deprecation notice, which is what that mechanism is for.
"""

from __future__ import annotations

import logging
import unittest
import warnings

import numpy as np

from nornir_imageregistration.layout import Layout, LayoutPosition

_LOGGER_NAME = 'nornir_imageregistration.layout'


def _connected_pair():
    """A node with one offset, to 1."""
    node = LayoutPosition(0, np.zeros(2))
    node.SetOffset(1, np.asarray((5.0, 7.0)), 1.0)
    return node


class TestTheWarningIsNowEmitted(unittest.TestCase):
    """It used to produce nothing at all."""

    def test_removing_an_absent_offset_logs(self):
        node = _connected_pair()
        with self.assertLogs(_LOGGER_NAME, level=logging.WARNING) as captured:
            node.RemoveOffset(99)

        self.assertEqual(1, len(captured.records))
        self.assertIn('non-existent offset', captured.output[0])

    def test_the_message_names_both_ends(self):
        node = LayoutPosition(7, np.zeros(2))
        node.SetOffset(8, np.asarray((1.0, 1.0)), 1.0)
        with self.assertLogs(_LOGGER_NAME, level=logging.WARNING) as captured:
            node.RemoveOffset(42)

        message = captured.output[0]
        self.assertIn('7', message, 'the node being modified should be named')
        self.assertIn('42', message, 'the missing ID should be named')

    def test_removing_from_a_node_with_no_offsets_at_all_logs(self):
        isolated = LayoutPosition(3, np.zeros(2))
        with self.assertLogs(_LOGGER_NAME, level=logging.WARNING) as captured:
            isolated.RemoveOffset(4)
        self.assertEqual(1, len(captured.records))


class TestSuccessIsSilent(unittest.TestCase):
    """The call sat outside the if, so it ran on success too."""

    def test_a_successful_removal_logs_nothing(self):
        node = _connected_pair()
        with self.assertNoLogs(_LOGGER_NAME, level=logging.WARNING):
            node.RemoveOffset(1)

    def test_the_offset_is_actually_removed(self):
        node = _connected_pair()
        self.assertIn(1, list(node.ConnectedIDs))
        node.RemoveOffset(1)
        self.assertNotIn(1, list(node.ConnectedIDs))

    def test_removing_one_of_several_is_silent_and_keeps_the_rest(self):
        node = LayoutPosition(0, np.zeros(2))
        for other in (1, 2, 3):
            node.SetOffset(other, np.asarray((float(other), 0.0)), 1.0)

        with self.assertNoLogs(_LOGGER_NAME, level=logging.WARNING):
            node.RemoveOffset(2)

        self.assertEqual([1, 3], sorted(int(i) for i in node.ConnectedIDs))

    def test_a_second_removal_of_the_same_id_does_warn(self):
        # The first succeeds silently, the second has nothing left to remove.
        node = _connected_pair()
        with self.assertNoLogs(_LOGGER_NAME, level=logging.WARNING):
            node.RemoveOffset(1)
        with self.assertLogs(_LOGGER_NAME, level=logging.WARNING):
            node.RemoveOffset(1)


class TestItGoesThroughLoggingNotWarnings(unittest.TestCase):
    """The unified logging rule, and why warnings.warn is the wrong mechanism here."""

    def test_it_does_not_raise_a_python_warning(self):
        node = _connected_pair()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            node.RemoveOffset(99)
        self.assertEqual([], [w for w in caught if 'non-existent' in str(w.message)])

    def test_repeated_misses_are_all_reported(self):
        # warnings.warn would show the first and suppress the rest by default.
        node = _connected_pair()
        with self.assertLogs(_LOGGER_NAME, level=logging.WARNING) as captured:
            for missing in (90, 91, 92):
                node.RemoveOffset(missing)
        self.assertEqual(3, len(captured.records))

    def test_the_removal_still_returns_none(self):
        node = _connected_pair()
        self.assertIsNone(node.RemoveOffset(99))
        self.assertIsNone(node.RemoveOffset(1))


class TestThroughTheLayoutApi(unittest.TestCase):
    """RemoveOverlap is how this is reached in practice."""

    @staticmethod
    def _two_nodes():
        layout = Layout()
        layout.CreateNode(0, np.zeros(2))
        layout.CreateNode(1, np.asarray((0.0, 32.0)))
        return layout

    def test_removing_an_overlap_that_was_never_offset_warns_once_per_direction(self):
        layout = self._two_nodes()
        with self.assertLogs(_LOGGER_NAME, level=logging.WARNING) as captured:
            layout.RemoveOverlap((0, 1))

        self.assertEqual(2, len(captured.records),
                         'RemoveOverlap removes both directions')

    def test_removing_a_real_overlap_is_silent(self):
        layout = self._two_nodes()
        layout.SetOffset(0, 1, np.asarray((0.0, 32.0)), 1.0)

        with self.assertNoLogs(_LOGGER_NAME, level=logging.WARNING):
            layout.RemoveOverlap((0, 1))

        self.assertFalse(layout.ContainsOffset((0, 1)))

    def test_removing_a_node_and_its_offsets_is_silent(self):
        layout = self._two_nodes()
        layout.CreateNode(2, np.asarray((32.0, 0.0)))
        layout.SetOffset(0, 1, np.asarray((0.0, 32.0)), 1.0)
        layout.SetOffset(0, 2, np.asarray((32.0, 0.0)), 1.0)

        with self.assertNoLogs(_LOGGER_NAME, level=logging.WARNING):
            self.assertTrue(layout.RemoveNode(0))

    def test_a_healthy_layout_produces_no_warnings(self):
        # The measured case: on a connected grid every removal finds its target.
        layout = Layout()
        for i in range(4):
            layout.CreateNode(i, np.asarray((float(i // 2) * 32, float(i % 2) * 32)))
        pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for a, b in pairs:
            layout.SetOffset(a, b, np.asarray((1.0, 1.0)), 1.0)

        with self.assertNoLogs(_LOGGER_NAME, level=logging.WARNING):
            for pair in pairs:
                layout.RemoveOverlap(pair)


if __name__ == '__main__':
    unittest.main()
