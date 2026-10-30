"""Regressions for #197: MergeDisconnectedLayouts nearest-pair + cost."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import layout as layout_mod
from nornir_imageregistration.layout import Layout, MergeDisconnectedLayouts


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


class TestMergeDisconnectedLayoutsNearestPair(unittest.TestCase):
    def test_links_global_nearest_pair(self) -> None:
        """Independent row/col argmin used to link A0–B0 at dist 10 instead of 1."""
        layout_a = Layout()
        layout_a.CreateNode(0, np.array([0.0, 0.0]))
        layout_a.CreateNode(1, np.array([3.0, 0.0]))
        layout_b = Layout()
        layout_b.CreateNode(10, np.array([3.0, 1.0]))
        layout_b.CreateNode(11, np.array([0.0, 1.0]))

        merged = MergeDisconnectedLayouts([layout_a, layout_b])
        connected = {
            int(nid): {int(x) for x in node.ConnectedIDs}
            for nid, node in merged.nodes.items()
        }
        # Either nearest pair (0,11) or (1,10) — both at sq dist 1 — not (0,10).
        cross = (0 in connected[10] or 10 in connected[0]
                 or 0 in connected[11] or 11 in connected[0]
                 or 1 in connected[10] or 10 in connected[1]
                 or 1 in connected[11] or 11 in connected[1])
        self.assertTrue(cross)
        self.assertNotIn(10, connected[0])
        self.assertNotIn(0, connected[10])

    def test_does_not_use_growing_pairwise_cdist(self) -> None:
        """#197: merging many singleton layouts must not accumulate O(N²) cdist cells."""
        layouts = []
        for i in range(40):
            layout = Layout()
            layout.CreateNode(i, np.array([float(i), 0.0]))
            layouts.append(layout)

        cells = {'n': 0}
        real = layout_mod.pairwise_cdist

        def _spy(a, b, **kwargs):
            cells['n'] += int(a.shape[0] * b.shape[0])
            return real(a, b, **kwargs)

        with patch.object(layout_mod, 'pairwise_cdist', side_effect=_spy):
            MergeDisconnectedLayouts(layouts)

        self.assertEqual(cells['n'], 0)


if __name__ == '__main__':
    unittest.main()
