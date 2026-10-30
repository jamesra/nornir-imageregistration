"""
A layout must never be merged against itself via a stale pair index.

``MergeDisconnectedLayoutsWithOffsets`` sorts layout pairs by offset count and pops
them one at a time. As each pair merges, every ``layout_list`` slot holding the
absorbed layout is rebound to the absorbing one, but the pending pair list and
``tile_to_layout`` keep the original indices. So ``assert iLayout_A != iLayout_B``
compares indices that no longer tell you whether the layouts are distinct.

Any three pairwise-connected layouts trigger it. With pairs (0,1), (0,2), (1,2):
after (0,1) and (0,2) merge, every slot holds the same object, so (1,2) passes the
index assertion while ``layout_list[1] is layout_list[2]``.

Measured before the fix, three layouts:

    Layout 0 absorbing 1
    Layout 0 absorbing 2
    Layout 1 absorbing 2
    merge helper called 3 time(s)
      call 2: same_object=True  A.ID=0 B.ID=0
              offset=[-965.66666667  972.33333333]
              *** translating a layout against itself ***

``MergeLayoutsWithAbsoluteOffset`` then runs ``layoutB.Translate(offset)`` followed
by ``layoutA.Merge(layoutB)`` on one object: the whole merged layout shifts by a
residual offset and the merge is a no-op. Node 0 moved from [0, 0] to
[-965.6667, 972.3333].

The translation is uniform, so *relative* geometry survives, and the production
caller runs ``TranslateToZeroOrigin()`` immediately afterwards
(``arrange_mosaic.py:231``), which cancels a global translation exactly. That is why
this went unnoticed. It still matters: any caller that skips that normalization gets
a displaced mosaic, each spurious pair does a full offset-average plus a translation
of every node, and the progress log claims absorptions that are not happening.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import layout as layout_mod
from nornir_imageregistration.layout import Layout, MergeDisconnectedLayoutsWithOffsets


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


def _linked_layout(ids, origin):
    """A chain layout, so no node is isolated."""
    lo = Layout()
    for k, node_id in enumerate(ids):
        lo.CreateNode(node_id, np.array([origin[0] + k * 10.0, origin[1] + k * 10.0]))
    for a, b in zip(ids, ids[1:]):
        lo.SetOffset(a, b, np.array([10.0, 10.0]))
    return lo


def _run_merge(layout_list, offsets):
    """Merge, recording whether each call received the same object as A and B."""
    calls = []
    real = layout_mod.MergeLayoutsWithAbsoluteOffset

    def spy(layoutA, layoutB, offset):
        calls.append({
            'same_object': layoutA is layoutB,
            'offset': np.asarray(offset, dtype=np.float64).copy(),
        })
        return real(layoutA, layoutB, offset)

    layout_mod.MergeLayoutsWithAbsoluteOffset = spy
    try:
        merged = MergeDisconnectedLayoutsWithOffsets(layout_list, dict(offsets))
    finally:
        layout_mod.MergeLayoutsWithAbsoluteOffset = real

    assert merged is not None, 'merging a non-empty layout list must return a layout'
    return merged, calls


def _three_pairwise_connected():
    """Three layouts with offsets between all three pairs."""
    layouts = [
        _linked_layout([0, 1], (0.0, 0.0)),
        _linked_layout([2, 3], (500.0, 0.0)),
        _linked_layout([4, 5], (0.0, 500.0)),
    ]
    offsets = {
        # A<->B, most keys so it merges first
        (0, 2): np.array([10.0, 0.0]),
        (1, 3): np.array([12.0, 0.0]),
        (1, 2): np.array([11.0, 0.0]),
        # A<->C
        (0, 4): np.array([0.0, 10.0]),
        (1, 5): np.array([0.0, 12.0]),
        # B<->C, the pair whose indices go stale
        (2, 4): np.array([-20.0, 20.0]),
    }
    return layouts, offsets


# --- the reported failure -------------------------------------------------------

def test_no_layout_is_merged_against_itself():
    layouts, offsets = _three_pairwise_connected()

    _, calls = _run_merge(layouts, offsets)

    self_merges = [c for c in calls if c['same_object']]
    assert not self_merges, \
        f'{len(self_merges)} self-merge(s), offsets {[c["offset"] for c in self_merges]}'


def test_one_merge_per_absorbed_layout():
    """Three layouts need exactly two merges, regardless of how many pairs exist."""
    layouts, offsets = _three_pairwise_connected()

    _, calls = _run_merge(layouts, offsets)

    assert len(calls) == len(layouts) - 1, \
        f'expected {len(layouts) - 1} merges, got {len(calls)}'


def test_anchor_node_is_not_spuriously_translated():
    """The absorbing layout is never translated, so its nodes must not move."""
    layouts, offsets = _three_pairwise_connected()
    anchor_before = np.asarray(layouts[0].GetPosition(0), dtype=np.float64).copy()

    merged, _ = _run_merge(layouts, offsets)

    np.testing.assert_allclose(
        np.asarray(merged.GetPosition(0), dtype=np.float64), anchor_before)


def test_four_pairwise_connected_layouts():
    """More layouts means more redundant pairs, so more chances to self-merge."""
    layouts = [
        _linked_layout([0, 1], (0.0, 0.0)),
        _linked_layout([2, 3], (500.0, 0.0)),
        _linked_layout([4, 5], (0.0, 500.0)),
        _linked_layout([6, 7], (500.0, 500.0)),
    ]
    offsets = {
        (0, 2): np.array([10.0, 0.0]),
        (1, 3): np.array([12.0, 0.0]),
        (0, 4): np.array([0.0, 10.0]),
        (1, 5): np.array([0.0, 12.0]),
        (0, 6): np.array([10.0, 10.0]),
        (2, 4): np.array([-20.0, 20.0]),
        (2, 6): np.array([0.0, 15.0]),
        (4, 6): np.array([15.0, 0.0]),
    }

    merged, calls = _run_merge(layouts, offsets)

    assert not [c for c in calls if c['same_object']]
    assert len(calls) == 3
    assert sorted(merged.nodes) == [0, 1, 2, 3, 4, 5, 6, 7]


# --- behaviour that must not change --------------------------------------------

def test_all_nodes_still_end_up_in_one_layout():
    layouts, offsets = _three_pairwise_connected()

    merged, _ = _run_merge(layouts, offsets)

    assert sorted(merged.nodes) == [0, 1, 2, 3, 4, 5]


def test_relative_geometry_is_unchanged_by_the_skip():
    """Skipping the self-merge must only remove a global translation.

    These are the relative offsets measured with the self-merge still happening,
    which confirms the spurious translation was uniform.
    """
    layouts, offsets = _three_pairwise_connected()

    merged, _ = _run_merge(layouts, offsets)

    origin = np.asarray(merged.GetPosition(0), dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(merged.GetPosition(2), dtype=np.float64) - origin,
        [985.6666666666666, -3.3333333333333335])
    np.testing.assert_allclose(
        np.asarray(merged.GetPosition(4), dtype=np.float64) - origin,
        [0.0, 989.0])


def test_two_layouts_are_unaffected():
    """The simple case has no redundant pair, so nothing should change."""
    layouts = [_linked_layout([0, 1], (0.0, 0.0)),
               _linked_layout([2, 3], (500.0, 0.0))]
    offsets = {(0, 2): np.array([10.0, 0.0])}

    merged, calls = _run_merge(layouts, offsets)

    assert len(calls) == 1
    assert not calls[0]['same_object']
    assert sorted(merged.nodes) == [0, 1, 2, 3]


def test_merged_positions_are_finite():
    layouts, offsets = _three_pairwise_connected()

    merged, _ = _run_merge(layouts, offsets)

    for node_id in merged.nodes:
        position = np.asarray(merged.GetPosition(node_id), dtype=np.float64)
        assert np.all(np.isfinite(position)), f'node {node_id} is {position}'
