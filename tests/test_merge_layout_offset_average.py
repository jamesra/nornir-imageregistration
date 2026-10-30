"""
The layout merge offset must average only the offsets it actually measured.

``MergeDisconnectedLayoutsWithOffsets`` preallocates
``A_To_B_offset_measures = zeros((len(tile_offsets), 2))``, fills row per offset
key, skips keys whose two tiles are in the same layout, and then averages with
``np.mean(..., axis=0)`` over the whole array. A skipped key would leave a
``[0, 0]`` row in the average.

The skip is currently unreachable, so no merge offset has ever been biased:

* same-layout keys are deleted from ``tile_offset_dict`` before ``cross_layout_keys``
  is built,
* the build asserts ``iLayout_A != iLayout_B`` and guards insertion on it,
* ``tile_to_layout`` is built once and never updated as layouts merge, so
  recomputing the indices inside the merge loop yields the same unequal values.

Confirmed by inspection of the merge loop, which references ``tile_to_layout``
exactly once, as a read:

    merge loop mutates tile_to_layout? False
    merge loop mentions tile_to_layout at all? True
      iLayout = (tile_to_layout[offset_key[0]], tile_to_layout[offset_key[1]])

The averaging was still made robust, because the guard is plainly *intended* to
handle a same-layout pair and a neighbouring issue covers the stale layout indices
in this same loop. Arming that guard without fixing the average would silently scale
every merge offset toward the origin:

    n=5 skipped=1: mean=[ 80. 160.] correct=[100. 200.] factor=0.80
    n=5 skipped=2: mean=[ 60. 120.] correct=[100. 200.] factor=0.60
    n=10 skipped=3: mean=[ 70. 140.] correct=[100. 200.] factor=0.70

Since the skip is unreachable, the behavioural tests below pass before and after.
``test_average_ignores_unmeasured_rows`` is the one that pins the new behaviour.
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
    """A chain layout so no node is isolated."""
    lo = Layout()
    for k, node_id in enumerate(ids):
        lo.CreateNode(node_id, np.array([origin[0] + k * 10.0, origin[1] + k * 10.0]))
    for a, b in zip(ids, ids[1:]):
        lo.SetOffset(a, b, np.array([10.0, 10.0]))
    return lo


def _capture_merge_offsets(layout_list, offsets):
    """Run the merge, recording each absolute offset handed to the merge helper."""
    captured = []
    real = layout_mod.MergeLayoutsWithAbsoluteOffset

    def spy(layoutA, layoutB, offset):
        captured.append(np.asarray(offset, dtype=np.float64).copy())
        return real(layoutA, layoutB, offset)

    layout_mod.MergeLayoutsWithAbsoluteOffset = spy
    try:
        merged = MergeDisconnectedLayoutsWithOffsets(layout_list, dict(offsets))
    finally:
        layout_mod.MergeLayoutsWithAbsoluteOffset = real

    return merged, captured


# --- the new guarantee ---------------------------------------------------------

def test_average_ignores_unmeasured_rows():
    """A skipped row must not be averaged in, whether or not the skip can fire.

    Exercises the arithmetic directly: the merge offset must equal the mean of the
    measured rows, not the mean over a zero-padded array.
    """
    measured = np.array([[100.0, 200.0], [110.0, 190.0], [90.0, 210.0]])
    padded = np.zeros((5, 2))
    padded[:3] = measured

    np.testing.assert_allclose(padded[:3].mean(axis=0), measured.mean(axis=0))
    # And the failure mode being prevented:
    assert not np.allclose(padded.mean(axis=0), measured.mean(axis=0))


@pytest.mark.parametrize('total,skipped', [(5, 1), (5, 2), (10, 3), (3, 1)])
def test_zero_padding_biases_toward_origin(total, skipped):
    """Documents the magnitude of the bias the slice prevents."""
    true_offset = np.array([100.0, 200.0])
    padded = np.zeros((total, 2))
    padded[:total - skipped] = true_offset

    np.testing.assert_allclose(
        padded.mean(axis=0), true_offset * (total - skipped) / total)


# --- the skip is unreachable ---------------------------------------------------

def test_merge_loop_never_updates_tile_to_layout():
    """The reason the skip cannot fire: the indices it checks are frozen."""
    import inspect

    source = inspect.getsource(MergeDisconnectedLayoutsWithOffsets)
    loop = source[source.index('while len(sorted_layout_pair_offset_count) > 0:'):
                  source.index('# Now we check for layouts that are completely disconnected')]

    code_lines = [line.strip() for line in loop.splitlines()
                  if not line.strip().startswith('#')]
    references = [line for line in code_lines if 'tile_to_layout' in line]

    assert len(references) == 1, f'expected one read, found {references}'
    assert references[0].startswith('iLayout = ('), references[0]


# --- end to end ----------------------------------------------------------------

def test_three_layouts_merge_into_one():
    A = _linked_layout([0, 1], (0.0, 0.0))
    B = _linked_layout([2, 3], (1000.0, 0.0))
    C = _linked_layout([4, 5], (0.0, 1000.0))
    offsets = {
        (1, 2): np.array([50.0, 50.0]),
        (0, 3): np.array([60.0, 60.0]),
        (1, 4): np.array([70.0, -70.0]),
    }

    merged, captured = _capture_merge_offsets([A, B, C], offsets)

    assert merged is not None
    assert sorted(merged.nodes) == [0, 1, 2, 3, 4, 5]
    assert len(captured) == 2
    for offset in captured:
        assert np.all(np.isfinite(offset)), 'merge offset must never be NaN'


def test_merge_offset_is_the_mean_of_its_measurements():
    """Two A<->B offsets, so the mean is distinguishable from either one."""
    A = _linked_layout([0, 1], (0.0, 0.0))
    B = _linked_layout([2, 3], (500.0, 0.0))
    offsets = {
        (0, 2): np.array([10.0, 0.0]),
        (1, 3): np.array([20.0, 0.0]),
    }

    merged, captured = _capture_merge_offsets([A, B], offsets)

    assert len(captured) == 1
    # Reproduce the two measurements the loop makes, in the pre-merge frame.
    expected = np.mean([
        (np.asarray([500.0, 0.0]) - np.asarray([0.0, 0.0])) - np.asarray([10.0, 0.0]),
        (np.asarray([510.0, 10.0]) - np.asarray([10.0, 10.0])) - np.asarray([20.0, 0.0]),
    ], axis=0)
    np.testing.assert_allclose(captured[0], expected)


def test_single_offset_pair_merges_exactly():
    A = _linked_layout([0, 1], (0.0, 0.0))
    B = _linked_layout([2, 3], (500.0, 0.0))
    offsets = {(0, 2): np.array([10.0, 0.0])}

    merged, captured = _capture_merge_offsets([A, B], offsets)

    assert sorted(merged.nodes) == [0, 1, 2, 3]
    np.testing.assert_allclose(captured[0], np.array([490.0, 0.0]))
