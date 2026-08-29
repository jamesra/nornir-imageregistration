"""
``Layout.RelaxNodes`` must not let isolated nodes leak into the movement loop.

``node_movement`` was preallocated as ``zeros((len(nodes), 2))`` with row i holding
``[node.ID, sort_weight]``, but the ``NumConnections == 0`` guard skipped the
assignment, leaving a ``[0.0, 0.0]`` row. The movement loop then walked *every* row
and read column 0 as a node ID, so each isolated node contributed a spurious visit
to node ID 0.

Measured before the fix, a 4-node star plus 2 isolated nodes:

    node_movement rows (id, weight):
      [ 0.  22.36067977]
      [ 1.  11.18033989]
      [ 2.  22.36067977]
      [ 3.  33.54101966]
      [0. 0.]
      [0. 0.]
    movement loop will visit ids: [0, 0, 1, 0, 2, 3]
    ids visited more than once: {0: 3}

Node 0 ended at [-1.111, -0.556] instead of [-3.333, -1.667], a 2.2 px error that
varies with how many nodes happen to be isolated. With no node 0 in the layout it
was a hard ``KeyError: 0`` instead.

Isolated nodes are expected rather than hypothetical: ``LayoutPosition`` documents
"Sometimes we have tiles which end up isolated, usually due to prune. When this
occurs they have no offsets."
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.layout import Layout


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


def _star(connected_ids, isolated_ids):
    """A star layout: connected_ids[0] is the hub, the rest are spokes under tension.

    Nodes in isolated_ids get no offsets, so NumConnections == 0 for them.
    """
    layout = Layout()
    for node_id in list(connected_ids) + list(isolated_ids):
        layout.CreateNode(node_id, np.array([0.0, 0.0]))

    hub = connected_ids[0]
    for k, spoke in enumerate(connected_ids[1:], start=1):
        layout.SetOffset(hub, spoke, np.array([10.0 * k, 5.0 * k]))

    return layout


def _visited_ids(node_movement):
    """Reproduce the order the movement loop walks, to count duplicate visits."""
    order = np.argsort(node_movement[:, 1])
    return [int(v) for v in node_movement[order, 0]]


# --- the reported failure modes -------------------------------------------------

@pytest.mark.parametrize('num_isolated', [1, 2, 5])
def test_isolated_nodes_do_not_add_spurious_rows(num_isolated):
    isolated = list(range(100, 100 + num_isolated))
    layout = _star([0, 1, 2, 3], isolated)

    node_movement = Layout.RelaxNodes(layout, vector_scalar=1.0)

    assert node_movement.shape[0] == 4, \
        f'expected one row per connected node, got {node_movement.shape[0]}'


@pytest.mark.parametrize('num_isolated', [1, 2, 5])
def test_no_node_is_relaxed_twice(num_isolated):
    isolated = list(range(100, 100 + num_isolated))
    layout = _star([0, 1, 2, 3], isolated)

    visited = _visited_ids(Layout.RelaxNodes(layout, vector_scalar=1.0))

    assert sorted(visited) == [0, 1, 2, 3], f'movement loop visited {visited}'


def test_layout_without_node_zero_does_not_raise():
    """The KeyError case: node IDs need not start at 0 after prune."""
    layout = _star([1, 2, 3, 4], [7, 8])

    node_movement = Layout.RelaxNodes(layout, vector_scalar=1.0)

    assert sorted(_visited_ids(node_movement)) == [1, 2, 3, 4]


def test_isolated_nodes_do_not_perturb_the_connected_result():
    """Positions must not depend on how many nodes happen to be isolated."""
    clean = _star([0, 1, 2, 3], [])
    with_isolated = _star([0, 1, 2, 3], [7, 8, 9])

    Layout.RelaxNodes(clean, vector_scalar=1.0)
    Layout.RelaxNodes(with_isolated, vector_scalar=1.0)

    for node_id in (0, 1, 2, 3):
        np.testing.assert_allclose(
            np.asarray(with_isolated.GetPosition(node_id), dtype=np.float64),
            np.asarray(clean.GetPosition(node_id), dtype=np.float64),
            err_msg=f'node {node_id} moved differently because of isolated nodes')


def test_isolated_nodes_are_left_where_they_were():
    layout = _star([0, 1, 2, 3], [7, 8])
    before = {i: np.asarray(layout.GetPosition(i), dtype=np.float64).copy()
              for i in (7, 8)}

    Layout.RelaxNodes(layout, vector_scalar=1.0)

    for node_id, position in before.items():
        np.testing.assert_allclose(
            np.asarray(layout.GetPosition(node_id), dtype=np.float64), position)


# --- edges ----------------------------------------------------------------------

def test_all_nodes_isolated_is_a_no_op():
    layout = Layout()
    for node_id in (3, 4, 5):
        layout.CreateNode(node_id, np.array([1.0, 2.0]))

    node_movement = Layout.RelaxNodes(layout, vector_scalar=1.0)

    assert node_movement.shape == (0, 2)
    for node_id in (3, 4, 5):
        np.testing.assert_allclose(
            np.asarray(layout.GetPosition(node_id), dtype=np.float64), [1.0, 2.0])


def test_returned_ids_are_exactly_the_connected_nodes():
    layout = _star([2, 5, 9], [7])

    node_movement = Layout.RelaxNodes(layout, vector_scalar=1.0)

    assert sorted(int(v) for v in node_movement[:, 0]) == [2, 5, 9]


def test_no_isolated_nodes_behaves_as_before():
    """Guards the common path against regression from this change."""
    layout = _star([0, 1, 2, 3], [])

    node_movement = Layout.RelaxNodes(layout, vector_scalar=1.0)

    assert node_movement.shape[0] == 4
    assert sorted(_visited_ids(node_movement)) == [0, 1, 2, 3]
