"""Two-way RBF transforms must advertise and honour control point add/remove.

`TwoWayRBFWithLinearCorrection` implemented AddPoint/RemovePoint but did not declare
`IControlPointAddRemove`. Pyre gates delete (Shift+right-click, Delete key) and point
creation on that interface, so in RBF mode the commands ran and silently did nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
from nornir_imageregistration.transforms import (
    IControlPointAddRemove,
    TwoWayRBFWithLinearCorrection,
)


def _rbf() -> TwoWayRBFWithLinearCorrection:
    """A 3x3 grid of control points with a small non-rigid offset."""
    fixed = np.array([[i * 100.0, j * 100.0] for i in range(3) for j in range(3)])
    warped = fixed + np.array([5.0, -3.0]) + np.linspace(0.0, 2.0, fixed.shape[0])[:, None]
    return TwoWayRBFWithLinearCorrection(WarpedPoints=warped, FixedPoints=fixed)


def test_declares_add_remove_interface() -> None:
    assert isinstance(_rbf(), IControlPointAddRemove)


def test_remove_point_keeps_inverse_in_step() -> None:
    transform = _rbf()
    removed_target = np.asarray(transform.TargetPoints[4], dtype=np.float64)
    transform.RemovePoint(4)
    assert transform.NumControlPoints == 8
    remaining = np.asarray(transform.TargetPoints, dtype=np.float64)
    assert not np.any(np.all(np.isclose(remaining, removed_target), axis=1))
    mapped = np.asarray(transform.Transform(np.asarray(transform.SourcePoints)), dtype=np.float64)
    np.testing.assert_allclose(mapped, transform.TargetPoints, atol=1e-4)
    inverse = np.asarray(transform.InverseTransform(np.asarray(transform.TargetPoints)), dtype=np.float64)
    np.testing.assert_allclose(inverse, transform.SourcePoints, atol=1e-4)


def test_add_point_returns_index_of_new_point() -> None:
    transform = _rbf()
    pair = np.array([150.0, 50.0, 153.0, 48.0])
    index = transform.AddPoint(pair)
    assert index is not None
    assert transform.NumControlPoints == 10
    np.testing.assert_allclose(transform.TargetPoints[int(index)], pair[:2])
    inverse = np.asarray(transform.InverseTransform(pair[:2].reshape(1, 2)), dtype=np.float64)
    np.testing.assert_allclose(inverse.ravel(), pair[2:], atol=1e-4)


def test_inverse_still_inverts_after_point_update() -> None:
    """Every point edit rebuilds the inverse RBF; it must map target back to source."""
    transform = _rbf()
    moved = np.asarray(transform.TargetPoints[4], dtype=np.float64) + np.array([1.5, -2.0])
    transform.UpdateTargetPointsByIndex(4, moved, remove_duplicates=False)
    inverse = np.asarray(transform.InverseTransform(np.asarray(transform.TargetPoints)), dtype=np.float64)
    np.testing.assert_allclose(inverse, transform.SourcePoints, atol=1e-4)


def test_points_setter_restores_snapshot() -> None:
    """Pyre's undo and drag-cancel restore control points by assigning ``points``."""
    transform = _rbf()
    snapshot = np.asarray(transform.points, dtype=np.float64).copy()
    moved = np.asarray(transform.TargetPoints[4], dtype=np.float64) + np.array([1.5, -2.0])
    transform.UpdateTargetPointsByIndex(4, moved, remove_duplicates=False)
    transform.points = snapshot
    np.testing.assert_allclose(transform.points, snapshot)
    inverse = np.asarray(transform.InverseTransform(np.asarray(transform.TargetPoints)), dtype=np.float64)
    np.testing.assert_allclose(inverse, transform.SourcePoints, atol=1e-4)


@pytest.mark.parametrize("count", [1, 2])
def test_remove_array_of_indices(count: int) -> None:
    transform = _rbf()
    transform.RemovePoint(np.arange(count))
    assert transform.NumControlPoints == 9 - count
