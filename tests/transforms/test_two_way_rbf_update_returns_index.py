"""Two-way RBF point edits must return the edited index, as the edit interfaces promise.

`TwoWayRBFWithLinearCorrection` forwarded `Update{Source,Target}PointsBy{Index,Position}`
to its forward RBF but dropped the return value, so callers got None. Pyre's
`TransformController.MovePoint` converts that result to session IDs, so every
registration applied to an RBF transform raised there: the point moved, but the views
were never refreshed and the status-bar progress never cleared.
"""
from __future__ import annotations

import numpy as np
import pytest
from nornir_imageregistration.transforms import TwoWayRBFWithLinearCorrection


def _rbf() -> TwoWayRBFWithLinearCorrection:
    """A 3x3 grid of control points with a small non-rigid offset."""
    fixed = np.array([[i * 100.0, j * 100.0] for i in range(3) for j in range(3)])
    warped = fixed + np.array([5.0, -3.0]) + np.linspace(0.0, 2.0, fixed.shape[0])[:, None]
    return TwoWayRBFWithLinearCorrection(WarpedPoints=warped, FixedPoints=fixed)


@pytest.mark.parametrize("method", ["UpdateTargetPointsByIndex", "UpdateSourcePointsByIndex"])
def test_update_by_index_returns_index(method: str) -> None:
    transform = _rbf()
    side = transform.TargetPoints if "Target" in method else transform.SourcePoints
    moved = np.asarray(side[4], dtype=np.float64) + np.array([1.5, -2.0])
    result = getattr(transform, method)(4, moved, remove_duplicates=False)
    assert result is not None
    assert int(np.atleast_1d(result)[0]) == 4
    side = transform.TargetPoints if "Target" in method else transform.SourcePoints
    np.testing.assert_allclose(side[4], moved)


@pytest.mark.parametrize("method", ["UpdateTargetPointsByPosition", "UpdateSourcePointsByPosition"])
def test_update_by_position_returns_index(method: str) -> None:
    transform = _rbf()
    side = transform.TargetPoints if "Target" in method else transform.SourcePoints
    old = np.asarray(side[4], dtype=np.float64)
    result = getattr(transform, method)(old, old + np.array([1.5, -2.0]))
    assert result is not None
    assert int(np.atleast_1d(result)[0]) == 4
