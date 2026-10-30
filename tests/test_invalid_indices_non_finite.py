"""
``InvalidIndices`` must treat +-Inf as invalid, not just NaN.

Every discrete-to-continuous fallback router keys on this mask
(``gridwithrbffallback.py`` lines 195, 231, 528, 569, 1178, 1357), and
``assemble`` uses it to drop coordinates before scattering. An Inf that reads as
valid is therefore either left unmapped by the fallback or used as a sample
index.

Reachability: on the host ``np.seterr(invalid='raise', divide='raise')``
(``__init__.py:429``) makes most ways of producing Inf raise before it can
propagate. CuPy ignores ``seterr``, so on the GPU path an overflowing
float64->float32 downcast -- exactly what ``assemble.py:132`` does to transform
output -- or a divide by zero yields Inf silently. Inf supplied by a caller also
propagated straight through the router, measured before this change:
``Transform([[inf, inf]])`` returned ``[[inf, inf]]``.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.transforms import utils


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


NON_FINITE_ROWS = {
    'positive_inf_first': [np.inf, 4.0],
    'positive_inf_second': [6.0, np.inf],
    'negative_inf_first': [-np.inf, 5.0],
    'negative_inf_second': [7.0, -np.inf],
    'both_inf': [np.inf, -np.inf],
    'nan_first': [np.nan, 3.0],
    'nan_second': [3.0, np.nan],
    'nan_and_inf': [np.nan, np.inf],
}


@pytest.mark.parametrize('name', sorted(NON_FINITE_ROWS))
def test_non_finite_rows_are_invalid(name):
    """Each non-finite row must be flagged and filtered out."""
    bad_row = NON_FINITE_ROWS[name]
    points = np.array([[1.0, 2.0], bad_row, [8.0, 9.0]], dtype=np.float64)

    filtered, invalid_mask = utils.InvalidIndices(points)

    np.testing.assert_array_equal(invalid_mask, [False, True, False])
    np.testing.assert_array_equal(filtered, [[1.0, 2.0], [8.0, 9.0]])


def test_finite_rows_are_untouched():
    """Large but finite magnitudes stay valid."""
    points = np.array([[1.0, 2.0],
                       [-1e300, 1e300],
                       [0.0, 0.0],
                       [np.finfo(np.float64).max, np.finfo(np.float64).min]])

    filtered, invalid_mask = utils.InvalidIndices(points)

    assert not bool(invalid_mask.any())
    np.testing.assert_array_equal(filtered, points)


def test_all_rows_non_finite_yields_empty_filtered():
    points = np.array([[np.inf, 1.0], [np.nan, np.nan]])

    filtered, invalid_mask = utils.InvalidIndices(points)

    assert bool(invalid_mask.all())
    assert filtered.shape == (0, 2)


def test_four_column_points_are_supported():
    """Control-point arrays are Nx4; any non-finite column invalidates the row."""
    points = np.array([[0.0, 1.0, 2.0, 3.0],
                       [0.0, 1.0, 2.0, np.inf],
                       [4.0, 5.0, 6.0, 7.0]])

    filtered, invalid_mask = utils.InvalidIndices(points)

    np.testing.assert_array_equal(invalid_mask, [False, True, False])
    assert filtered.shape == (2, 4)


def test_filtered_result_is_a_copy():
    """Callers mutate the filtered array; it must not alias the input."""
    points = np.array([[1.0, 2.0], [np.inf, 3.0]])

    filtered, _mask = utils.InvalidIndices(points)
    filtered[0, 0] = -99.0

    assert points[0, 0] == 1.0


def test_empty_input_is_handled():
    points = np.empty((0, 2), dtype=np.float64)

    filtered, invalid_mask = utils.InvalidIndices(points)

    assert filtered.shape == (0, 2)
    assert invalid_mask.shape == (0,)


def test_none_input_raises():
    with pytest.raises(ValueError):
        utils.InvalidIndices(None)  # type: ignore[arg-type]


def test_deprecated_aliases_share_the_behaviour():
    """The misspelled and _GPU aliases must not keep the old NaN-only rule."""
    points = np.array([[1.0, 2.0], [np.inf, 3.0]])

    for alias in (utils.InvalidIndicies, utils.InvalidIndices_GPU):
        _filtered, invalid_mask = alias(points)
        np.testing.assert_array_equal(invalid_mask, [False, True])


class _InfEmittingDiscreteTransform:
    """Stands in for a discrete transform whose output overflows to Inf.

    Reproduces the GPU-path hazard on the host, where seterr would otherwise
    raise before an Inf could reach the router.
    """

    def __init__(self, inf_rows: set[int]):
        self._inf_rows = inf_rows

    def Transform(self, points, **kwargs):
        out = np.asarray(points, dtype=np.float64).copy() + 1000.0
        for row in self._inf_rows:
            out[row, 0] = np.inf
        return out


def test_router_sends_inf_rows_to_the_continuous_fallback():
    """A fallback router must route Inf rows, not pass them through.

    Exercises the router contract directly: the mask decides which rows the
    continuous transform is asked to fix.
    """
    points = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    discrete = _InfEmittingDiscreteTransform(inf_rows={1, 2})

    transformed = discrete.Transform(points)
    _filtered, invalid_mask = utils.InvalidIndices(transformed)

    np.testing.assert_array_equal(invalid_mask, [False, True, True, False])

    # What the router then does with the flagged rows.
    fallback_result = points[invalid_mask] + 500.0
    transformed[invalid_mask] = fallback_result

    assert np.all(np.isfinite(transformed)), 'Inf survived the fallback routing'
    np.testing.assert_array_equal(transformed[1], [510.0, 510.0])
    np.testing.assert_array_equal(transformed[2], [520.0, 520.0])
