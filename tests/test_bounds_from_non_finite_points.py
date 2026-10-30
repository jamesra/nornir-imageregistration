"""
``BoundsArrayFromPoints`` must reject non-finite points instead of propagating them.

Bare ``min``/``max`` let NaN and Inf through silently. ``np.seterr(invalid='raise')``
(``__init__.py:429``) does not fire on a reduction, so measured before this change:

    clean  -> [ 0.  0. 30. 20.]
    nan    -> [nan  0. nan 20.]
    inf    -> [ 0.  0. inf 20.]
    -inf   -> [-inf   0.  30.  20.]

Only the affected axis degrades, so the damage is easy to miss. Downstream, a NaN
axis produced a Rectangle with ``Area=nan`` and ``Height=nan`` that
``Rectangle.SafeRound`` preserved unchanged, surfacing far away as
"cannot convert float NaN to integer". An Inf axis is worse: it *passes*
``IsValidBoundingBox``, since that predicate only checks ``min < max`` and
``0 < inf`` holds.

The path the finding cites was confirmed: a single NaN source coordinate in a
transform's control points gave ``MappedBoundingBox = MinX: 0 MinY: nan MaxX: 100
MaxY: nan``.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import spatial
from nornir_imageregistration.spatial.converters import BoundsArrayFromPoints


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


CLEAN_2D = np.array([[0.0, 0.0], [10.0, 20.0], [30.0, 5.0]])

BAD_VALUES = {
    'nan': np.nan,
    'positive_inf': np.inf,
    'negative_inf': -np.inf,
}


def test_clean_points_give_expected_bounds():
    np.testing.assert_array_equal(BoundsArrayFromPoints(CLEAN_2D), [0.0, 0.0, 30.0, 20.0])


@pytest.mark.parametrize('name', sorted(BAD_VALUES))
@pytest.mark.parametrize('column', [0, 1])
def test_non_finite_point_is_rejected(name, column):
    """Any non-finite coordinate, in either axis, must raise."""
    points = CLEAN_2D.copy()
    points[1, column] = BAD_VALUES[name]

    with pytest.raises(ValueError, match='non-finite'):
        BoundsArrayFromPoints(points)


def test_error_message_locates_the_offending_row():
    """Diagnosis is the point of the guard, so the message must be specific."""
    points = np.array([[0.0, 0.0], [1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

    with pytest.raises(ValueError) as excinfo:
        BoundsArrayFromPoints(points)

    message = str(excinfo.value)
    assert '1 of 4 rows' in message
    assert 'index 2' in message


def test_multiple_bad_rows_are_counted():
    points = np.array([[0.0, 0.0], [np.nan, 2.0], [3.0, np.inf], [-np.inf, 6.0]])

    with pytest.raises(ValueError) as excinfo:
        BoundsArrayFromPoints(points)

    assert '3 of 4 rows' in str(excinfo.value)


def test_three_dimensional_points_are_guarded():
    points = np.array([[0.0, 1.0, 2.0], [3.0, np.nan, 5.0], [6.0, 7.0, 8.0]])

    with pytest.raises(ValueError, match='non-finite'):
        BoundsArrayFromPoints(points)


def test_three_dimensional_clean_points_still_work():
    points = np.array([[0.0, 1.0, 2.0], [9.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    np.testing.assert_array_equal(
        BoundsArrayFromPoints(points), [0.0, 1.0, 2.0, 9.0, 7.0, 8.0])


def test_large_finite_magnitudes_are_accepted():
    """The guard must reject non-finite values, not merely large ones."""
    points = np.array([[-1e300, -1e300], [1e300, 1e300]])

    bounds = BoundsArrayFromPoints(points)

    assert np.all(np.isfinite(bounds))


def test_rectangle_construction_no_longer_yields_a_nan_rectangle():
    """The consumer that previously produced Area=nan now fails loudly."""
    points = CLEAN_2D.copy()
    points[1, 0] = np.nan

    with pytest.raises(ValueError, match='non-finite'):
        spatial.BoundingPrimitiveFromPoints(points)


def test_inf_bounds_would_have_passed_the_validity_check():
    """Records why Inf needed guarding: IsValidBoundingBox cannot catch it.

    ``IsValidBoundingBox`` only tests ``min < max``, and ``0 < inf`` is True, so an
    infinite bounding box was accepted as valid while a NaN one was rejected only
    incidentally, by NaN comparisons being False.
    """
    assert bool(spatial.IsValidBoundingBox(np.array([0.0, 0.0, np.inf, 20.0]))) is True
    assert bool(spatial.IsValidBoundingBox(np.array([np.nan, 0.0, np.nan, 20.0]))) is False


def test_transform_bounding_box_rejects_a_nan_control_point():
    """The path named in the finding: control points feed MappedBoundingBox."""
    from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback

    rows = []
    for y in (0.0, 50.0, 100.0):
        for x in (0.0, 50.0, 100.0):
            rows.append([y + 3.0, x + 5.0, y, x])
    control_points = np.array(rows)
    control_points[4, 2] = np.nan  # one source coordinate goes bad

    transform = MeshWithRBFFallback(control_points)

    with pytest.raises(ValueError, match='non-finite'):
        _ = transform.MappedBoundingBox
