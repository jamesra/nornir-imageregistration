"""RBF transforms answered the first query differently from every later one.

Two compounding problems in `one_way_rbftransform.py`:

1. `Transform` read `UseRigidTransform` before touching the lazy `Weights` property.
   The weight solve is what sets `_rigid_transform`, so on a fresh transform the first
   call saw `UseRigidTransform == False` and went down the RBF branch, while every later
   call saw `True` and went down the rigid branch. Same object, same input array,
   different answer.

2. The rigid shortcut was chosen by `np.allclose(deviation_weights, 0)`, whose default
   absolute tolerance is 1e-8. Those weights multiply the basis function `r^2*log r`,
   which is about 1.15e11 at a 100k-pixel section extent, so a weight of 1e-9 -- "zero"
   to allclose -- still displaces a point by roughly 115 px. Genuinely non-rigid warps
   were therefore declared rigid, and the larger the section the worse it got.

Together these made the bug visible *and* determined which answer was wrong. At a 100k
extent with a 2 px non-rigid component, measured at a control point where the correct
answer is known exactly:

    call 1 (RBF branch)   error 0.0006 px
    call 2 (rigid branch) error 2.6386 px

So the first answer was the correct one and every later answer was wrong. Forcing the
weight solve earlier without also fixing the shortcut test would have made every call
return the 2.6 px answer -- turning a visible inconsistency into a uniform silent error.

The shortcut is now confirmed against the control points in pixels before it is adopted.
A genuinely rigid warp still takes it at every extent tested; the cases the weight test
wrongly accepted no longer do, and pay the full RBF evaluation instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from nornir_imageregistration.transforms import TwoWayRBFWithLinearCorrection
from nornir_imageregistration.transforms.one_way_rbftransform import (
    _MAX_RIGID_EQUIVALENT_ERROR_PIXELS,
    OneWayRBFWithLinearCorrection,
    _rigid_transform_is_equivalent,
)

ANGLE = np.radians(2.0)
ROTATION = np.array([[np.cos(ANGLE), -np.sin(ANGLE)],
                     [np.sin(ANGLE), np.cos(ANGLE)]])
TRANSLATION = np.array([25.0, -40.0])

# A section-scale extent. The bug is scale dependent and invisible at small extents.
SECTION_EXTENT = 100000.0


def _control_points(extent=SECTION_EXTENT, bump=0.0, n=6):
    """Control points related by a rotation plus an optional non-rigid perturbation."""
    g = np.linspace(0.0, extent, n)
    source = np.array([[y, x] for y in g for x in g], dtype=np.float64)
    target = source @ ROTATION.T + TRANSLATION
    if bump:
        phase = 2.0 * np.pi * source[:, 0] / extent
        target = target + np.column_stack((bump * np.sin(phase), bump * np.cos(phase)))
    return source, target


def _query(extent=SECTION_EXTENT):
    return np.array([[extent * 0.5, extent * 0.5],
                     [extent * 0.25, extent * 0.75],
                     [-extent * 0.1, -extent * 0.1],
                     [extent * 1.1, extent * 1.1]], dtype=np.float64)


# --- idempotency ---------------------------------------------------------------

@pytest.mark.parametrize('bump', [0.0, 0.1, 0.5, 2.0, 10.0, 50.0])
def test_the_first_call_matches_the_second(bump):
    source, target = _control_points(bump=bump)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    query = _query()

    first = np.asarray(transform.Transform(query.copy()), dtype=np.float64).copy()
    second = np.asarray(transform.Transform(query.copy()), dtype=np.float64).copy()

    np.testing.assert_array_equal(first, second)


def test_later_calls_remain_stable():
    source, target = _control_points(bump=2.0)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    query = _query()

    transform.Transform(query.copy())
    second = np.asarray(transform.Transform(query.copy()), dtype=np.float64).copy()
    third = np.asarray(transform.Transform(query.copy()), dtype=np.float64).copy()

    np.testing.assert_array_equal(second, third)


def test_the_first_inverse_call_matches_the_second():
    source, target = _control_points(bump=2.0)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    query = _query()

    first = np.asarray(transform.InverseTransform(query.copy()), dtype=np.float64).copy()
    second = np.asarray(transform.InverseTransform(query.copy()), dtype=np.float64).copy()

    np.testing.assert_array_equal(first, second)


def test_precomputing_weights_does_not_change_the_answer():
    """A caller that warms the transform must get the same answer as one that does not."""
    source, target = _control_points(bump=2.0)
    query = _query()

    cold = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    warm = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    warm._forward_rbf.PrecomputeWeights()

    np.testing.assert_array_equal(
        np.asarray(cold.Transform(query.copy()), dtype=np.float64),
        np.asarray(warm.Transform(query.copy()), dtype=np.float64))


def test_the_branch_is_decided_before_the_first_query():
    source, target = _control_points(bump=2.0)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    forward = transform._forward_rbf

    transform.Transform(_query())
    after_first_call = forward.UseRigidTransform

    settled = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    settled._forward_rbf.PrecomputeWeights()

    assert after_first_call == settled._forward_rbf.UseRigidTransform


# --- correctness: the transform must reproduce its own control points ----------

@pytest.mark.parametrize('bump', [0.1, 0.5, 2.0, 10.0])
def test_a_non_rigid_warp_reproduces_its_control_points(bump):
    """This is the assertion the shortcut was silently failing, by up to 13 px."""
    source, target = _control_points(bump=bump)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)

    # Deliberately the second call: the wrong answer only appeared after the first.
    transform.Transform(source.copy())
    mapped = np.asarray(transform.Transform(source.copy()), dtype=np.float64)

    error = np.max(np.abs(mapped - target))
    assert error < 0.05, f'control points off by {error:.4f} px'


def test_a_rigid_warp_still_reproduces_its_control_points():
    source, target = _control_points(bump=0.0)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)

    mapped = np.asarray(transform.Transform(source.copy()), dtype=np.float64)

    assert np.max(np.abs(mapped - target)) < 0.05


# --- the shortcut is still used where it is genuinely valid --------------------

@pytest.mark.parametrize('extent', [1e3, 1e4, SECTION_EXTENT])
def test_a_genuinely_rigid_warp_still_takes_the_shortcut(extent):
    """The guard must not simply disable the fast path."""
    source, target = _control_points(extent=extent, bump=0.0)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    transform._forward_rbf.PrecomputeWeights()

    assert transform._forward_rbf.UseRigidTransform


@pytest.mark.parametrize('bump', [0.1, 0.5, 2.0, 10.0])
def test_a_non_rigid_warp_does_not_take_the_shortcut(bump):
    source, target = _control_points(bump=bump)
    transform = TwoWayRBFWithLinearCorrection(WarpedPoints=source, FixedPoints=target)
    transform._forward_rbf.PrecomputeWeights()

    assert not transform._forward_rbf.UseRigidTransform


def test_the_weight_test_alone_would_have_accepted_these():
    """Guards the premise: without the pixel check these warps look rigid."""
    accepted_by_weights = []
    for bump in (0.1, 0.5, 2.0, 10.0):
        source, target = _control_points(bump=bump)
        _, verdict = OneWayRBFWithLinearCorrection.CalculateRBFWeights(
            source, target, OneWayRBFWithLinearCorrection.DefaultBasisFunction)
        accepted_by_weights.append(bool(verdict))

    assert all(accepted_by_weights), \
        'premise broken: the deviation-weight test no longer accepts these warps'


# --- the equivalence check itself ---------------------------------------------

def test_the_equivalence_check_accepts_an_exact_match():
    source, target = _control_points(bump=0.0)

    class Exact:
        def Transform(self, points):
            return np.asarray(points) @ ROTATION.T + TRANSLATION

    assert _rigid_transform_is_equivalent(Exact(), source, target)


def test_the_equivalence_check_rejects_a_visible_offset():
    source, target = _control_points(bump=0.0)
    offset = _MAX_RIGID_EQUIVALENT_ERROR_PIXELS * 10.0

    class Offset:
        def Transform(self, points):
            return np.asarray(points) @ ROTATION.T + TRANSLATION + offset

    assert not _rigid_transform_is_equivalent(Offset(), source, target)


def test_the_equivalence_threshold_is_sub_pixel():
    """A threshold at or above a pixel would defeat the point of the check."""
    assert 0.0 < _MAX_RIGID_EQUIVALENT_ERROR_PIXELS < 1.0
