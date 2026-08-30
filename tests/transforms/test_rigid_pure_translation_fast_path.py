"""Rigid.Transform paid a fixed per-call cost on geometrically pure translations.

Two independent problems, at ``rigid.py`` in ``Transform`` / ``InverseTransform``:

1. The fast path was gated on ``self.angle == 0``. Fitting a pure translation returns
   an angle of about ``-5.551115123125783e-17``, so the exact comparison was False for
   a transform that is geometrically a pure translation, and the vector-add shortcut
   was skipped.
2. The matmul branch then called ``_to_xp_array(self.forward_matrix, xp)`` on every
   call. The matrix is built on whichever module ``GetComputationModule`` reports, so
   a host-side caller against a device-resident matrix paid a CuPy ``.get()`` per call.

Measured, host side, matrix CuPy-resident:

                       before      after
    residue,   1k pts   0.1429 ms   0.0042 ms    34x
    residue, 100k pts   2.1081 ms   0.2535 ms   8.3x
    rotating,  1k pts   0.1454 ms   0.0283 ms   5.1x   (matmul, cache only)
    rotating, 100k pts  2.4460 ms   2.1822 ms   1.1x

The second pair matters because it shows the cache helps transforms that genuinely
rotate and so cannot take the fast path at all.

Taking the fast path skips the ``xp.around`` the matmul branch applies, so results
move by up to 3.8e-06 of a pixel for a transform carrying the residue -- far below any
pixel effect, and it makes a near-zero angle agree exactly with a zero angle where
before the two disagreed.

``Rigid`` and ``CenteredSimilarity2DTransform`` share this ``Transform``, and ``xp``
follows the caller's points, so there is no separate GPU mirror to keep in step.
"""
from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.transforms import rigid

# The residue an actual pure-translation fit produces.
RESIDUE_ANGLE = -5.551115123125783e-17
OFFSET = (7.0, -3.0)
CENTER = (50.0, 50.0)


def _points(n=512, seed=0):
    return (np.random.default_rng(seed).random((n, 2)).astype(np.float32) * 100.0)


def _residue():
    return rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                       angle=RESIDUE_ANGLE)


def _exact():
    return rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER, angle=0.0)


# --- the predicate -------------------------------------------------------------

def test_the_residue_still_does_not_equal_zero():
    """Guards the premise: an exact comparison really is False here."""
    assert _residue().angle != 0


def test_a_residue_angle_counts_as_a_pure_translation():
    assert _residue()._is_pure_translation()


def test_an_exact_zero_angle_counts_as_a_pure_translation():
    assert _exact()._is_pure_translation()


def test_a_real_rotation_does_not_count():
    assert not rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                           angle=0.3)._is_pure_translation()


def test_a_flip_never_counts_however_small_the_angle():
    flipped = rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                          angle=RESIDUE_ANGLE, flip_ud=True)
    assert not flipped._is_pure_translation()


def test_a_scaling_transform_does_not_count():
    scaled = rigid.CenteredSimilarity2DTransform(
        target_offset=OFFSET, source_rotation_center=CENTER, angle=0.0, scalar=2.0)
    assert not scaled._is_pure_translation()


def test_an_angle_large_enough_to_matter_is_not_treated_as_zero():
    """A degree of arc must never be mistaken for a translation."""
    assert not rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                           angle=1e-3)._is_pure_translation()


# --- the residue now agrees with exact zero ------------------------------------

def test_a_residue_angle_transforms_like_an_exact_zero():
    points = _points()

    residue = np.asarray(_residue().Transform(points), dtype=np.float64)
    exact = np.asarray(_exact().Transform(points), dtype=np.float64)

    np.testing.assert_array_equal(residue, exact)


def test_a_residue_angle_inverts_like_an_exact_zero():
    points = _points(seed=1)

    residue = np.asarray(_residue().InverseTransform(points), dtype=np.float64)
    exact = np.asarray(_exact().InverseTransform(points), dtype=np.float64)

    np.testing.assert_array_equal(residue, exact)


def test_the_translation_is_still_applied():
    points = _points(seed=2)

    moved = np.asarray(_residue().Transform(points), dtype=np.float64)

    # atol is float32 resolution at these coordinate magnitudes, not a real tolerance.
    expected = np.broadcast_to(np.asarray(OFFSET, dtype=np.float64), moved.shape)
    np.testing.assert_allclose(moved - np.asarray(points, dtype=np.float64),
                               expected, atol=1e-3)


def test_the_fast_path_round_trips():
    points = _points(seed=3)
    transform = _residue()

    back = transform.InverseTransform(transform.Transform(points))

    np.testing.assert_allclose(np.asarray(back, dtype=np.float64),
                               np.asarray(points, dtype=np.float64), atol=1e-4)


# --- the matrix cache ----------------------------------------------------------

def test_a_rotation_still_round_trips():
    """The cache must not change what the matmul branch computes."""
    points = _points(seed=4)
    transform = rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                            angle=0.3)

    back = transform.InverseTransform(transform.Transform(points))

    np.testing.assert_allclose(np.asarray(back, dtype=np.float64),
                               np.asarray(points, dtype=np.float64), atol=1e-4)


def test_the_cached_matrix_matches_the_live_one():
    transform = rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                            angle=0.3)

    forward, inverse = transform._matrices_for(np)

    np.testing.assert_allclose(
        np.asarray(forward, dtype=np.float64),
        np.asarray(nornir_imageregistration.EnsureNumpyArray(transform.forward_matrix),
                   dtype=np.float64))
    np.testing.assert_allclose(
        np.asarray(inverse, dtype=np.float64),
        np.asarray(nornir_imageregistration.EnsureNumpyArray(transform.inverse_matrix),
                   dtype=np.float64))


def test_rotating_the_transform_invalidates_the_cache():
    """A stale cache would keep returning the pre-rotation mapping."""
    points = _points(seed=5)
    transform = rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                            angle=0.3)

    before = np.asarray(transform.Transform(points), dtype=np.float64).copy()
    transform.RotateSourcePoints(0.6, None)
    after = np.asarray(transform.Transform(points), dtype=np.float64)

    assert not np.allclose(before, after), 'cache outlived a rotation'

    transform.RotateSourcePoints(-0.6, None)
    restored = np.asarray(transform.Transform(points), dtype=np.float64)
    np.testing.assert_allclose(restored, before, atol=1e-4)


def test_restoring_state_invalidates_the_cache():
    points = _points(seed=6)
    transform = rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                            angle=0.3)
    snapshot = transform.GetRigidState()

    before = np.asarray(transform.Transform(points), dtype=np.float64).copy()
    transform.RotateSourcePoints(0.6, None)
    transform.SetRigidState(snapshot)
    after = np.asarray(transform.Transform(points), dtype=np.float64)

    np.testing.assert_allclose(after, before, atol=1e-4)


def test_a_pickled_transform_still_transforms():
    """The cache must not travel to a pool worker, where CuPy cannot initialise."""
    import pickle

    points = _points(seed=7)
    transform = rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                            angle=0.3)
    expected = np.asarray(transform.Transform(points), dtype=np.float64)

    revived = pickle.loads(pickle.dumps(transform))
    actual = np.asarray(revived.Transform(points), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_the_cache_is_not_pickled():
    import pickle

    transform = rigid.Rigid(target_offset=OFFSET, source_rotation_center=CENTER,
                            angle=0.3)
    transform._matrices_for(np)

    assert '_matrix_cache' not in pickle.loads(pickle.dumps(transform)).__dict__ or \
        pickle.loads(pickle.dumps(transform))._matrix_cache == {}
