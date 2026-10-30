"""
``_FindTileOffsets`` must not read ``f_score`` outside the guard that binds it.

At ``arrange_mosaic.py:630-637`` the weight code read:

    feature_scores = tile_overlap.normalized_feature_scores
    if feature_scores is not None:
        f_score = min(feature_scores)

    final_weight = offset.weight  # * f_score

    if use_feature_score:
        final_weight *= f_score

``f_score`` was bound only inside the ``is not None`` test but read under a
different condition, which fails two ways:

1. ``UnboundLocalError`` when the first overlap with an alignment result has no
   normalized scores and ``use_feature_score`` is on.
2. Worse, and not in the original finding: this is a loop body, so ``f_score``
   survives between iterations. An unscored overlap following a scored one
   silently reused the *previous* overlap's texture measurement. Measured with
   overlap (0,1) scored 0.25 and overlap (2,3) unscored, (2,3) received weight
   0.125 == 0.5 * 0.25 rather than its own 0.5.

``TranslateSettings.feature_score_calculations_required`` is
``feature_score_threshold is not None or use_feature_score is True``, and
``ArrangeTilesWithTranslate`` runs ``ScoreTileOverlaps`` plus
``NormalizeOverlapFeatureScores`` whenever it is set, so the supported pipeline
always populates the scores. ``_FindTileOffsets`` is reachable directly though
(``use_feature_score`` is a plain keyword argument, default False), so the fix
raises ValueError naming the missing call rather than guessing a score.

These tests drive ``_FindTileOffsets`` with a canned alignment result so only the
weight arithmetic runs. That keeps them fast and independent of image data, and it
is the code path the bug lives in.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration import arrange_mosaic

Rectangle = nornir_imageregistration.Rectangle


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


class _FakeTile:
    def __init__(self, tile_id: int):
        self.ID = tile_id
        self.ImagePath = f'fake_{tile_id}.png'
        self.FixedBoundingBox = Rectangle.CreateFromBounds(
            np.array([tile_id * 100.0, 0.0, tile_id * 100.0 + 100.0, 100.0]))


class _FakeOverlap:
    """Only the attributes _FindTileOffsets reads."""

    def __init__(self, a_id: int, b_id: int,
                 normalized_feature_scores: tuple[float, float] | None):
        self.A = _FakeTile(a_id)
        self.B = _FakeTile(b_id)
        self.ID = (a_id, b_id)
        self.normalized_feature_scores = normalized_feature_scores
        self.scaled_offset = np.array([10.0, 0.0])
        self.scaled_overlapping_source_rect_A = Rectangle.CreateFromBounds(
            np.array([0.0, 0.0, 50.0, 50.0]))
        self.scaled_overlapping_source_rect_B = Rectangle.CreateFromBounds(
            np.array([0.0, 0.0, 50.0, 50.0]))


_ALIGNMENT_WEIGHT = 0.5


def _find_offsets(overlaps, use_feature_score: bool):
    """Run _FindTileOffsets with the phase correlation replaced by a fixed record."""

    def canned(*args, **kwargs):
        return nornir_imageregistration.AlignmentRecord(
            peak=np.array([10.0, 0.0]), weight=_ALIGNMENT_WEIGHT)

    real_remote = getattr(arrange_mosaic, '__tile_offset_remote')
    real_pool = nornir_pools.GetGlobalMultithreadingPool
    setattr(arrange_mosaic, '__tile_offset_remote', canned)
    # The canned closure is not picklable and the weight logic does not care which
    # pool ran the alignment, so keep everything on the serial pool.
    nornir_pools.GetGlobalMultithreadingPool = nornir_pools.GetGlobalSerialPool
    try:
        return arrange_mosaic._FindTileOffsets(
            overlaps, excess_scalar=1.0, image_to_source_space_scale=1.0,
            use_feature_score=use_feature_score)
    finally:
        setattr(arrange_mosaic, '__tile_offset_remote', real_remote)
        nornir_pools.GetGlobalMultithreadingPool = real_pool


def _weight(layout, a_id: int, b_id: int) -> float:
    return layout.nodes[a_id].GetWeight(b_id)


# --- the reported failure -------------------------------------------------------

def test_missing_scores_raises_a_named_error():
    """Was UnboundLocalError, which named f_score rather than the real problem."""
    overlaps = [_FakeOverlap(0, 1, None)]

    with pytest.raises(ValueError, match='normalized_feature_scores'):
        _find_offsets(overlaps, use_feature_score=True)


def test_unbound_local_error_is_gone():
    overlaps = [_FakeOverlap(0, 1, None)]

    with pytest.raises(Exception) as caught:
        _find_offsets(overlaps, use_feature_score=True)

    assert not isinstance(caught.value, UnboundLocalError), caught.value


def test_score_does_not_leak_from_the_previous_overlap():
    """The stale-value half of the bug.

    Before the fix, overlap (2, 3) silently took overlap (0, 1)'s score and got
    weight 0.125 instead of 0.5. It must not be assigned a score that is not its own.
    """
    overlaps = [_FakeOverlap(0, 1, (0.25, 0.25)), _FakeOverlap(2, 3, None)]

    with pytest.raises(ValueError, match=r'\(2, 3\)'):
        _find_offsets(overlaps, use_feature_score=True)


def test_error_names_the_offending_overlap():
    """Not the scored one that happens to precede it."""
    overlaps = [_FakeOverlap(0, 1, (0.25, 0.25)), _FakeOverlap(2, 3, None)]

    with pytest.raises(ValueError) as caught:
        _find_offsets(overlaps, use_feature_score=True)

    assert '(2, 3)' in str(caught.value)
    assert '(0, 1)' not in str(caught.value)


# --- behaviour that must not change --------------------------------------------

def test_scores_are_applied_per_overlap():
    overlaps = [_FakeOverlap(0, 1, (0.25, 0.25)), _FakeOverlap(2, 3, (0.8, 0.8))]

    layout = _find_offsets(overlaps, use_feature_score=True)

    assert _weight(layout, 0, 1) == pytest.approx(_ALIGNMENT_WEIGHT * 0.25)
    assert _weight(layout, 2, 3) == pytest.approx(_ALIGNMENT_WEIGHT * 0.8)


def test_min_of_the_two_scores_is_used():
    overlaps = [_FakeOverlap(0, 1, (0.9, 0.3))]

    layout = _find_offsets(overlaps, use_feature_score=True)

    assert _weight(layout, 0, 1) == pytest.approx(_ALIGNMENT_WEIGHT * 0.3)


def test_feature_score_off_ignores_missing_scores():
    """The default path must not care that scores were never computed."""
    overlaps = [_FakeOverlap(0, 1, None)]

    layout = _find_offsets(overlaps, use_feature_score=False)

    assert _weight(layout, 0, 1) == pytest.approx(_ALIGNMENT_WEIGHT)


def test_feature_score_off_ignores_present_scores():
    """Scores present but unused must not scale the weight."""
    overlaps = [_FakeOverlap(0, 1, (0.25, 0.25))]

    layout = _find_offsets(overlaps, use_feature_score=False)

    assert _weight(layout, 0, 1) == pytest.approx(_ALIGNMENT_WEIGHT)


def test_the_supported_pipeline_always_populates_scores():
    """Why this cannot fire through ArrangeTilesWithTranslate."""
    settings = nornir_imageregistration.settings.TranslateSettings(
        use_feature_score=True)

    assert settings.feature_score_calculations_required is True


# --- LayoutPosition.GetWeight --------------------------------------------------

def test_get_weight_returns_a_scalar():
    """float() on the shape-(1,) boolean selection raised under NumPy 2.

    GetWeight had no production callers, only test helpers in test_arrange.py, so
    the break went unnoticed. These tests read weights through it.
    """
    from nornir_imageregistration.layout import Layout

    layout = Layout()
    layout.CreateNode(0, np.array([0.0, 0.0]))
    layout.CreateNode(1, np.array([10.0, 0.0]))
    layout.SetOffset(0, 1, np.array([10.0, 0.0]), 0.25)

    weight = layout.nodes[0].GetWeight(1)

    assert isinstance(weight, float)
    assert weight == pytest.approx(0.25)
