"""Unit tests for locked-anchor mesh smoothing."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.anchor_smooth import (
    AnchorSmoothSettings,
    compute_locked_displacement_field,
    should_use_anchor_smooth_mesh,
    smooth_peaks_from_locked_anchors,
)
from nornir_imageregistration.transforms.rigid import RigidTranslation


def _rec(
        key: tuple[int, int],
        *,
        peak: tuple[float, float] = (0.0, 0.0),
        weight: float = 10.0,
        source: tuple[float, float] = (10.0, 10.0),
        target: tuple[float, float] | None = None) -> EnhancedAlignmentRecord:
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target if target is not None else source, dtype=np.float64)
    return EnhancedAlignmentRecord(
        ID=key,
        TargetPoint=tgt,
        SourcePoint=src,
        peak=np.asarray(peak, dtype=np.float64),
        weight=weight,
        angle=0.0,
        flipped_ud=False,
    )


class _TranslateTransform:
    """Simple transform: target = source + offset."""

    def __init__(self, offset: tuple[float, float]) -> None:
        self._offset = np.asarray(offset, dtype=np.float64)

    def Transform(self, points: np.ndarray, **kwargs) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return pts + self._offset


class TestComputeLockedDisplacementField(unittest.TestCase):
    """Locked displacement seeding."""

    def test_baked_lock_shift_matches_target_minus_predicted(self) -> None:
        transform = _TranslateTransform((2.0, 3.0))
        finalized = {
            (0, 0): _rec((0, 0), source=(0.0, 0.0), target=(5.0, 8.0), peak=(0.0, 0.0)),
        }
        shifts, measured = compute_locked_displacement_field(finalized, transform, (1, 1))
        self.assertTrue(measured[0])
        np.testing.assert_allclose(shifts[0], (3.0, 5.0), rtol=0, atol=1e-6)


class TestSmoothPeaksFromLockedAnchors(unittest.TestCase):
    """Gap-fill and smoothed peak emission."""

    def test_locked_only_seed_gap_fills_center(self) -> None:
        """Three locks in a line; center unlocked cell gets smoothed non-raw peak."""
        transform = RigidTranslation((0.0, 0.0))
        finalized = {
            (0, 0): _rec((0, 0), source=(0.0, 0.0), target=(0.0, 0.0), peak=(0.0, 0.0)),
            (0, 2): _rec((0, 2), source=(20.0, 0.0), target=(25.0, 0.0), peak=(0.0, 0.0)),
            (0, 4): _rec((0, 4), source=(40.0, 0.0), target=(50.0, 0.0), peak=(0.0, 0.0)),
        }
        center = _rec((0, 1), source=(10.0, 0.0), peak=(50.0, 0.0), weight=5.0)
        alignment_points = [center]
        settings = AnchorSmoothSettings(min_anchor_count=3, median_radius=1)
        smoothed = smooth_peaks_from_locked_anchors(
            finalized, alignment_points, transform, settings)
        by_id = {rec.ID: rec for rec in smoothed}
        self.assertIn((0, 1), by_id)
        raw_peak = float(np.linalg.norm(center.peak))
        smooth_peak = float(np.linalg.norm(by_id[(0, 1)].peak))
        self.assertLess(smooth_peak, raw_peak)
        self.assertGreater(smooth_peak, 0.0)

    def test_locked_cell_peak_matches_baked_displacement(self) -> None:
        transform = _TranslateTransform((1.0, 2.0))
        finalized = {
            (1, 1): _rec((1, 1), source=(10.0, 10.0), target=(14.0, 17.0), peak=(0.0, 0.0)),
        }
        settings = AnchorSmoothSettings(min_anchor_count=1, median_radius=0)
        smoothed = smooth_peaks_from_locked_anchors(
            finalized, [], transform, settings)
        self.assertEqual(len(smoothed), 1)
        np.testing.assert_allclose(smoothed[0].peak, (3.0, 5.0), rtol=0, atol=1e-5)


class TestShouldUseAnchorSmoothMesh(unittest.TestCase):
    """Threshold gating for anchor-smooth mesh path."""

    def test_below_threshold_returns_false(self) -> None:
        finalized = {
            (0, 0): _rec((0, 0)),
            (0, 1): _rec((0, 1)),
        }
        settings = AnchorSmoothSettings(min_anchor_count=3)
        self.assertFalse(should_use_anchor_smooth_mesh(finalized, settings))

    def test_at_threshold_returns_true(self) -> None:
        finalized = {(i, 0): _rec((i, 0)) for i in range(3)}
        settings = AnchorSmoothSettings(min_anchor_count=3)
        self.assertTrue(should_use_anchor_smooth_mesh(finalized, settings))


if __name__ == '__main__':
    unittest.main()
