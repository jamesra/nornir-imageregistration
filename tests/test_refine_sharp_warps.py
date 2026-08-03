"""Unit tests for discontinuity tagging and edge-preserving sharp warps."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.anchor_smooth import (
    AnchorSmoothSettings,
    smooth_peaks_from_locked_anchors,
)
from nornir_imageregistration.refine_shared.discontinuity import (
    per_record_max_travel,
    tag_discontinuities,
)
from nornir_imageregistration.refine_shared.finalize import (
    FinalizeSettings,
    evaluate_finalize_candidates,
    filter_records_for_mesh_inclusion,
)
from nornir_imageregistration.refine_shared.pass_diagnostics import (
    build_pass_diagnostic_rows,
    write_pass_diagnostics,
)
from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config
from nornir_imageregistration.transforms.rigid import RigidTranslation


def _rec(
        key: tuple[int, int],
        *,
        peak: tuple[float, float] = (0.0, 0.0),
        weight: float = 10.0,
        source: tuple[float, float] | None = None,
        peak_ratio: float | None = 2.0,
) -> EnhancedAlignmentRecord:
    row, col = key
    src = np.asarray(source if source is not None else (float(row) * 10.0, float(col) * 10.0),
                     dtype=np.float64)
    return EnhancedAlignmentRecord(
        ID=key,
        TargetPoint=src.copy(),
        SourcePoint=src,
        peak=np.asarray(peak, dtype=np.float64),
        weight=weight,
        angle=0.0,
        flipped_ud=False,
        peak_ratio=peak_ratio,
    )


class TestTagDiscontinuities(unittest.TestCase):
    """Neighbor-median discontinuity detection."""

    def setUp(self) -> None:
        os.environ.pop('NORNIR_REFINE_SHARP_WARPS', None)
        get_runtime_config(refresh=True)

    def test_fold_peak_tagged_against_zero_neighbors(self) -> None:
        records = [
            _rec((0, 0), peak=(0.0, 0.0)),
            _rec((0, 1), peak=(30.0, 0.0)),  # fold
            _rec((0, 2), peak=(0.0, 0.0)),
            _rec((1, 0), peak=(0.0, 0.0)),
            _rec((1, 1), peak=(0.0, 0.0)),
            _rec((1, 2), peak=(0.0, 0.0)),
        ]
        tagged = tag_discontinuities(records, max_travel=10.0, discontinuity_k=1.5)
        self.assertIn((0, 1), tagged)
        self.assertNotIn((0, 0), tagged)

    def test_sharp_warps_off_returns_empty(self) -> None:
        os.environ['NORNIR_REFINE_SHARP_WARPS'] = '0'
        get_runtime_config(refresh=True)
        try:
            records = [_rec((0, 0)), _rec((0, 1), peak=(40.0, 0.0)), _rec((0, 2))]
            self.assertEqual(tag_discontinuities(records, max_travel=10.0), set())
        finally:
            os.environ.pop('NORNIR_REFINE_SHARP_WARPS', None)
            get_runtime_config(refresh=True)


class TestEdgePreservingAnchorSmooth(unittest.TestCase):
    """Discontinuity cells keep raw peaks through anchor-smooth."""

    def test_gaussian_does_not_pull_fold_peak(self) -> None:
        transform = RigidTranslation((0.0, 0.0))
        finalized = {
            (0, 0): _rec((0, 0), peak=(0.0, 0.0), source=(0.0, 0.0)),
            (0, 2): _rec((0, 2), peak=(0.0, 0.0), source=(0.0, 20.0)),
            (0, 4): _rec((0, 4), peak=(0.0, 0.0), source=(0.0, 40.0)),
        }
        # Bake locks as target=source (zero displacement) so smooth field is near 0.
        for key, rec in list(finalized.items()):
            finalized[key] = EnhancedAlignmentRecord(
                ID=key,
                TargetPoint=np.asarray(rec.SourcePoint, dtype=np.float64),
                SourcePoint=np.asarray(rec.SourcePoint, dtype=np.float64),
                peak=np.zeros(2, dtype=np.float64),
                weight=10.0,
                angle=0.0,
                flipped_ud=False,
            )
        fold = _rec((0, 1), peak=(30.0, 0.0), source=(0.0, 10.0), weight=5.0)
        settings = AnchorSmoothSettings(min_anchor_count=3, median_radius=1)
        without = smooth_peaks_from_locked_anchors(
            finalized, [fold], transform, settings, discontinuity_ids=None)
        with_disc = smooth_peaks_from_locked_anchors(
            finalized, [fold], transform, settings, discontinuity_ids={(0, 1)})
        by_without = {r.ID: r for r in without}
        by_with = {r.ID: r for r in with_disc}
        self.assertLess(float(np.linalg.norm(by_without[(0, 1)].peak)), 15.0)
        np.testing.assert_allclose(by_with[(0, 1)].peak, (30.0, 0.0), rtol=0, atol=1e-4)


class TestNeighborhoodAwareTravel(unittest.TestCase):
    """Per-record travel limits for mesh inclusion / finalize."""

    def test_filter_keeps_discontinuity_large_travel(self) -> None:
        records = [
            _rec((0, 0), peak=(1.0, 0.0), weight=20.0),
            _rec((0, 1), peak=(25.0, 0.0), weight=8.0),
            _rec((0, 2), peak=(1.0, 0.0), weight=20.0),
        ]
        limits = per_record_max_travel(
            records, base_max_travel=10.0, discontinuity_ids={(0, 1)}, relax_mult=3.0)
        kept, dropped = filter_records_for_mesh_inclusion(
            records, max_travel=10.0, min_keep=1, per_record_max_travel=limits)
        kept_ids = {tuple(r.ID) for r in kept}
        self.assertIn((0, 1), kept_ids)
        self.assertEqual(dropped, 0)

    def test_finalize_uses_soft_weight_for_discontinuity(self) -> None:
        records = [
            _rec((0, 0), peak=(1.0, 0.0), weight=20.0, peak_ratio=2.0),
            _rec((0, 1), peak=(5.0, 0.0), weight=3.0, peak_ratio=1.5),  # below inflection 10, above soft 2
        ]
        settings = FinalizeSettings(
            max_travel_for_finalization=10.0,
            min_finalize_pass=1,
            finalize_stability_passes=1,
        )
        limits = per_record_max_travel(
            records, base_max_travel=10.0, discontinuity_ids={(0, 1)}, relax_mult=2.5)
        result = evaluate_finalize_candidates(
            records,
            transform_cutoff=10.0,
            settings=settings,
            pass_index=2,
            per_record_max_travel=limits,
            soft_weight_cutoff=2.0,
            discontinuity_ids={(0, 1)},
        )
        self.assertTrue(bool(result.lock_mask[1]))

    def test_finalize_denies_soft_weight_for_ambiguous_disc(self) -> None:
        records = [
            _rec((0, 0), peak=(1.0, 0.0), weight=20.0, peak_ratio=2.0),
            _rec((0, 1), peak=(5.0, 0.0), weight=3.0, peak_ratio=1.05),
        ]
        settings = FinalizeSettings(
            max_travel_for_finalization=10.0,
            min_finalize_pass=1,
            finalize_stability_passes=1,
        )
        # Caller filters soft_disc_ids; ambiguous cell must not get soft floors.
        result = evaluate_finalize_candidates(
            records,
            transform_cutoff=10.0,
            settings=settings,
            pass_index=2,
            soft_weight_cutoff=2.0,
            discontinuity_ids=set(),
        )
        self.assertFalse(bool(result.lock_mask[1]))
        self.assertEqual(result.rejected_ambiguous_count, 1)

    def test_anchor_smooth_preserves_only_ratio_eligible_disc(self) -> None:
        """Ambiguous disc peaks are smoothed; unique disc keeps raw."""
        from nornir_imageregistration.refine_shared.peak_ratio_gates import soft_discontinuity_ids

        transform = RigidTranslation((0.0, 0.0))
        finalized = {
            (0, 0): _rec((0, 0), peak=(0.0, 0.0), source=(0.0, 0.0), peak_ratio=2.0),
            (0, 2): _rec((0, 2), peak=(0.0, 0.0), source=(0.0, 20.0), peak_ratio=2.0),
            (0, 4): _rec((0, 4), peak=(0.0, 0.0), source=(0.0, 40.0), peak_ratio=2.0),
        }
        for key, rec in list(finalized.items()):
            finalized[key] = EnhancedAlignmentRecord(
                ID=key,
                TargetPoint=np.asarray(rec.SourcePoint, dtype=np.float64),
                SourcePoint=np.asarray(rec.SourcePoint, dtype=np.float64),
                peak=np.zeros(2, dtype=np.float64),
                weight=10.0,
                angle=0.0,
                flipped_ud=False,
                peak_ratio=2.0,
            )
        unique_fold = _rec((0, 1), peak=(30.0, 0.0), source=(0.0, 10.0), weight=5.0, peak_ratio=1.6)
        amb_dirt = _rec((0, 3), peak=(40.0, 0.0), source=(0.0, 30.0), weight=5.0, peak_ratio=1.05)
        disc = {(0, 1), (0, 3)}
        soft_ids = soft_discontinuity_ids([unique_fold, amb_dirt], disc)
        self.assertEqual(soft_ids, {(0, 1)})
        settings = AnchorSmoothSettings(min_anchor_count=3, median_radius=1)
        out = smooth_peaks_from_locked_anchors(
            finalized, [unique_fold, amb_dirt], transform, settings, discontinuity_ids=soft_ids)
        by_id = {r.ID: r for r in out}
        np.testing.assert_allclose(by_id[(0, 1)].peak, (30.0, 0.0), rtol=0, atol=1e-4)
        self.assertLess(float(np.linalg.norm(by_id[(0, 3)].peak)), 15.0)


class TestPassDiagnostics(unittest.TestCase):
    """NPZ/CSV diagnostic export."""

    def test_write_pass_diagnostics_files(self) -> None:
        records = [
            _rec((0, 0), peak=(2.0, 0.0), peak_ratio=None),
            _rec((0, 1), peak=(4.0, 0.0), peak_ratio=None),
        ]
        rows = build_pass_diagnostic_rows(
            alignment_points=records,
            finalized={},
            included_ids={(0, 0)},
            travel_dropped_ids={(0, 1)},
            unlocked_ids=set(),
            transform_cutoff=1.0,
            finalize_candidates={},
            transform=RigidTranslation((0.0, 0.0)),
            discontinuity_ids={(0, 1)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = write_pass_diagnostics(tmp, 1, rows, pair_label='unit', write_heatmaps=True)
            self.assertTrue(os.path.isfile(written['npz']))
            self.assertTrue(os.path.isfile(written['csv']))
            self.assertGreater(float(written['tables_s']), 0.0)
            data = np.load(written['npz'])
            self.assertEqual(int(data['grid_col'][1]), 1)
            self.assertTrue(bool(data['discontinuity'][1]))
            self.assertTrue(bool(data['travel_dropped'][1]))
            self.assertIn('peak_ratio', data.files)
            self.assertTrue(np.all(np.isnan(data['peak_ratio'])))  # unset on synthetic records

    def test_write_pass_diagnostics_tables_only_skips_png(self) -> None:
        records = [
            _rec((0, 0), peak=(2.0, 0.0), peak_ratio=1.5),
            _rec((0, 1), peak=(4.0, 0.0), peak_ratio=1.1),
        ]
        rows = build_pass_diagnostic_rows(
            alignment_points=records,
            finalized={},
            included_ids={(0, 0)},
            travel_dropped_ids=set(),
            unlocked_ids=set(),
            transform_cutoff=1.0,
            finalize_candidates={},
            transform=RigidTranslation((0.0, 0.0)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = write_pass_diagnostics(tmp, 2, rows, pair_label='unit', write_heatmaps=False)
            self.assertTrue(os.path.isfile(written['npz']))
            self.assertTrue(os.path.isfile(written['csv']))
            self.assertEqual(float(written['heatmaps_s']), 0.0)
            pngs = [n for n in os.listdir(tmp) if n.endswith('.png')]
            self.assertEqual(pngs, [])

    def test_write_heatmaps_single_savefig_per_map(self) -> None:
        records = [
            _rec((0, 0), peak=(2.0, 0.0), peak_ratio=2.0),
            _rec((0, 1), peak=(4.0, 0.0), peak_ratio=1.2),
        ]
        rows = build_pass_diagnostic_rows(
            alignment_points=records,
            finalized={(0, 0): records[0]},
            included_ids={(0, 0), (0, 1)},
            travel_dropped_ids=set(),
            unlocked_ids=set(),
            transform_cutoff=1.0,
            finalize_candidates={},
            transform=RigidTranslation((0.0, 0.0)),
            discontinuity_ids={(0, 1)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = write_pass_diagnostics(tmp, 3, rows, pair_label='unit', write_heatmaps=True)
            png_names = sorted(n for n in os.listdir(tmp) if n.endswith('.png'))
            # One PNG per map name returned (no duplicate artifact replot).
            map_keys = sorted(k for k in written if k not in ('npz', 'csv', 'tables_s', 'heatmaps_s'))
            self.assertEqual(len(png_names), len(map_keys))
            self.assertGreater(len(png_names), 0)
            self.assertGreater(float(written['heatmaps_s']), 0.0)
            for name in map_keys:
                self.assertTrue(os.path.isfile(written[name]))
                basename = os.path.basename(str(written[name]))
                self.assertEqual(png_names.count(basename), 1)


if __name__ == '__main__':
    unittest.main()
