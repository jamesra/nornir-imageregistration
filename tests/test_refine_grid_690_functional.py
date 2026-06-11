"""Functional test for RefineGridMosaic on RC2_4Square_Assembled section 0690."""

from __future__ import annotations

import os
import unittest

import nornir_imageregistration

from grid690_diagnostics import (
    GRID690_TEST_CLASS,
    REGISTRATION_DOWNSAMPLE,
    clear_diagnostics_output_dir,
    edge_collinearity_by_tile,
    write_grid690_comparison_diagnostics,
    write_grid690_diagnostics,
)
from grid_seam_metrics import (
    GOLDEN_GRID_MOSAIC_NAME,
    GOLDEN_TARGET_DELTA_MAX,
    GRID690_DATASET,
    REFINE_MAX_PASSES,
    SEAM_MIN_OVERLAP,
    assert_final_pass_best_of_all,
    assert_monotonic_refinement_passes,
    assert_mosaic_target_bounds_sane,
    assert_refined_beats_translated,
    build_registration_comparison,
    compare_mosaic_target_points_to_golden,
    grid690_fixture_is_usable,
    grid690_fixture_root,
    grid690_golden_mosaic_path,
    load_translated_grid690_mosaic,
    measure_mosaic_seam_scores,
    refine_grid690,
    refine_grid690_passes,
    _grid690_tile_dir,
)

# Calibrated on RC2_4Square_Assembled 0690 when golden mosaic is available.
MAX_SEAM_MAE = 35.0


def _require_fixture() -> str:
    """Return fixture root or skip when the Grid690 fixture is unavailable."""
    fixture_root = grid690_fixture_root()
    if not grid690_fixture_is_usable(fixture_root):
        translated = os.path.join(fixture_root, "Translated_Prune_Max0.5.mosaic")
        tile_dir = _grid690_tile_dir(fixture_root)
        raise unittest.SkipTest(
            f"Grid690 fixture incomplete at {fixture_root}. "
            f"Need Translated_Prune_Max0.5.mosaic and mosaic tiles under {tile_dir}. "
            f"Install {GRID690_DATASET} under TESTINPUTPATH/PlatformRaw/IDOC/.")
    grid_paths = [
        os.path.join(fixture_root, name)
        for name in os.listdir(fixture_root)
        if name.startswith("Grid_") and name.endswith(".mosaic")
        and name != GOLDEN_GRID_MOSAIC_NAME
    ]
    if grid_paths:
        raise unittest.SkipTest(
            "Grid690 fixture must not include non-golden Grid_*.mosaic; test runs RefineGridMosaic fresh.")
    return fixture_root


def _assert_golden_target_points_if_present(
        test_case: unittest.TestCase,
        refined: nornir_imageregistration.mosaic.Mosaic,
        fixture_root: str) -> None:
    """Compare refined target points to the C++ golden mosaic when it is bundled."""
    golden_path = grid690_golden_mosaic_path(fixture_root)
    if not os.path.isfile(golden_path):
        return
    golden = nornir_imageregistration.Mosaic.LoadFromMosaicFile(golden_path)
    mean_delta, per_tile = compare_mosaic_target_points_to_golden(refined, golden)
    test_case.assertLessEqual(
        mean_delta,
        GOLDEN_TARGET_DELTA_MAX,
        f"Mean target delta {mean_delta:.3f}px vs golden {GOLDEN_GRID_MOSAIC_NAME}; "
        f"per_tile={per_tile}")


class TestRefineGrid690Functional(unittest.TestCase):
    """End-to-end grid refine validation on RC2_4Square_Assembled section 0690 data."""

    def setUp(self) -> None:
        """Clear stale diagnostic artifacts for this test method before it runs."""
        clear_diagnostics_output_dir(GRID690_TEST_CLASS, self._testMethodName)

    def test_refine_grid_690_produces_sane_transforms(self) -> None:
        """RefineGridMosaic must emit grid transforms with plausible target bounds."""
        fixture_root = _require_fixture()
        refined, diagnostics = refine_grid690(fixture_root)

        self.assertGreaterEqual(diagnostics.iterations_completed, 1)
        for image_name, transform in refined.ImageToTransform.items():
            self.assertTrue(
                isinstance(transform, nornir_imageregistration.transforms.IGridTransform),
                f"{image_name} is not a grid transform")

        target_bounds = assert_mosaic_target_bounds_sane(refined)
        _assert_golden_target_points_if_present(self, refined, fixture_root)
        edge_collinearity_by_tile(refined)
        seam_summary = measure_mosaic_seam_scores(
            refined,
            os.path.join(fixture_root, "Leveled", "TilePyramid", "004"),
            REGISTRATION_DOWNSAMPLE,
            min_overlap=SEAM_MIN_OVERLAP,
            retain_worst_images=True)
        write_grid690_diagnostics(
            refined,
            os.path.join(fixture_root, "Leveled", "TilePyramid", "004"),
            seam_summary,
            diagnostics,
            target_bounds,
            max_seam_mae_limit=MAX_SEAM_MAE,
            test_class=GRID690_TEST_CLASS,
            test_method=self._testMethodName)

    def test_refine_grid_690_seams_within_threshold(self) -> None:
        """Refined overlap seams must stay within calibrated MAE limits."""
        fixture_root = _require_fixture()
        tile_dir = os.path.join(fixture_root, "Leveled", "TilePyramid", "004")
        refined, diagnostics = refine_grid690(fixture_root)
        assert_mosaic_target_bounds_sane(refined)

        refined_seams = measure_mosaic_seam_scores(
            refined,
            tile_dir,
            REGISTRATION_DOWNSAMPLE,
            min_overlap=SEAM_MIN_OVERLAP,
            retain_worst_images=True)

        self.assertGreaterEqual(len(refined_seams.pair_scores), 3)
        self.assertLess(
            refined_seams.max_mae,
            MAX_SEAM_MAE,
            msg=(
                f"Refined max seam MAE {refined_seams.max_mae:.2f} exceeds {MAX_SEAM_MAE}; "
                f"pairs={[(s.pair_label, s.mae) for s in refined_seams.pair_scores]}"))

        target_bounds = assert_mosaic_target_bounds_sane(refined)
        write_grid690_diagnostics(
            refined,
            tile_dir,
            refined_seams,
            diagnostics,
            target_bounds,
            max_seam_mae_limit=MAX_SEAM_MAE,
            test_class=GRID690_TEST_CLASS,
            test_method=self._testMethodName)

    def test_refine_grid_690_beats_translated_baseline(self) -> None:
        """Grid refine must improve monotonically and beat the translated baseline."""
        fixture_root = _require_fixture()
        tile_dir = os.path.join(fixture_root, "Leveled", "TilePyramid", "004")
        translated_mosaic = load_translated_grid690_mosaic(fixture_root)
        refined, diagnostics, pass_scores = refine_grid690_passes(
            fixture_root,
            max_passes=REFINE_MAX_PASSES)

        for image_name, transform in refined.ImageToTransform.items():
            self.assertTrue(
                isinstance(transform, nornir_imageregistration.transforms.IGridTransform),
                f"{image_name} is not a grid transform")

        translated_bounds = assert_mosaic_target_bounds_sane(translated_mosaic)
        refined_bounds = assert_mosaic_target_bounds_sane(refined)
        comparison = build_registration_comparison(
            translated_mosaic,
            refined,
            tile_dir,
            REGISTRATION_DOWNSAMPLE,
            min_overlap=SEAM_MIN_OVERLAP)

        write_grid690_comparison_diagnostics(
            translated_mosaic,
            refined,
            tile_dir,
            comparison,
            pass_scores,
            diagnostics,
            translated_bounds,
            refined_bounds,
            max_seam_mae_limit=MAX_SEAM_MAE,
            test_class=GRID690_TEST_CLASS,
            test_method=self._testMethodName)

        assert_monotonic_refinement_passes(pass_scores[1:])
        assert_final_pass_best_of_all(pass_scores)
        assert_refined_beats_translated(comparison)
        _assert_golden_target_points_if_present(self, refined, fixture_root)


if __name__ == "__main__":
    unittest.main()
