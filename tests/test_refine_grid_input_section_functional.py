"""Functional test for RefineGridMosaic on RC2_4Square_Assembled section 0690."""

from __future__ import annotations

import os
import unittest

import nornir_imageregistration

from grid_refine_input_section_diagnostics import (
    GRID_REFINE_INPUT_SECTION_TEST_CLASS,
    REGISTRATION_DOWNSAMPLE,
    clear_diagnostics_output_dir,
    edge_collinearity_by_tile,
    write_grid_refine_input_section_comparison_diagnostics,
    write_grid_refine_input_section_diagnostics,
)
from grid_seam_metrics import (
    GOLDEN_GRID_MOSAIC_NAME,
    GOLDEN_TARGET_DELTA_MAX,
    REFINE_MAX_PASSES,
    SEAM_MIN_OVERLAP,
    assert_final_pass_best_of_all,
    assert_monotonic_refinement_passes,
    assert_mosaic_target_bounds_sane,
    assert_refined_beats_translated,
    build_registration_comparison,
    compare_mosaic_target_points_to_golden,
    grid_refine_input_section_golden_mosaic_path,
    load_translated_grid_refine_input_section_mosaic,
    measure_mosaic_seam_scores,
    refine_grid_input_section,
    refine_grid_input_section_passes,
    require_grid_refine_input_section_fixture,
)

# Calibrated on RC2_4Square_Assembled 0690 when golden mosaic is available.
MAX_SEAM_MAE = 35.0


def _assert_golden_target_points_if_present(
        test_case: unittest.TestCase,
        refined: nornir_imageregistration.mosaic.Mosaic,
        fixture_root: str) -> None:
    """Compare refined target points to the C++ golden mosaic when it is bundled."""
    golden_path = grid_refine_input_section_golden_mosaic_path(fixture_root)
    if not os.path.isfile(golden_path):
        return
    golden = nornir_imageregistration.Mosaic.LoadFromMosaicFile(golden_path)
    mean_delta, per_tile = compare_mosaic_target_points_to_golden(refined, golden)
    test_case.assertLessEqual(
        mean_delta,
        GOLDEN_TARGET_DELTA_MAX,
        f"Mean target delta {mean_delta:.3f}px vs golden {GOLDEN_GRID_MOSAIC_NAME}; "
        f"per_tile={per_tile}")


class TestRefineGridInputSectionFunctional(unittest.TestCase):
    """End-to-end grid refine validation on RC2_4Square_Assembled section 0690 data."""

    def setUp(self) -> None:
        """Clear stale diagnostic artifacts for this test method before it runs."""
        clear_diagnostics_output_dir(GRID_REFINE_INPUT_SECTION_TEST_CLASS, self._testMethodName)

    def test_refine_grid_input_section_produces_sane_transforms(self) -> None:
        """RefineGridMosaic must emit grid transforms with plausible target bounds."""
        fixture_root = require_grid_refine_input_section_fixture()
        refined, diagnostics = refine_grid_input_section(fixture_root)

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
        write_grid_refine_input_section_diagnostics(
            refined,
            os.path.join(fixture_root, "Leveled", "TilePyramid", "004"),
            seam_summary,
            diagnostics,
            target_bounds,
            max_seam_mae_limit=MAX_SEAM_MAE,
            test_class=GRID_REFINE_INPUT_SECTION_TEST_CLASS,
            test_method=self._testMethodName)

    def test_refine_grid_input_section_seams_within_threshold(self) -> None:
        """Refined overlap seams must stay within calibrated MAE limits."""
        fixture_root = require_grid_refine_input_section_fixture()
        tile_dir = os.path.join(fixture_root, "Leveled", "TilePyramid", "004")
        refined, diagnostics = refine_grid_input_section(fixture_root)
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
        write_grid_refine_input_section_diagnostics(
            refined,
            tile_dir,
            refined_seams,
            diagnostics,
            target_bounds,
            max_seam_mae_limit=MAX_SEAM_MAE,
            test_class=GRID_REFINE_INPUT_SECTION_TEST_CLASS,
            test_method=self._testMethodName)

    def test_refine_grid_input_section_beats_translated_baseline(self) -> None:
        """Grid refine must improve monotonically and beat the translated baseline."""
        fixture_root = require_grid_refine_input_section_fixture()
        tile_dir = os.path.join(fixture_root, "Leveled", "TilePyramid", "004")
        translated_mosaic = load_translated_grid_refine_input_section_mosaic(fixture_root)
        refined, diagnostics, pass_scores = refine_grid_input_section_passes(
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

        write_grid_refine_input_section_comparison_diagnostics(
            translated_mosaic,
            refined,
            tile_dir,
            comparison,
            pass_scores,
            diagnostics,
            translated_bounds,
            refined_bounds,
            max_seam_mae_limit=MAX_SEAM_MAE,
            test_class=GRID_REFINE_INPUT_SECTION_TEST_CLASS,
            test_method=self._testMethodName)

        assert_monotonic_refinement_passes(pass_scores[1:])
        assert_final_pass_best_of_all(pass_scores)
        assert_refined_beats_translated(comparison)
        _assert_golden_target_points_if_present(self, refined, fixture_root)


if __name__ == "__main__":
    unittest.main()
