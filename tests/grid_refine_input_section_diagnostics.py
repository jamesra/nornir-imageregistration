"""Non-blocking diagnostic artifacts for the grid refine input section functional test."""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_imageregistration.local_distortion_correction
import nornir_imageregistration.mosaic_tileset
from nornir_imageregistration.headless import inspect_png_output

from grid_seam_metrics import (
    RefinePassScore,
    RegistrationComparisonSummary,
    SEAM_MIN_OVERLAP,
    SEAM_WIDTH_PIXELS,
    SeamPairScore,
    SeamScoreSummary,
    TargetBoundsSummary,
    _to_numpy,
    measure_mosaic_seam_scores,
    score_mosaic_on_translated_overlaps,
)


logger = logging.getLogger(__name__)

REGISTRATION_DOWNSAMPLE = 4
GRID_REFINE_INPUT_SECTION_TEST_CLASS = "TestRefineGridInputSectionFunctional"


def _diagnostics_output_path(test_class: str, test_method: str) -> str | None:
    """Return the diagnostic output path when a test output root is configured."""
    root = (
        os.environ.get("TEST_OUTPUT_DIR", "").strip()
        or os.environ.get("TESTOUTPUTPATH", "").strip()
    )
    if not root:
        return None
    return os.path.join(root, test_class, test_method)


def clear_diagnostics_output_dir(test_class: str, test_method: str) -> None:
    """Remove the diagnostic output directory for a test method if it exists."""
    output_dir = _diagnostics_output_path(test_class, test_method)
    if output_dir is None or not os.path.isdir(output_dir):
        return
    shutil.rmtree(output_dir)


def _diagnostics_output_dir(test_class: str, test_method: str) -> str | None:
    """Return the diagnostic output directory when a test output root is configured."""
    output_dir = _diagnostics_output_path(test_class, test_method)
    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_figure(fig, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to a stable path and validate the PNG."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    try:
        inspect_png_output(path)
    except AssertionError as error:
        logger.warning("Diagnostic PNG validation failed for %s: %s", path, error)


def _draw_grid_mesh(
        axis: plt.Axes,
        target_points: NDArray[np.floating],
        grid_dims: tuple[int, int],
        color: str) -> None:
    """Draw grid control-point mesh lines on an axis."""
    rows, cols = int(grid_dims[0]), int(grid_dims[1])
    grid = np.asarray(target_points, dtype=np.float64).reshape(rows, cols, 2)
    for row in range(rows):
        axis.plot(grid[row, :, 1], grid[row, :, 0], color=color, alpha=0.5, linewidth=0.5)
    for col in range(cols):
        axis.plot(grid[:, col, 1], grid[:, col, 0], color=color, alpha=0.5, linewidth=0.5)


def _edge_collinearity_deviation(
        target_points: NDArray[np.floating],
        grid_dims: tuple[int, int]) -> float:
    """Return the maximum perpendicular deviation of outer-edge grid nodes from a fitted line."""
    rows, cols = int(grid_dims[0]), int(grid_dims[1])
    grid = np.asarray(target_points, dtype=np.float64).reshape(rows, cols, 2)
    edge_segments = (
        grid[0, :, :],
        grid[-1, :, :],
        grid[:, 0, :],
        grid[:, -1, :],
    )
    max_deviation = 0.0
    for edge in edge_segments:
        if edge.shape[0] < 2:
            continue
        coordinates = edge[:, ::-1]  # (x, y) for line fit
        origin = coordinates[0]
        direction = coordinates[-1] - origin
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0:
            continue
        unit = direction / direction_norm
        for point in coordinates[1:-1]:
            offset = point - origin
            projection_length = float(np.dot(offset, unit))
            projection = origin + unit * projection_length
            deviation = float(np.linalg.norm(point - projection))
            max_deviation = max(max_deviation, deviation)
    return max_deviation


def edge_collinearity_by_tile(
        mosaic: nornir_imageregistration.mosaic.Mosaic) -> dict[str, float]:
    """Measure outer-edge grid waviness for each grid transform in a mosaic."""
    scores: dict[str, float] = {}
    for image_name, transform in mosaic.ImageToTransform.items():
        if not isinstance(transform, nornir_imageregistration.transforms.IGridTransform):
            continue
        scores[image_name] = _edge_collinearity_deviation(
            _to_numpy(transform.TargetPoints),
            transform.grid_dims)
    return scores


def _assembled_image_target_extent(
        background_image: NDArray[np.floating]) -> tuple[float, float, float, float]:
    """Map an L4 assembled raster to full target-space bounds for overlay with TargetPoints."""
    target_space_scale = 1.0 / float(REGISTRATION_DOWNSAMPLE)
    height_px, width_px = background_image.shape[:2]
    return (
        0.0,
        float(width_px) / target_space_scale,
        float(height_px) / target_space_scale,
        0.0,
    )


def plot_mosaic_grid_target_points(
        refined_mosaic: nornir_imageregistration.mosaic.Mosaic,
        output_path: str,
        *,
        background_image: NDArray[np.floating] | None = None,
        max_seam_mae: float | None = None) -> None:
    """Render target-space grid control points with one color per tile."""
    fig, axis = plt.subplots(figsize=(10, 10), dpi=150)
    if background_image is not None:
        axis.imshow(
            background_image,
            cmap="gray",
            alpha=0.35,
            aspect="equal",
            extent=_assembled_image_target_extent(background_image))
    colors = plt.get_cmap("tab10").colors
    for index, (image_name, transform) in enumerate(refined_mosaic.ImageToTransform.items()):
        if not isinstance(transform, nornir_imageregistration.transforms.IGridTransform):
            continue
        color = colors[index % len(colors)]
        target_points = _to_numpy(transform.TargetPoints)
        grid_dims = transform.grid_dims
        axis.scatter(
            target_points[:, 1],
            target_points[:, 0],
            c=[color],
            s=8,
            label=image_name)
        _draw_grid_mesh(axis, target_points, grid_dims, color)
    title = "Grid refine input section target control points (L4)"
    if max_seam_mae is not None:
        title += f" — max seam MAE {max_seam_mae:.1f}"
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.legend(loc="upper right", fontsize=8)
    _save_figure(fig, output_path)


def plot_worst_seam_diff(
        worst_score: SeamPairScore,
        output_path: str) -> None:
    """Save a three-panel figure for the worst overlap seam pair."""
    if worst_score.image_a is None or worst_score.image_b is None:
        return
    image_a = worst_score.image_a
    image_b = worst_score.image_b
    diff = np.abs(image_a - image_b)
    if worst_score.strip_mask is not None:
        diff = np.where(worst_score.strip_mask, diff, 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(image_a, cmap="gray")
    axes[0].set_title(f"A ({worst_score.tile_a_id})")
    axes[1].imshow(image_b, cmap="gray")
    axes[1].set_title(f"B ({worst_score.tile_b_id})")
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title(f"|A - B| on {SEAM_WIDTH_PIXELS}px seam strip")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(
        f"Worst seam {worst_score.pair_label} — MAE {worst_score.mae:.2f}")
    for axis in axes:
        axis.axis("off")
    _save_figure(fig, output_path)


def _assemble_mosaic_l4(
        mosaic: nornir_imageregistration.mosaic.Mosaic,
        tile_dir: str) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Assemble a mosaic at the grid refine input section registration downsample for diagnostics."""
    tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
        mosaic,
        tile_dir,
        image_to_source_space_scale=float(REGISTRATION_DOWNSAMPLE))
    tileset.TranslateToZeroOrigin()
    assembled_image, assembled_mask = tileset.AssembleImage(
        target_space_scale=1.0 / float(REGISTRATION_DOWNSAMPLE))
    return _to_numpy(assembled_image), _to_numpy(assembled_mask)  # type: ignore[arg-type]


def write_grid_refine_input_section_diagnostics(
        refined_mosaic: nornir_imageregistration.mosaic.Mosaic,
        tile_dir: str,
        seam_summary: SeamScoreSummary,
        refine_diagnostics: nornir_imageregistration.local_distortion_correction.MosaicRefinementDiagnostics,
        target_bounds: Mapping[str, TargetBoundsSummary],
        *,
        max_seam_mae_limit: float,
        test_class: str = GRID_REFINE_INPUT_SECTION_TEST_CLASS,
        test_method: str) -> dict[str, str]:
    """Write JSON and PNG diagnostics when TESTOUTPUTPATH is configured."""
    output_dir = _diagnostics_output_dir(test_class, test_method)
    if output_dir is None:
        return {}

    artifact_paths: dict[str, str] = {}
    assembled_path = os.path.join(output_dir, "grid_refine_input_section_assembled_L4.png")
    mask_path = os.path.join(output_dir, "grid_refine_input_section_assembled_mask_L4.png")
    grid_points_path = os.path.join(output_dir, "grid_refine_input_section_target_grid_points.png")
    worst_diff_path = os.path.join(output_dir, "grid_refine_input_section_seam_worst_diff.png")
    report_path = os.path.join(output_dir, "grid_refine_input_section_seam_report.json")

    assembled_np, mask_np = _assemble_mosaic_l4(refined_mosaic, tile_dir)
    nornir_imageregistration.SaveImage(assembled_path, assembled_np, bpp=8)
    nornir_imageregistration.SaveImage(mask_path, mask_np, bpp=8)
    artifact_paths["assembled"] = assembled_path
    artifact_paths["assembled_mask"] = mask_path

    plot_mosaic_grid_target_points(
        refined_mosaic,
        grid_points_path,
        background_image=assembled_np,
        max_seam_mae=seam_summary.max_mae)
    artifact_paths["grid_points"] = grid_points_path

    worst_score = max(
        (score for score in seam_summary.pair_scores if np.isfinite(score.mae)),
        key=lambda item: item.mae)
    plot_worst_seam_diff(worst_score, worst_diff_path)
    artifact_paths["worst_diff"] = worst_diff_path

    report = {
        "fixture_section": "0690",
        "registration_downsample": REGISTRATION_DOWNSAMPLE,
        "refine": {
            "iterations_completed": refine_diagnostics.iterations_completed,
            "converged": refine_diagnostics.converged,
            "average_displacement_per_iteration": refine_diagnostics.average_displacement_per_iteration,
            "resolved_cell_size": list(refine_diagnostics.resolved_cell_size),
            "resolved_mesh_shape": list(refine_diagnostics.resolved_mesh_shape),
        },
        "target_bounds": {
            tile_name: {
                "y_min": bounds.y_min,
                "y_max": bounds.y_max,
                "x_min": bounds.x_min,
                "x_max": bounds.x_max,
            }
            for tile_name, bounds in target_bounds.items()
        },
        "seam_scores": {
            score.pair_label: {
                "mae": score.mae,
                "strip_pixels": score.strip_pixels,
                "pair_label": score.pair_label,
            }
            for score in seam_summary.pair_scores
        },
        "seam_aggregate": {
            "mean_mae": seam_summary.mean_mae,
            "max_mae": seam_summary.max_mae,
            "worst_pair": seam_summary.worst_pair,
        },
        "edge_collinearity": edge_collinearity_by_tile(refined_mosaic),
        "thresholds": {"max_seam_mae_limit": max_seam_mae_limit},
        "artifact_paths": artifact_paths,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    artifact_paths["report"] = report_path
    return artifact_paths


def write_grid_refine_input_section_comparison_diagnostics(
        translated_mosaic: nornir_imageregistration.mosaic.Mosaic,
        refined_mosaic: nornir_imageregistration.mosaic.Mosaic,
        tile_dir: str,
        comparison: RegistrationComparisonSummary,
        refine_passes: tuple[RefinePassScore, ...],
        refine_diagnostics: nornir_imageregistration.local_distortion_correction.MosaicRefinementDiagnostics,
        translated_target_bounds: Mapping[str, TargetBoundsSummary],
        refined_target_bounds: Mapping[str, TargetBoundsSummary],
        *,
        max_seam_mae_limit: float,
        test_class: str = GRID_REFINE_INPUT_SECTION_TEST_CLASS,
        test_method: str) -> dict[str, str]:
    """Write translated-vs-refined comparison artifacts when TESTOUTPUTPATH is configured."""
    output_dir = _diagnostics_output_dir(test_class, test_method)
    if output_dir is None:
        return {}

    artifact_paths = write_grid_refine_input_section_diagnostics(
        refined_mosaic,
        tile_dir,
        score_mosaic_on_translated_overlaps(
            refined_mosaic,
            translated_mosaic,
            tile_dir,
            REGISTRATION_DOWNSAMPLE,
            min_overlap=SEAM_MIN_OVERLAP,
            retain_worst_images=True),
        refine_diagnostics,
        refined_target_bounds,
        max_seam_mae_limit=max_seam_mae_limit,
        test_class=test_class,
        test_method=test_method)

    translated_assembled_path = os.path.join(output_dir, "grid_refine_input_section_translated_assembled_L4.png")
    translated_mask_path = os.path.join(output_dir, "grid_refine_input_section_translated_assembled_mask_L4.png")
    translated_worst_diff_path = os.path.join(output_dir, "grid_refine_input_section_translated_seam_worst_diff.png")
    report_path = artifact_paths.get("report", os.path.join(output_dir, "grid_refine_input_section_seam_report.json"))

    translated_assembled, translated_mask = _assemble_mosaic_l4(translated_mosaic, tile_dir)
    nornir_imageregistration.SaveImage(translated_assembled_path, translated_assembled, bpp=8)
    nornir_imageregistration.SaveImage(translated_mask_path, translated_mask, bpp=8)
    artifact_paths["translated_assembled"] = translated_assembled_path
    artifact_paths["translated_assembled_mask"] = translated_mask_path

    if comparison.translated.pair_scores:
        translated_with_images = score_mosaic_on_translated_overlaps(
            translated_mosaic,
            translated_mosaic,
            tile_dir,
            REGISTRATION_DOWNSAMPLE,
            min_overlap=SEAM_MIN_OVERLAP,
            retain_worst_images=True)
        translated_worst = max(
            (score for score in translated_with_images.pair_scores if np.isfinite(score.mae)),
            key=lambda item: item.mae)
        plot_worst_seam_diff(translated_worst, translated_worst_diff_path)
        artifact_paths["translated_worst_diff"] = translated_worst_diff_path

    report: dict[str, object] = {}
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)

    report["translated_seam_scores"] = {
        score.pair_label: {
            "mae": score.mae,
            "strip_pixels": score.strip_pixels,
            "pair_label": score.pair_label,
        }
        for score in comparison.translated.pair_scores
    }
    report["translated_seam_aggregate"] = {
        "mean_mae": comparison.translated.mean_mae,
        "max_mae": comparison.translated.max_mae,
        "worst_pair": comparison.translated.worst_pair,
    }
    report["translated_target_bounds"] = {
        tile_name: {
            "y_min": bounds.y_min,
            "y_max": bounds.y_max,
            "x_min": bounds.x_min,
            "x_max": bounds.x_max,
        }
        for tile_name, bounds in translated_target_bounds.items()
    }
    report["refinement_trajectory"] = [
        {
            "pass": pass_score.pass_index,
            "mean_mae": pass_score.mean_mae,
            "max_mae": pass_score.max_mae,
            "displacement": pass_score.displacement,
            "pair_count": len(pass_score.pair_scores),
        }
        for pass_score in refine_passes
    ]
    report["comparison"] = {
        "translated_mean": comparison.translated.mean_mae,
        "refined_mean": comparison.refined.mean_mae,
        "translated_max": comparison.translated.max_mae,
        "refined_max": comparison.refined.max_mae,
        "shared_pair_deltas": [
            {
                "pair_label": delta.pair_label,
                "translated_mae": delta.translated_mae,
                "refined_mae": delta.refined_mae,
                "delta": delta.delta,
            }
            for delta in comparison.shared_pair_deltas
        ],
        "grid_beats_translation": comparison.grid_beats_translation,
    }
    report["artifact_paths"] = artifact_paths

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    artifact_paths["report"] = report_path
    return artifact_paths
