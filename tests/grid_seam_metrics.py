"""Seam alignment metrics for grid-refined mosaic functional tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles
import nornir_imageregistration.mosaic_tileset
import nornir_imageregistration.tile_overlap


# Lower than RefineGridMosaic default (0.03): refined target boxes can shrink overlap area.
SEAM_MIN_OVERLAP = 0.001

SEAM_WIDTH_PIXELS = 30
SEAM_HALF_WIDTH = SEAM_WIDTH_PIXELS / 2.0

# Legacy ir-refine-grid default -it 10.
REFINE_MAX_PASSES = 10
REFINE_PASS_TOLERANCE = 0.10

GRID690_DATASET = "RC2_4Square_Assembled"
GOLDEN_GRID_MOSAIC_NAME = "Grid_Cel96_Mes8_sp4_Mes8_Thr0.5.mosaic"
REFINE_CELL_SIZE = (96, 96)
REFINE_MESH_SHAPE = (8, 8)
REFINE_DISPLACEMENT_THRESHOLD = 0.2
REFINE_IMAGE_SCALE = 0.25
GOLDEN_TARGET_DELTA_MAX = 2.0
SEAM_COMPARE_EPSILON = 1e-4


@dataclass(frozen=True)
class SeamPairScore:
    """Seam mismatch score for one overlapping tile pair."""

    pair_label: str
    tile_a_id: str
    tile_b_id: str
    mae: float
    strip_pixels: int
    image_a: NDArray[np.floating] | None = None
    image_b: NDArray[np.floating] | None = None
    strip_mask: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class SeamScoreSummary:
    """Aggregate seam scores across all overlap pairs."""

    pair_scores: tuple[SeamPairScore, ...]
    mean_mae: float
    max_mae: float
    worst_pair: str


@dataclass(frozen=True)
class TargetBoundsSummary:
    """Target-space bounds for one tile transform."""

    y_min: float
    y_max: float
    x_min: float
    x_max: float


@dataclass(frozen=True)
class RefinePassScore:
    """Seam metrics for one refinement pass (pass 0 is the translated baseline)."""

    pass_index: int
    mean_mae: float
    max_mae: float
    pair_scores: tuple[SeamPairScore, ...]
    displacement: float | None


@dataclass(frozen=True)
class SharedPairDelta:
    """Per-pair seam MAE comparison between translated and refined mosaics."""

    pair_label: str
    translated_mae: float
    refined_mae: float
    delta: float


@dataclass(frozen=True)
class RegistrationComparisonSummary:
    """Translated baseline vs final refined registration quality."""

    translated: SeamScoreSummary
    refined: SeamScoreSummary
    shared_pair_deltas: tuple[SharedPairDelta, ...]
    grid_beats_translation: bool


def grid690_fixture_is_usable(fixture_root: str) -> bool:
    """Return True when translated mosaic and L4 tiles referenced by the mosaic exist."""
    translated_path = os.path.join(fixture_root, "Translated_Prune_Max0.5.mosaic")
    if not os.path.isfile(translated_path):
        return False
    try:
        mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(translated_path)
    except (OSError, ValueError):
        return False
    tile_dir = _grid690_tile_dir(fixture_root)
    for image_name in mosaic.ImageToTransform.keys():
        if not os.path.isfile(os.path.join(tile_dir, image_name)):
            return False
    return True


def grid690_fixture_root() -> str:
    """Return the RC2 section 0690 fixture directory when present."""
    relative_tail = os.path.join(
        "PlatformRaw", "IDOC", GRID690_DATASET, "TEM", "0690", "TEM")
    candidates: list[str] = []
    testinput = os.environ.get("TESTINPUTPATH", "").strip()
    if testinput:
        candidates.append(os.path.join(testinput, relative_tail))
    candidates.extend([
        os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "RC2_4Square_Assembled_Grid690",
            "TEM",
            "0690",
            "TEM",
        ),
        os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "IDocBuildTest_Grid690",
            "TEM",
            "0690",
            "TEM",
        ),
    ])
    for candidate in candidates:
        if grid690_fixture_is_usable(candidate):
            return candidate
    return candidates[0] if candidates else relative_tail


def grid690_golden_mosaic_path(fixture_root: str) -> str:
    """Return the C++ golden grid mosaic path for the Grid690 fixture."""
    return os.path.join(fixture_root, GOLDEN_GRID_MOSAIC_NAME)


def compare_mosaic_target_points_to_golden(
        refined_mosaic: nornir_imageregistration.mosaic.Mosaic,
        golden_mosaic: nornir_imageregistration.mosaic.Mosaic) -> tuple[float, dict[str, float]]:
    """
    Return mean per-tile target-point delta and per-tile deltas vs a golden mosaic.

    Mosaic placement is translation-gauge-free (all tiles move during refinement), so the
    global mean target-point offset between the two mosaics is removed before measuring
    per-tile deltas; only relative tile placement is compared.
    """
    delta_by_tile: dict[str, NDArray[np.floating]] = {}
    for image_key in golden_mosaic.ImageToTransform.keys():
        golden_transform = golden_mosaic.ImageToTransform[image_key]
        refined_transform = refined_mosaic.ImageToTransform.get(image_key)
        if refined_transform is None:
            continue
        if not isinstance(golden_transform, nornir_imageregistration.transforms.IGridTransform):
            continue
        if not isinstance(refined_transform, nornir_imageregistration.transforms.IGridTransform):
            continue
        golden_targets = np.asarray(golden_transform.TargetPoints, dtype=np.float64)
        refined_targets = np.asarray(refined_transform.TargetPoints, dtype=np.float64)
        if golden_targets.shape != refined_targets.shape:
            continue
        delta_by_tile[image_key] = refined_targets - golden_targets

    if not delta_by_tile:
        raise ValueError("No comparable grid transforms found between refined and golden mosaics")

    global_offset = np.vstack(list(delta_by_tile.values())).mean(axis=0)
    per_tile = {
        image_key: float(np.mean(np.linalg.norm(delta - global_offset, axis=1)))
        for image_key, delta in delta_by_tile.items()}
    return float(np.mean(list(per_tile.values()))), per_tile


def _to_numpy(array: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert CuPy arrays to NumPy when needed."""
    if hasattr(array, "get"):
        return np.asarray(array.get(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def target_bounds_for_transform(
        transform: nornir_imageregistration.ITransform) -> TargetBoundsSummary:
    """Return target-space bounds for a mosaic transform."""
    if hasattr(transform, "TargetPoints"):
        points = _to_numpy(transform.TargetPoints)  # type: ignore[attr-defined]
        return TargetBoundsSummary(
            y_min=float(np.min(points[:, 0])),
            y_max=float(np.max(points[:, 0])),
            x_min=float(np.min(points[:, 1])),
            x_max=float(np.max(points[:, 1])),
        )
    source_corners = np.array(
        [[0.0, 0.0], [0.0, 4080.0], [4080.0, 4080.0], [4080.0, 0.0]],
        dtype=np.float64)
    target_corners = _to_numpy(transform.Transform(source_corners))
    return TargetBoundsSummary(
        y_min=float(np.min(target_corners[:, 0])),
        y_max=float(np.max(target_corners[:, 0])),
        x_min=float(np.min(target_corners[:, 1])),
        x_max=float(np.max(target_corners[:, 1])),
    )


def assert_mosaic_target_bounds_sane(
        mosaic: nornir_imageregistration.mosaic.Mosaic,
        max_span_factor: float = 2.0,
        reference_source_span: float = 4080.0) -> dict[str, TargetBoundsSummary]:
    """
    Return per-tile target bounds and raise if any control point span is implausible.

    Guards against exploded grid target coordinates that would force huge assemble buffers.
    """
    max_allowed = float(reference_source_span * max_span_factor)
    bounds_by_tile: dict[str, TargetBoundsSummary] = {}
    for image_name, transform in mosaic.ImageToTransform.items():
        summary = target_bounds_for_transform(transform)
        bounds_by_tile[image_name] = summary
        if summary.y_max > max_allowed or summary.x_max > max_allowed:
            raise AssertionError(
                f"Tile {image_name} target bounds exceed {max_allowed:.0f}: "
                f"Y=[{summary.y_min:.0f}, {summary.y_max:.0f}] "
                f"X=[{summary.x_min:.0f}, {summary.x_max:.0f}]")
    return bounds_by_tile


def _pair_label(tile_a: nornir_imageregistration.Tile,
                tile_b: nornir_imageregistration.Tile) -> str:
    """Build a stable overlap-pair label from tile image names."""
    a_name = os.path.basename(str(tile_a.ImagePath or tile_a.ID))
    b_name = os.path.basename(str(tile_b.ImagePath or tile_b.ID))
    a_stem = os.path.splitext(a_name)[0]
    b_stem = os.path.splitext(b_name)[0]
    return f"{a_stem}-{b_stem}"


def _warped_overlap_valid_mask(
        warped: nornir_imageregistration.transformed_image_data.ITransformedImageData) -> NDArray[np.bool_]:
    """Return True where the warped patch samples from inside the source tile bounds."""
    distance_image = _to_numpy(warped.centerDistanceImage)  # type: ignore[attr-defined]
    distance_max = np.finfo(distance_image.dtype).max
    return distance_image < distance_max


def _seam_strip_mask(
        image_shape: tuple[int, int],
        overlap: nornir_imageregistration.tile_overlap.TileOverlap,
        seam_half_width: float) -> NDArray[np.bool_]:
    """
    Mask pixels in a strip centered on the overlap patch, spanning the seam interface.

    The seam normal follows the target-space offset from tile A to tile B (the direction
    tiles are separated). For side-by-side tiles the strip is a thin vertical band; for
    vertically stacked tiles it is a thin horizontal band. Pixels outside the patch are
    excluded by construction.
    """
    height, width = image_shape
    yy, xx = np.mgrid[0:height, 0:width]
    center_y = (float(height) - 1.0) * 0.5
    center_x = (float(width) - 1.0) * 0.5
    offset = np.asarray(overlap.offset, dtype=np.float64)
    offset_norm = float(np.linalg.norm(offset))
    if offset_norm <= 1e-6:
        seam_normal = np.array([0.0, 1.0], dtype=np.float64)
    else:
        seam_normal = offset / offset_norm
    rel_y = yy - center_y
    rel_x = xx - center_x
    distance = np.abs(rel_y * seam_normal[0] + rel_x * seam_normal[1])
    return distance <= seam_half_width


def measure_overlap_seam_mae(
        tile_a: nornir_imageregistration.Tile,
        tile_b: nornir_imageregistration.Tile,
        overlap: nornir_imageregistration.tile_overlap.TileOverlap,
        target_space_scale: float,
        *,
        seam_half_width: float = SEAM_HALF_WIDTH,
        intensity_floor: float = 0.01,
        retain_images: bool = False) -> SeamPairScore:
    """
    Measure mean absolute seam difference for one overlap in target space.

    Only pixels inside the seam strip and valid on both warped tiles count toward the score.
    Pixels outside source-tile bounds or with no overlapping coverage are excluded.
    """
    overlap_rect = overlap.overlapping_target_rect
    if overlap_rect is None:
        raise ValueError(f"Tiles {tile_a.ID} and {tile_b.ID} have no overlapping target rect")

    warped_a = nornir_imageregistration.assemble_tiles.TransformTile(
        tile_a,
        distanceImage=None,
        target_space_scale=target_space_scale,
        TargetRegion=overlap_rect,
        SingleThreadedInvoke=True)
    warped_b = nornir_imageregistration.assemble_tiles.TransformTile(
        tile_b,
        distanceImage=None,
        target_space_scale=target_space_scale,
        TargetRegion=overlap_rect,
        SingleThreadedInvoke=True)

    if isinstance(warped_a, nornir_imageregistration.transformed_image_data.TransformedImageDataError):
        raise ValueError(str(warped_a.error_msg))
    if isinstance(warped_b, nornir_imageregistration.transformed_image_data.TransformedImageDataError):
        raise ValueError(str(warped_b.error_msg))

    image_a = _to_numpy(warped_a.image)  # type: ignore[arg-type]
    image_b = _to_numpy(warped_b.image)  # type: ignore[arg-type]
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Overlap warp shape mismatch for {tile_a.ID}/{tile_b.ID}: "
            f"{image_a.shape} vs {image_b.shape}")

    overlap_valid = (
        _warped_overlap_valid_mask(warped_a)
        & _warped_overlap_valid_mask(warped_b)
        & (image_a > intensity_floor)
        & (image_b > intensity_floor))
    strip = _seam_strip_mask(image_a.shape, overlap, seam_half_width)
    sample = overlap_valid & strip
    strip_pixels = int(np.count_nonzero(sample))
    if strip_pixels == 0:
        mae = float("inf")
    else:
        mae = float(np.mean(np.abs(image_a[sample] - image_b[sample])))

    return SeamPairScore(
        pair_label=_pair_label(tile_a, tile_b),
        tile_a_id=str(tile_a.ID),
        tile_b_id=str(tile_b.ID),
        mae=mae,
        strip_pixels=strip_pixels,
        image_a=image_a if retain_images else None,
        image_b=image_b if retain_images else None,
        strip_mask=sample if retain_images else None,
    )


def measure_mosaic_seam_scores(
        mosaic: nornir_imageregistration.mosaic.Mosaic,
        tile_dir: str,
        registration_downsample: int,
        *,
        min_overlap: float = 0.03,
        seam_half_width: float = SEAM_HALF_WIDTH,
        retain_worst_images: bool = False) -> SeamScoreSummary:
    """Compute seam MAE for every overlapping tile pair in a mosaic."""
    tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
        mosaic,
        tile_dir,
        image_to_source_space_scale=float(registration_downsample))
    target_space_scale = 1.0 / float(registration_downsample)
    tiles = list(tileset.values())
    pair_scores: list[SeamPairScore] = []

    overlaps = list(nornir_imageregistration.tile_overlap.IterateTileOverlaps(
        tiles,
        image_to_source_space_scale=float(registration_downsample),
        min_overlap=min_overlap))

    for overlap in overlaps:
        pair_scores.append(measure_overlap_seam_mae(
            overlap.A,
            overlap.B,
            overlap,
            target_space_scale,
            seam_half_width=seam_half_width,
            retain_images=False))

    if retain_worst_images and pair_scores:
        worst = max(pair_scores, key=lambda item: item.mae)
        worst_overlap = next(
            overlap for overlap in overlaps
            if _pair_label(overlap.A, overlap.B) == worst.pair_label)
        retained = measure_overlap_seam_mae(
            worst_overlap.A,
            worst_overlap.B,
            worst_overlap,
            target_space_scale,
            seam_half_width=seam_half_width,
            retain_images=True)
        pair_scores = [
            retained if score.pair_label == worst.pair_label else score
            for score in pair_scores]

    if not pair_scores:
        raise AssertionError("No overlapping tile pairs found for seam scoring")

    maes = np.asarray([score.mae for score in pair_scores], dtype=np.float64)
    finite = maes[np.isfinite(maes)]
    if finite.size == 0:
        raise AssertionError("No finite seam scores computed for any overlap pair")
    worst_pair = max(
        (score for score in pair_scores if np.isfinite(score.mae)),
        key=lambda item: item.mae).pair_label
    return SeamScoreSummary(
        pair_scores=tuple(pair_scores),
        mean_mae=float(np.mean(finite)),
        max_mae=float(np.max(finite)),
        worst_pair=worst_pair,
    )


def load_translated_grid690_mosaic(
        fixture_root: str) -> nornir_imageregistration.mosaic.Mosaic:
    """Load the translated input mosaic for the Grid690 functional fixture."""
    translated_path = os.path.join(fixture_root, "Translated_Prune_Max0.5.mosaic")
    return nornir_imageregistration.Mosaic.LoadFromMosaicFile(translated_path)


def _grid690_tile_dir(fixture_root: str) -> str:
    """Return the L4 tile directory for the Grid690 fixture."""
    return os.path.join(fixture_root, "Leveled", "TilePyramid", "004")


def _summarize_pair_scores(pair_scores: list[SeamPairScore]) -> SeamScoreSummary:
    """Aggregate per-pair seam scores into mean/max summary statistics."""
    if not pair_scores:
        raise AssertionError("No overlapping tile pairs found for seam scoring")

    maes = np.asarray([score.mae for score in pair_scores], dtype=np.float64)
    finite = maes[np.isfinite(maes)]
    if finite.size == 0:
        raise AssertionError("No finite seam scores computed for any overlap pair")
    worst_pair = max(
        (score for score in pair_scores if np.isfinite(score.mae)),
        key=lambda item: item.mae).pair_label
    return SeamScoreSummary(
        pair_scores=tuple(pair_scores),
        mean_mae=float(np.mean(finite)),
        max_mae=float(np.max(finite)),
        worst_pair=worst_pair,
    )


def score_mosaic_on_translated_overlaps(
        mosaic: nornir_imageregistration.mosaic.Mosaic,
        translated_mosaic: nornir_imageregistration.mosaic.Mosaic,
        tile_dir: str,
        registration_downsample: int,
        *,
        min_overlap: float = SEAM_MIN_OVERLAP,
        seam_half_width: float = SEAM_HALF_WIDTH,
        retain_worst_images: bool = False) -> SeamScoreSummary:
    """
    Score seam MAE using overlap pairs discovered on the translated mosaic.

    Pair labels stay stable when the scored mosaic loses overlap pairs after refine.
    """
    target_space_scale = 1.0 / float(registration_downsample)
    scale = float(registration_downsample)

    translated_tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
        translated_mosaic,
        tile_dir,
        image_to_source_space_scale=scale)
    mosaic_tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
        mosaic,
        tile_dir,
        image_to_source_space_scale=scale)
    mosaic_tiles = {tile.ID: tile for tile in mosaic_tileset.values()}

    canonical_overlaps = list(nornir_imageregistration.tile_overlap.IterateTileOverlaps(
        list(translated_tileset.values()),
        image_to_source_space_scale=scale,
        min_overlap=min_overlap))

    pair_scores: list[SeamPairScore] = []
    scored_overlaps: list[nornir_imageregistration.tile_overlap.TileOverlap] = []
    for canonical_overlap in canonical_overlaps:
        tile_a = mosaic_tiles.get(canonical_overlap.A.ID)
        tile_b = mosaic_tiles.get(canonical_overlap.B.ID)
        if tile_a is None or tile_b is None:
            continue

        mosaic_overlap = next(
            (
                overlap for overlap in nornir_imageregistration.tile_overlap.IterateTileOverlaps(
                    [tile_a, tile_b],
                    image_to_source_space_scale=scale,
                    min_overlap=min_overlap)
                if overlap.A.ID == canonical_overlap.A.ID and overlap.B.ID == canonical_overlap.B.ID
            ),
            None)
        if mosaic_overlap is None:
            continue

        pair_scores.append(measure_overlap_seam_mae(
            mosaic_overlap.A,
            mosaic_overlap.B,
            mosaic_overlap,
            target_space_scale,
            seam_half_width=seam_half_width,
            retain_images=False))
        scored_overlaps.append(mosaic_overlap)

    if retain_worst_images and pair_scores:
        worst = max(pair_scores, key=lambda item: item.mae)
        worst_overlap = next(
            overlap for overlap in scored_overlaps
            if _pair_label(overlap.A, overlap.B) == worst.pair_label)
        retained = measure_overlap_seam_mae(
            worst_overlap.A,
            worst_overlap.B,
            worst_overlap,
            target_space_scale,
            seam_half_width=seam_half_width,
            retain_images=True)
        pair_scores = [
            retained if score.pair_label == worst.pair_label else score
            for score in pair_scores]

    return _summarize_pair_scores(pair_scores)


def build_registration_comparison(
        translated_mosaic: nornir_imageregistration.mosaic.Mosaic,
        refined_mosaic: nornir_imageregistration.mosaic.Mosaic,
        tile_dir: str,
        registration_downsample: int,
        *,
        min_overlap: float = SEAM_MIN_OVERLAP) -> RegistrationComparisonSummary:
    """Compare translated and refined seam scores on shared translated overlap pairs."""
    translated_scores = score_mosaic_on_translated_overlaps(
        translated_mosaic,
        translated_mosaic,
        tile_dir,
        registration_downsample,
        min_overlap=min_overlap)
    refined_scores = score_mosaic_on_translated_overlaps(
        refined_mosaic,
        translated_mosaic,
        tile_dir,
        registration_downsample,
        min_overlap=min_overlap)

    translated_by_label = {score.pair_label: score for score in translated_scores.pair_scores}
    refined_by_label = {score.pair_label: score for score in refined_scores.pair_scores}
    shared_labels = sorted(set(translated_by_label) & set(refined_by_label))
    shared_pair_deltas = tuple(
        SharedPairDelta(
            pair_label=label,
            translated_mae=translated_by_label[label].mae,
            refined_mae=refined_by_label[label].mae,
            delta=refined_by_label[label].mae - translated_by_label[label].mae,
        )
        for label in shared_labels)

    grid_beats_translation = (
        refined_scores.mean_mae <= translated_scores.mean_mae + SEAM_COMPARE_EPSILON
        and refined_scores.max_mae <= translated_scores.max_mae + SEAM_COMPARE_EPSILON)
    return RegistrationComparisonSummary(
        translated=translated_scores,
        refined=refined_scores,
        shared_pair_deltas=shared_pair_deltas,
        grid_beats_translation=grid_beats_translation,
    )


def assert_monotonic_refinement_passes(
        refine_passes: tuple[RefinePassScore, ...],
        *,
        tolerance: float = REFINE_PASS_TOLERANCE) -> None:
    """Require each refine pass mean MAE to be no worse than the prior pass within tolerance."""
    if len(refine_passes) < 2:
        return
    for prior_pass, next_pass in zip(refine_passes, refine_passes[1:]):
        allowed = prior_pass.mean_mae * (1.0 + tolerance)
        if next_pass.mean_mae > allowed:
            raise AssertionError(
                f"Refine pass {next_pass.pass_index} mean seam MAE {next_pass.mean_mae:.4f} "
                f"exceeds pass {prior_pass.pass_index} allowance {allowed:.4f} "
                f"(prior={prior_pass.mean_mae:.4f}, tolerance={tolerance:.0%})")


def assert_final_pass_best_of_all(passes: tuple[RefinePassScore, ...]) -> None:
    """Require the final pass to have the lowest mean seam MAE across all passes."""
    if not passes:
        raise AssertionError("No refinement passes were recorded")
    final_pass = passes[-1]
    best_mean = min(pass_score.mean_mae for pass_score in passes)
    if final_pass.mean_mae > best_mean + SEAM_COMPARE_EPSILON:
        raise AssertionError(
            f"Final refine pass {final_pass.pass_index} mean seam MAE {final_pass.mean_mae:.4f} "
            f"is not the best of all passes (best={best_mean:.4f})")


def assert_refined_beats_translated(
        comparison: RegistrationComparisonSummary) -> None:
    """Require refined mean and max seam MAE to beat or match the translated baseline."""
    translated = comparison.translated
    refined = comparison.refined
    if refined.mean_mae > translated.mean_mae + SEAM_COMPARE_EPSILON:
        raise AssertionError(
            f"Refined mean seam MAE {refined.mean_mae:.4f} must be <= translated "
            f"{translated.mean_mae:.4f}; shared_pairs="
            f"{[(delta.pair_label, delta.translated_mae, delta.refined_mae) for delta in comparison.shared_pair_deltas]}")
    if refined.max_mae > translated.max_mae + SEAM_COMPARE_EPSILON:
        raise AssertionError(
            f"Refined max seam MAE {refined.max_mae:.4f} must be <= translated "
            f"{translated.max_mae:.4f}; pairs="
            f"{[(score.pair_label, score.mae) for score in refined.pair_scores]}")


def refine_grid690_passes(
        fixture_root: str,
        *,
        max_passes: int = REFINE_MAX_PASSES) -> tuple[
    nornir_imageregistration.mosaic.Mosaic,
    nornir_imageregistration.local_distortion_correction.MosaicRefinementDiagnostics,
    tuple[RefinePassScore, ...]]:
    """
    Run chained single-iteration refines and record seam scores after each pass.

    Pass 0 scores the translated input; passes 1..max_passes apply RefineGridMosaic once each.
    Matching legacy ir-refine-grid, chaining stops once a pass's average displacement falls
    to or below the displacement threshold (C++ breaks after applying that pass's update).
    """
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")

    tile_dir = _grid690_tile_dir(fixture_root)
    translated_mosaic = load_translated_grid690_mosaic(fixture_root)
    pass_scores: list[RefinePassScore] = []

    translated_summary = score_mosaic_on_translated_overlaps(
        translated_mosaic,
        translated_mosaic,
        tile_dir,
        4,
        min_overlap=SEAM_MIN_OVERLAP)
    pass_scores.append(RefinePassScore(
        pass_index=0,
        mean_mae=translated_summary.mean_mae,
        max_mae=translated_summary.max_mae,
        pair_scores=translated_summary.pair_scores,
        displacement=None))

    mosaic = translated_mosaic
    final_diagnostics: nornir_imageregistration.local_distortion_correction.MosaicRefinementDiagnostics | None = None
    for pass_index in range(1, max_passes + 1):
        mosaic, pass_diagnostics = nornir_imageregistration.RefineGridMosaic(
            mosaic,
            tile_dir,
            iterations=1,
            cell_size=REFINE_CELL_SIZE,
            mesh_shape=REFINE_MESH_SHAPE,
            displacement_threshold=REFINE_DISPLACEMENT_THRESHOLD,
            imageScale=REFINE_IMAGE_SCALE,
            return_diagnostics=True)
        final_diagnostics = pass_diagnostics
        displacement = float(pass_diagnostics.average_displacement_per_iteration[-1])
        pass_summary = score_mosaic_on_translated_overlaps(
            mosaic,
            translated_mosaic,
            tile_dir,
            4,
            min_overlap=SEAM_MIN_OVERLAP)
        pass_scores.append(RefinePassScore(
            pass_index=pass_index,
            mean_mae=pass_summary.mean_mae,
            max_mae=pass_summary.max_mae,
            pair_scores=pass_summary.pair_scores,
            displacement=displacement))
        if displacement <= REFINE_DISPLACEMENT_THRESHOLD:
            break

    if final_diagnostics is None:
        raise RuntimeError("refine_grid690_passes did not run any refinement passes")

    return mosaic, final_diagnostics, tuple(pass_scores)


def refine_grid690(
        fixture_root: str) -> tuple[
    nornir_imageregistration.mosaic.Mosaic,
    nornir_imageregistration.local_distortion_correction.MosaicRefinementDiagnostics]:
    """Run grid refinement on the section 0690 fixture with repro pipeline parameters."""
    translated_path = os.path.join(fixture_root, "Translated_Prune_Max0.5.mosaic")
    tile_dir = _grid690_tile_dir(fixture_root)
    refined, diagnostics = nornir_imageregistration.RefineGridMosaic(
        translated_path,
        tile_dir,
        iterations=REFINE_MAX_PASSES,
        cell_size=REFINE_CELL_SIZE,
        mesh_shape=REFINE_MESH_SHAPE,
        displacement_threshold=REFINE_DISPLACEMENT_THRESHOLD,
        imageScale=REFINE_IMAGE_SCALE,
        return_diagnostics=True)
    return refined, diagnostics
