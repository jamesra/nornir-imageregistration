"""
Created on Apr 7, 2015

@author: u0490822

This module performs local distortions of images to refine alignments of mosaics and sections.

Public APIs are also re-exported from ``nornir_imageregistration.mosaic_refine`` and
``nornir_imageregistration.stos_refine``. Shared measurement, cutoff, displacement
regularization, and ``NORNIR_REFINE_*`` runtime config live in
``nornir_imageregistration.refine_shared``. See
``docs/grid_refine_stos_vs_mosaic.md`` for the STOS vs mosaic comparison.
"""
import gc
import logging
import os
import enum
import copy
import time
import threading
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence, cast

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration import cp
import nornir_imageregistration.assemble
import nornir_imageregistration.assemble_tiles
from nornir_imageregistration.mathfuncs import EMA
import nornir_imageregistration.phasecorrelation
import nornir_imageregistration.batched_phase_correlation
from nornir_imageregistration.settings import SliceToSliceMethod
from nornir_imageregistration.registration_control import (
    ProgressCallback,
    check_cancelled,
    report_progress,
)
from nornir_imageregistration.refine_shared import (
    get_runtime_config,
    get_phase_timer,
    is_alignable_cell,
    filter_records_by_registration_weight,
    filter_weights_by_estimate_cutoff,
    estimate_registration_weight_cutoff,
    measure_translation_cell,
    measure_translation_cells_batched,
    regularize_displacements as shared_regularize_displacements,
    FinalizeCandidateState,
    FinalizeSettings,
    evaluate_finalize_candidates,
    filter_records_for_mesh_inclusion,
    legacy_finalize_mask,
    unlock_stale_finalized,
    use_legacy_finalize_gate,
    AnchorSmoothSettings,
    should_use_anchor_smooth_mesh,
    smooth_peaks_from_locked_anchors,
    RefineGridProgressReporter,
    count_initial_grid_points,
    report_pass_transform,
    classify_field,
    classify_roles,
    exclude_reject_mesh_records,
    field_brand_identity_suspect_ids,
    unique_large_travel_raw_preserve_ids,
    coherent_discontinuity_raw_preserve_ids,
    masked_zncc,
    Role,
    SourceContentCache,
    crop_source_cell_std,
    crop_source_cell_stds_batched,
    CellPassHistoryStore,
    write_cell_history_plots,
)
from nornir_imageregistration.refine_shared.ring_pose_limits import (
    RING_ALLOW_FLIP_CHANGE,
    RING_ANGLE_MAX_DEGREES,
    RING_SCALE_FRACTION_MAX,
    RingReferencePose,
    clamp_similarity_arrays,
    reference_pose_from_transform,
)
from nornir_imageregistration.refine_shared.discontinuity import (
    per_record_max_travel,
    sharp_warps_enabled,
    tag_discontinuities,
)
from nornir_imageregistration.refine_shared.peak_ratio_gates import (
    soft_discontinuity_ids,
)
from nornir_imageregistration.refine_shared.coherent_residual import (
    LOCK_FRAC_TRIGGER,
    MIN_MESH_ABS_AFTER_RESIDUAL,
    MIN_MESH_FRAC_AFTER_RESIDUAL,
    diagnose_coherent_residual_translation,
    should_attempt_global_fov_recovery,
    estimate_global_fov_residual_translation,
    should_keep_prior_sparse_mesh,
)
from nornir_imageregistration.refine_shared.pass_diagnostics import (
    build_pass_diagnostic_rows,
    pass_diagnostics_enabled,
    write_pass_diagnostics,
)
from nornir_imageregistration.refine_shared.adaptive_cell_size import (
    can_grow_cell_size_on_pass,
    cell_size_cap_from_shapes,
    cell_size_exceeds_requested,
    clamp_cell_size_to_cap,
    next_cell_size_after_failure,
    pass_found_no_usable_alignments,
    pass_found_registrations,
)
from nornir_imageregistration.refine_shared.phase_timer import RefinePhaseTimer as _RefinePhaseTimer
from nornir_imageregistration.refine_shared.peak_ratio_gates import finite_peak_ratio, PEAK_RATIO_MIN
import nornir_pools
from nornir_imageregistration.transforms.triangulation import Triangulation
from nornir_shared import prettyoutput

try:
    import cupyx
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupyx_thunk as cupyx

# import nornir_imageregistration.views.grid_data

# Summary type used for typing
AlignmentRecordDict = dict[tuple[int, int], nornir_imageregistration.EnhancedAlignmentRecord]
AlignmentRecordList = Sequence[nornir_imageregistration.EnhancedAlignmentRecord]
AlignmentRecordKey = tuple[int, int]


_PHASE_TIMER = get_phase_timer()


def _use_batched_vertex_measurement() -> bool:
    """Return True when the batched vertex-measurement path should be used.

    Delegates to ``RefineRuntimeConfig`` (``NORNIR_REFINE_BATCHED`` /
    ``NORNIR_REFINE_BATCHED_GPU``). Default ON for both backends.
    """
    return get_runtime_config(refresh=True).batched_vertex_measurement


def _log_phase_breakdown(label: str, baseline: dict[str, float]) -> None:
    """Log the per-phase wall-time split since *baseline* (no-op when disabled)."""
    if not _PHASE_TIMER.enabled:
        return
    deltas = {name: _PHASE_TIMER.totals.get(name, 0.0) - baseline.get(name, 0.0)
              for name in _RefinePhaseTimer.PHASES}
    measured = sum(deltas.values())
    backend = 'cupy' if nornir_imageregistration.UsingCupy() else 'numpy'
    lines = [f'{label} phase timing [{backend}] (s):']
    for name in _RefinePhaseTimer.PHASES:
        value = deltas[name]
        pct = (100.0 * value / measured) if measured > 0 else 0.0
        count = _PHASE_TIMER.counts.get(name, 0)
        lines.append(f'  {name:<12} {value:9.3f}  {pct:5.1f}%  (n={count})')
    lines.append(f'  {"measured":<12} {measured:9.3f}')
    prettyoutput.Log('\n'.join(lines))


class WeightMethod(enum.IntEnum):
    Registration = 0  # The registration score
    Distance = 1  # The distance score is the distance of the updated registration point to the original registration point
    Composite = 2  # The composite score is the distance score / max distance * registration score, distances less than 1 are set to 1


class DistortionCorrection:

    def __init__(self):
        """Initialize distortion-correction state containers."""
        self.PointsForTile = {}


@dataclass
class MosaicRefinementDiagnostics:
    iterations_completed: int
    converged: bool
    # Per-pass convergence metric: max |shift component| over all vertices (working-res px).
    average_displacement_per_iteration: list[float]
    overlap_count_per_iteration: list[int]
    control_points_per_tile: dict[int, int]
    resolved_cell_size: tuple[int, int]
    resolved_mesh_shape: tuple[int, int]
    # Per pass, per tile: counts of mesh vertices measured by FFT, filled by
    # regularization gap-fill, and updated with a non-zero shift.
    vertex_diagnostics_per_pass: list[dict[int, dict[str, int]]] = field(default_factory=list)


@dataclass(frozen=True)
class PaddedOverlapGeometry:
    """Padded overlap rectangles used for grid-refinement warping and FFT cells."""
    grid_dim: NDArray[np.int64]
    padded_scaled_source_rect_a: nornir_imageregistration.Rectangle
    padded_scaled_source_rect_b: nornir_imageregistration.Rectangle
    target_region_rect: nornir_imageregistration.Rectangle


def _normalize_pair(
        value: int | float | Sequence[int | float] | NDArray[np.integer] | NDArray[np.floating],
        param_name: str,
        minimum: int = 1) -> tuple[int, int]:
    """Normalize scalar/pair-like input into a validated integer pair."""
    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value, value]

    if len(values) == 1:
        values = [values[0], values[0]]
    if len(values) != 2:
        raise ValueError(f"{param_name} must contain exactly two values")

    out = []
    for item in values:
        if item is None:
            raise ValueError(f"{param_name} cannot contain None")
        numeric = int(round(float(cast(int | float, item))))
        if numeric < minimum:
            numeric = minimum
        out.append(numeric)

    return int(out[0]), int(out[1])


def _resolve_mesh_shape_and_cell_size(tile_shape: np.ndarray,
                                      cell_size: NDArray[np.integer] | Sequence[int] | int | None,
                                      mesh_shape: NDArray[np.integer] | Sequence[int] | int | None) -> tuple[
    tuple[int, int], tuple[int, int]]:
    """Resolve legacy-compatible mesh and cell sizing from partial inputs."""
    tile_shape = np.asarray(tile_shape, dtype=np.int64)
    if tile_shape.shape[0] != 2:
        raise ValueError("tile_shape must have two elements")

    normalized_cell_size: tuple[int, int] | None = None
    normalized_mesh_shape: tuple[int, int] | None = None

    if cell_size is not None:
        normalized_cell_size = _normalize_pair(cell_size, "cell_size", minimum=4)

    if mesh_shape is not None:
        normalized_mesh_shape = _normalize_pair(mesh_shape, "mesh_shape", minimum=2)

    if normalized_cell_size is None and normalized_mesh_shape is None:
        normalized_cell_size = (128, 128)

    if normalized_mesh_shape is None and normalized_cell_size is not None:
        # Legacy ir-refine-grid relationship: mesh = 1 + (3 * tile_dim / cell_size)
        rows = max(2, int(1 + ((3 * tile_shape[0]) / normalized_cell_size[0])))
        cols = max(2, int(1 + ((3 * tile_shape[1]) / normalized_cell_size[1])))
        normalized_mesh_shape = (rows, cols)

    if normalized_cell_size is None and normalized_mesh_shape is not None:
        # Legacy ir-refine-grid relationship: cell ~= 3 * tile_dim / (mesh - 1)
        rows, cols = normalized_mesh_shape
        cell_h = max(4, int(np.ceil((3 * tile_shape[0]) / max(1, rows - 1))))
        cell_w = max(4, int(np.ceil((3 * tile_shape[1]) / max(1, cols - 1))))
        normalized_cell_size = (cell_h, cell_w)

    assert normalized_cell_size is not None
    assert normalized_mesh_shape is not None
    return normalized_cell_size, normalized_mesh_shape


def _merge_weighted_point_pairs(point_pairs: np.ndarray, merge_distance: float) -> np.ndarray:
    """Merge nearby weighted correspondence pairs into per-bucket centroids."""
    if point_pairs.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    # Legacy ir-refine-grid semantics: only measured/usable cells should drive
    # the distortion fit. Zero-weight cells are placeholders for missing overlap.
    valid = point_pairs['Weight'] > 0
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.float32)
    point_pairs = point_pairs[valid]

    if merge_distance <= 0:
        merge_distance = 1.0

    source_points = np.vstack((point_pairs['SourceY'], point_pairs['SourceX'])).T
    target_points = np.vstack((point_pairs['TargetY'], point_pairs['TargetX'])).T
    weights = np.maximum(point_pairs['Weight'].astype(np.float64, copy=False), 0.0)

    bucket_coords = np.rint(source_points / merge_distance).astype(np.int64, copy=False)
    bucket_to_indices: dict[tuple[int, int], list[int]] = {}
    for i, bucket in enumerate(bucket_coords):
        key = (int(bucket[0]), int(bucket[1]))
        bucket_to_indices.setdefault(key, []).append(i)

    merged_pairs: list[list[float]] = []
    for indices in bucket_to_indices.values():
        idx = np.asarray(indices, dtype=np.int64)
        local_weights = weights[idx]
        if np.sum(local_weights) <= 0:
            continue

        merged_source = np.average(source_points[idx], axis=0, weights=local_weights)
        merged_target = np.average(target_points[idx], axis=0, weights=local_weights)
        merged_pairs.append([merged_target[0], merged_target[1], merged_source[0], merged_source[1]])

    if len(merged_pairs) == 0:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(merged_pairs, dtype=np.float32)


def _tile_source_shape_for_grid(tile: nornir_imageregistration.Tile) -> np.ndarray:
    """Return a valid source-image shape for grid transform resampling."""
    source_shape = np.asarray(tile.ImageSize, dtype=np.float64) * float(tile.image_to_source_space_scale)
    source_shape = np.asarray(np.ceil(source_shape), dtype=np.int64)

    return np.maximum(source_shape, 2)


def _target_center_for_refinement_cell(
        subregion_offset: NDArray[np.floating],
        overlapping_target_region: nornir_imageregistration.Rectangle,
        image_scale: float) -> NDArray[np.floating]:
    """Map a warped-overlap subregion center to full target-space coordinates."""
    downsample = 1.0 / float(image_scale)
    return np.asarray(overlapping_target_region.BottomLeft, dtype=np.float64) + (
        np.asarray(subregion_offset, dtype=np.float64) * downsample)


def _full_source_points_for_refinement_cell(
        subregion_offset: NDArray[np.floating],
        scaled_overlapping_source_rect_A: nornir_imageregistration.Rectangle,
        scaled_overlapping_source_rect_B: nornir_imageregistration.Rectangle,
        source_space_scale: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Map a refinement subregion center to full-resolution source coordinates for both tiles (linear fallback)."""
    image_source_a = subregion_offset + scaled_overlapping_source_rect_A.BottomLeft
    image_source_b = subregion_offset + scaled_overlapping_source_rect_B.BottomLeft
    full_source_a = image_source_a * source_space_scale
    full_source_b = image_source_b * source_space_scale
    return full_source_a, full_source_b


def _refinement_cell_geometry(
        subregion_offset: NDArray[np.floating],
        overlapping_target_region: nornir_imageregistration.Rectangle,
        image_scale: float,
        tile_a: nornir_imageregistration.Tile,
        tile_b: nornir_imageregistration.Tile) -> tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Map a refinement FFT cell to target center and per-tile source anchors.

    Legacy ir-refine-grid anchors each measurement at the target-space cell center and
    inverse-maps that location into each tile's source space.
    """
    target_center = _target_center_for_refinement_cell(
        subregion_offset, overlapping_target_region, image_scale)
    target_batch = np.asarray([target_center], dtype=np.float64)
    full_source_a = nornir_imageregistration.EnsureNumpyArray(
        tile_a.Transform.InverseTransform(target_batch)[0]).astype(np.float64)
    full_source_b = nornir_imageregistration.EnsureNumpyArray(
        tile_b.Transform.InverseTransform(target_batch)[0]).astype(np.float64)
    return target_center, full_source_a, full_source_b, target_center


def _base_target_for_refinement_cell(
        A: nornir_imageregistration.Tile,
        B: nornir_imageregistration.Tile,
        full_source_a: NDArray[np.floating],
        full_source_b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the nominal target-space center for a refinement subregion.

    Accepts NumPy or CuPy source points; Transform I/O is host-snapshotted for the 2-float mean.
    """
    src_a = nornir_imageregistration.EnsureNumpyArray(full_source_a, dtype=np.float64).reshape(1, 2)
    src_b = nornir_imageregistration.EnsureNumpyArray(full_source_b, dtype=np.float64).reshape(1, 2)
    target_a = nornir_imageregistration.EnsureNumpyArray(A.Transform.Transform(src_a)[0])
    target_b = nornir_imageregistration.EnsureNumpyArray(B.Transform.Transform(src_b)[0])
    return (target_a + target_b) * 0.5


def _filter_weighted_point_pair_updates(point_pair_updates: np.ndarray) -> np.ndarray:
    """
    Drop low-confidence overlap updates using the shared estimate_cutoff helper.
    """
    return filter_records_by_registration_weight(point_pair_updates)


def _phase_correlate_refinement_cell(
        cell_a: NDArray[np.floating],
        cell_b: NDArray[np.floating],
        subregion_shape: NDArray[np.integer],
        *,
        min_overlap: float = 0.25,
        max_overlap: float = 1.0) -> nornir_imageregistration.AlignmentRecord:
    """Preprocess and phase-correlate one refinement FFT cell (translate-path parity)."""
    return measure_translation_cell(
        cell_a, cell_b, subregion_shape, min_overlap=min_overlap, max_overlap=max_overlap)


@dataclass
class _PrewarpedTile:
    """A tile fully warped into (scaled) mosaic space for one refinement pass."""
    image: NDArray[np.floating]  # invalid pixels are zero, matching legacy extraction
    valid_mask: NDArray[np.bool_]
    origin: NDArray[np.int64]  # (y, x) of pixel [0, 0] in scaled target space


def _prewarp_tile_for_grid_refine(
        tile: nornir_imageregistration.Tile,
        target_space_scale: float) -> _PrewarpedTile:
    """
    Warp a full tile into scaled mosaic space using its current transform.

    Mirrors legacy ir-refine-grid ``prewarp_tiles=true``: every pass re-renders each
    tile so mesh-vertex neighborhoods can be sampled on a common mosaic grid.
    """
    full_source_image = cast(
        NDArray[np.floating],
        nornir_imageregistration.ForceGrayscale(
            nornir_imageregistration.ImageParamToImageArray(tile.Image)))

    transform, _ = _scale_tile_transform_for_warp(tile, target_space_scale)

    if target_space_scale != 1.0:
        scaled_target_region = nornir_imageregistration.Rectangle.scale_on_origin(
            tile.FixedBoundingBox, target_space_scale)
        scaled_rounded_target_region = nornir_imageregistration.Rectangle.SnapRound(scaled_target_region)
    else:
        scaled_rounded_target_region = nornir_imageregistration.Rectangle.SnapRound(tile.FixedBoundingBox)

    target_height = int(scaled_rounded_target_region.Height)
    target_width = int(scaled_rounded_target_region.Width)
    target_min_y = int(scaled_rounded_target_region.MinY)
    target_min_x = int(scaled_rounded_target_region.MinX)

    read_coords, write_coords = nornir_imageregistration.assemble.write_to_target_roi_coords(
        transform,
        (target_min_y, target_min_x),
        (target_height, target_width),
        extrapolate=False)

    # Legacy ir-refine-grid prewarp uses itk::NearestNeighborInterpolateImageFunction
    # (common.hxx warp<>), not cubic resampling.
    warp_kwargs = {
        'output_origin': (target_min_y, target_min_x),
        'output_area': (target_height, target_width),
        'cval': 0,
        'interpolation_order': 0,
    }
    if _prewarp_single_warp_coverage_enabled():
        # Opt-in: derive the true coverage from the warp's scatter indices in a
        # single warp instead of warping a second ones-image.
        warped_image, valid_mask = cast(
            tuple[NDArray[np.floating], NDArray[np.bool_]],
            nornir_imageregistration.assemble._TransformImageUsingCoords(
                write_coords,
                read_coords,
                full_source_image,
                return_valid_mask=True,
                **warp_kwargs))
        # The warp clips its output to the source min/max, which can raise the
        # cval=0 background above zero, so re-zero non-covered pixels.
        xp = cp.get_array_module(warped_image)
        warped_image = xp.where(valid_mask, warped_image, 0)

        del read_coords
        del write_coords
        del full_source_image
    else:
        warped_image = cast(
            NDArray[np.floating],
            nornir_imageregistration.assemble._TransformImageUsingCoords(
                write_coords,
                read_coords,
                full_source_image,
                **warp_kwargs))

        # Warp a ones-image to obtain coverage: pixels outside the transform domain or
        # outside the source image read the cval and drop below the validity threshold.
        xp = cp.get_array_module(warped_image)
        coverage_source = xp.ones(full_source_image.shape[0:2], dtype=np.float32)
        coverage = cast(
            NDArray[np.floating],
            nornir_imageregistration.assemble._TransformImageUsingCoords(
                write_coords,
                read_coords,
                coverage_source,
                **warp_kwargs))

        valid_mask = coverage > 0.999
        warped_image = xp.where(valid_mask, warped_image, 0)

        del read_coords
        del write_coords
        del coverage
        del coverage_source
        del full_source_image

    return _PrewarpedTile(
        image=warped_image,
        valid_mask=valid_mask,
        origin=np.asarray((target_min_y, target_min_x), dtype=np.int64))


def _store_prewarped_tile_cache(
        prewarp_cache: dict[int, _PrewarpedTile],
        tile_id: int,
        warped: _PrewarpedTile) -> None:
    """Replace one cached prewarp, dropping the previous tile warp first."""
    stale = prewarp_cache.pop(tile_id, None)
    del stale
    prewarp_cache[tile_id] = warped


def _prewarp_all_tiles_for_grid_refine(
        tiles: list[nornir_imageregistration.Tile],
        target_space_scale: float,
        prewarp_cache: dict[int, _PrewarpedTile] | None = None,
        revision_cache: dict[int, int] | None = None) -> dict[int, _PrewarpedTile]:
    """
    Prewarp every tile for one grid-refinement pass, using workers on CPU hosts.

    When ``prewarp_cache`` and ``revision_cache`` are supplied, tiles whose lattice
    revision is unchanged since the last pass reuse the cached warp.
    """
    if prewarp_cache is None:
        prewarp_cache = {}
    if revision_cache is None:
        revision_cache = {}

    use_cache = _prewarp_cache_enabled()
    tiles_to_prewarp: list[nornir_imageregistration.Tile] = []
    prewarped: dict[int, _PrewarpedTile] = {}

    for tile in tiles:
        grid_transform = tile.Transform
        if not isinstance(grid_transform, nornir_imageregistration.transforms.IGridTransform):
            tiles_to_prewarp.append(tile)
            continue

        if use_cache:
            tile_revision = revision_cache.get(tile.ID, 0)
            cached = prewarp_cache.get(tile.ID)
            if cached is not None and getattr(cached, '_revision', None) == tile_revision:
                prewarped[tile.ID] = cached
                continue

        tiles_to_prewarp.append(tile)

    if len(tiles_to_prewarp) == 0:
        return prewarped

    # Thread dispatch overlaps host tile-load + coord compute (CPU SciPy inverse)
    # with warp kernels. It is the CuPy default (output-neutral, ~20% faster);
    # 'serial' opts out. CPU keeps its multiprocess-pool path unless 'thread'.
    use_thread_dispatch = _prewarp_thread_dispatch_enabled()
    if len(tiles_to_prewarp) <= 1 or (nornir_imageregistration.UsingCupy() and not use_thread_dispatch):
        for tile in tiles_to_prewarp:
            warped = _prewarp_tile_for_grid_refine(tile, target_space_scale)
            if use_cache:
                warped._revision = revision_cache.get(tile.ID, 0)  # type: ignore[attr-defined]
                _store_prewarped_tile_cache(prewarp_cache, tile.ID, warped)
            prewarped[tile.ID] = warped
        return prewarped

    if use_thread_dispatch:
        pool = nornir_pools.GetGlobalThreadPool()
    else:
        pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = [
        pool.add_task(
            f"grid_prewarp_{tile.ID}",
            _prewarp_tile_for_grid_refine,
            tile,
            target_space_scale)
        for tile in tiles_to_prewarp
    ]
    pool.wait_completion()
    for tile, task in zip(tiles_to_prewarp, tasks):
        warped = task.wait_return()
        if use_cache:
            warped._revision = revision_cache.get(tile.ID, 0)  # type: ignore[attr-defined]
            _store_prewarped_tile_cache(prewarp_cache, tile.ID, warped)
        prewarped[tile.ID] = warped
    return prewarped


def _extract_refinement_cell(
        prewarped: _PrewarpedTile,
        center_scaled: NDArray[np.floating],
        cell_shape: NDArray[np.integer]) -> tuple[NDArray[np.floating], float]:
    """
    Extract one cell-sized mosaic-space neighborhood centered on a mesh vertex.

    Pixels outside the warped tile are zero (legacy extraction convention).
    Returns the cell and the fraction of valid pixels within it.
    """
    xp = cp.get_array_module(prewarped.image)
    cell_shape = np.asarray(cell_shape, dtype=np.int64)
    cell, valid_cell = _extract_refinement_cell_and_mask(prewarped, center_scaled, cell_shape)
    valid_count = float(xp.count_nonzero(valid_cell))
    return cell, valid_count / float(cell_shape.prod())


_CellWindow = tuple[tuple[int, int, int, int], tuple[int, int, int, int]]


def _refinement_cell_window(
        prewarped: _PrewarpedTile,
        center_scaled: NDArray[np.floating],
        cell_shape: NDArray[np.integer]) -> _CellWindow | None:
    """
    Resolve the pure-host slice geometry for one refinement cell.

    Returns ``((ws0, we0, ws1, we1), (rs0, re0, rs1, re1))`` naming the
    destination window within a ``cell_shape`` cell and the source window within
    the prewarped tile, or None when the cell falls entirely off the tile. Split
    out from extraction so callers can copy straight into a preallocated batch
    stack rather than materializing one array per cell.
    """
    cell_shape = np.asarray(cell_shape, dtype=np.int64)
    # Legacy: origin = center - 0.5 * cell; pixel index i samples origin + i.
    start = np.floor(
        nornir_imageregistration.EnsureNumpyArray(center_scaled, dtype=np.float64)
        - prewarped.origin
        - (cell_shape.astype(np.float64) / 2.0)).astype(np.int64)
    stop = start + cell_shape

    image_shape = np.asarray(prewarped.image.shape, dtype=np.int64)
    clipped_start = np.maximum(start, 0)
    clipped_stop = np.minimum(stop, image_shape)
    if np.any(clipped_start >= clipped_stop):
        return None

    write_start = clipped_start - start
    write_stop = write_start + (clipped_stop - clipped_start)
    return ((int(write_start[0]), int(write_stop[0]),
             int(write_start[1]), int(write_stop[1])),
            (int(clipped_start[0]), int(clipped_stop[0]),
             int(clipped_start[1]), int(clipped_stop[1])))


def _extract_refinement_cell_and_mask(
        prewarped: _PrewarpedTile,
        center_scaled: NDArray[np.floating],
        cell_shape: NDArray[np.integer]) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Extract one cell and its full-size validity mask without any host sync.

    Returns ``(cell, valid_cell)`` both of shape ``cell_shape`` on the prewarped
    tile's array module. ``valid_cell`` is True where the cell sampled a covered
    prewarped pixel. The validity fraction is intentionally NOT computed here so
    the batched caller can reduce many cells in a single device->host transfer.
    """
    xp = cp.get_array_module(prewarped.image)
    cell_tuple = tuple(int(v) for v in np.asarray(cell_shape, dtype=np.int64))

    cell = xp.zeros(cell_tuple, dtype=prewarped.image.dtype)
    valid_cell = xp.zeros(cell_tuple, dtype=bool)

    window = _refinement_cell_window(prewarped, center_scaled, cell_shape)
    if window is None:
        return cell, valid_cell

    (ws0, we0, ws1, we1), (rs0, re0, rs1, re1) = window
    cell[ws0:we0, ws1:we1] = prewarped.image[rs0:re0, rs1:re1]
    valid_cell[ws0:we0, ws1:we1] = prewarped.valid_mask[rs0:re0, rs1:re1]
    return cell, valid_cell


def _measure_grid_vertex_displacements(
        moving: _PrewarpedTile,
        fixed: _PrewarpedTile,
        centers_scaled: NDArray[np.floating],
        cell_shape: NDArray[np.integer],
        cell_min_overlap: float) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Measure mosaic-space shifts for every mesh vertex of the moving tile against one neighbor.

    Port of legacy ``calc_displacements``: each vertex's current mosaic position is the
    neighborhood center; both prewarped tiles are sampled on the same mosaic window so the
    measured phase-correlation peak is the residual shift to apply to the moving tile.
    Returns per-vertex (y, x) shifts in scaled mosaic pixels and a measured flag.
    """
    centers_scaled = nornir_imageregistration.EnsureNumpyArray(centers_scaled, dtype=np.float64)
    cell_shape = np.asarray(cell_shape, dtype=np.int64)
    num_vertices = centers_scaled.shape[0]
    shifts = np.zeros((num_vertices, 2), dtype=np.float64)
    measured = np.zeros(num_vertices, dtype=bool)
    weights = np.zeros(num_vertices, dtype=np.float64)

    fixed_shape = np.asarray(fixed.image.shape, dtype=np.float64)
    for k in range(num_vertices):
        center = centers_scaled[k]
        # Legacy skips vertices whose center is outside the fixed tile's buffer.
        local_fixed = center - fixed.origin
        if np.any(local_fixed < 0) or np.any(local_fixed >= fixed_shape):
            continue

        with _PHASE_TIMER.section('cell_extract'):
            fixed_cell, fixed_fraction = _extract_refinement_cell(fixed, center, cell_shape)
        if fixed_fraction < cell_min_overlap:
            continue
        with _PHASE_TIMER.section('cell_extract'):
            moving_cell, moving_fraction = _extract_refinement_cell(moving, center, cell_shape)
        if moving_fraction < cell_min_overlap:
            continue

        try:
            # find_offset peak = shift to apply to the source (second) image's content;
            # the moving tile is the source so its vertices receive +peak.
            with _PHASE_TIMER.section('fft'):
                record = _phase_correlate_refinement_cell(
                    fixed_cell,
                    moving_cell,
                    cell_shape,
                    min_overlap=cell_min_overlap)
        except Exception as e:
            prettyoutput.LogErr(f'Exception phase-correlating mesh vertex {k}:\n{e}')
            continue

        with _PHASE_TIMER.section('host_sync'):
            peak = np.asarray(
                nornir_imageregistration.EnsureNumpyArray(record.peak), dtype=np.float64).reshape(-1)
        if record.weight <= 0 or np.any(np.isnan(peak)):
            continue

        shifts[k, :] = peak
        measured[k] = True
        weights[k] = float(record.weight)

    return _apply_optional_mosaic_weight_cutoff(shifts, measured, weights)


def _apply_optional_mosaic_weight_cutoff(
        shifts: NDArray[np.floating],
        measured: NDArray[np.bool_],
        weights: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """When ``NORNIR_REFINE_MOSAIC_CUTOFF`` is set, drop low-weight vertex measurements."""
    if not get_runtime_config(refresh=True).mosaic_cutoff:
        return shifts, measured
    keep = filter_weights_by_estimate_cutoff(weights)
    drop = measured & ~keep
    if np.any(drop):
        shifts = shifts.copy()
        measured = measured.copy()
        shifts[drop, :] = 0.0
        measured[drop] = False
    return shifts, measured


def _measure_grid_vertex_displacements_batched(
        moving: _PrewarpedTile,
        fixed: _PrewarpedTile,
        centers_scaled: NDArray[np.floating],
        cell_shape: NDArray[np.integer],
        cell_min_overlap: float) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Batched analog of ``_measure_grid_vertex_displacements`` (prototype).

    Applies the same per-vertex eligibility gating as the serial path, then
    correlates all eligible cells in a single batched FFT + vectorized peak
    pass (one host transfer for the whole batch) instead of one tiny FFT and
    one ``.get()`` per vertex. Returns identical ``(shifts, measured)`` arrays.
    """
    centers_scaled = nornir_imageregistration.EnsureNumpyArray(centers_scaled, dtype=np.float64)
    cell_shape = np.asarray(cell_shape, dtype=np.int64)
    num_vertices = centers_scaled.shape[0]
    shifts = np.zeros((num_vertices, 2), dtype=np.float64)
    measured = np.zeros(num_vertices, dtype=bool)

    fixed_shape = np.asarray(fixed.image.shape, dtype=np.float64)
    xp = cp.get_array_module(fixed.image)
    cell_tuple = (int(cell_shape[0]), int(cell_shape[1]))
    candidate_indices: list[int] = []
    candidate_windows: list[tuple[_CellWindow | None, _CellWindow | None]] = []

    with _PHASE_TIMER.section('cell_extract'):
        # Two passes so no per-cell array is ever materialized: gate on validity
        # using only the bool masks, then copy the surviving image cells straight
        # into an exactly-sized stack. Accumulating one array per cell and then
        # stacking held the whole candidate set several times over at the moment
        # of the stack, and freeing the pieces afterwards cannot undo that peak.
        # Tile measurement can run on a thread pool, so the peak is multiplied by
        # the worker count.
        fixed_valid_stack = xp.zeros((num_vertices,) + cell_tuple, dtype=bool)
        moving_valid_stack = xp.zeros((num_vertices,) + cell_tuple, dtype=bool)

        for k in range(num_vertices):
            center = centers_scaled[k]
            local_fixed = center - fixed.origin
            if np.any(local_fixed < 0) or np.any(local_fixed >= fixed_shape):
                continue

            row = len(candidate_indices)
            fixed_window = _refinement_cell_window(fixed, center, cell_shape)
            moving_window = _refinement_cell_window(moving, center, cell_shape)
            if fixed_window is not None:
                (ws0, we0, ws1, we1), (rs0, re0, rs1, re1) = fixed_window
                fixed_valid_stack[row, ws0:we0, ws1:we1] = fixed.valid_mask[rs0:re0, rs1:re1]
            if moving_window is not None:
                (ws0, we0, ws1, we1), (rs0, re0, rs1, re1) = moving_window
                moving_valid_stack[row, ws0:we0, ws1:we1] = moving.valid_mask[rs0:re0, rs1:re1]
            candidate_indices.append(k)
            candidate_windows.append((fixed_window, moving_window))

        num_candidates = len(candidate_indices)
        if num_candidates == 0:
            return shifts, measured

        cell_area = float(cell_shape.prod())
        # One batched validity reduction + a single device->host transfer for the
        # whole candidate set, instead of one count_nonzero sync per cell. The
        # count vectors are (N,), so stacking them keeps it to one transfer.
        counts_host = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(
                xp.stack((xp.count_nonzero(fixed_valid_stack[:num_candidates], axis=(1, 2)),
                          xp.count_nonzero(moving_valid_stack[:num_candidates], axis=(1, 2))),
                         axis=0)),
            dtype=np.float64)
        del fixed_valid_stack, moving_valid_stack
        fixed_fraction = counts_host[0] / cell_area
        moving_fraction = counts_host[1] / cell_area
        eligible_mask = (fixed_fraction >= cell_min_overlap) & (moving_fraction >= cell_min_overlap)

    eligible_positions = [i for i in range(num_candidates) if eligible_mask[i]]
    eligible_indices = [candidate_indices[i] for i in eligible_positions]
    if len(eligible_indices) == 0:
        return shifts, measured

    with _PHASE_TIMER.section('cell_extract'):
        fixed_stack = xp.zeros((len(eligible_positions),) + cell_tuple, dtype=fixed.image.dtype)
        moving_stack = xp.zeros((len(eligible_positions),) + cell_tuple, dtype=moving.image.dtype)
        for row, position in enumerate(eligible_positions):
            fixed_window, moving_window = candidate_windows[position]
            if fixed_window is not None:
                (ws0, we0, ws1, we1), (rs0, re0, rs1, re1) = fixed_window
                fixed_stack[row, ws0:we0, ws1:we1] = fixed.image[rs0:re0, rs1:re1]
            if moving_window is not None:
                (ws0, we0, ws1, we1), (rs0, re0, rs1, re1) = moving_window
                moving_stack[row, ws0:we0, ws1:we1] = moving.image[rs0:re0, rs1:re1]

    with _PHASE_TIMER.section('fft'):
        peaks_dev, weights_dev, _peak_ratios_dev = measure_translation_cells_batched(
            fixed_stack,
            moving_stack,
            cell_shape,
            min_overlap=cell_min_overlap,
            max_overlap=1.0)

    with _PHASE_TIMER.section('host_sync'):
        peaks = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(peaks_dev), dtype=np.float64).reshape(-1, 2)
        weights = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(weights_dev), dtype=np.float64).reshape(-1)

    weight_field = np.zeros(num_vertices, dtype=np.float64)
    for batch_pos, k in enumerate(eligible_indices):
        peak = peaks[batch_pos]
        if weights[batch_pos] <= 0 or np.any(np.isnan(peak)):
            continue
        shifts[k, :] = peak
        measured[k] = True
        weight_field[k] = float(weights[batch_pos])

    return _apply_optional_mosaic_weight_cutoff(shifts, measured, weight_field)


def _regularize_displacements(
        shifts: NDArray[np.floating],
        measured: NDArray[np.bool_],
        mesh_dims: tuple[int, int],
        median_radius: int = 1) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Port of legacy ``regularize_displacements``; delegates to refine_shared."""
    return shared_regularize_displacements(shifts, measured, mesh_dims, median_radius=median_radius)


def _grid_refine_neighbors(
        list_tiles: Sequence[nornir_imageregistration.Tile]) -> dict[int, list[nornir_imageregistration.Tile]]:
    """Find overlapping neighbors per tile via target-space bounding-box intersection (legacy rule)."""
    neighbors: dict[int, list[nornir_imageregistration.Tile]] = {tile.ID: [] for tile in list_tiles}
    for i, tile_i in enumerate(list_tiles):
        for j, tile_j in enumerate(list_tiles):
            if i == j:
                continue
            if nornir_imageregistration.Rectangle.contains(
                    tile_i.FixedBoundingBox, tile_j.FixedBoundingBox):
                neighbors[tile_i.ID].append(tile_j)
    return neighbors


def _legacy_overlap_target_region(
        tile_a: nornir_imageregistration.Tile,
        tile_b: nornir_imageregistration.Tile,
        grid_dim: NDArray[np.integer],
        subregion_shape: NDArray[np.integer],
        image_scale: float) -> nornir_imageregistration.Rectangle:
    """Return the legacy ir-refine-grid target-space ROI used for overlap warping."""
    downsample = 1.0 / float(image_scale)
    overlapping_rect = nornir_imageregistration.Rectangle.overlap_rect(
        tile_a.FixedBoundingBox, tile_b.FixedBoundingBox)
    if overlapping_rect is None:
        raise ValueError(f"Tiles {tile_a.ID} and {tile_b.ID} do not overlap in target space")
    target_size = (int(grid_dim[0] * subregion_shape[0] * downsample),
                   int(grid_dim[1] * subregion_shape[1] * downsample))
    return nornir_imageregistration.Rectangle.change_area(overlapping_rect, target_size)


def _padded_scaled_overlap_source_rects(
        scaled_overlapping_source_rect_a: nornir_imageregistration.Rectangle,
        scaled_overlapping_source_rect_b: nornir_imageregistration.Rectangle,
        grid_dim: NDArray[np.integer],
        subregion_shape: NDArray[np.integer]) -> tuple[
    nornir_imageregistration.Rectangle, nornir_imageregistration.Rectangle]:
    """Pad scaled overlap source rectangles to the refinement FFT grid."""
    padded_image_size = (int(grid_dim[0] * subregion_shape[0]), int(grid_dim[1] * subregion_shape[1]))
    padded_a = nornir_imageregistration.Rectangle.change_area(
        scaled_overlapping_source_rect_a, padded_image_size)
    padded_b = nornir_imageregistration.Rectangle.change_area(
        scaled_overlapping_source_rect_b, padded_image_size)
    padded_b = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
        padded_b.BottomLeft, padded_a.Size)
    return padded_a, padded_b


def _max_refinement_displacement(
        overlapping_target_rect: nornir_imageregistration.Rectangle,
        subregion_shape: NDArray[np.integer],
        image_scale: float) -> float:
    """Upper bound for a plausible refinement displacement magnitude in full target space."""
    downsample = 1.0 / float(image_scale)
    cell_extent = float(np.max(subregion_shape) * downsample)
    overlap_extent = float(max(overlapping_target_rect.Width, overlapping_target_rect.Height))
    return max(cell_extent * 4.0, overlap_extent * 0.5)


def _refinement_pool_for_overlap_tasks(target_space_scale: float):
    """Return a pool sized for overlap refinement without multiplying warp memory by worker count."""
    if nornir_imageregistration.UsingCupy() or target_space_scale >= 1.0:
        return get_runtime_config(refresh=True).pool_for_cell_tasks(True)
    return get_runtime_config(refresh=True).pool_for_cell_tasks(False)


def _angles_are_translation_only(angles_to_search: Iterable[float] | None) -> bool:
    """Return True when the angle search is empty or only zero (translation-only cells)."""
    if angles_to_search is None:
        return False
    angles = np.asarray(list(angles_to_search), dtype=np.float64).reshape(-1)
    if angles.size == 0:
        return True
    return bool(np.all(np.isclose(angles, 0.0)))


def _maybe_regularize_stos_alignment_peaks(
        alignment_points: list[nornir_imageregistration.EnhancedAlignmentRecord]
) -> list[nornir_imageregistration.EnhancedAlignmentRecord]:
    """Optionally spatially regularize STOS peaks (``NORNIR_REFINE_STOS_REGULARIZE=1``)."""
    if not get_runtime_config(refresh=True).stos_regularize:
        return alignment_points
    if len(alignment_points) < 3:
        return alignment_points

    rows = [int(rec.ID[0]) for rec in alignment_points]
    cols = [int(rec.ID[1]) for rec in alignment_points]
    mesh_rows = max(rows) + 1
    mesh_cols = max(cols) + 1
    shifts = np.zeros((mesh_rows * mesh_cols, 2), dtype=np.float64)
    measured = np.zeros(mesh_rows * mesh_cols, dtype=bool)
    index_of: dict[tuple[int, int], int] = {}
    for rec in alignment_points:
        r, c = int(rec.ID[0]), int(rec.ID[1])
        idx = r * mesh_cols + c
        shifts[idx, :] = np.asarray(rec.peak, dtype=np.float64).reshape(2)
        measured[idx] = True
        index_of[(r, c)] = idx

    regularized, _ = shared_regularize_displacements(
        shifts, measured, (mesh_rows, mesh_cols), median_radius=1)

    updated: list[nornir_imageregistration.EnhancedAlignmentRecord] = []
    for rec in alignment_points:
        r, c = int(rec.ID[0]), int(rec.ID[1])
        peak = regularized[index_of[(r, c)], :]
        updated.append(nornir_imageregistration.EnhancedAlignmentRecord(
            ID=rec.ID,
            TargetPoint=rec.TargetPoint,
            SourcePoint=rec.SourcePoint,
            peak=peak,
            weight=rec.weight,
            angle=rec.angle,
            flipped_ud=rec.flippedud,
            peak_ratio=getattr(rec, 'peak_ratio', None)))
    return updated


def _batched_roi_sample_budget(cell_h: int, cell_w: int) -> int:
    """Max (cells * H * W) samples per ``map_coordinates`` launch for batched ROI extract."""
    from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_roi_sample_budget
    return batched_roi_sample_budget(cell_h, cell_w)


def _alignment_roi_botlefts(target_points: NDArray,
                            alignment_area: NDArray | tuple[float, float]) -> NDArray:
    """Return integer-origin botlefts ``(N, 2)`` matching ``BuildAlignmentROIs`` geometry.

    Accepts NumPy or CuPy; ops follow ``cp.get_array_module``.
    """
    xp = cp.get_array_module(target_points)
    area = xp.asarray(alignment_area, dtype=xp.float64).ravel()[:2]
    points = xp.asarray(target_points, dtype=xp.float64).reshape(-1, 2).copy()
    adjust_mask = xp.mod(area, 2) > 0
    points[:, adjust_mask] += 0.5
    # Same as Rectangle.CreateFromPointAndArea → SnapRound → change_area(..., integer_origin=True)
    botleft = points - (area / 2.0)
    return xp.floor(botleft)


def _rigid_inverse_matrices(rigid_transforms: Sequence[nornir_imageregistration.ITransform],
                            xp) -> tuple[NDArray, NDArray[np.bool_]] | None:
    """Stack 3x3 inverse affine matrices and a pure-translation mask, or ``None`` if unsupported."""
    matrices: list[NDArray[np.float64]] = []
    pure_flags: list[bool] = []
    for transform in rigid_transforms:
        if not isinstance(transform, nornir_imageregistration.transforms.IRigidTransform):
            return None
        angle = float(getattr(transform, 'angle', 0.0) or 0.0)
        scalar = float(getattr(transform, 'scalar', 1.0) or 1.0)
        flip_ud = bool(getattr(transform, 'flip_ud', False))
        offset = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(getattr(transform, 'target_offset')),
            dtype=np.float64).ravel()[:2]
        if angle == 0.0 and scalar == 1.0 and not flip_ud:
            # Match Rigid.InverseTransform's pure-translation branch (no rounding).
            matrix = np.eye(3, dtype=np.float64)
            matrix[0, 2] = -offset[0]
            matrix[1, 2] = -offset[1]
            matrices.append(matrix)
            pure_flags.append(True)
            continue
        inverse = getattr(transform, 'inverse_matrix', None)
        if inverse is None:
            return None
        matrices.append(np.asarray(nornir_imageregistration.EnsureNumpyArray(inverse), dtype=np.float64))
        pure_flags.append(False)
    return (xp.asarray(np.stack(matrices, axis=0), dtype=np.float64),
            np.asarray(pure_flags, dtype=bool))


def _global_sample_bounds(relative: NDArray,
                          botlefts_dev: NDArray,
                          linear_t: NDArray,
                          offset: NDArray,
                          src_h: int,
                          src_w: int,
                          xp) -> tuple[int, int, int, int]:
    """Source-image AABB covering every cell's samples, as ``(y0, x0, y1, x1)``.

    Taken from the four corners of each cell's sample grid rather than from the grid
    itself: the maps are affine, so a rectangle's image is a parallelogram whose
    extreme points are the images of its corners. That makes this exact while costing
    an ``(n, 4, 2)`` array instead of the ``(n, H*W, 2)`` the full coordinate set needs
    -- the whole reason the sampling is chunked in the first place.

    Returns an empty box (``y1 <= y0``) when no cell produces a finite coordinate.
    """
    if int(botlefts_dev.shape[0]) == 0:
        return 0, 0, 0, 0

    rel_min = relative.min(axis=0)
    rel_max = relative.max(axis=0)
    corners = xp.stack((
        xp.stack((rel_min[0], rel_min[1])),
        xp.stack((rel_min[0], rel_max[1])),
        xp.stack((rel_max[0], rel_min[1])),
        xp.stack((rel_max[0], rel_max[1])),
    ))                                                          # (4, 2)

    write = corners[None, :, :] + botlefts_dev[:, None, :]      # (n, 4, 2)
    mapped = xp.matmul(write.astype(np.float64, copy=False), linear_t)
    mapped += offset[:, None, :]

    finite = xp.where(xp.isfinite(mapped), mapped, xp.nan)
    flat = finite.reshape(-1, 2)
    mins_maxs = nornir_imageregistration.EnsureNumpyArray(xp.stack((
        xp.nanmin(flat, axis=0),
        xp.nanmax(flat, axis=0),
    )))
    if np.any(np.isnan(mins_maxs)):
        return 0, 0, 0, 0

    # Same floor/ceil widening the per-chunk crop used, so a single-chunk batch lands on
    # exactly the box it used to. The rounding applied to non-pure-translation cells
    # moves coordinates by well under the one pixel this already grants.
    mins = np.floor(mins_maxs[0]).astype(np.int64)
    maxs = np.ceil(mins_maxs[1]).astype(np.int64)
    return (int(max(0, mins[0])),
            int(max(0, mins[1])),
            int(min(src_h, maxs[0] + 1)),
            int(min(src_w, maxs[1] + 1)))


def _sample_source_rois_batched(source_image: NDArray,
                                inverse_matrices: NDArray,
                                pure_translation_mask: NDArray[np.bool_],
                                botlefts: NDArray,
                                cell_h: int,
                                cell_w: int,
                                xp,
                                sp,
                                oob_cval: float = np.nan) -> NDArray:
    """Inverse-map each cell's target grid through its rigid matrix and sample once (chunked)."""
    num_cells = int(botlefts.shape[0])
    relative = nornir_imageregistration.assemble.GetROICoords((0.0, 0.0), (cell_h, cell_w), xp=xp)
    relative = xp.asarray(relative, dtype=np.float32)
    botlefts_dev = _ensure_on_array_module(botlefts, xp).astype(np.float32, copy=False)
    matrices_dev = _ensure_on_array_module(inverse_matrices, xp).astype(np.float64, copy=False)
    pure_mask = np.asarray(pure_translation_mask, dtype=bool).reshape(-1)

    source_image = _ensure_on_array_module(source_image, xp)
    original_dtype = source_image.dtype
    if source_image.dtype == np.float16:
        source_image = source_image.astype(np.float32, copy=False)

    any_nan_values = bool(xp.any(xp.isnan(source_image)))
    # Match _TransformImageUsingCoords: order 1 when source contains NaN (or bool).
    order = 1 if any_nan_values or source_image.dtype == bool else 3
    prefilter = order > 1
    def _scalar_to_float(value) -> float:
        return float(nornir_imageregistration.EnsureNumpyArray(xp.asarray(value).reshape((1,)))[0])

    if any_nan_values:
        finite_src = source_image[~xp.isnan(source_image)]
        min_val = _scalar_to_float(finite_src.min())
        max_val = _scalar_to_float(finite_src.max())
    else:
        min_val = _scalar_to_float(source_image.min())
        max_val = _scalar_to_float(source_image.max())

    oob_cval_float = float(oob_cval)
    preserve_oob_sentinel = bool(np.isnan(oob_cval_float) or oob_cval_float < min_val or oob_cval_float > max_val)

    samples_per_cell = cell_h * cell_w
    chunk_cells = max(1, _batched_roi_sample_budget(cell_h, cell_w) // max(1, samples_per_cell))
    # Write each chunk into the final stack instead of collecting chunks and
    # concatenating, which held every chunk plus a full second copy of the result.
    sampled_stack = xp.empty((num_cells, cell_h, cell_w), dtype=original_dtype)
    use_gpu = xp is not np
    lock = nornir_imageregistration.assemble._gpu_warp_lock if use_gpu else contextlib.nullcontext()
    src_h = int(source_image.shape[0])
    src_w = int(source_image.shape[1])

    # The inverse maps are affine, so apply the 2x2 linear part and the translation
    # column directly instead of expanding to homogeneous coordinates. Building the
    # (n, HW, 3) float64 homogeneous array and taking a (n, HW, 3) matmul result to
    # then discard its last column cost two of the largest allocations in this
    # function; the coordinate machinery, not the ROI stacks, sets the peak here.
    linear_t = xp.swapaxes(matrices_dev[:, :2, :2], -1, -2)
    offset = matrices_dev[:, :2, 2]

    # Crop and spline-prefilter ONCE over the union of every cell's samples, before the
    # chunk loop, rather than per chunk. With order=3 the prefilter's boundary
    # conditions depend on the extent it runs over, so a per-chunk crop made a cell's
    # sampled values depend on which other cells happened to share its chunk -- up to
    # 8% of full intensity range. chunk_cells is a memory-budget knob, so that let a
    # tuning parameter change registration output. Production grids run in one chunk,
    # where the union crop *is* the chunk crop, so those values are unchanged.
    gy0, gx0, gy1, gx1 = _global_sample_bounds(
        relative, botlefts_dev, linear_t, offset, src_h, src_w, xp)
    if gy1 <= gy0 or gx1 <= gx0:
        # No cell lands on the image; every sample is out of bounds.
        return xp.full((num_cells, cell_h, cell_w), oob_cval_float, dtype=original_dtype)

    sample_source = source_image[gy0:gy1, gx0:gx1]
    if prefilter:
        # scipy applies no prepadding for mode='constant' (_prepad_for_spline_filter
        # returns npad=0), so filtering here and sampling with prefilter=False is
        # equivalent to what map_coordinates did internally, only once and over a
        # domain that no longer moves with the chunking.
        sample_source = sp.ndimage.spline_filter(
            sample_source, order=order, output=np.float64, mode='constant')
    crop_origin = xp.asarray((gy0, gx0), dtype=np.float32)

    for start in range(0, num_cells, chunk_cells):
        stop = min(num_cells, start + chunk_cells)
        chunk_n = stop - start
        write = relative[None, :, :] + botlefts_dev[start:stop, None, :]  # (n, HW, 2)
        # (n, HW, 2) @ (n, 2, 2)^T + (n, 1, 2) -> (n, HW, 2)
        source_yx = xp.matmul(write.astype(np.float64, copy=False), linear_t[start:stop])
        source_yx += offset[start:stop][:, None, :]
        chunk_pure = pure_mask[start:stop]
        if np.all(chunk_pure):
            pass
        elif np.any(chunk_pure):
            rounded = xp.around(
                source_yx,
                nornir_imageregistration.RoundingPrecision(source_yx.dtype))
            pure_dev = xp.asarray(chunk_pure)[:, None, None]
            source_yx = xp.where(pure_dev, source_yx, rounded)
        else:
            source_yx = xp.around(
                source_yx,
                nornir_imageregistration.RoundingPrecision(source_yx.dtype))
        sample_coords = source_yx.reshape(chunk_n * samples_per_cell, 2).astype(np.float32, copy=False)

        # Sample the shared prefiltered crop. No per-chunk bounds reduction, which also
        # drops a four-scalar device-to-host sync per chunk.
        local_coords = sample_coords - crop_origin
        with lock:
            sampled_flat = sp.ndimage.map_coordinates(
                sample_source,
                local_coords.transpose(),
                mode='constant',
                order=order,
                cval=oob_cval_float,
                prefilter=False).astype(original_dtype, copy=False)
        sampled = sampled_flat.reshape(chunk_n, cell_h, cell_w)
        # Match _TransformImageUsingCoords clipping; preserve OOB sentinels when they
        # lie outside the source intensity range (NaN or explicit fill).
        if preserve_oob_sentinel:
            if np.isnan(oob_cval_float):
                finite = xp.logical_not(xp.isnan(sampled))
            else:
                finite = sampled != sampled.dtype.type(oob_cval_float)
            clipped = xp.clip(sampled, a_min=min_val, a_max=max_val)
            sampled = xp.where(finite, clipped, sampled)
        else:
            xp.clip(sampled, a_min=min_val, a_max=max_val, out=sampled)
        sampled_stack[start:stop] = sampled
        del write, source_yx, sample_coords, sampled

    return sampled_stack


def _crop_target_rois_batched(target_image: NDArray,
                              botlefts: NDArray,
                              cell_h: int,
                              cell_w: int,
                              target_image_stats: nornir_imageregistration.ImageStats | None,
                              xp) -> NDArray:
    """Crop target-space cells into a ``(N, H, W)`` stack (still one CropImage per cell).

    Accepts NumPy or CuPy images; ``CropImage`` origins are host-converted once.
    """
    cval: float | int | str | None = False if target_image_stats is None else 'random'
    botlefts_host = nornir_imageregistration.EnsureNumpyArray(botlefts)
    num_cells = int(botlefts_host.shape[0])
    # Each crop is copied into the stack and released immediately; collecting all of
    # them first and then stacking kept the whole grid resident twice over.
    if num_cells == 0:
        return xp.empty((0, cell_h, cell_w), dtype=target_image.dtype)

    stack: NDArray | None = None
    for i in range(num_cells):
        yo = int(botlefts_host[i, 0])
        xo = int(botlefts_host[i, 1])
        crop = nornir_imageregistration.CropImage(
            target_image, xo, yo, cell_w, cell_h,
            cval=cval, image_stats=target_image_stats)
        if stack is None:
            stack = xp.empty((num_cells, cell_h, cell_w), dtype=crop.dtype)
        stack[i] = xp.asarray(crop)
    return stack


def _apply_noise_mask_batched(source_stack: NDArray,
                              nan_mask: NDArray,
                              source_image_stats: nornir_imageregistration.ImageStats,
                              xp) -> NDArray:
    """Replace NaN (OOB) pixels with stats-matched noise, matching ``RandomNoiseMask``."""
    invalid = nan_mask.ravel()
    num_invalid = int(nornir_imageregistration.EnsureNumpyArray(
        xp.asarray(xp.sum(invalid)).reshape((1,)))[0])
    if num_invalid == 0:
        return source_stack
    noise = source_image_stats.GenerateNoise(num_invalid, dtype=source_stack.dtype, xp=xp)
    if cp.get_array_module(noise) is not xp:
        noise = xp.asarray(noise) if xp is not np else nornir_imageregistration.EnsureNumpyArray(
            noise, dtype=source_stack.dtype)
    flat = source_stack.ravel().copy()
    flat[invalid] = noise
    return flat.reshape(source_stack.shape)


def _ensure_on_array_module(array: NDArray, xp) -> NDArray:
    """Return *array* on *xp* without copying when already resident there.

    Accepts NumPy or CuPy; ``np.asarray`` is never used on a CuPy input.
    """
    if cp.get_array_module(array) is xp:
        return array
    if xp is np:
        return nornir_imageregistration.EnsureNumpyArray(array)
    return xp.asarray(array)


def _stos_settings_images(
        settings: nornir_imageregistration.settings.GridRefinement
) -> tuple[NDArray, NDArray]:
    """Return target/source images on their stored array module.

    Does not upgrade host arrays to CuPy from ``GetComputationModule()``.
    """
    dtype = nornir_imageregistration.default_image_dtype()
    target = nornir_imageregistration.ImageParamToImageArray(
        settings.target_image, dtype=dtype)
    source = nornir_imageregistration.ImageParamToImageArray(
        settings.source_image, dtype=dtype)
    return target, source


def BuildAlignmentROIsBatched(
        rigid_transforms: Sequence[nornir_imageregistration.ITransform],
        target_image: NDArray,
        source_image: NDArray,
        target_image_stats: nornir_imageregistration.ImageStats | None,
        source_image_stats: nornir_imageregistration.ImageStats | None,
        target_points: NDArray,
        alignment_area: NDArray | tuple[float, float],
) -> tuple[NDArray, NDArray, NDArray | None] | None:
    """Batched rigid ROI extract for translation-only STOS grid refine.

    Accepts NumPy or CuPy; ops follow ``cp.get_array_module``.

    Replaces the per-cell ``BuildAlignmentROIs`` → ``SourceImageToTargetSpace`` loop with
    stacked inverse-affine maps and one (chunked) ``map_coordinates`` over the source image.
    Returns ``(fixed_stack, moving_stack, nan_mask_stack)`` on the image array module, or
    ``None`` when transforms are not rigid-compatible (caller should fall back).
    """
    if len(rigid_transforms) == 0:
        return None

    area = np.asarray(alignment_area, dtype=np.int64).ravel()[:2]
    cell_h = int(area[0])
    cell_w = int(area[1])
    if cell_h <= 0 or cell_w <= 0:
        return None

    # Follow the image array module so host STOS images stay on the CPU even when
    # the process-wide lib is CuPy (Pyre display). CuPy-in stays on device.
    xp = cp.get_array_module(target_image)
    source_image = _ensure_on_array_module(source_image, xp)
    target_points = _ensure_on_array_module(target_points, xp)
    sp = cupyx.scipy if xp is not np else scipy

    botlefts = _alignment_roi_botlefts(target_points, alignment_area)
    matrix_info = _rigid_inverse_matrices(rigid_transforms, xp)
    if matrix_info is None:
        return None
    inverse_matrices, pure_translation_mask = matrix_info

    fixed_stack = _crop_target_rois_batched(
        target_image, botlefts, cell_h, cell_w, target_image_stats, xp)
    # Match BuildAlignmentROIs: NaN OOB sentinel only when noise-fill stats are available;
    # otherwise fill OOB with 0 (cval=False → 0).
    oob_cval: float = np.nan if source_image_stats is not None else 0.0
    moving_stack = _sample_source_rois_batched(
        source_image, inverse_matrices, pure_translation_mask, botlefts, cell_h, cell_w, xp, sp,
        oob_cval=oob_cval)

    nan_mask_stack: NDArray | None = None
    if source_image_stats is not None:
        nan_mask_stack = xp.isnan(moving_stack)
        moving_stack = _apply_noise_mask_batched(
            moving_stack, nan_mask_stack, source_image_stats, xp)

    return fixed_stack, moving_stack, nan_mask_stack


def _attempt_align_points_translation_batched(
        keys: list[tuple[int, int]],
        source_points: np.ndarray,
        target_points: np.ndarray,
        rigid_transforms: Sequence[nornir_imageregistration.ITransform],
        settings: nornir_imageregistration.settings.GridRefinement
) -> list[nornir_imageregistration.EnhancedAlignmentRecord] | None:
    """Measure translation-only STOS cells with the shared batched FFT helper.

    Returns ``None`` only when the batched path could not run: ROI extraction
    failed for too many cells, cell shapes disagree, or too few cells survive to
    batch. The caller treats ``None`` as "batched unavailable" and re-measures the
    whole grid with the serial peak finder.

    Returns an empty list when the batched path *did* run and legitimately found
    nothing to align -- every cell rejected as unalignable, or every measured peak
    unusable. That is an answer, not a failure, so it must not be reported as
    ``None``: doing so sent the caller through a different peak finder and made the
    control points depend on whether the batched result happened to be empty.

    Under CuPy, ROIs stay on-device through ``xp.stack`` and the batched FFT; peaks
    and the source/target lattices sync to host once each (same pattern as mosaic
    ``_measure_grid_vertex_displacements_batched``).
    """
    with _PHASE_TIMER.section('cell_extract'):
        target_image, source_image = _stos_settings_images(settings)

        batched = BuildAlignmentROIsBatched(
            rigid_transforms=rigid_transforms,
            target_image=target_image,
            source_image=source_image,
            target_image_stats=settings.target_image_stats,
            source_image_stats=settings.source_image_stats,
            target_points=target_points,
            alignment_area=settings.cell_size)

        kept_indices: list[int]
        if batched is not None:
            fixed_stack, moving_stack, nan_mask_stack = batched
            kept_indices = list(range(len(keys)))
            xp = cp.get_array_module(fixed_stack)
            if fixed_stack.dtype != np.float64:
                fixed_stack = xp.asarray(fixed_stack, dtype=np.float64)
            if moving_stack.dtype != np.float64:
                moving_stack = xp.asarray(moving_stack, dtype=np.float64)
        else:
            # Non-rigid / unsupported transforms: legacy per-cell extract with deferred OOB.
            fixed_cells: list[NDArray] = []
            moving_cells: list[NDArray] = []
            nan_masks: list[NDArray | None] = []
            kept_indices = []
            for i, _key in enumerate(keys):
                try:
                    target_roi, source_roi, nan_mask = BuildAlignmentROIs(
                        transform=rigid_transforms[i],
                        targetImage_param=settings.target_image,
                        sourceImage_param=settings.source_image,
                        target_image_stats=settings.target_image_stats,
                        source_image_stats=settings.source_image_stats,
                        target_controlpoint=target_points[i, :],
                        alignmentArea=settings.cell_size,
                        description='',
                        defer_oob_check=True)
                except ValueError:
                    continue
                if target_roi is None or target_roi.size == 0 or source_roi is None or source_roi.size == 0:
                    continue
                xp_roi = cp.get_array_module(target_roi)
                fixed_cells.append(xp_roi.asarray(target_roi, dtype=np.float64))
                moving_cells.append(xp_roi.asarray(source_roi, dtype=np.float64))
                nan_masks.append(nan_mask)
                kept_indices.append(i)

            if len(kept_indices) < 3:
                return None

            shapes = {tuple(int(s) for s in cell.shape) for cell in fixed_cells + moving_cells}
            if len(shapes) != 1:
                return None

            xp = cp.get_array_module(fixed_cells[0])
            fixed_stack = xp.stack(fixed_cells, axis=0)
            moving_stack = xp.stack(moving_cells, axis=0)
            if any(mask is not None for mask in nan_masks):
                cell_shape_tmp = next(iter(shapes))
                nan_mask_stack = xp.stack([
                    mask if mask is not None else xp.zeros(cell_shape_tmp, dtype=bool)
                    for mask in nan_masks], axis=0)
            else:
                nan_mask_stack = None

        if len(kept_indices) < 3:
            return None

        xp = cp.get_array_module(fixed_stack)
        cell_shape = np.asarray(fixed_stack.shape[1:], dtype=np.int64)

        # Batched accept/reject: alignability + entirely-OOB, one host sync.
        flat_fixed = fixed_stack.reshape(fixed_stack.shape[0], -1)
        flat_moving = moving_stack.reshape(moving_stack.shape[0], -1)
        from nornir_imageregistration.refine_shared.cell_validity import low_content_std_min_threshold
        min_std = float(low_content_std_min_threshold())
        std_fixed = flat_fixed.std(axis=1)
        std_moving = flat_moving.std(axis=1)
        alignable = ((flat_fixed.min(axis=1) != flat_fixed.max(axis=1)) & (flat_fixed.max(axis=1) != 0)
                     & (flat_moving.min(axis=1) != flat_moving.max(axis=1)) & (flat_moving.max(axis=1) != 0)
                     & (std_fixed >= min_std) & (std_moving >= min_std))

        if nan_mask_stack is not None:
            fully_oob = nan_mask_stack.reshape(nan_mask_stack.shape[0], -1).all(axis=1)
            alignable = alignable & ~fully_oob

        keep_mask = nornir_imageregistration.EnsureNumpyArray(alignable).astype(bool).reshape(-1)

        if not np.any(keep_mask):
            # Decided, not unavailable: the serial path applies the same
            # alignability gate, so re-measuring the grid can only spend a full
            # serial pass to arrive back at nothing.
            return []

        if not np.all(keep_mask):
            keep_mask_dev = xp.asarray(keep_mask)
            fixed_stack = fixed_stack[keep_mask_dev]
            moving_stack = moving_stack[keep_mask_dev]
            kept_indices = [idx for idx, keep in zip(kept_indices, keep_mask) if keep]

        # Genuinely unavailable rather than empty: batching needs at least three
        # cells. The serial pass will reject the same cells this gate rejected, so
        # it arrives at the same records, only slower.
        if len(kept_indices) < 3:
            return None

    with _PHASE_TIMER.section('fft'):
        peaks_dev, weights_dev, peak_ratios_dev = measure_translation_cells_batched(
            fixed_stack,
            moving_stack,
            cell_shape,
            min_overlap=float(settings.min_alignment_overlap),
            max_overlap=1.0)

    with _PHASE_TIMER.section('host_sync'):
        peaks = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(peaks_dev), dtype=np.float64).reshape(-1, 2)
        weights = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(weights_dev), dtype=np.float64).reshape(-1)
        peak_ratios = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(peak_ratios_dev), dtype=np.float64).reshape(-1)

    with _PHASE_TIMER.section('record_assemble'):
        records: list[nornir_imageregistration.EnhancedAlignmentRecord] = []
        # AlignmentRecord is host-only; one lattice transfer instead of per-cell D2H.
        target_host = nornir_imageregistration.EnsureNumpyArray(target_points)
        source_host = nornir_imageregistration.EnsureNumpyArray(source_points)
        for batch_pos, i in enumerate(kept_indices):
            if weights[batch_pos] <= 0 or np.any(np.isnan(peaks[batch_pos])):
                continue
            records.append(nornir_imageregistration.EnhancedAlignmentRecord(
                ID=keys[i],
                TargetPoint=target_host[i, :],
                SourcePoint=source_host[i, :],
                peak=peaks[batch_pos],
                weight=float(weights[batch_pos]),
                angle=0.0,
                flipped_ud=False,
                peak_ratio=float(peak_ratios[batch_pos])))
    # An empty list is returned as-is. The measurement ran; every peak was simply
    # unusable. Reporting None here re-ran the whole grid through the serial peak
    # finder, which can disagree, so the control points depended on whether the
    # batched result happened to come back empty.
    return records


def _cupy_memory_pool_stats() -> tuple[int | None, int | None]:
    """Return ``(used_bytes, total_bytes)`` for the default CuPy pool, or ``(None, None)``."""
    if not nornir_imageregistration.UsingCupy():
        return None, None
    pool_factory = getattr(cp, 'get_default_memory_pool', None)
    if pool_factory is None:
        return None, None
    try:
        pool = pool_factory()
        return int(pool.used_bytes()), int(pool.total_bytes())
    except Exception:
        return None, None


def _log_refinement_gpu_memory(label: str) -> None:
    """Log CuPy pool usage when ``NORNIR_LOG_GPU_MEM=1``."""
    if os.environ.get('NORNIR_LOG_GPU_MEM', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return
    used_bytes, total_bytes = _cupy_memory_pool_stats()
    if used_bytes is None:
        return
    total_label = 'unknown' if total_bytes is None else str(total_bytes)
    prettyoutput.Log(
        f'GPU mem [{label}]: used_bytes={used_bytes} total_bytes={total_label}')


def _release_refinement_worker_memory() -> None:
    """Drop transient warp allocations after an overlap refinement worker task."""
    gc.collect()
    if nornir_imageregistration.UsingCupy():
        free_all_blocks = getattr(cp, 'get_default_memory_pool', None)
        if free_all_blocks is not None:
            try:
                free_all_blocks().free_all_blocks()
            except Exception:
                pass


def _prewarp_dispatch_mode() -> str:
    """Return the prewarp dispatch mode tokens from RefineRuntimeConfig."""
    return get_runtime_config(refresh=True).prewarp_mode


def _prewarp_thread_dispatch_enabled() -> bool:
    """Return True when per-tile prewarp should run on the shared thread pool."""
    return get_runtime_config(refresh=True).prewarp_thread_dispatch_enabled(
        nornir_imageregistration.UsingCupy())


def _tile_measure_parallel_enabled() -> bool:
    """Return True when per-tile vertex measurement should run on a thread pool."""
    return get_runtime_config(refresh=True).tile_measure_parallel


def _prewarp_single_warp_coverage_enabled() -> bool:
    """Return True for the opt-in single-warp prewarp coverage path."""
    return get_runtime_config(refresh=True).prewarp_single_warp_coverage_enabled()


def _refine_gpu_transform_enabled() -> bool:
    """Return True for the opt-in on-device inverse transform during prewarp."""
    cfg = get_runtime_config(refresh=True)
    if not cfg.gpu_transform:
        return False
    if not nornir_imageregistration.UsingCupy():
        return False
    from nornir_imageregistration.transforms.gridtransform import cuLinearNDInterpolator
    return cuLinearNDInterpolator is not None


def _prewarp_cache_enabled() -> bool:
    """Return False when cross-pass prewarp caching is disabled."""
    return get_runtime_config(refresh=True).prewarp_cache_enabled(
        nornir_imageregistration.UsingCupy())


def _compute_padded_overlap_geometry(
        scaled_overlapping_source_rect_a: nornir_imageregistration.Rectangle,
        scaled_overlapping_source_rect_b: nornir_imageregistration.Rectangle,
        overlapping_target_rect: nornir_imageregistration.Rectangle | None,
        subregion_shape: NDArray[np.integer],
        image_scale: float) -> PaddedOverlapGeometry:
    """Pad overlap source/target rectangles to the refinement FFT grid (translate-style footprint)."""
    subregion_shape = np.asarray(subregion_shape, dtype=np.int64)
    downsample = 1.0 / float(image_scale)
    grid_dim = np.asarray(
        nornir_imageregistration.TileGridShape(scaled_overlapping_source_rect_a.Size, subregion_shape),
        dtype=np.int64)
    padded_image_size = (int(grid_dim[0] * subregion_shape[0]), int(grid_dim[1] * subregion_shape[1]))

    padded_scaled_source_rect_a = nornir_imageregistration.Rectangle.change_area(
        scaled_overlapping_source_rect_a, padded_image_size)
    padded_scaled_source_rect_b = nornir_imageregistration.Rectangle.change_area(
        scaled_overlapping_source_rect_b, padded_image_size)
    padded_scaled_source_rect_b = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
        padded_scaled_source_rect_b.BottomLeft, padded_scaled_source_rect_a.Size)

    target_size = (int(grid_dim[0] * subregion_shape[0] * downsample),
                   int(grid_dim[1] * subregion_shape[1] * downsample))
    if overlapping_target_rect is None:
        raise ValueError("overlapping_target_rect is required for grid overlap refinement")
    target_region_rect = nornir_imageregistration.Rectangle.change_area(overlapping_target_rect, target_size)

    return PaddedOverlapGeometry(
        grid_dim=grid_dim,
        padded_scaled_source_rect_a=padded_scaled_source_rect_a,
        padded_scaled_source_rect_b=padded_scaled_source_rect_b,
        target_region_rect=target_region_rect)


def _scale_tile_transform_for_warp(tile: nornir_imageregistration.Tile,
                                   target_space_scale: float) -> tuple[
    nornir_imageregistration.ITransform, float]:
    """Apply the same source/target scaling rules as ``TransformTile``."""
    source_space_scale = 1.0 / tile.image_to_source_space_scale
    transform = tile.Transform
    if source_space_scale == target_space_scale:
        if source_space_scale != 1.0:
            scaled_transform = nornir_imageregistration.assemble_tiles.__CreateScalableTransformCopy(tile.Transform)
            scaled_transform.Scale(source_space_scale)
            transform = scaled_transform
    else:
        if source_space_scale != 1.0:
            scaled_transform = nornir_imageregistration.assemble_tiles.__CreateScalableTransformCopy(tile.Transform)
            scaled_transform.ScaleWarped(source_space_scale)  # type: ignore[attr-defined]
            transform = scaled_transform
        if target_space_scale != 1.0:
            scaled_transform = nornir_imageregistration.assemble_tiles.__CreateScalableTransformCopy(tile.Transform)
            scaled_transform.ScaleFixed(target_space_scale)  # type: ignore[attr-defined]
            transform = scaled_transform
    return transform, source_space_scale


def _warp_overlap_for_grid_refine(
        tile: nornir_imageregistration.Tile,
        padded_source_rect: nornir_imageregistration.Rectangle,
        target_region_rect: nornir_imageregistration.Rectangle,
        target_space_scale: float,
        single_threaded_invoke: bool) -> nornir_imageregistration.transformed_image_data.ITransformedImageData:
    """Crop a tile to its overlap source rect and warp only the padded target ROI."""
    try:
        full_source_image = nornir_imageregistration.ImageParamToImageArray(tile.Image)
    except IOError:
        return nornir_imageregistration.transformed_image_data.TransformedImageDataError(
            error_msg=f'Tile does not exist {tile.ImagePath}')
    except ValueError as ve:
        return nornir_imageregistration.transformed_image_data.TransformedImageDataError(error_msg=f'{ve}')

    full_source_image = nornir_imageregistration.ForceGrayscale(full_source_image)
    crop_x = int(padded_source_rect.BottomLeft[1])
    crop_y = int(padded_source_rect.BottomLeft[0])
    crop_w = int(padded_source_rect.Width)
    crop_h = int(padded_source_rect.Height)

    full_distance_image = nornir_imageregistration.assemble_tiles.distance_image_cache.KeepGetOrCreate(
        None, full_source_image.shape[0:2])
    cropped_source = nornir_imageregistration.CropImage(
        full_source_image, Xo=crop_x, Yo=crop_y, Width=crop_w, Height=crop_h, cval=0)
    cropped_distance = nornir_imageregistration.CropImage(
        full_distance_image, Xo=crop_x, Yo=crop_y, Width=crop_w, Height=crop_h, cval=0)
    del full_source_image
    del full_distance_image

    transform, source_space_scale = _scale_tile_transform_for_warp(tile, target_space_scale)

    if target_space_scale != 1.0:
        scaled_target_region = nornir_imageregistration.Rectangle.scale_on_origin(
            target_region_rect, target_space_scale)
        scaled_rounded_target_region = nornir_imageregistration.Rectangle.SnapRound(scaled_target_region)
    else:
        scaled_rounded_target_region = nornir_imageregistration.Rectangle.SnapRound(target_region_rect)

    target_width = int(scaled_rounded_target_region.Width)
    target_height = int(scaled_rounded_target_region.Height)
    target_min_x = int(scaled_rounded_target_region.MinX)
    target_min_y = int(scaled_rounded_target_region.MinY)

    distance_image = cropped_distance
    distance_cval = float(np.sum(distance_image.shape) * 32.0)

    read_coords, write_coords = nornir_imageregistration.assemble.write_to_target_roi_coords(
        transform,
        (target_min_y, target_min_x),
        (target_height, target_width),
        extrapolate=True)
    crop_origin = np.asarray(padded_source_rect.BottomLeft, dtype=np.float32)
    read_module = cp.get_array_module(read_coords)
    if read_module is not np:
        read_coords = read_coords - read_module.asarray(crop_origin, dtype=read_coords.dtype)
    else:
        read_coords = read_coords - crop_origin

    fixed_image = nornir_imageregistration.assemble._TransformImageUsingCoords(
        write_coords,
        read_coords,
        cropped_source,
        output_origin=(target_min_y, target_min_x),
        output_area=(target_height, target_width),
        cval=0)
    center_distance_image = nornir_imageregistration.assemble._TransformImageUsingCoords(
        write_coords,
        read_coords,
        distance_image,
        output_origin=(target_min_y, target_min_x),
        output_area=(target_height, target_width),
        cval=distance_cval)

    del cropped_source
    del distance_image
    del read_coords
    del write_coords

    logging.getLogger(__name__).info(
        "Grid refine warp tile %s crop %dx%d -> target %dx%d (scale=%g, backend=%s)",
        tile.ID,
        int(padded_source_rect.Height),
        int(padded_source_rect.Width),
        target_height,
        target_width,
        target_space_scale,
        "cupy" if nornir_imageregistration.UsingCupy() else "numpy")

    return nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile.Create(
        fixed_image,  # type: ignore[arg-type]
        center_distance_image,  # type: ignore[arg-type]
        transform,
        source_space_scale,
        target_space_scale,
        rendered_target_space_origin=(
            target_min_y * (1.0 / target_space_scale),
            target_min_x * (1.0 / target_space_scale)),
        SingleThreadedInvoke=single_threaded_invoke)


def _grid_transform_matches_lattice(
        transform: nornir_imageregistration.ITransform,
        resolved_mesh_shape: tuple[int, int],
        resolved_cell_size: tuple[int, int],
        source_shape: NDArray[np.integer]) -> bool:
    """Return True if transform is already a grid on the requested output lattice."""
    if not isinstance(transform, nornir_imageregistration.transforms.IGridTransform):
        return False

    grid = transform.grid
    grid_dims = (int(grid.grid_dims[0]), int(grid.grid_dims[1]))
    cell_size = (int(grid.cell_size[0]), int(grid.cell_size[1]))
    if grid_dims != resolved_mesh_shape or cell_size != resolved_cell_size:
        return False

    existing_source_shape = np.asarray(grid.source_shape, dtype=np.int64)
    return bool(np.array_equal(existing_source_shape, np.asarray(source_shape, dtype=np.int64)))


def _initialize_tile_grid_transforms(
        list_tiles: Sequence[nornir_imageregistration.Tile],
        resolved_cell_size: tuple[int, int],
        resolved_mesh_shape: tuple[int, int]) -> None:
    """Resample each tile transform onto the legacy-compatible output grid lattice once."""
    cell_size_array = np.asarray(resolved_cell_size, dtype=np.int64)
    grid_dims_array = np.asarray(resolved_mesh_shape, dtype=np.int64)

    for tile in list_tiles:
        source_shape = _tile_source_shape_for_grid(tile)
        if _grid_transform_matches_lattice(
                tile.Transform, resolved_mesh_shape, resolved_cell_size, source_shape):
            continue

        tile.Transform = nornir_imageregistration.transforms.converters.ConvertTransformToGridTransform(
            tile.Transform,
            source_image_shape=source_shape,
            cell_size=cell_size_array,
            grid_dims=grid_dims_array,
            prefer_gpu=_refine_gpu_transform_enabled())


def _resample_transform_to_output_grid(
        transform: nornir_imageregistration.ITransform,
        source_shape: NDArray[np.integer],
        resolved_cell_size: tuple[int, int],
        resolved_mesh_shape: tuple[int, int],
        *,
        prefer_gpu: bool = False) -> nornir_imageregistration.ITransform:
    """Resample a transform onto the legacy-compatible output grid lattice.

    Defaults to a host-backed grid so mosaic/STOS save stays NumPy. Pass
    ``prefer_gpu=True`` when the resampled transform will be used for further
    on-device warps in this process.
    """
    return nornir_imageregistration.transforms.converters.ConvertTransformToGridTransform(
        transform,
        source_image_shape=source_shape,
        cell_size=np.asarray(resolved_cell_size, dtype=np.int64),
        grid_dims=np.asarray(resolved_mesh_shape, dtype=np.int64),
        prefer_gpu=prefer_gpu)


def _update_tile_transform_from_merged_overlap_pairs(
        tile: nornir_imageregistration.Tile,
        merged_point_pairs: np.ndarray,
        resolved_cell_size: tuple[int, int],
        resolved_mesh_shape: tuple[int, int]) -> int:
    """
    Fit a smooth mesh from overlap control pairs and resample onto the output grid.

    Used by unit tests and optional mesh-rebuild paths only; the mosaic refine loop
    uses the legacy per-mesh-vertex update model (see ``_refine_tileset``).
    """
    if merged_point_pairs.size == 0:
        return 0

    mesh_transform = _build_mesh_transform_from_pairs(tile.Transform, merged_point_pairs)
    source_shape = _tile_source_shape_for_grid(tile)
    tile.Transform = _resample_transform_to_output_grid(
        mesh_transform,
        source_shape,
        resolved_cell_size,
        resolved_mesh_shape)
    return int(merged_point_pairs.shape[0])


def _apply_point_pair_updates_to_grid_transform(
        grid_transform: nornir_imageregistration.transforms.IGridTransform,
        point_pairs: np.ndarray,
        max_assign_distance: float | None = None) -> int:
    """
    Nudge grid target control points for measured source locations.

    Each row of ``point_pairs`` is ``[target_y, target_x, source_y, source_x]``.
    Measured cells receive the weighted-average displacement from the current
    transform; unmeasured cells are unchanged. Retained for unit tests; the mosaic
    refine loop now ports the legacy full-mesh vertex update (``_refine_tileset``).
    """
    if point_pairs.size == 0:
        return 0

    pairs = np.asarray(point_pairs, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError("point_pairs must have shape (N, 4)")

    source_points = pairs[:, 2:4]
    target_points = pairs[:, 0:2]
    predicted_targets = nornir_imageregistration.EnsureNumpyArray(
        grid_transform.Transform(source_points), dtype=np.float64)
    target_delta = target_points - predicted_targets
    delta_norm = np.linalg.norm(target_delta, axis=1)
    source_span = np.asarray(grid_transform.grid.source_shape, dtype=np.float64)
    max_target_jump = float(np.max(source_span) * 2.0)
    valid_measurement = delta_norm <= max_target_jump
    if not np.any(valid_measurement):
        return 0

    source_points = source_points[valid_measurement]
    target_delta = target_delta[valid_measurement]

    nearest_result = grid_transform.NearestSourcePoint(source_points)
    if isinstance(nearest_result[1], (int, np.integer)):
        nearest_distances = np.asarray([float(nearest_result[0])], dtype=np.float64)
        nearest_indices = np.asarray([int(nearest_result[1])], dtype=np.int64)
    else:
        nearest_distances = np.atleast_1d(
            nornir_imageregistration.EnsureNumpyArray(nearest_result[0])).astype(np.float64).reshape(-1)
        nearest_indices = np.atleast_1d(
            nornir_imageregistration.EnsureNumpyArray(nearest_result[1])).astype(np.int64).reshape(-1)

    grid_spacing = np.asarray(grid_transform.grid.grid_spacing, dtype=np.float64)
    cell_size = np.asarray(grid_transform.grid.cell_size, dtype=np.float64)
    if max_assign_distance is None:
        max_assign_distance = float(min(np.min(grid_spacing) * 0.5, np.min(cell_size) * 0.5))
    else:
        max_assign_distance = float(max_assign_distance)
    assigned_to_grid_node = nearest_distances <= max_assign_distance
    if not np.any(assigned_to_grid_node):
        return 0

    source_points = source_points[assigned_to_grid_node]
    target_delta = target_delta[assigned_to_grid_node]
    nearest_indices = nearest_indices[assigned_to_grid_node]

    index_to_deltas: dict[int, list[NDArray[np.floating]]] = {}
    for i, grid_index in enumerate(nearest_indices):
        index_to_deltas.setdefault(int(grid_index), []).append(target_delta[i])

    updated_cells = 0
    current_targets = nornir_imageregistration.EnsureNumpyArray(
        grid_transform.TargetPoints, dtype=np.float64)
    for grid_index, deltas in index_to_deltas.items():
        averaged_delta = np.mean(np.asarray(deltas, dtype=np.float64), axis=0)
        grid_transform.UpdateTargetPointsByIndex(
            grid_index,
            current_targets[grid_index] + averaged_delta)
        updated_cells += 1

    return updated_cells


def _build_mesh_transform_from_pairs(existing_transform: nornir_imageregistration.ITransform,
                                     point_pairs: np.ndarray) -> nornir_imageregistration.ITransform:
    """Build a mesh transform from control pairs, falling back on failure."""
    if point_pairs.shape[0] < 3:
        return existing_transform

    # Deduplicate exact source points and exact target points to avoid triangulation failures.
    _, unique_source_idx = np.unique(np.round(point_pairs[:, 2:4], decimals=3), axis=0, return_index=True)
    point_pairs = point_pairs[np.sort(unique_source_idx)]
    _, unique_target_idx = np.unique(np.round(point_pairs[:, 0:2], decimals=3), axis=0, return_index=True)
    point_pairs = point_pairs[np.sort(unique_target_idx)]

    if point_pairs.shape[0] < 3:
        return existing_transform

    try:
        return nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(point_pairs)
    except Exception as e:
        prettyoutput.LogErr(f"Unable to build mesh transform from {point_pairs.shape[0]} points: {e}")
        return existing_transform


def _create_tileset_for_refinement(
        mosaic_or_transforms: str | nornir_imageregistration.mosaic.Mosaic | nornir_imageregistration.mosaic_tileset.MosaicTileset | Sequence[
            nornir_imageregistration.ITransform],
        image_source: str | Sequence[str] | None,
        image_to_source_space_scale: float = 1.0) -> nornir_imageregistration.mosaic_tileset.MosaicTileset:
    """Create a mutable tileset view over mosaic/transforms for refinement."""
    if isinstance(mosaic_or_transforms, nornir_imageregistration.mosaic_tileset.MosaicTileset):
        return copy.deepcopy(mosaic_or_transforms)

    if isinstance(mosaic_or_transforms, (str, nornir_imageregistration.mosaic.Mosaic)):
        if image_source is None:
            if isinstance(mosaic_or_transforms, str):
                image_source = os.path.dirname(mosaic_or_transforms)
            else:
                raise ValueError("image_source must be provided when refining a Mosaic object")

        if not isinstance(image_source, str):
            raise ValueError("image_source must be a directory path for Mosaic input")

        return nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
            mosaic_or_transforms,  # type: ignore[arg-type]
            image_folder=image_source,
            image_to_source_space_scale=image_to_source_space_scale)

    if image_source is None:
        raise ValueError("image_source must be a sequence of image paths when passing transforms")

    return cast(
        nornir_imageregistration.mosaic_tileset.MosaicTileset,
        nornir_imageregistration.mosaic_tileset.Create(
            mosaic_or_transforms,
            image_source,
            image_to_source_space_scale=image_to_source_space_scale)
    )


def _measure_tile_grid_update(tile: nornir_imageregistration.Tile,
                              tile_neighbors: list,
                              prewarped: dict,
                              target_space_scale: float,
                              cell_shape,
                              min_overlap: float,
                              median_radius,
                              measure_vertex_displacements):
    """Measure one tile's regularized vertex displacement for the current pass.

    Pure with respect to shared state: reads ``prewarped`` (read-only) and the
    tile's own transform, returns the computed update. Safe to run on a thread
    pool because no shared structure is mutated (the caller assembles results in
    tile order and applies them after the loop).

    Returns ``(targets, applied_scaled, measured_count, diagnostics)``.
    """
    grid_transform = tile.Transform
    if not isinstance(grid_transform, nornir_imageregistration.transforms.IGridTransform):
        raise ValueError(f"Tile {tile.ID} transform is not a grid transform after initialization")

    targets = np.asarray(
        nornir_imageregistration.EnsureNumpyArray(
            grid_transform.TargetPoints),  # type: ignore[attr-defined]
        dtype=np.float64)
    centers_scaled = targets * float(target_space_scale)
    mesh_dims = (int(grid_transform.grid.grid_dims[0]), int(grid_transform.grid.grid_dims[1]))
    num_vertices = targets.shape[0]

    total_shift = np.zeros((num_vertices, 2), dtype=np.float64)
    mass = np.zeros(num_vertices, dtype=np.float64)
    measured_count = 0
    filled_count = 0

    for neighbor in tile_neighbors:
        shifts, measured = measure_vertex_displacements(
            moving=prewarped[tile.ID],
            fixed=prewarped[neighbor.ID],
            centers_scaled=centers_scaled,
            cell_shape=cell_shape,
            cell_min_overlap=min_overlap)
        measured_count += int(np.count_nonzero(measured))
        with _PHASE_TIMER.section('regularize'):
            regularized_shifts, regularized_db = _regularize_displacements(
                shifts, measured, mesh_dims, median_radius=median_radius)
        filled_count += int(np.count_nonzero(regularized_db)) - int(np.count_nonzero(measured))
        total_shift += regularized_shifts
        mass += regularized_db

    # Legacy blend: scale = 1 / (1 + mass) (all tiles moving).
    applied_scaled = total_shift * (1.0 / (1.0 + mass))[:, None]
    diagnostics = {
        'measured': measured_count,
        'gap_filled': filled_count,
        'updated': int(np.count_nonzero(np.any(applied_scaled != 0, axis=1))),
    }
    return targets, applied_scaled, measured_count, diagnostics


def _refine_tileset(tiles: nornir_imageregistration.mosaic_tileset.MosaicTileset,
                    target_space_scale: float,
                    iterations: int,
                    cell_size: NDArray[np.integer] | Sequence[int] | int | None,
                    mesh_shape: NDArray[np.integer] | Sequence[int] | int | None,
                    displacement_threshold: float,
                    min_overlap: float,
                    merge_distance: float | None = None,
                    median_radius: int = 1) -> MosaicRefinementDiagnostics:
    """
    Run iterative per-mesh-vertex refinement over all tiles (legacy refine_mosaic_mt port).

    Each pass: prewarp every tile into scaled mosaic space, measure a phase-correlation
    shift at every mesh vertex against each overlapping neighbor, regularize the
    per-neighbor displacement fields (median + gap-fill + Gaussian blur), blend with
    1/(1+mass) normalization, then add the blended shifts to every tile's grid target
    points.  Stops early when the max vertex displacement reaches the threshold or stops
    improving (dual condition; legacy C++ uses mean for the threshold comparison).
    """
    del merge_distance  # retained for API compatibility; unused by the legacy-parity model
    list_tiles = list(tiles.values())
    if len(list_tiles) == 0:
        raise ValueError("No tiles available for refinement")

    resolved_cell_size, resolved_mesh_shape = _resolve_mesh_shape_and_cell_size(
        np.asarray(list_tiles[0].ImageSize, dtype=np.int64),
        cell_size=cell_size,
        mesh_shape=mesh_shape)
    cell_shape = np.asarray(resolved_cell_size, dtype=np.int64)

    average_displacement_per_iteration: list[float] = []
    overlap_count_per_iteration: list[int] = []
    control_points_per_tile = {tile.ID: 0 for tile in list_tiles}
    vertex_diagnostics_per_pass: list[dict[int, dict[str, int]]] = []
    converged = False

    _initialize_tile_grid_transforms(list_tiles, resolved_cell_size, resolved_mesh_shape)

    downsample = 1.0 / float(target_space_scale)
    last_pass_displacement = float('inf')
    prewarp_cache: dict[int, _PrewarpedTile] = {}
    prewarp_revision_cache: dict[int, int] = {tile.ID: 0 for tile in list_tiles}

    _log_refinement_gpu_memory('_refine_tileset start')
    _PHASE_TIMER.reset()
    for pass_index in range(iterations):
        pass_phase_baseline = _PHASE_TIMER.snapshot()
        # Legacy prewarp_tiles=true: re-render tiles whose grid moved since the last pass.
        _log_refinement_gpu_memory(f'_refine_tileset pass {pass_index + 1} before prewarp')
        with _PHASE_TIMER.section('prewarp'):
            prewarped = _prewarp_all_tiles_for_grid_refine(
                list_tiles,
                target_space_scale,
                prewarp_cache=prewarp_cache,
                revision_cache=prewarp_revision_cache)
        _log_refinement_gpu_memory(f'_refine_tileset pass {pass_index + 1} after prewarp')
        neighbors = _grid_refine_neighbors(list_tiles)
        overlap_count_per_iteration.append(
            sum(len(neighbor_list) for neighbor_list in neighbors.values()))

        pass_vertex_diagnostics: dict[int, dict[str, int]] = {}
        pending_updates: list[tuple[nornir_imageregistration.Tile, NDArray[np.floating], NDArray[np.floating]]] = []
        all_applied_components: list[NDArray[np.floating]] = []

        measure_vertex_displacements = (
            _measure_grid_vertex_displacements_batched
            if _use_batched_vertex_measurement()
            else _measure_grid_vertex_displacements)

        def _measure(tile):
            return _measure_tile_grid_update(
                tile, neighbors[tile.ID], prewarped, target_space_scale,
                cell_shape, min_overlap, median_radius, measure_vertex_displacements)

        # Each tile's measurement is independent; optionally dispatch across a
        # thread pool. Results are always assembled in list_tiles order so the
        # reductions below are bit-for-bit identical to the serial path.
        if _tile_measure_parallel_enabled() and len(list_tiles) > 1:
            pool = nornir_pools.GetGlobalThreadPool()
            tasks = [pool.add_task(f"grid_measure_{tile.ID}", _measure, tile)
                     for tile in list_tiles]
            pool.wait_completion()
            tile_results = [task.wait_return() for task in tasks]
        else:
            tile_results = [_measure(tile) for tile in list_tiles]

        for tile, (targets, applied_scaled, measured_count, diagnostics) in zip(list_tiles, tile_results):
            pending_updates.append((tile, targets, applied_scaled))
            all_applied_components.append(np.abs(applied_scaled).reshape(-1))
            control_points_per_tile[tile.ID] = measured_count
            pass_vertex_diagnostics[tile.ID] = diagnostics

        # Apply only after all tiles are measured (legacy updates grids between passes).
        with _PHASE_TIMER.section('apply'):
            for tile, targets, applied_scaled in pending_updates:
                grid_transform = cast(nornir_imageregistration.transforms.IGridTransform, tile.Transform)
                new_targets = targets + (applied_scaled * downsample)
                grid_transform.UpdateTargetPointsByIndex(  # type: ignore[attr-defined]
                    np.arange(targets.shape[0], dtype=np.int64), new_targets)
                if np.any(applied_scaled != 0):
                    prewarp_revision_cache[tile.ID] = prewarp_revision_cache.get(tile.ID, 0) + 1

        del prewarped
        _release_refinement_worker_memory()
        _log_refinement_gpu_memory(f'_refine_tileset pass {pass_index + 1} after release')
        _log_phase_breakdown(f'_refine_tileset pass {pass_index + 1}', pass_phase_baseline)

        vertex_diagnostics_per_pass.append(pass_vertex_diagnostics)

        # Convergence metric: max |sy| or |sx| over all vertices of all tiles, in
        # working-resolution (scaled) pixels. Legacy C++ uses mean; max is stricter and
        # keeps refining while any vertex still moves above the threshold.
        components = np.concatenate(all_applied_components) if all_applied_components else np.zeros(0)
        pass_displacement = float(np.max(components)) if components.size > 0 else 0.0
        average_displacement_per_iteration.append(pass_displacement)

        if components.size > 0:
            if pass_displacement <= displacement_threshold:
                converged = True
                break
            if pass_displacement >= last_pass_displacement:
                # Dual stop: a pass that fails to improve ends refinement.
                break
            last_pass_displacement = pass_displacement

    prewarp_cache.clear()
    prewarp_revision_cache.clear()
    _release_refinement_worker_memory()
    _log_refinement_gpu_memory('_refine_tileset end')
    _log_phase_breakdown('_refine_tileset total', {})

    return MosaicRefinementDiagnostics(
        iterations_completed=len(average_displacement_per_iteration),
        converged=converged,
        average_displacement_per_iteration=average_displacement_per_iteration,
        overlap_count_per_iteration=overlap_count_per_iteration,
        control_points_per_tile=control_points_per_tile,
        resolved_cell_size=resolved_cell_size,
        resolved_mesh_shape=resolved_mesh_shape,
        vertex_diagnostics_per_pass=vertex_diagnostics_per_pass)


def RefineGridMosaic(
                     mosaic_or_transforms: str | nornir_imageregistration.mosaic.Mosaic | nornir_imageregistration.mosaic_tileset.MosaicTileset | Sequence[
                         nornir_imageregistration.ITransform],
                     image_source: str | Sequence[str] | None,
                     iterations: int = 10,
                     cell_size: NDArray[np.integer] | Sequence[int] | int | None = None,
                     mesh_shape: NDArray[np.integer] | Sequence[int] | int | None = None,
                     displacement_threshold: float = 1.0,
                     min_overlap: float = 0.25,
                     imageScale: float | None = None,
                     merge_distance: float | None = None,
                     return_diagnostics: bool = False) -> nornir_imageregistration.mosaic.Mosaic | tuple[
    nornir_imageregistration.mosaic.Mosaic, MosaicRefinementDiagnostics]:
    """
    Refine a mosaic's tile transforms using per-mesh-vertex phase correlation and output
    legacy-compatible grid transforms.

    This is a Python port of legacy `ir-refine-grid`: each pass prewarps every tile into
    mosaic space, measures a local offset at every grid-transform vertex against each
    overlapping neighbor, regularizes the per-neighbor displacement fields (median filter,
    gap fill, Gaussian blur), blends them with 1/(1+mass) normalization, and adds the
    blended shifts directly to the grid target points. No end-of-run resample occurs; the
    refined grid is the output transform.

    Parameters
    ----------
    mosaic_or_transforms:
        One of:
        - `.mosaic` path
        - `Mosaic` object
        - `MosaicTileset`
        - sequence of transforms (requires `image_source` as image-path sequence)
    image_source:
        - directory containing tile images (for mosaic input), or
        - sequence of image paths (for transform-sequence input)
    iterations:
        Max number of refinement passes.
    cell_size:
        Refinement neighborhood size as scalar or `(height, width)`.
    mesh_shape:
        Optional `(rows, cols)` mesh density. If omitted, derived from `cell_size` using legacy equations.
    displacement_threshold:
        Early-stop threshold on the max |shift component| per pass, in working-resolution
        pixels (legacy `-displacement_threshold` applies to mean; this port uses max).
        Refinement also stops when a pass fails to improve on the prior pass.
    min_overlap:
        Minimum valid-pixel fraction required of each cell-sized vertex neighborhood
        (legacy hardcodes 0.25).
    imageScale:
        Target-space scaling used while warping tiles; defaults to `1.0`. Use `1/sp`
        to match a legacy `-sp` pixel spacing.
    merge_distance:
        Deprecated; unused by the legacy-parity vertex-update model.
    return_diagnostics:
        If True, return `(Mosaic, MosaicRefinementDiagnostics)`; otherwise return `Mosaic`.
    """

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if imageScale is None:
        imageScale = 1.0
    if imageScale <= 0:
        raise ValueError("imageScale must be > 0")

    image_to_source_space_scale = 1.0 / float(imageScale)
    tiles = _create_tileset_for_refinement(
        mosaic_or_transforms,
        image_source,
        image_to_source_space_scale=image_to_source_space_scale)
    diagnostics = _refine_tileset(tiles=tiles,
                                  target_space_scale=imageScale,
                                  iterations=iterations,
                                  cell_size=cell_size,
                                  mesh_shape=mesh_shape,
                                  displacement_threshold=displacement_threshold,
                                  min_overlap=min_overlap,
                                  merge_distance=merge_distance)
    _release_refinement_worker_memory()
    _log_refinement_gpu_memory('RefineGridMosaic after _refine_tileset')

    # Legacy ir-refine-grid never resamples at the end: the refined grid IS the output
    # transform. Resample only if a tile somehow is not on the output lattice, or to
    # downconvert an on-device GPU grid transform (NORNIR_REFINE_GPU_TRANSFORM) back to
    # the host-backed CPU grid so the saved mosaic stays NumPy (CPU Grid constructor
    # packs TargetPoints onto the host).
    for tile in tiles.values():
        source_shape = _tile_source_shape_for_grid(tile)
        is_gpu_grid = isinstance(
            tile.Transform,
            nornir_imageregistration.transforms.GridWithRBFFallback_GPUComponent)
        if not is_gpu_grid and _grid_transform_matches_lattice(
                tile.Transform,
                diagnostics.resolved_mesh_shape,
                diagnostics.resolved_cell_size,
                source_shape):
            continue
        tile.Transform = _resample_transform_to_output_grid(
            tile.Transform,
            source_shape,
            diagnostics.resolved_cell_size,
            diagnostics.resolved_mesh_shape)

    output_mosaic = tiles.ToMosaic()
    if return_diagnostics:
        return output_mosaic, diagnostics

    return output_mosaic


def RefineMosaic(transforms: Sequence[nornir_imageregistration.ITransform],
                imagepaths: Sequence[str],
                imageScale: float | None = None,
                subregion_shape: NDArray[np.integer] | Sequence[int] | int | None = None) -> tuple[
    nornir_imageregistration.layout.Layout, nornir_imageregistration.mosaic_tileset.MosaicTileset]:
    """
    Locate overlapping regions between tiles in a mosaic and align multiple small subregions within.  This generates a set of control points.

    More than one tile may overlap, for example corners.  To solve this the set of control points is merged into a KD tree.  points closer than a set distance (less than subregion size) are averaged to create a single offset.

    Using the remaining points a mesh transform is generated for the tile.
    """

    if imageScale is None:
        imageScale = 1.0
    if imageScale <= 0:
        raise ValueError("imageScale must be > 0")

    tiles = cast(
        nornir_imageregistration.mosaic_tileset.MosaicTileset,
        nornir_imageregistration.mosaic_tileset.Create(
            transforms, imagepaths, image_to_source_space_scale=(1.0 / float(imageScale)))  # type: ignore[arg-type]
    )
    _refine_tileset(tiles=tiles,
                    target_space_scale=imageScale,
                    iterations=1,
                    cell_size=subregion_shape,
                    mesh_shape=None,
                    displacement_threshold=0.0,
                    min_overlap=0.03,
                    merge_distance=None)

    layout = nornir_imageregistration.layout.Layout()
    for t in list(tiles.values()):
        layout.CreateNode(t.ID, t.FixedBoundingBox.Center)
    return layout, tiles


def _refine_single_tile_overlap_pair(
        A: nornir_imageregistration.Tile,
        B: nornir_imageregistration.Tile,
        scaled_overlapping_source_rect_A: nornir_imageregistration.Rectangle,
        scaled_overlapping_source_rect_B: nornir_imageregistration.Rectangle,
        overlapping_target_rect: nornir_imageregistration.Rectangle,
        image_scale: float,
        subregion_shape: NDArray[np.integer]) -> tuple[np.ndarray, np.ndarray]:
    """Measure local phase-correlation offsets for one overlapping tile pair."""
    downsample = 1.0 / image_scale
    subregion_shape = np.asarray(subregion_shape, dtype=np.int64)

    geometry = _compute_padded_overlap_geometry(
        scaled_overlapping_source_rect_A,
        scaled_overlapping_source_rect_B,
        overlapping_target_rect,
        subregion_shape,
        image_scale)
    grid_dim = geometry.grid_dim
    padded_source_rect_a = geometry.padded_scaled_source_rect_a
    padded_source_rect_b = geometry.padded_scaled_source_rect_b
    overlapping_target_region = geometry.target_region_rect
    max_displacement = _max_refinement_displacement(
        overlapping_target_region, subregion_shape, image_scale)

    a_transformed = _warp_overlap_for_grid_refine(
        A,
        padded_source_rect_a,
        overlapping_target_region,
        image_scale,
        single_threaded_invoke=False)
    b_transformed = _warp_overlap_for_grid_refine(
        B,
        padded_source_rect_b,
        overlapping_target_region,
        image_scale,
        single_threaded_invoke=False)

    if isinstance(a_transformed, nornir_imageregistration.transformed_image_data.TransformedImageDataError):
        raise ValueError(str(a_transformed.error_msg))
    if isinstance(b_transformed, nornir_imageregistration.transformed_image_data.TransformedImageDataError):
        raise ValueError(str(b_transformed.error_msg))

    distance_max = np.finfo(a_transformed.centerDistanceImage.dtype).max  # type: ignore[union-attr]
    valid_mask_a = a_transformed.centerDistanceImage < distance_max  # type: ignore[operator]
    valid_mask_b = b_transformed.centerDistanceImage < distance_max  # type: ignore[operator]

    a_image = nornir_imageregistration.RandomNoiseMask(
        a_transformed.image,  # type: ignore[arg-type]
        valid_mask_a,
        Copy=False)
    b_image = nornir_imageregistration.RandomNoiseMask(
        b_transformed.image,  # type: ignore[arg-type]
        valid_mask_b,
        Copy=False)

    a_tiles = nornir_imageregistration.ImageToTiles(a_image, subregion_shape, cval='random')  # type: ignore[arg-type]
    b_tiles = nornir_imageregistration.ImageToTiles(b_image, subregion_shape, cval='random')  # type: ignore[arg-type]

    refine_dtype = np.dtype([('SourceAY', 'f4'),
                             ('SourceAX', 'f4'),
                             ('SourceBY', 'f4'),
                             ('SourceBX', 'f4'),
                             ('BaseTargetY', 'f4'),
                             ('BaseTargetX', 'f4'),
                             ('TargetY', 'f4'),
                             ('TargetX', 'f4'),
                             ('DisplacementY', 'f4'),
                             ('DisplacementX', 'f4'),
                             ('Weight', 'f4'),
                             ('Angle', 'f4')])

    point_pairs = np.empty(grid_dim, dtype=refine_dtype)
    net_displacement = np.empty((int(grid_dim.prod()), 3), dtype=np.float32)
    cell_center_offset = subregion_shape / 2.0

    try:
        for i_row in range(0, int(grid_dim[0])):
            for i_col in range(0, int(grid_dim[1])):
                subregion_offset = (np.array([i_row, i_col]) * subregion_shape) + cell_center_offset
                _, full_source_a, full_source_b, global_offset = _refinement_cell_geometry(
                    subregion_offset,
                    overlapping_target_region,
                    image_scale,
                    A,
                    B)

                if (i_row, i_col) not in a_tiles or (i_row, i_col) not in b_tiles:
                    net_displacement[(i_row * grid_dim[1]) + i_col, :] = np.array([0, 0, 0])
                    point_pairs[i_row, i_col] = np.array(
                        (full_source_a[0], full_source_a[1],
                         full_source_b[0], full_source_b[1],
                         global_offset[0], global_offset[1],
                         global_offset[0], global_offset[1],
                         0, 0,
                         0, 0),
                        dtype=refine_dtype)
                    continue

                try:
                    record = _phase_correlate_refinement_cell(
                        a_tiles[i_row, i_col],
                        b_tiles[i_row, i_col],
                        subregion_shape)
                except Exception as e:
                    prettyoutput.LogErr(f'Exception on row: {i_row} col: {i_col} when finding offset:\n{e}')
                    net_displacement[(i_row * grid_dim[1]) + i_col, :] = np.array([0, 0, 0])
                    point_pairs[i_row, i_col] = np.array(
                        (full_source_a[0], full_source_a[1],
                         full_source_b[0], full_source_b[1],
                         global_offset[0], global_offset[1],
                         global_offset[0], global_offset[1],
                         0, 0,
                         0, 0),
                        dtype=refine_dtype)
                    continue

                adjusted_record = nornir_imageregistration.AlignmentRecord(
                    np.array(record.peak) * downsample,
                    record.weight)
                displacement_norm = float(np.linalg.norm(adjusted_record.peak))
                if displacement_norm > max_displacement:
                    prettyoutput.LogErr(
                        f'Ignoring refinement displacement {displacement_norm:.1f} > {max_displacement:.1f} '
                        f'at row {i_row} col {i_col}')
                    net_displacement[(i_row * grid_dim[1]) + i_col, :] = np.array([0, 0, 0])
                    point_pairs[i_row, i_col] = np.array(
                        (full_source_a[0], full_source_a[1],
                         full_source_b[0], full_source_b[1],
                         global_offset[0], global_offset[1],
                         global_offset[0], global_offset[1],
                         0, 0,
                         0, 0),
                        dtype=refine_dtype)
                    continue

                if np.any(np.isnan(record.peak)):
                    net_displacement[(i_row * grid_dim[1]) + i_col, :] = np.array([0, 0, 0])
                    point_pairs[i_row, i_col] = np.array(
                        (full_source_a[0], full_source_a[1],
                         full_source_b[0], full_source_b[1],
                         global_offset[0], global_offset[1],
                         global_offset[0], global_offset[1],
                         0, 0,
                         0, record.angle),
                        dtype=refine_dtype)
                else:
                    net_displacement[(i_row * grid_dim[1]) + i_col, :] = np.array(
                        [adjusted_record.peak[0], adjusted_record.peak[1], record.weight])
                    point_pairs[i_row, i_col] = np.array(
                        (full_source_a[0], full_source_a[1],
                         full_source_b[0], full_source_b[1],
                         global_offset[0], global_offset[1],
                         adjusted_record.peak[0] + global_offset[0],
                         adjusted_record.peak[1] + global_offset[1],
                         adjusted_record.peak[0],
                         adjusted_record.peak[1],
                         record.weight,
                         record.angle),
                        dtype=refine_dtype)
    finally:
        del a_transformed
        del b_transformed
        del a_tiles
        del b_tiles
        A._image = None
        A._paddedimage = None
        B._image = None
        B._paddedimage = None

    weighted_net_offset = np.copy(net_displacement)
    weight_sum = np.sum(net_displacement[:, 2])
    if weight_sum > 0:
        weighted_net_offset[:, 2] /= weight_sum
    weighted_net_offset[:, 0] *= weighted_net_offset[:, 2]
    weighted_net_offset[:, 1] *= weighted_net_offset[:, 2]

    net_offset = np.sum(weighted_net_offset[:, 0:2], axis=0)
    meaningful_weights = weighted_net_offset[weighted_net_offset[:, 2] > 0, 2]
    weight = 0.0
    if meaningful_weights.shape[0] > 0:
        weight = float(np.median(meaningful_weights))

    net_offset = np.hstack((np.around(net_offset, 3), weight))
    return point_pairs, net_offset


def __RefineTileOverlapBatchRemote(
        anchor_tile: nornir_imageregistration.Tile,
        overlap_batch: Sequence[nornir_imageregistration.tile_overlap.TileOverlap],
        image_scale: float,
        subregion_shape: NDArray[np.integer] | Sequence[int] | int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Refine all overlaps for which ``anchor_tile`` is the lower-ID (A) tile."""
    if subregion_shape is None:
        subregion_shape = np.array([128, 128], dtype=np.int64)
    else:
        subregion_shape = np.asarray(_normalize_pair(subregion_shape, "subregion_shape", minimum=4), dtype=np.int64)

    results: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        for tile_overlap in overlap_batch:
            if tile_overlap.A.ID != anchor_tile.ID:
                raise ValueError(
                    f"Overlap batch anchor {anchor_tile.ID} does not match pair A {tile_overlap.A.ID}")
            results.append(_refine_single_tile_overlap_pair(
                tile_overlap.A,
                tile_overlap.B,
                tile_overlap.scaled_overlapping_source_rect_A,
                tile_overlap.scaled_overlapping_source_rect_B,
                tile_overlap.overlapping_target_rect,  # type: ignore[arg-type]
                image_scale,
                subregion_shape))
    finally:
        anchor_tile._image = None
        anchor_tile._paddedimage = None
        _release_refinement_worker_memory()

    return results


def __RefineTileAlignmentRemote(
        A: nornir_imageregistration.Tile,
        B: nornir_imageregistration.Tile,
        scaled_overlapping_source_rect_A: nornir_imageregistration.Rectangle,
        scaled_overlapping_source_rect_B: nornir_imageregistration.Rectangle,
        OffsetAdjustment: NDArray[np.floating],
        imageScale: float,
        subregion_shape: NDArray[np.integer] | Sequence[int] | int | None = None,
        overlapping_target_rect: nornir_imageregistration.Rectangle | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Refine local offsets for one overlapping tile pair in worker context."""
    if subregion_shape is None:
        subregion_shape = np.array([128, 128], dtype=np.int64)
    else:
        subregion_shape = np.asarray(_normalize_pair(subregion_shape, "subregion_shape", minimum=4), dtype=np.int64)

    if overlapping_target_rect is None:
        overlapping_target_rect = nornir_imageregistration.Rectangle.overlap_rect(
            A.FixedBoundingBox, B.FixedBoundingBox)
        if overlapping_target_rect is None:
            raise ValueError(f"Tiles {A.ID} and {B.ID} do not overlap")

    try:
        return _refine_single_tile_overlap_pair(
            A,
            B,
            scaled_overlapping_source_rect_A,
            scaled_overlapping_source_rect_B,
            overlapping_target_rect,
            imageScale,
            subregion_shape)
    finally:
        _release_refinement_worker_memory()


def SplitDisplacements(
        A: nornir_imageregistration.Tile | None,
        B: nornir_imageregistration.Tile | None,
        point_pairs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Split pairwise displacement updates into per-tile source/target updates.

    Parameters
    ----------
    A:
        Reference tile for one side of each pair.
    B:
        Paired tile for the opposite side of each pair.
    point_pairs:
        Structured point-pair array produced by local refinement.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two structured arrays of updates for tile A and tile B.
    """

    if point_pairs is None:
        return np.empty(0), np.empty(0)

    flattened = point_pairs.reshape(-1)
    if flattened.size == 0:
        return np.empty(0), np.empty(0)

    output_dtype = np.dtype([('SourceY', 'f4'),
                             ('SourceX', 'f4'),
                             ('TargetY', 'f4'),
                             ('TargetX', 'f4'),
                             ('Weight', 'f4')])

    a_updates = np.empty(flattened.shape[0], dtype=output_dtype)
    b_updates = np.empty(flattened.shape[0], dtype=output_dtype)

    half_displacement_y = flattened['DisplacementY'] / 2.0
    half_displacement_x = flattened['DisplacementX'] / 2.0

    a_updates['SourceY'] = flattened['SourceAY']
    a_updates['SourceX'] = flattened['SourceAX']
    a_updates['TargetY'] = flattened['BaseTargetY'] + half_displacement_y
    a_updates['TargetX'] = flattened['BaseTargetX'] + half_displacement_x
    a_updates['Weight'] = flattened['Weight']

    b_updates['SourceY'] = flattened['SourceBY']
    b_updates['SourceX'] = flattened['SourceBX']
    b_updates['TargetY'] = flattened['BaseTargetY'] - half_displacement_y
    b_updates['TargetX'] = flattened['BaseTargetX'] - half_displacement_x
    b_updates['Weight'] = flattened['Weight']

    valid = flattened['Weight'] > 0
    return a_updates[valid], b_updates[valid]


def RefineStosFile(InputStos: str | nornir_imageregistration.StosFile,
                   OutputStosPath: str,
                   num_iterations: int | None = None,
                   cell_size: NDArray[np.integer] | tuple[int, int] | None = None,
                   grid_spacing: NDArray[np.integer] | tuple[int, int] | None = None,
                   angles_to_search: NDArray[np.floating] | Sequence[float] | None = None,
                   final_pass_angles: NDArray[np.floating] | Sequence[float] | None = None,
                   max_travel_for_finalization: float | None = None,
                   max_travel_for_finalization_improvement: float | None = None,
                   min_finalize_pass: int | None = None,
                   finalize_stability_passes: int | None = None,
                   finalize_stability_epsilon_px: float | None = None,
                   finalize_unlock_travel_multiplier: float | None = None,
                   inclusion_travel_multiplier: float | None = None,
                   anchor_smooth_min_locks: int | None = None,
                   anchor_smooth_median_radius: int | None = None,
                   min_alignment_overlap: float | None = None,
                   min_unmasked_area: float | None = None,
                   SaveImages: bool = False,
                   SavePlots: bool = False,
                   **kwargs: Any) -> None:
    """
    Refine one STOS transform and save the refined output file.

    This loads source/target image data and masks, runs iterative grid refinement
    using the configured settings, converts the result to a grid transform, and
    writes the updated STOS file to ``OutputStosPath``.
    """

    progress_depth_base = int(kwargs.pop("progress_depth_base", 0))

    for k, v in kwargs.items():
        prettyoutput.Log(f"\tUnused parameter to RefineStosFile: {k}:{v}\n")

    stos_parent_dir = os.path.dirname(OutputStosPath)
    stos_stem = os.path.splitext(os.path.basename(OutputStosPath))[0]
    # Nest pass diagnostics / plots under refine_diagnostics/<stos stem> so
    # multi-pair Grid runs do not clobber shared refine_passNN_* filenames.
    write_artifacts = bool(SavePlots or SaveImages or pass_diagnostics_enabled(SavePlots))
    outputDir: str | None = None
    if write_artifacts:
        outputDir = os.path.join(stos_parent_dir, 'refine_diagnostics', stos_stem)
        os.makedirs(outputDir, exist_ok=True)
        prettyoutput.Log(f'Writing refine pass artifacts to {outputDir}')

    # Load the input stos file if it is not already loaded
    if not isinstance(InputStos, nornir_imageregistration.StosFile):
        stosDir = os.path.dirname(InputStos)
        InputStos = nornir_imageregistration.files.StosFile.Load(InputStos)
        InputStos.TryConvertRelativePathsToAbsolutePaths(stosDir)

    stosTransform = nornir_imageregistration.transforms.factory.LoadTransform(InputStos.Transform, 1)  # type: ignore[arg-type]
    if stosTransform is None:
        raise ValueError(f"Could not load transform: {InputStos} - {InputStos.Transform}")

    target_image_data = nornir_imageregistration.ImagePermutationHelper(img=InputStos.ControlImageFullPath,
                                                                        mask=InputStos.ControlMaskFullPath,
                                                                        extrema_mask_size_cuttoff=None,
                                                                        dtype=nornir_imageregistration.default_image_dtype())

    source_image_data = nornir_imageregistration.ImagePermutationHelper(img=InputStos.MappedImageFullPath,
                                                                        mask=InputStos.MappedMaskFullPath,
                                                                        extrema_mask_size_cuttoff=None,
                                                                        dtype=nornir_imageregistration.default_image_dtype())

    with nornir_imageregistration.settings.GridRefinement.CreateWithPreprocessedImages(
            target_img_data=target_image_data,
            source_img_data=source_image_data,
            num_iterations=num_iterations, cell_size=cell_size,  # type: ignore[arg-type]
            grid_spacing=grid_spacing,  # type: ignore[arg-type]
            angles_to_search=angles_to_search,  # type: ignore[arg-type]
            final_pass_angles=final_pass_angles,  # type: ignore[arg-type]
            max_travel_for_finalization=max_travel_for_finalization,  # type: ignore[arg-type]
            max_travel_for_finalization_improvement=max_travel_for_finalization_improvement,  # type: ignore[arg-type]
            min_finalize_pass=min_finalize_pass,
            finalize_stability_passes=finalize_stability_passes,
            finalize_stability_epsilon_px=finalize_stability_epsilon_px,
            finalize_unlock_travel_multiplier=finalize_unlock_travel_multiplier,
            inclusion_travel_multiplier=inclusion_travel_multiplier,
            anchor_smooth_min_locks=anchor_smooth_min_locks,
            anchor_smooth_median_radius=anchor_smooth_median_radius,
            min_alignment_overlap=min_alignment_overlap,  # type: ignore[arg-type]
            min_unmasked_area=min_unmasked_area,  # type: ignore[arg-type]
            single_thread_processing=False) as settings:

        try:
            output_transform = RefineTransform(stosTransform,
                                               settings,
                                               SaveImages=SaveImages,
                                               SavePlots=SavePlots,
                                               outputDir=outputDir,
                                               progress_depth_base=progress_depth_base)
        finally:
            _release_refinement_worker_memory()

        InputStos.Transform = nornir_imageregistration.transforms.ConvertTransformToGridTransform(
            output_transform,
            source_image_shape=settings.source_image.shape,  # type: ignore[arg-type]
            cell_size=settings.cell_size,
            grid_spacing=settings.grid_spacing,
            prefer_gpu=False)
        InputStos.Save(OutputStosPath)


def _zncc_at_claimed_peak(
        target_roi: NDArray,
        source_roi: NDArray,
        peak: NDArray[np.floating] | tuple[float, float],
        *,
        travel_eps: float = 0.5,
) -> float:
    """Masked ZNCC of fixed vs moving after shifting moving by *peak*.

    Near-identity peaks (``‖peak‖ < travel_eps``) skip ``CropImage`` and compare
    ROIs in place.
    """
    peak_arr = np.asarray(peak, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(peak_arr)) < float(travel_eps):
        return masked_zncc(target_roi, source_roi)

    xp = cp.get_array_module(source_roi)
    h = int(source_roi.shape[0])
    w = int(source_roi.shape[1])
    med = float(xp.median(source_roi))
    shifted = nornir_imageregistration.CropImage(
        source_roi,
        int(np.floor(-peak_arr[1])),
        int(np.floor(-peak_arr[0])),
        w,
        h,
        cval=med)
    return masked_zncc(target_roi, shifted)


def _shift_moving_stack_by_peaks(
        moving_stack: NDArray,
        peaks: NDArray[np.floating],
        *,
        travel_eps: float = 0.5,
) -> NDArray:
    """Return a moving stack shifted by each peak; identity peaks are copied."""
    xp = cp.get_array_module(moving_stack)
    peaks_np = np.asarray(peaks, dtype=np.float64).reshape(-1, 2)
    n = int(moving_stack.shape[0])
    h = int(moving_stack.shape[1])
    w = int(moving_stack.shape[2])
    out = xp.empty_like(moving_stack)
    meds = xp.median(moving_stack, axis=(1, 2))
    for i in range(n):
        peak = peaks_np[i]
        if float(np.linalg.norm(peak)) < float(travel_eps):
            out[i] = moving_stack[i]
            continue
        out[i] = nornir_imageregistration.CropImage(
            moving_stack[i],
            int(np.floor(-peak[1])),
            int(np.floor(-peak[0])),
            w,
            h,
            cval=float(meds[i]))
    return out


def _masked_zncc_stack(fixed_stack: NDArray, moving_stack: NDArray) -> NDArray[np.float64]:
    """Per-cell masked ZNCC for stacked ROIs; returns host float64 length N."""
    xp = cp.get_array_module(fixed_stack)
    a = xp.asarray(fixed_stack, dtype=xp.float64)
    b = xp.asarray(moving_stack, dtype=xp.float64)
    valid = xp.isfinite(a) & xp.isfinite(b)
    n_valid = xp.count_nonzero(valid, axis=(1, 2))
    a_z = xp.where(valid, a, 0)
    b_z = xp.where(valid, b, 0)
    n = xp.maximum(n_valid.astype(xp.float64), 1.0)
    a_mean = a_z.sum(axis=(1, 2)) / n
    b_mean = b_z.sum(axis=(1, 2)) / n
    av = xp.where(valid, a - a_mean[:, None, None], 0)
    bv = xp.where(valid, b - b_mean[:, None, None], 0)
    num = (av * bv).sum(axis=(1, 2))
    denom = xp.sqrt((av * av).sum(axis=(1, 2)) * (bv * bv).sum(axis=(1, 2)))
    ok = (n_valid >= 2) & (denom > 0) & xp.isfinite(denom)
    scores = xp.where(ok, num / denom, 0.0)
    scores = xp.where(xp.isfinite(scores), scores, 0.0)
    return nornir_imageregistration.EnsureNumpyArray(scores).astype(np.float64, copy=False)


#: Per-candidate ZNCC failures reported individually before switching to a count.
_ZNCC_FAILURE_LOG_LIMIT: int = 3


def _compute_zncc_for_candidates(
        records: Sequence,
        candidate_ids: set[tuple[int, int]],
        transform: nornir_imageregistration.ITransform,
        settings: nornir_imageregistration.settings.GridRefinement,
        *,
        travel_eps: float = 0.5,
        reference_pose: RingReferencePose | None = None,
) -> dict[tuple[int, int], float]:
    """Score masked ZNCC at each lock-candidate peak.

    Prefer one batched ``ApproximateRigidTransformBySourcePoints`` plus
    ``BuildAlignmentROIsBatched``; fall back to per-candidate extract when the
    batched path is unavailable.
    """
    scores: dict[tuple[int, int], float] = {}
    if not candidate_ids:
        return scores

    cand_recs = [
        rec for rec in records
        if (int(rec.ID[0]), int(rec.ID[1])) in candidate_ids
    ]
    if not cand_recs:
        return scores

    keys = [(int(rec.ID[0]), int(rec.ID[1])) for rec in cand_recs]
    source_points = np.asarray(
        [np.asarray(rec.SourcePoint, dtype=np.float64).reshape(2) for rec in cand_recs],
        dtype=np.float64)
    target_points = np.asarray(
        [np.asarray(rec.TargetPoint, dtype=np.float64).reshape(2) for rec in cand_recs],
        dtype=np.float64)
    peaks = np.asarray(
        [np.asarray(rec.peak, dtype=np.float64).reshape(2) for rec in cand_recs],
        dtype=np.float64)

    try:
        rigid_transforms = ApproximateRigidTransformBySourcePoints(
            input_transform=transform,
            source_points=source_points,
            cell_size=settings.cell_size,
            reference_pose=reference_pose,
            ring_scale_fraction_max=settings.ring_scale_fraction_max,
            ring_angle_max_degrees=settings.ring_angle_max_degrees,
            ring_allow_flip_change=settings.ring_allow_flip_change)
        target_image, source_image = _stos_settings_images(settings)
        batched = BuildAlignmentROIsBatched(
            rigid_transforms=rigid_transforms,
            target_image=target_image,
            source_image=source_image,
            target_image_stats=settings.target_image_stats,
            source_image_stats=settings.source_image_stats,
            target_points=target_points,
            alignment_area=settings.cell_size)
        if batched is not None:
            fixed_stack, moving_stack, _nan_mask = batched
            shifted = _shift_moving_stack_by_peaks(
                moving_stack, peaks, travel_eps=travel_eps)
            cell_scores = _masked_zncc_stack(fixed_stack, shifted)
            for key, score in zip(keys, cell_scores):
                scores[key] = float(score)
            return scores
    except Exception as e:
        # Recoverable: the per-candidate fallback below still produces scores. Logged
        # because silence here made a genuine defect in the batched ZNCC path look
        # like normal operation, permanently degraded to the slow path.
        prettyoutput.LogErr(
            f'Batched ZNCC failed for {len(keys)} lock candidates, '
            f'falling back to per-candidate scoring:\n{e}')

    # Fallback: per-candidate extract (batched path unavailable or failed).
    # A systematic failure fails every candidate, so report the first few in full
    # and then a count. Grids run to thousands of cells.
    failed_keys: list[tuple[int, int]] = []
    for i, _rec in enumerate(cand_recs):
        key = keys[i]
        try:
            rigid = ApproximateRigidTransformBySourcePoints(
                input_transform=transform,
                source_points=source_points[i:i + 1],
                cell_size=settings.cell_size,
                reference_pose=reference_pose,
                ring_scale_fraction_max=settings.ring_scale_fraction_max,
                ring_angle_max_degrees=settings.ring_angle_max_degrees,
                ring_allow_flip_change=settings.ring_allow_flip_change)[0]
            rois = BuildAlignmentROIs(
                transform=rigid,
                targetImage_param=settings.target_image,
                sourceImage_param=settings.source_image,
                target_image_stats=settings.target_image_stats,
                source_image_stats=settings.source_image_stats,
                target_controlpoint=target_points[i],
                alignmentArea=settings.cell_size,
                defer_oob_check=False)
            scores[key] = _zncc_at_claimed_peak(
                rois[0], rois[1], peaks[i], travel_eps=travel_eps)
        except Exception as e:
            # Leave the key absent rather than storing 0.0. Zero is a legitimate
            # ZNCC meaning "does not correlate", so recording it made an
            # infrastructure failure indistinguishable from a measured verdict and
            # wrote a score that was never measured into the pass diagnostics.
            # classify_roles fails closed on a missing key (IDENTITY_SUSPECT, never
            # locks) and pass_diagnostics already reports missing keys as NaN.
            failed_keys.append(key)
            if len(failed_keys) <= _ZNCC_FAILURE_LOG_LIMIT:
                prettyoutput.LogErr(f'ZNCC scoring failed for lock candidate {key}:\n{e}')

    if len(failed_keys) > _ZNCC_FAILURE_LOG_LIMIT:
        prettyoutput.LogErr(
            f'ZNCC scoring failed for {len(failed_keys)} of {len(cand_recs)} lock '
            f'candidates; {_ZNCC_FAILURE_LOG_LIMIT} reported above. Those cells have '
            f'no ZNCC score and cannot lock.')
    return scores


def _lock_candidate_ids_preview(
        records: Sequence,
        transform_cutoff: float,
        max_travel: float,
        per_record_max_travel: NDArray[np.floating] | None,
        soft_weight_cutoff: float | None,
        discontinuity_ids: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """IDs that pass weight/travel lock bars and are not peak-ambiguous."""
    ids: set[tuple[int, int]] = set()
    n = len(records)
    if n == 0:
        return ids
    if per_record_max_travel is not None:
        limits = np.asarray(per_record_max_travel, dtype=np.float64).reshape(-1)
    else:
        limits = np.full(n, float(max_travel), dtype=np.float64)
    for i, rec in enumerate(records):
        ratio = finite_peak_ratio(rec)
        if ratio is not None and float(ratio) < float(PEAK_RATIO_MIN):
            continue
        key = (int(rec.ID[0]), int(rec.ID[1]))
        weight_bar = float(transform_cutoff)
        if (soft_weight_cutoff is not None and key in discontinuity_ids
                and ratio is not None and float(ratio) >= float(PEAK_RATIO_MIN)):
            weight_bar = float(soft_weight_cutoff)
        travel = float(np.linalg.norm(np.asarray(rec.peak, dtype=np.float64).reshape(2)))
        if float(rec.weight) >= weight_bar and travel <= float(limits[i]):
            ids.add(key)
    return ids


def should_finish_on_empty_alignment_pass(pass_index: int, n_finalized: int) -> bool:
    """Return True when an empty remasure should end refine instead of aborting.

    Pass 1 with no locked points is still a hard failure after cell-size
    doubling cannot continue. Later passes, or any pass that already locked
    cells, keep the current transform.
    """
    return int(n_finalized) > 0 or int(pass_index) > 1


def _clamp_settings_cell_size(
        settings: nornir_imageregistration.settings.GridRefinement,
) -> NDArray[np.int64]:
    """Clamp ``settings.cell_size`` to the image/1024 cap. Returns the clamped size."""
    cap = cell_size_cap_from_shapes(
        settings.source_image.shape, settings.target_image.shape)
    current = np.asarray(settings.cell_size, dtype=np.int64).ravel()[:2].copy()
    clamped = clamp_cell_size_to_cap(current, cap)
    if bool(np.any(current > cap)):
        prettyoutput.Log(
            f'cell_size {current.tolist()} exceeds cap {cap.tolist()}; clamping')
        settings.cell_size = clamped
    return clamped


def _grow_refine_cell_size_after_failure(
        settings: nornir_imageregistration.settings.GridRefinement,
        source_content_cache: SourceContentCache,
        *,
        pass_index: int,
        final_pass: bool,
) -> bool:
    """Double ``settings.cell_size`` for remaining passes after a failed measure.

    Returns True when cell size grew. Sticky LOW_CONTENT cache entries from the
    smaller crop must be dropped so larger ROIs are remeasured.
    """
    if not can_grow_cell_size_on_pass(
            pass_index=pass_index,
            num_iterations=int(settings.num_iterations),
            final_pass=final_pass):
        return False
    previous = np.asarray(settings.cell_size, dtype=np.int64).copy()
    grown = next_cell_size_after_failure(
        previous,
        cell_size_cap_from_shapes(
            settings.source_image.shape, settings.target_image.shape),
    )
    if grown is None:
        prettyoutput.Log(
            f'Pass {pass_index}: no usable alignments; cell_size already at cap '
            f'{previous.tolist()}')
        return False
    settings.cell_size = grown
    source_content_cache.clear()
    prettyoutput.Log(
        f'Pass {pass_index}: no usable alignments; doubling cell_size '
        f'{previous.tolist()} -> {grown.tolist()} for remaining passes')
    return True


def _restore_refine_cell_size_after_success(
        settings: nornir_imageregistration.settings.GridRefinement,
        source_content_cache: SourceContentCache,
        requested_cell_size: NDArray[np.integer] | Sequence[int],
        *,
        pass_index: int,
        final_pass: bool,
) -> bool:
    """Return ``settings.cell_size`` to the caller's requested size after registrations.

    Returns True when cell size shrank. Source-content cache entries from the
    larger crop are dropped so the original ROI size is remeasured.
    """
    if not can_grow_cell_size_on_pass(
            pass_index=pass_index,
            num_iterations=int(settings.num_iterations),
            final_pass=final_pass):
        return False
    if not cell_size_exceeds_requested(settings.cell_size, requested_cell_size):
        return False
    previous = np.asarray(settings.cell_size, dtype=np.int64).ravel()[:2].copy()
    restored = np.asarray(requested_cell_size, dtype=np.int64).ravel()[:2].copy()
    settings.cell_size = restored
    source_content_cache.clear()
    prettyoutput.Log(
        f'Pass {pass_index}: found registrations; restoring cell_size '
        f'{previous.tolist()} -> {restored.tolist()} for remaining passes')
    return True


def RefineTransform(stosTransform: nornir_imageregistration.ITransform,
                    settings: nornir_imageregistration.settings.GridRefinement,
                    SaveImages: bool = False,
                    SavePlots: bool = False,
                    outputDir: str | None = None,
                    progress_depth_base: int = 0,
                    cancel_event: threading.Event | None = None,
                    progress_callback: ProgressCallback | None = None) -> nornir_imageregistration.ITransform:
    """
    Iteratively refine a source-to-target transform from local alignment points.

    The routine alternates between generating candidate alignments, building an
    updated transform from cutoff-selected points, and finalizing stable points
    until convergence or pass limits are reached.

    When *progress_callback* accepts a fourth argument, each completed pass
    (and residual ``TranslateFixed``) delivers a deep-copied transform so a UI
    can preview the working mesh without sharing the worker's live object.
    MQTT dashboard progress is separate and does not carry the transform.

    Pass state (locks, ring pose, residual check, cell-size adaptation) lives
    in this call. Repeating ``num_iterations=1`` N times is not equivalent.

    When a pass measures nothing usable (empty grid or all REJECT), remaining
    iterations double ``settings.cell_size`` up to ``MAX_REFINE_CELL_SIZE``
    (1024), further limited by the smaller image shape. When a later pass finds
    FREE/LOCKABLE registrations, remaining iterations return to the original
    requested cell size. Grid spacing is unchanged. The caller's ``cell_size``
    is restored on exit.
    """

    if (SavePlots or SaveImages) and outputDir is None:
        raise ValueError("outputDir must be specified if SavePlots or SaveImages is true.")

    # Frozen input pose for ring linearization. Do not re-Kabsch from later meshes.
    ring_reference_pose = reference_pose_from_transform(stosTransform)
    # Same object as *stosTransform* until a mesh rebuild replaces it. TranslateFixed
    # mutations remain visible here; collapsed 3-point meshes must not.
    refine_input_transform = stosTransform

    # Convert inputs to numpy arrays

    final_pass = False  # True if this is the last iteration the loop will perform

    finalized_points = {}  # type: AlignmentRecordDict
    finalize_candidates: dict[tuple[int, int], FinalizeCandidateState] = {}
    finalize_settings = FinalizeSettings.from_grid_refinement(settings)
    legacy_finalize = use_legacy_finalize_gate()
    coherent_residual_checked = False  # Track A/B attempted (once per refine)
    coherent_residual_translated = False  # TranslateFixed actually ran (for preserve gate)
    global_pose_recovery_applied = False
    last_residual_translation: NDArray[np.floating] | None = None
    residual_undo_attempted = False
    source_content_cache = SourceContentCache()
    original_cell_size = settings.cell_size
    requested_cell_size = _clamp_settings_cell_size(settings)
    cell_history = CellPassHistoryStore()
    if legacy_finalize:
        prettyoutput.Log(
            'NORNIR_REFINE_FINALIZE_LEGACY=1: using distance-primary finalize with 2% weight floor')

    CutoffPercentilePerIteration = 10.0

    FirstPassWeightScoreCutoff = None
    FirstPassCompositeScoreCutoff = None
    # FirstPassFinalizeValue = None  # The score required to finalize a control point on the first pass.
    # The first score is recorded to prevent the best scores from being finalized and then later
    # groups of poor scores looking falsely good because the correct registrations are all finalized
    first_pass_weight_distance_composite_scores = None

    # transform_inclusion_percentile = 66  # - (CutoffPercentilePerIteration * i)
    # transform_inclusion_range = 20.0
    # finalize_percentile = 80
    # finalize_range = 46.6
    updatedTransform = None  # type: nornir_imageregistration.ITransform | None

    i = 1
    combined_records_this_pass: AlignmentRecordDict = {}
    role_result = None

    # num_iterations // 2 is 0 for a single-pass refine, which EMA cannot average over.
    ema_window = max(1, settings.num_iterations // 2)
    finalize_ema = EMA(ema_window, 2)  # Track the cutoff values over the last three passes
    cutoff_ema = EMA(ema_window, 2)
    first_cutoff = None  # The first cutoff value, we use this to decide which points make it into the final transform

    _PHASE_TIMER.reset()
    pair_t0 = time.perf_counter()

    initial_grid_count = count_initial_grid_points(stosTransform, settings)
    progress_reporter = RefineGridProgressReporter(
        settings.num_iterations,
        initial_grid_count,
        depth_base=progress_depth_base,
    )

    try:
        while i <= settings.num_iterations:
            check_cancelled(cancel_event)
            report_progress(
                progress_callback,
                i,
                settings.num_iterations,
                f"Refine pass {i}/{settings.num_iterations}")
            pass_t0 = time.perf_counter()
            pass_phase_baseline = _PHASE_TIMER.snapshot()
            measure_s = 0.0
            finalize_s = 0.0
            diagnostics_tables_s = 0.0
            diagnostics_heatmaps_s = 0.0

            if i == settings.num_iterations:
                final_pass = True

            measure_t0 = time.perf_counter()
            alignment_points = _RefineGridPointsForTwoImages(
                stosTransform,
                settings=settings,
                finalized=finalized_points,
                source_content_cache=source_content_cache,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                reference_pose=ring_reference_pose)
            measure_s = time.perf_counter() - measure_t0

            if len(alignment_points) == 0:
                if _grow_refine_cell_size_after_failure(
                        settings,
                        source_content_cache,
                        pass_index=i,
                        final_pass=final_pass):
                    i += 1
                    continue
                if should_finish_on_empty_alignment_pass(i, len(finalized_points)):
                    if len(finalized_points) > 0:
                        prettyoutput.Log(
                            f"Pass {i}: no remaining unfinalized points meet mask/bounds criteria; "
                            f"finishing with {len(finalized_points)} locked points")
                    else:
                        prettyoutput.Log(
                            f"Pass {i}: no alignment points generated; "
                            f"keeping transform from pass {i - 1}")
                    break
                raise ValueError(f"No alignment points generated at pass #{i}")

            alignment_points = _maybe_regularize_stos_alignment_peaks(alignment_points)

            prettyoutput.Log(f"Pass {i} aligned {len(alignment_points)} points")

            # FOV / coherent residual can leave almost no cells registering. Undo once
            # so a bad TranslateFixed does not starve mesh construction.
            if (coherent_residual_translated
                    and not residual_undo_attempted
                    and last_residual_translation is not None
                    and len(finalized_points) == 0
                    and len(alignment_points) < 3):
                translate_fixed = getattr(stosTransform, 'TranslateFixed', None)
                if callable(translate_fixed):
                    residual_undo_attempted = True
                    translate_fixed(-np.asarray(last_residual_translation, dtype=np.float64))
                    coherent_residual_translated = False
                    prettyoutput.Log(
                        f'Reverted residual translation after sparse remasure '
                        f'({len(alignment_points)} alignments, 0 locks); remeasuring')
                    report_pass_transform(
                        progress_callback,
                        stosTransform,
                        i,
                        settings.num_iterations,
                        label=f"Refine pass {i}: residual revert")
                    measure_t0 = time.perf_counter()
                    alignment_points = _RefineGridPointsForTwoImages(
                        stosTransform,
                        settings=settings,
                        finalized=finalized_points,
                        source_content_cache=source_content_cache,
                        cancel_event=cancel_event,
                        progress_callback=progress_callback,
                        reference_pose=ring_reference_pose)
                    measure_s += time.perf_counter() - measure_t0
                    if len(alignment_points) == 0:
                        if _grow_refine_cell_size_after_failure(
                                settings,
                                source_content_cache,
                                pass_index=i,
                                final_pass=final_pass):
                            i += 1
                            continue
                        if should_finish_on_empty_alignment_pass(i, len(finalized_points)):
                            prettyoutput.Log(
                                f"Pass {i}: no alignment points after residual revert; "
                                f"keeping current transform")
                            break
                        raise ValueError(
                            f"No alignment points generated at pass #{i} after residual revert")
                    alignment_points = _maybe_regularize_stos_alignment_peaks(alignment_points)
                    prettyoutput.Log(
                        f"Pass {i} aligned {len(alignment_points)} points after residual revert")

            # Track A: once per refine, absorb a coherent residual translation when
            # almost nothing has locked. If unique peaks are too scarce/incoherent
            # (wrap-like), try downsampled whole-FOV phase-correlation once.
            # coherent_residual_checked gates the attempt; coherent_residual_translated
            # is True only when TranslateFixed ran (preserve must not treat "checked"
            # as "translated" or sparse meshes discard a non-existent residual).
            if not coherent_residual_checked:
                grid_n = max(1, len(alignment_points) + len(finalized_points))
                lock_fraction = float(len(finalized_points)) / float(grid_n)
                diagnosis = diagnose_coherent_residual_translation(
                    alignment_points, lock_fraction=lock_fraction)
                residual = diagnosis.result
                translate_fixed = getattr(stosTransform, 'TranslateFixed', None)
                if residual is not None and callable(translate_fixed):
                    translate_fixed(residual.translation)
                    finalize_candidates.clear()
                    coherent_residual_translated = True
                    coherent_residual_checked = True
                    last_residual_translation = np.asarray(residual.translation, dtype=np.float64).copy()
                    prettyoutput.Log(
                        f'Coherent residual translation: (dy, dx)=('
                        f'{float(residual.translation[0]):.2f}, '
                        f'{float(residual.translation[1]):.2f}) '
                        f'coherence={residual.coherence:.3f} '
                        f'n_unique={residual.n_unique} '
                        f'n_inliers={residual.n_inliers}')
                    report_pass_transform(
                        progress_callback,
                        stosTransform,
                        i,
                        settings.num_iterations,
                        label=f"Refine pass {i}: residual translation")
                    _log_phase_breakdown(f'RefineTransform pass {i} (coherent residual)', pass_phase_baseline)
                    continue
                if lock_fraction < float(LOCK_FRAC_TRIGGER):
                    prettyoutput.Log(
                        f'Coherent residual skipped: reason={diagnosis.skip_reason} '
                        f'n_unique={diagnosis.n_unique} n_inliers={diagnosis.n_inliers} '
                        f'coherence={diagnosis.coherence:.3f}')
                    if (not global_pose_recovery_applied
                            and should_attempt_global_fov_recovery(diagnosis, lock_fraction)
                            and callable(translate_fixed)):
                        global_peak = estimate_global_fov_residual_translation(
                            stosTransform,
                            settings.target_image,
                            settings.source_image,
                        )
                        global_pose_recovery_applied = True
                        if global_peak is not None and float(np.linalg.norm(global_peak)) >= 1.0:
                            translate_fixed(global_peak)
                            finalize_candidates.clear()
                            coherent_residual_translated = True
                            coherent_residual_checked = True
                            last_residual_translation = np.asarray(global_peak, dtype=np.float64).copy()
                            prettyoutput.Log(
                                f'Global FOV residual translation: (dy, dx)=('
                                f'{float(global_peak[0]):.2f}, {float(global_peak[1]):.2f})')
                            report_pass_transform(
                                progress_callback,
                                stosTransform,
                                i,
                                settings.num_iterations,
                                label=f"Refine pass {i}: residual translation")
                            _log_phase_breakdown(
                                f'RefineTransform pass {i} (global FOV residual)',
                                pass_phase_baseline)
                            continue
                        prettyoutput.Log(
                            'Global FOV residual skipped: no usable peak')
                    elif (not global_pose_recovery_applied
                          and int(diagnosis.n_unique) <= 0):
                        prettyoutput.Log(
                            'Global FOV residual skipped: n_unique=0 '
                            '(all cell peaks rejected; no TranslateFixed)')
                    # Pathological recovery attempts finished for this refine.
                    coherent_residual_checked = True
                else:
                    # Locks already healthy — no residual recovery needed.
                    coherent_residual_checked = True

            progress_reporter.on_pass_start(i)
            report_progress(
                progress_callback,
                i,
                settings.num_iterations,
                f"Refine pass {i}: scoring / finalize")

            updated_and_finalized_alignment_points = alignment_points + list(finalized_points.values())
            updated_and_finalized_weights_distance = _alignment_records_to_composite_scores(
                updated_and_finalized_alignment_points,
                max_distance=max(settings.cell_size))

            # What fraction of the maximum number of iterations have been completed?
            adjustment_scalar = (i - 1) / settings.num_iterations

            # transform_inclusion_percentile_this_pass = transform_inclusion_percentile
            # if adjustment_scalar != 0:
            #     transform_inclusion_percentile_this_pass -= (transform_inclusion_range * adjustment_scalar)
            #
            # transform_inclusion_percentile_this_pass = float(np.clip(transform_inclusion_percentile_this_pass, 10.0,
            #                                                          100.0))  # This is a float, so don't bother with out parameter
            #
            # transform_cutoff_this_pass = np.percentile(updated_and_finalized_weights_distance[:, 2],
            #                                            # Do not include finalize points because they have a distance of zero which throws off the composite scores
            #                                            transform_inclusion_percentile_this_pass)

            # finalize_percentile_this_pass = finalize_percentile
            # if adjustment_scalar != 0:
            #     finalize_percentile_this_pass -= (finalize_range * adjustment_scalar)
            #
            # finalize_percentile_this_pass = float(
            #     np.clip(finalize_percentile_this_pass, 10.0, 100.0))  # This is a float, so don't bother with out parameter
            #
            # finalize_cutoff_this_pass = np.percentile(updated_and_finalized_weights_distance[:, 0],
            #                                           finalize_percentile_this_pass)

            # Using the set of alignment record scores, estimate the cutoff value that separates successful registrations from failed registrations
            weight_cutoff = estimate_registration_weight_cutoff(
                updated_and_finalized_weights_distance[:, WeightMethod.Registration])
            cutoff_percentile_this_pass = weight_cutoff.cutoff_percentile_index
            inflection_percentile = weight_cutoff.inflection_percentile_index
            cutoff_value_this_pass = weight_cutoff.cutoff_value
            polyfit_weights = weight_cutoff.percentile_curve
            if weight_cutoff.used_fallback:
                prettyoutput.Log(
                    "No inflection in registration weights; including nearly all measured points this pass")

            # Diagnostic-only inflection (kept for logs / EMA history). Mesh inclusion,
            # lock candidacy, and finalize no longer gate on registration weight —
            # travel + Role/ZNCC secondary are the active bars.
            transform_cutoff_percentile = inflection_percentile
            diagnostic_inflection_value = float(polyfit_weights[transform_cutoff_percentile])
            cutoff_ema.add(diagnostic_inflection_value)
            if first_cutoff is None:
                first_cutoff = float(cutoff_value_this_pass)

            transform_cutoff_value = float('-inf')
            cutoff_value = transform_cutoff_value

            prettyoutput.Log(
                f'#######\n'
                f'Registration weight cutoff disabled; gating via travel + Role/ZNCC\n'
                f'Diagnostic inflection (unused): {transform_cutoff_percentile}% -> '
                f'{diagnostic_inflection_value}\n'
                f'Exponential Moving Average diagnostic inflection: '
                f'{cutoff_ema.ema_value if cutoff_ema.has_samples else "n/a"}\n')

            finalize_t0 = time.perf_counter()
            preserve_post_residual = False
            with _PHASE_TIMER.section_wall('finalize'):
                anchor_smooth_active = should_use_anchor_smooth_mesh(finalized_points, settings)
                n_travel_dropped = 0
                inclusion_travel = float(settings.max_travel_for_finalization) * float(
                    getattr(settings, 'inclusion_travel_multiplier', 1.0))
                discontinuity_ids: set[tuple[int, int]] = set()
                if sharp_warps_enabled() and len(alignment_points) > 0:
                    discontinuity_ids = tag_discontinuities(
                        alignment_points,
                        max_travel=float(settings.max_travel_for_finalization),
                        stable_ids=None,
                    )
                    if discontinuity_ids:
                        prettyoutput.Log(
                            f'Sharp-warp discontinuities: {len(discontinuity_ids)} cells '
                            f'(neighbor peak disagreement)')

                # Soft travel/weight only for disc cells with finite peak_ratio >= min.
                soft_disc_ids = soft_discontinuity_ids(alignment_points, discontinuity_ids)
                if discontinuity_ids and len(soft_disc_ids) < len(discontinuity_ids):
                    prettyoutput.Log(
                        f'Sharp-warp soft floors: {len(soft_disc_ids)}/{len(discontinuity_ids)} '
                        f'disc cells eligible (peak_ratio >= min)')

                # Mesh raw-preserve: ratio-eligible soft-disc ∪ unique large-travel
                # ∪ coherent active disc fronts (cluster + direction; no pr floor).
                # Do NOT raw-preserve all discontinuity tags — ambiguous wrap-like
                # disc peaks (pr≈1.03) create a feedback loop (disc 78→489 on 252-254)
                # unless they form a spatially coherent front.
                unique_raw_ids = unique_large_travel_raw_preserve_ids(
                    alignment_points,
                    max_travel=float(settings.max_travel_for_finalization),
                )
                coherent_disc_ids = coherent_discontinuity_raw_preserve_ids(
                    alignment_points,
                    discontinuity_ids,
                    max_travel=float(settings.max_travel_for_finalization),
                )
                raw_preserve_ids = set(soft_disc_ids) | unique_raw_ids | coherent_disc_ids
                if unique_raw_ids or soft_disc_ids or coherent_disc_ids:
                    prettyoutput.Log(
                        f'Mesh raw-preserve: soft-disc={len(soft_disc_ids)} '
                        f'unique_large={len(unique_raw_ids)} '
                        f'coherent_front={len(coherent_disc_ids)} -> {len(raw_preserve_ids)} '
                        f'(tagged disc={len(discontinuity_ids)})')

                travel_limits = per_record_max_travel(
                    alignment_points,
                    base_max_travel=inclusion_travel,
                    discontinuity_ids=raw_preserve_ids,
                )
                # Soft weight floor unused while primary registration-weight bar is off.
                soft_weight_cutoff = None

                # Role + FieldMode classification (once per pass).
                finalize_cutoff_preview = float(transform_cutoff_value)
                finalize_travel_limits = per_record_max_travel(
                    alignment_points,
                    base_max_travel=float(settings.max_travel_for_finalization),
                    discontinuity_ids=soft_disc_ids,
                )
                classify_t0 = time.perf_counter()
                grid_n = max(1, len(alignment_points) + len(finalized_points))
                lock_fraction = float(len(finalized_points)) / float(grid_n)
                with _PHASE_TIMER.section_wall('classify'):
                    field_mode = classify_field(
                        alignment_points,
                        lock_fraction=lock_fraction,
                        max_travel=float(settings.max_travel_for_finalization),
                        travel_eps=float(finalize_settings.finalize_stability_epsilon_px),
                        locked_records=list(finalized_points.values()),
                    )
                    cand_ids = _lock_candidate_ids_preview(
                        alignment_points,
                        transform_cutoff=finalize_cutoff_preview,
                        max_travel=float(settings.max_travel_for_finalization),
                        per_record_max_travel=finalize_travel_limits if soft_disc_ids else None,
                        soft_weight_cutoff=None,
                        discontinuity_ids=soft_disc_ids,
                    )
                    field_suspect_ids = field_brand_identity_suspect_ids(
                        alignment_points,
                        field_mode=field_mode,
                        max_travel=float(settings.max_travel_for_finalization),
                        travel_eps=float(finalize_settings.finalize_stability_epsilon_px),
                        discontinuity_ids=discontinuity_ids if discontinuity_ids else None,
                    )
                    # Skip expensive ZNCC ROI re-extract when field already brands suspect.
                    zncc_cand_ids = cand_ids - field_suspect_ids
                zncc_t0 = time.perf_counter()
                with _PHASE_TIMER.section_wall('zncc_secondary'):
                    zncc_by_id = _compute_zncc_for_candidates(
                        alignment_points,
                        zncc_cand_ids,
                        stosTransform,
                        settings,
                        travel_eps=float(finalize_settings.finalize_stability_epsilon_px),
                        reference_pose=ring_reference_pose,
                    )
                zncc_s = time.perf_counter() - zncc_t0
                with _PHASE_TIMER.section_wall('classify'):
                    role_result = classify_roles(
                        alignment_points,
                        transform_cutoff=finalize_cutoff_preview,
                        max_travel=float(settings.max_travel_for_finalization),
                        per_record_max_travel=finalize_travel_limits if soft_disc_ids else None,
                        soft_weight_cutoff=None,
                        discontinuity_ids=soft_disc_ids if soft_disc_ids else None,
                        zncc_by_id=zncc_by_id,
                        low_content_ids=source_content_cache.low_content_ids,
                        field_mode=field_mode,
                        field_suspect_ids=field_suspect_ids,
                        travel_eps=float(finalize_settings.finalize_stability_epsilon_px),
                    )
                classify_s = time.perf_counter() - classify_t0
                lockable_ids = {
                    key for key, role in role_result.role_by_id.items() if role == Role.LOCKABLE
                }
                prettyoutput.Log(
                    f'field_mode={field_mode.name} '
                    f'reject={role_result.n_reject} free={role_result.n_free} '
                    f'lockable={role_result.n_lockable} '
                    f'identity_suspect={role_result.n_identity_suspect} '
                    f'peak_amb={role_result.n_peak_ambiguous} '
                    f'low_content={role_result.n_low_content} '
                    f'lock_cand={role_result.n_lock_cand} '
                    f'zncc_eval={role_result.n_zncc_eval} '
                    f'zncc_pass={role_result.n_zncc_pass} '
                    f'zncc_fail={role_result.n_zncc_fail} '
                    f'source_low_content_skip={len(source_content_cache.low_content_ids)} '
                    f'classify_s={classify_s:.3f} zncc_s={zncc_s:.3f}')

                if anchor_smooth_active:
                    # Soft-disc + unique large-travel keep raw peaks; ambiguous disc
                    # are gap-filled from locked anchors (avoids disc feedback loop).
                    mesh_alignment_points = smooth_peaks_from_locked_anchors(
                        finalized_points,
                        alignment_points,
                        stosTransform,
                        settings,
                        discontinuity_ids=raw_preserve_ids if raw_preserve_ids else None,
                    )
                    prettyoutput.Log(
                        f'Anchor-smooth mesh: {len(finalized_points)} locked seeds, '
                        f'{len(mesh_alignment_points)} cells in smoothed field '
                        f'(raw-preserve={len(raw_preserve_ids)})')
                    (updatedTransform, included_alignment_records, weight_distance_composite_scores) = (
                        _build_mesh_transform_or_keep(
                            mesh_alignment_points,
                            prior_transform=stosTransform,
                            fixed_points=None,
                        ))
                else:
                    # Exclude free points whose residual travel exceeds the inclusion travel bar.
                    # Weight-only inclusion previously folded meshes when ~1900 high-weight / ~60px-peak
                    # outliers reshaped the triangulation while locks stayed near 2% (228-229 Grid16).
                    # Disc / unique large-travel cells get a relaxed limit for mesh inclusion.
                    # (Soft-disc only — not all tagged discontinuities.)
                    mesh_alignment_points, n_travel_dropped = filter_records_for_mesh_inclusion(
                        alignment_points,
                        max_travel=inclusion_travel,
                        min_keep=0,
                        per_record_max_travel=travel_limits if raw_preserve_ids else None,
                    )
                    if n_travel_dropped > 0:
                        prettyoutput.Log(
                            f'Dropped {n_travel_dropped} free points from mesh inclusion '
                            f'(peak travel > limit); '
                            f'{len(mesh_alignment_points)} remain after travel filter')

                    # Drop REJECT roles from mesh (peak-ambiguous / low-content).
                    # min_keep=0: do not emergency-fill with rejects; _build_mesh_transform_or_keep
                    # retains the prior pose when fewer than three clear cells remain.
                    mesh_roles = [
                        role_result.role_by_id.get(
                            (int(rec.ID[0]), int(rec.ID[1])), Role.FREE)
                        for rec in mesh_alignment_points
                    ]
                    mesh_alignment_points, n_rej_dropped = exclude_reject_mesh_records(
                        mesh_alignment_points, mesh_roles, min_keep=0)
                    if n_rej_dropped > 0:
                        prettyoutput.Log(
                            f'Dropped {n_rej_dropped} REJECT free points from mesh inclusion; '
                            f'{len(mesh_alignment_points)} remain')

                    # No registration-weight bar: travel + REJECT already filtered;
                    # raw-preserve cells keep residuals via relaxed travel only.
                    (updatedTransform, included_alignment_records, weight_distance_composite_scores) = (
                        _build_mesh_transform_or_keep(
                            mesh_alignment_points,
                            prior_transform=stosTransform,
                            fixed_points=AlignRecordsToControlPoints(finalized_points.values()),
                        ))

                # Sparse inclusion (reject/travel soup, or post-residual wrap peaks)
                # must not replace a usable prior pose with a 3-point triangulation.
                preserve_post_residual = should_keep_prior_sparse_mesh(
                    residual_applied=coherent_residual_translated,
                    n_mesh=len(included_alignment_records),
                    n_grid=grid_n,
                    n_locks=len(finalized_points),
                    n_prior_points=_control_point_count(stosTransform),
                )
                if preserve_post_residual:
                    min_mesh = max(
                        int(MIN_MESH_ABS_AFTER_RESIDUAL),
                        int(float(MIN_MESH_FRAC_AFTER_RESIDUAL) * float(grid_n)),
                    )
                    prettyoutput.Log(
                        f'Keeping prior transform; mesh only has '
                        f'{len(included_alignment_records)} points (min {min_mesh})')
                    updatedTransform = stosTransform

                prettyoutput.Log(f'{len(included_alignment_records)} points included in updated transform after cutoff')

                finalize_percentile_this_pass = cutoff_percentile_this_pass
                finalize_cutoff_this_pass = float(np.percentile(polyfit_weights,  # type: ignore[arg-type]
                                                                finalize_percentile_this_pass))
                finalize_ema.add(finalize_cutoff_this_pass)

                if final_pass:
                    finalize_cutoff_this_pass = first_cutoff  # type: ignore[assignment]

                # Unlock stale locks that disagree with the updated mesh before adding new locks.
                unlocked_keys: list[tuple[int, int]] = []
                if updatedTransform is not None and len(finalized_points) > 0 and not legacy_finalize:
                    finalized_points, unlocked_keys = unlock_stale_finalized(
                        finalized_points, updatedTransform, finalize_settings)

                if legacy_finalize:
                    finalize_cutoff = float(np.percentile(polyfit_weights, 2.0))
                    new_finalized_points = legacy_finalize_mask(
                        alignment_points,
                        max_travel_distance=settings.max_travel_for_finalization,
                        polyfit_weights=polyfit_weights,
                        floor_percentile=2.0)
                    deferred_stability = 0
                    prettyoutput.Log(
                        f'Finalize cutoff this pass (legacy): 2% -> {finalize_cutoff}\n#####\n')
                else:
                    # Only Role.LOCKABLE cells may lock (PC-pass ∧ ZNCC-pass).
                    # Registration-weight bar is disabled (-inf); travel + Role/ZNCC gate.
                    finalize_cutoff = float(transform_cutoff_value)
                    eval_result = evaluate_finalize_candidates(
                        alignment_points,
                        transform_cutoff=finalize_cutoff,
                        settings=finalize_settings,
                        pass_index=i,
                        prior_candidates=finalize_candidates,
                        per_record_max_travel=finalize_travel_limits if soft_disc_ids else None,
                        soft_weight_cutoff=None,
                        discontinuity_ids=soft_disc_ids if soft_disc_ids else None,
                        lockable_ids=lockable_ids,
                    )
                    new_finalized_points = eval_result.lock_mask
                    finalize_candidates = eval_result.candidates
                    deferred_stability = eval_result.deferred_stability_count
                    prettyoutput.Log(
                        f'Finalize cutoff this pass: weight bar disabled '
                        f'(travel + Role/ZNCC)\n'
                        f'  rejected weight={eval_result.rejected_weight_count} '
                        f'travel={eval_result.rejected_travel_count} '
                        f'pass={eval_result.rejected_pass_count} '
                        f'ambiguous={eval_result.rejected_ambiguous_count} '
                        f'identity_suspect={eval_result.rejected_identity_suspect_count} '
                        f'deferred_stability={deferred_stability}\n#####\n')
            finalize_s = time.perf_counter() - finalize_t0

            new_finalized_alignments_list = list(
                filter(lambda index_item: new_finalized_points[index_item[0]], enumerate(alignment_points)))

            if first_pass_weight_distance_composite_scores is None:
                first_pass_weight_distance_composite_scores = weight_distance_composite_scores

            new_finalized_alignments_dict = {fp[1].ID: fp[1] for fp in new_finalized_alignments_list}
            new_finalization_count = len(new_finalized_alignments_dict)

            # remove finalized points from alignment_points
            non_final_alignment_points = list(filter(lambda r: r.ID not in new_finalized_alignments_dict, alignment_points))

            # Check previous finalizations to see if we can do better now
            (finalized_points, improved_alignments) = TryToImproveAlignments(updatedTransform,
                                                                             finalized_points,
                                                                             settings)

            # Bake new locks immediately (peak -> TargetPoint) so unlock/mesh see Adjusted==Target.
            baked_new_locks: AlignmentRecordDict = {}
            for key, rec in new_finalized_alignments_dict.items():
                baked_new_locks[key] = nornir_imageregistration.EnhancedAlignmentRecord(
                    rec.ID,
                    TargetPoint=rec.AdjustedTargetPoint,
                    SourcePoint=rec.SourcePoint,
                    peak=np.asarray((0, 0), dtype=np.float32),
                    weight=rec.weight,
                    angle=0,
                    flipped_ud=rec.flippedud,
                    peak_ratio=getattr(rec, 'peak_ratio', None),
                )
            finalized_points = {**finalized_points, **baked_new_locks}
            # Drop candidate tracking for cells that just locked.
            for key in baked_new_locks:
                finalize_candidates.pop(key, None)

            prettyoutput.Log(
                f"Pass {i} has locked {new_finalization_count} new points, "
                f"unlocked {len(unlocked_keys)}, deferred_stability {deferred_stability}; "
                f"{len(finalized_points)} of {len(updated_and_finalized_alignment_points)} are locked")

            prettyoutput.Log(
                f"  Improved {len(improved_alignments)} finalized points using latest transform")

            progress_reporter.on_pass_locked(
                i,
                len(finalized_points),
                len(updated_and_finalized_alignment_points),
            )

            if SavePlots:
                np.savez(os.path.join(outputDir,  # type: ignore[arg-type]
                                      f'weight_distance_composite_scores_pass{i}.npz'),
                         updated_and_finalized_weights_distance=updated_and_finalized_weights_distance,
                         weight_distance_composite_scores=weight_distance_composite_scores,
                         )
                mesh_scores = np.asarray(weight_distance_composite_scores)
                if mesh_scores.size == 0 or mesh_scores.shape[0] == 0:
                    prettyoutput.Log(
                        f'Pass {i}: skipping percentile plot; no mesh inclusion scores')
                else:
                    percentile_filename = os.path.join(outputDir, f'percentile_pass{i}.svg')  # type: ignore[arg-type]
                    nornir_imageregistration.views.plot_percentiles(
                        mesh_scores[:, 0],
                        percentile_filename,
                        title=f"Value at percentile",
                        horz_line_pos_list=[(diagnostic_inflection_value,
                                             {'label': 'Diagnostic inflection (unused)',
                                              'color': 'green'}),
                                            (finalize_cutoff,
                                             {'label': 'Finalize Cutoff',
                                              'color': 'brown'})])

                histogram_filename = os.path.join(outputDir, f'weight_histogram_pass{i}.svg')  # type: ignore[arg-type]
                nornir_imageregistration.views.PlotWeightHistogram(alignment_points, filename=histogram_filename,
                                                                   transform_cutoff=transform_cutoff_percentile / 100.0,
                                                                   finalize_cutoff=finalize_percentile_this_pass / 100.0,
                                                                   line_pos_list=[diagnostic_inflection_value,
                                                                                  finalize_cutoff_this_pass],
                                                                   title=f"Histogram of Weights, pass #{i}")

                vector_field_filename = os.path.join(outputDir, f'Vector_field_pass{i}.svg')  # type: ignore[arg-type]
                nornir_imageregistration.views.PlotPeakList(non_final_alignment_points, list(finalized_points.values()),
                                                            vector_field_filename,
                                                            ylim=(0, settings.target_image.shape[1]),
                                                            xlim=(0, settings.target_image.shape[0]))
                # vector_field_filename = os.path.join(outputDir, f'Vector_field_pass_delta{i}.png')
                # nornir_imageregistration.views.PlotPeakList(alignment_points, list(finalized_points.values()),
                #                                             vector_field_filename,
                #                                             ylim=(0, settings.target_image.shape[1]),
                #                                             xlim=(0, settings.target_image.shape[0]),
                #                                             attrib='PSDDelta')

            # Always merge locks into the pass control set (previously only when TryToImprove
            # reported improvements, which dropped fixed anchors on many passes).
            smoothed_by_id: dict[tuple[int, int], object] = {}
            if anchor_smooth_active:
                smoothed_pass_records = smooth_peaks_from_locked_anchors(
                    finalized_points,
                    list({a.ID: a for a in included_alignment_records}.values()),
                    updatedTransform,
                    settings,
                    discontinuity_ids=raw_preserve_ids if raw_preserve_ids else None,
                )
                combined_records_this_pass = {rec.ID: rec for rec in smoothed_pass_records}
                smoothed_by_id = combined_records_this_pass
            else:
                combined_records_this_pass = {a.ID: a for a in included_alignment_records}
                combined_records_this_pass.update(finalized_points)

            if pass_diagnostics_enabled(SavePlots) and outputDir is not None:
                included_ids = {(int(r.ID[0]), int(r.ID[1])) for r in included_alignment_records}
                travel_dropped_ids: set[tuple[int, int]] = set()
                if n_travel_dropped > 0:
                    kept_ids = {(int(r.ID[0]), int(r.ID[1])) for r in mesh_alignment_points}
                    travel_dropped_ids = {
                        (int(r.ID[0]), int(r.ID[1])) for r in alignment_points
                        if (int(r.ID[0]), int(r.ID[1])) not in kept_ids
                    }
                pair_label = os.path.basename(outputDir.rstrip(os.sep)) or 'stos'
                diag_rows = build_pass_diagnostic_rows(
                    alignment_points=alignment_points,
                    finalized=finalized_points,
                    included_ids=included_ids,
                    travel_dropped_ids=travel_dropped_ids,
                    unlocked_ids=set(unlocked_keys),
                    transform_cutoff=float(diagnostic_inflection_value),
                    finalize_candidates=finalize_candidates,
                    transform=updatedTransform,
                    discontinuity_ids=discontinuity_ids,
                    smoothed_by_id=smoothed_by_id,
                    role_by_id={k: int(v) for k, v in role_result.role_by_id.items()},
                    reject_reason_by_id={
                        (int(rec.ID[0]), int(rec.ID[1])): int(reason)
                        for rec, reason in zip(alignment_points, role_result.reject_reasons)
                    },
                    zncc_by_id=zncc_by_id,
                    lock_candidate_ids={
                        (int(rec.ID[0]), int(rec.ID[1]))
                        for rec, ok in zip(alignment_points, role_result.lock_candidate)
                        if bool(ok)
                    },
                    source_content_by_id=dict(source_content_cache.as_mapping()),
                )
                written_diag = write_pass_diagnostics(
                    outputDir,
                    i,
                    diag_rows,
                    pair_label=pair_label,
                    write_heatmaps=bool(SavePlots),
                )
                cell_history.append_pass(i, diag_rows)
                diagnostics_tables_s = float(written_diag.get('tables_s', 0.0))
                diagnostics_heatmaps_s = float(written_diag.get('heatmaps_s', 0.0))
                _PHASE_TIMER.add('diagnostics_tables', diagnostics_tables_s)
                if diagnostics_heatmaps_s > 0:
                    _PHASE_TIMER.add('diagnostics_heatmaps', diagnostics_heatmaps_s)

            pass_wall_s = time.perf_counter() - pass_t0
            prettyoutput.Log(
                f'RefineTransform pass {i} pass_wall_s={pass_wall_s:.2f} '
                f'measure_s={measure_s:.2f} finalize_s={finalize_s:.2f} '
                f'diagnostics_tables_s={diagnostics_tables_s:.2f} '
                f'diagnostics_heatmaps_s={diagnostics_heatmaps_s:.2f}')
            # Log after classify / zncc_secondary so PHASE_TIMING buckets are non-zero.
            _log_phase_breakdown(f'RefineTransform pass {i}', pass_phase_baseline)

            if len(combined_records_this_pass) > 2:
                # Recompute preserve against current locks/mesh in case finalize
                # path already kept stosTransform — still skip sparse rebuild.
                preserve_end = should_keep_prior_sparse_mesh(
                    residual_applied=coherent_residual_translated,
                    n_mesh=len(combined_records_this_pass),
                    n_grid=max(1, len(alignment_points) + len(finalized_points)),
                    n_locks=len(finalized_points),
                    n_prior_points=_control_point_count(stosTransform),
                )
                if preserve_post_residual or preserve_end:
                    prettyoutput.Log(
                        f'Keeping prior transform for next round; '
                        f'combined mesh would have {len(combined_records_this_pass)} points')
                    updatedTransform = stosTransform
                else:
                    prettyoutput.Log(
                        f'Building transform for next round with {len(included_alignment_records)} free '
                        f'and {len(finalized_points)} finalized points')
                    updatedTransform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
                        AlignRecordsToControlPoints(combined_records_this_pass.values()))  # type: ignore[arg-type]

            report_pass_transform(
                progress_callback,
                updatedTransform,
                i,
                settings.num_iterations)

            if SaveImages:
                # InputStos.Save(os.path.join(outputDir, "UpdatedTransform_pass{0}.stos".format(i)))

                warpedToFixedImage = nornir_imageregistration.assemble.TransformStos(updatedTransform,
                                                                                     fixedImage=settings.target_image,
                                                                                     warpedImage=settings.source_image)

                Delta = warpedToFixedImage - settings.source_image  # type: ignore[operator]
                ComparisonImage = np.abs(Delta)
                if ComparisonImage.max() != 0:
                    ComparisonImage = ComparisonImage / ComparisonImage.max()

                # nornir_imageregistration.SaveImage(os.path.join(outputDir, f'delta_pass{i}.png'), ComparisonImage, bpp=8)
                # nornir_imageregistration.SaveImage(os.path.join(outputDir, f'image_pass{i}.png'), warpedToFixedImage, bpp=8)
                pool = nornir_pools.GetGlobalThreadPool()
                pool.add_task(f'delta_pass{i}.png', nornir_imageregistration.SaveImage,
                              os.path.join(outputDir, f'delta_pass{i}.png'), np.copy(ComparisonImage), bpp=8)  # type: ignore[call-overload, arg-type]
                pool.add_task(f'image_pass{i}.png', nornir_imageregistration.SaveImage,
                              os.path.join(outputDir, f'image_pass{i}.png'), np.copy(warpedToFixedImage), bpp=8)  # type: ignore[call-overload, arg-type]

            i += 1

            if final_pass:
                break

            grew_cell_size = False
            if pass_found_no_usable_alignments(
                    role_result, n_measured=len(alignment_points)):
                grew_cell_size = _grow_refine_cell_size_after_failure(
                    settings,
                    source_content_cache,
                    pass_index=i - 1,
                    final_pass=final_pass)
            elif pass_found_registrations(
                    role_result, n_measured=len(alignment_points)):
                _restore_refine_cell_size_after_success(
                    settings,
                    source_content_cache,
                    requested_cell_size,
                    pass_index=i - 1,
                    final_pass=final_pass)

            if grew_cell_size:
                stosTransform = updatedTransform
                continue

            # No "next pass is the last" check here. `i` was already incremented above,
            # so it holds the *next* pass number (the cell-size calls just above pass
            # `pass_index=i - 1` for exactly that reason). Testing num_iterations - 1
            # therefore declared the final pass one pass too early and refine stopped a
            # pass short: with cell_size 256 and num_iterations=4 it ran passes 1-3 only.
            # The top of the loop already sets final_pass when `i == num_iterations`, and
            # it does so on every path -- including the `continue` above, which this block
            # is skipped by.

            # If we've locked 10% of the points and have not locked any new ones we are done
            if len(finalized_points) > len(updated_and_finalized_alignment_points) * 0.1 and new_finalization_count == 0:
                final_pass = True

            # If we've locked 90% of the points we are done
            if len(finalized_points) > len(updated_and_finalized_alignment_points) * 0.9:
                final_pass = True

            if len(finalized_points) >= len(updated_and_finalized_alignment_points):
                break  # There are no more points to align, everything is finalized

            stosTransform = updatedTransform

            # Make one more pass to see if we can improve finalized points
        # Todo: This code remained untouched after an optimization pass.  I think it would be worth examining whether it can be improved.
        # if len(finalized_points) >= 3:
        #     final_transform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
        #         AlignRecordsToControlPoints(finalized_points.values()))
        # else:
        #     final_transform = updatedTransform
        final_transform = stosTransform

        final_anchor_smooth_active = should_use_anchor_smooth_mesh(finalized_points, settings)
        if final_anchor_smooth_active:
            final_disc_ids: set[tuple[int, int]] = set()
            final_soft_disc_ids: set[tuple[int, int]] = set()
            final_raw_preserve_ids: set[tuple[int, int]] = set()
            if sharp_warps_enabled() and combined_records_this_pass:
                final_disc_ids = tag_discontinuities(
                    list(combined_records_this_pass.values()),
                    max_travel=float(settings.max_travel_for_finalization),
                    stable_ids=None,
                )
                final_soft_disc_ids = soft_discontinuity_ids(
                    list(combined_records_this_pass.values()), final_disc_ids)
                final_unique_raw = unique_large_travel_raw_preserve_ids(
                    list(combined_records_this_pass.values()),
                    max_travel=float(settings.max_travel_for_finalization),
                )
                final_coherent_disc = coherent_discontinuity_raw_preserve_ids(
                    list(combined_records_this_pass.values()),
                    final_disc_ids,
                    max_travel=float(settings.max_travel_for_finalization),
                )
                final_raw_preserve_ids = (
                    set(final_soft_disc_ids) | final_unique_raw | final_coherent_disc)
            final_mesh_records = smooth_peaks_from_locked_anchors(
                finalized_points,
                list(combined_records_this_pass.values()),
                stosTransform,
                settings,
                discontinuity_ids=final_raw_preserve_ids if final_raw_preserve_ids else None,
            )
            final_control_records = {rec.ID: rec for rec in final_mesh_records}
            prettyoutput.Log(
                f'Final anchor-smooth mesh: {len(finalized_points)} locked seeds, '
                f'{len(final_mesh_records)} cells '
                f'(raw-preserve={len(final_raw_preserve_ids)}, '
                f'tagged disc={len(final_disc_ids)})')
        else:
            final_control_records = combined_records_this_pass

        (nudged_final_points, nudged_point_keys) = TryToImproveAlignments(
            stosTransform, final_control_records, settings)
        prettyoutput.Log(
            f'Final tuning of points adjusted {len(nudged_point_keys)} of '
            f'{len(final_control_records)} points')

        # Return a transform built from control points, unless a post-residual
        # sparse set would discard TranslateFixed. Use last-pass FOV grid size for
        # lock_frac / min_mesh (not len(final_control_records) — that is ~12 when
        # preserve already kept a sparse combined set and falsely raises lock_frac).
        final_grid_n = max(1, len(alignment_points) + len(finalized_points))
        n_nudged = len(nudged_final_points)
        n_stos_pts = _control_point_count(stosTransform)
        n_input_pts = _control_point_count(refine_input_transform)
        preserve_final = should_keep_prior_sparse_mesh(
            residual_applied=coherent_residual_translated,
            n_mesh=n_nudged,
            n_grid=final_grid_n,
            n_locks=len(finalized_points),
            n_prior_points=n_stos_pts,
        )
        if preserve_final:
            prettyoutput.Log(
                f'Keeping prior transform as final; control set only has '
                f'{n_nudged} points')
            final_transform = stosTransform
        elif should_keep_prior_sparse_mesh(
                residual_applied=coherent_residual_translated,
                n_mesh=n_nudged,
                n_grid=final_grid_n,
                n_locks=len(finalized_points),
                n_prior_points=n_input_pts,
        ):
            prettyoutput.Log(
                f'Keeping input transform as final; nudged set only has '
                f'{n_nudged} points, pass mesh {n_stos_pts}')
            final_transform = refine_input_transform
        elif n_nudged >= 3:
            min_keep = max(
                int(MIN_MESH_ABS_AFTER_RESIDUAL),
                int(float(MIN_MESH_FRAC_AFTER_RESIDUAL) * float(final_grid_n)),
            )
            if n_nudged < min_keep and n_stos_pts > n_nudged:
                prettyoutput.Log(
                    f'Keeping pass transform ({n_stos_pts} points); '
                    f'nudged final only has {n_nudged} points (min {min_keep})')
                final_transform = stosTransform
            else:
                final_transform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
                    AlignRecordsToControlPoints(nudged_final_points.values()))  # type: ignore[arg-type]

        progress_reporter.on_complete()
        report_pass_transform(
            progress_callback,
            final_transform,
            settings.num_iterations,
            settings.num_iterations,
            label="Refine complete")

        pair_wall_s = time.perf_counter() - pair_t0
        prettyoutput.Log(f'RefineTransform pair_wall_s={pair_wall_s:.2f}')
        _log_phase_breakdown('RefineTransform total', {})

        if outputDir is not None and cell_history.pass_index:
            hist_path = cell_history.write_npz(outputDir)
            if hist_path is not None:
                prettyoutput.Log(f'Wrote cell history NPZ: {hist_path}')
            if SavePlots:
                pair_label = os.path.basename(outputDir.rstrip(os.sep)) or 'stos'
                written_hist = write_cell_history_plots(
                    outputDir,
                    cell_history.as_arrays(),
                    pair_label=pair_label,
                    travel_eps=float(finalize_settings.finalize_stability_epsilon_px),
                )
                for name, path in written_hist.items():
                    prettyoutput.Log(f'Wrote cell history plot ({name}): {path}')

        return final_transform
    finally:
        settings.cell_size = original_cell_size


def _RefineGridPointsForTwoImages(transform: nornir_imageregistration.transforms.ITransform,
                                  finalized: AlignmentRecordDict,
                                  settings: nornir_imageregistration.settings.GridRefinement,
                                  source_content_cache: SourceContentCache | None = None,
                                  cancel_event: threading.Event | None = None,
                                  progress_callback: ProgressCallback | None = None,
                                  reference_pose: RingReferencePose | None = None) -> list[
    nornir_imageregistration.EnhancedAlignmentRecord]:
    """
    Build a refinement grid, remove masked/finalized cells, and align remaining cells.
    """

    with _PHASE_TIMER.section('grid_build'):
        # Regular grid on the source image (SourcePoints), then target positions via Transform(SourcePoints).

        grid_data = nornir_imageregistration.grid_subdivision.CenteredGridDivision(settings.source_image.shape,  # type: ignore[attr-defined]
                                                                                   cell_size=settings.cell_size,
                                                                                   grid_spacing=settings.grid_spacing,
                                                                                   transform=transform)
        # grid_data = nornir_imageregistration.ITKGridDivision(settings.source_image.shape,
        #                                                                       cell_size=settings.cell_size,
        #                                                                       grid_spacing=settings.grid_spacing,
        #                                                                       transform=transform)

        # Remove finalized points from refinement consideration
        allow_empty = finalized is not None and len(finalized) > 0
        if allow_empty:
            not_finalized = [tuple(grid_data.coords[i, :]) not in finalized for i in range(grid_data.coords.shape[0])]
            valid = np.asarray(not_finalized, bool)
            if grid_data.RemoveMaskedPoints(
                    valid,
                    allow_empty=True,
                    context=f"excluding {len(finalized)} already-finalized points") == 0:
                prettyoutput.Log(
                    f"All grid points already finalized ({len(finalized)}); nothing left to measure")
                return []

        grid_data.FilterOutofBoundsSourcePoints(settings.source_image.shape, allow_empty=allow_empty)
        grid_data.RemoveCellsUsingSourceImageMask(settings.source_mask, settings.min_unmasked_area,
                                                  allow_empty=allow_empty)
        # nornir_imageregistration.views.grid_data.PlotGridPositionsAndMask(grid_data.SourcePoints, source_mask, OutputFilename=None)

        if grid_data.num_points == 0:
            if allow_empty:
                prettyoutput.Log(
                    "No unfinalized points remain after source-mask / bounds filtering; "
                    f"continuing with {len(finalized)} locked points")
                return []
            # There is nothing to refine, perhaps the image is too small for the grid cell size?
            msg = (
                "No points meet criteria for grid refinement after source-mask / "
                f"bounds filtering (cell_size={np.asarray(settings.cell_size).tolist()}, "
                f"grid_spacing={np.asarray(settings.grid_spacing).tolist()}, "
                f"min_unmasked_area={float(settings.min_unmasked_area):g})"
            )
            prettyoutput.LogErr(msg)
            raise ValueError(msg)

        # Sticky source LOW_CONTENT: never remasure those grid IDs this refine.
        if source_content_cache is not None and grid_data.num_points > 0:
            with _PHASE_TIMER.section_wall('low_content_gate'):
                content_ok = np.ones(grid_data.num_points, dtype=bool)
                uncached_indices: list[int] = []
                uncached_keys: list[tuple[int, int]] = []
                for i in range(grid_data.num_points):
                    key = tuple(int(v) for v in grid_data.coords[i, :])
                    if source_content_cache.is_low_content(key):
                        content_ok[i] = False
                        continue
                    if source_content_cache.get(key) is not None:
                        continue
                    uncached_indices.append(i)
                    uncached_keys.append(key)
                if uncached_indices:
                    pts = grid_data.SourcePoints[uncached_indices]
                    stds = crop_source_cell_stds_batched(
                        settings.source_image, pts, settings.cell_size)
                    for j, i in enumerate(uncached_indices):
                        if not source_content_cache.remember(uncached_keys[j], float(stds[j])):
                            content_ok[i] = False
                if not np.all(content_ok):
                    n_skip = int(np.count_nonzero(~content_ok))
                    grid_data.RemoveMaskedPoints(
                        content_ok,
                        allow_empty=True,
                        context=f"excluding {n_skip} source-low-content cells")
                    if grid_data.num_points == 0:
                        prettyoutput.Log(
                            f"All remaining cells are source-low-content "
                            f"(skip={len(source_content_cache.low_content_ids)}); "
                            f"continuing with {len(finalized)} locked points")
                        return []

        grid_data.PopulateTargetPoints(transform)
        remaining = grid_data.RemoveCellsUsingTargetImageMask(
            settings.target_mask, settings.min_unmasked_area, allow_empty=allow_empty)
        if remaining == 0:
            if allow_empty:
                prettyoutput.Log(
                    "No unfinalized points remain after target-mask filtering; "
                    f"continuing with {len(finalized)} locked points")
                return []
            # RemoveMaskedPoints already raised when allow_empty is False
            raise ValueError("No points meet criteria for grid refinement after target-mask filtering")

        coords = [tuple(row) for row in grid_data.coords]
        # Cell coordinates are host metadata for ROI origins and alignment records.
        source_points = nornir_imageregistration.EnsureNumpyArray(
            grid_data.SourcePoints, dtype=np.float64)
        target_points = nornir_imageregistration.EnsureNumpyArray(
            grid_data.TargetPoints, dtype=np.float64)

    return _RefinePointsForTwoImages(
        transform, coords, source_points, target_points, settings,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        reference_pose=reference_pose)


def _RefinePointsForTwoImages(transform: nornir_imageregistration.transforms.ITransform,
                              keys: list[tuple[int, int]],
                              sourcePoints: np.ndarray,
                              targetPoints: np.ndarray,
                              settings: nornir_imageregistration.settings.GridRefinement,
                              cancel_event: threading.Event | None = None,
                              progress_callback: ProgressCallback | None = None,
                              reference_pose: RingReferencePose | None = None) -> list[
    nornir_imageregistration.EnhancedAlignmentRecord]:
    """
    Register corresponding source/target neighborhoods for each control-point key.
    """

    if len(keys) != targetPoints.shape[0]:
        raise ValueError("keys must have equal number of entries as points")

    nPoints = len(keys)

    pool = get_runtime_config(refresh=True).pool_for_cell_tasks(
        settings.cupy_processing)
    # pool = nornir_pools.GetGlobalThreadPool()
    tasks = list()
    alignment_records = list()

    with _PHASE_TIMER.section('approx_rigid'):
        rigid_transforms = ApproximateRigidTransformBySourcePoints(
            input_transform=transform, source_points=sourcePoints,
            cell_size=settings.cell_size,
            reference_pose=reference_pose,
            ring_scale_fraction_max=settings.ring_scale_fraction_max,
            ring_angle_max_degrees=settings.ring_angle_max_degrees,
            ring_allow_flip_change=settings.ring_allow_flip_change)

    if (_use_batched_vertex_measurement()
            and _angles_are_translation_only(settings.angles_to_search)
            and nPoints > 0):
        batched = _attempt_align_points_translation_batched(
            keys=keys,
            source_points=sourcePoints,
            target_points=targetPoints,
            rigid_transforms=rigid_transforms,
            settings=settings)
        # None means the batched path could not run, so fall through and measure
        # serially. An empty list means it ran and found nothing alignable, which is
        # an answer: re-measuring with a different peak finder would make the
        # control points depend on whether the batched result came back empty.
        if batched is not None:
            return batched

    if settings.single_thread_processing:
        target_image = settings.target_image
        source_image = settings.source_image
        for i in range(nPoints):
            check_cancelled(cancel_event)
            report_progress(
                progress_callback,
                i + 1,
                nPoints,
                f"Align cell {i + 1}/{nPoints}")
            targetPoint = targetPoints[i, :]
            sourcePoint = sourcePoints[i, :]
            key = keys[i]
            arecord = AttemptAlignPoint(
                transform=rigid_transforms[i],
                targetImage=target_image,
                sourceImage=source_image,
                target_image_stats=settings.target_image_stats,
                source_image_stats=settings.source_image_stats,
                target_controlpoint=targetPoint,
                alignmentArea=settings.cell_size,
                anglesToSearch=settings.angles_to_search,
                min_alignment_overlap=settings.min_alignment_overlap)
            if arecord is None:
                continue

            erec = nornir_imageregistration.EnhancedAlignmentRecord(
                ID=key,
                TargetPoint=targetPoint,
                SourcePoint=sourcePoint,
                peak=arecord.peak,
                weight=arecord.weight,
                angle=arecord.angle,
                flipped_ud=arecord.flippedud,
                peak_ratio=arecord.peak_ratio)

            if nornir_imageregistration.in_debug_mode():
                erec.TargetROI = arecord.TargetROI  # type: ignore[attr-defined]
                erec.SourceROI = arecord.SourceROI  # type: ignore[attr-defined]
                source_roi = erec.SourceROI  # type: ignore[attr-defined]
                xp = cp.get_array_module(source_roi)
                erec.TranslatedSourceROI = nornir_imageregistration.CropImage(
                    source_roi,
                    int(np.floor(-erec.peak[1])),
                    int(np.floor(-erec.peak[0])),
                    source_roi.shape[1],
                    source_roi.shape[0],
                    cval=float(xp.median(source_roi)))

            alignment_records.append(erec)

        return alignment_records

    for i in range(nPoints):
        check_cancelled(cancel_event)
        report_progress(
            progress_callback,
            i + 1,
            nPoints,
            f"Queue cell {i + 1}/{nPoints}")
        targetPoint = targetPoints[i, :]
        sourcePoint = sourcePoints[i, :]
        key = keys[i]
        # So... the way I'm handling refine is backwards.  I'm not sure how much it matters.
        # I'm passing the target point, but the point that is fixed in the transform I am building is the source point.
        # So the transform runs an inverse transform to obtain the source point, which may be slightly off.
        AlignTask = pool.add_task(f"Align {key}",
                                  AttemptAlignPoint,
                                  transform=rigid_transforms[i],
                                  targetImage=settings.target_image_meta,  # Send the shared file to the task
                                  sourceImage=settings.source_image_meta,  # Send the shared file to the task
                                  # settings.target_mask,
                                  # settings.source_mask,
                                  target_image_stats=settings.target_image_stats,
                                  source_image_stats=settings.source_image_stats,
                                  target_controlpoint=targetPoint,
                                  alignmentArea=settings.cell_size,
                                  anglesToSearch=settings.angles_to_search,
                                  min_alignment_overlap=settings.min_alignment_overlap)
        # AlignTask = StartAttemptAlignPoint(pool,
        # f"Align {key}",
        # rigid_transforms[i],

        if AlignTask is None:
            continue

        AlignTask.ID = i  # type: ignore[attr-defined]
        AlignTask.key = key  # type: ignore[attr-defined]
        tasks.append(AlignTask)

    for t_index, t in enumerate(tasks):
        check_cancelled(cancel_event)
        report_progress(
            progress_callback,
            t_index + 1,
            max(1, len(tasks)),
            f"Collect cell {t_index + 1}/{len(tasks)}")
        arecord = t.wait_return()
        if arecord is None:
            continue

        erec = nornir_imageregistration.EnhancedAlignmentRecord(ID=t.key,
                                                                TargetPoint=targetPoints[t.ID, :],
                                                                SourcePoint=sourcePoints[t.ID, :],
                                                                peak=arecord.peak,
                                                                weight=arecord.weight,
                                                                angle=arecord.angle,
                                                                flipped_ud=arecord.flippedud,
                                                                peak_ratio=arecord.peak_ratio)

        if nornir_imageregistration.in_debug_mode():
            erec.TargetROI = arecord.TargetROI  # type: ignore[attr-defined]
            erec.SourceROI = arecord.SourceROI  # type: ignore[attr-defined]
            source_roi = erec.SourceROI  # type: ignore[attr-defined]
            xp = cp.get_array_module(source_roi)
            erec.TranslatedSourceROI = nornir_imageregistration.CropImage(
                source_roi,
                int(np.floor(-erec.peak[1])),
                int(np.floor(-erec.peak[0])),
                source_roi.shape[1],
                source_roi.shape[0],
                cval=float(xp.median(source_roi)))

        # erec.TargetPSDScore = nornir_imageregistration.image_stats.ScoreImageWithPowerSpectralDensity(t.TargetROI)
        # erec.SourcePSDScore = nornir_imageregistration.image_stats.ScoreImageWithPowerSpectralDensity(t.SourceROI)

        # erec.PSDDelta = abs(erec.TargetPSDScore - erec.SourcePSDScore)
        # erec.PSDDelta = (erec.TargetROI - np.mean(erec.TargetROI.flat)) - (
        #        erec.SourceROI - np.mean(erec.SourceROI.flat))
        # erec.PSDDelta = np.sum(np.abs(erec.PSDDelta))
        # erec.CalculatedWarpedPoint = Transform.InverseTransform(erec.AdjustedTargetPoint).reshape(2)
        # arecord.ID = (iRow, iCol)
        # arecord.TargetPoint = t.TargetPoint
        # arecord.WarpedPoint = t.WarpedPoint
        # arecord.AdjustedWarpedPoint = t.WarpedPoint + arecord.peak

        alignment_records.append(erec)
    #
    #     del shared_warped_image
    #     del shared_fixed_image

    return alignment_records
    # Cull the worst of the alignment records

    # Build a new transform using our alignment points
    # peaks = np.list(map(lambda a: a.peak, alignment_records))
    # updatedTransform = _PeakListToTransform(alignment_records)

    # Re-run the loop

    # return updatedTransform


def AlignRecordsToControlPoints(
        alignment_records: AlignmentRecordList) -> NDArray[np.floating]:
    """
    Convert alignment records into ``[target_y, target_x, source_y, source_x]`` pairs.

    Host-only: SciPy Qhull / file I/O. Point pairs feed CPU mesh constructors.
    """
    records = list(alignment_records)
    if not records:
        return np.zeros((0, 4), dtype=np.float64)

    SourcePoints = np.asarray(
        [nornir_imageregistration.EnsureNumpyArray(a.SourcePoint) for a in records])
    TargetPoints = np.asarray(
        [nornir_imageregistration.EnsureNumpyArray(a.AdjustedTargetPoint) for a in records])

    PointPairs = np.hstack((TargetPoints, SourcePoints))
    return PointPairs


def _control_point_count(transform: nornir_imageregistration.ITransform) -> int:
    """Return the number of control pairs on *transform*, or 0 for rigid poses."""
    points = getattr(transform, 'points', None)
    if points is None:
        return 0
    host = nornir_imageregistration.EnsureNumpyArray(points)
    if host.ndim != 2 or host.shape[0] == 0:
        return 0
    return int(host.shape[0])


def _fixed_point_count(fixed_points: NDArray | None) -> int:
    """Return the number of fixed control pairs, treating empty arrays as zero."""
    if fixed_points is None:
        return 0
    if not isinstance(fixed_points, np.ndarray):
        raise ValueError("fixed_points must be an ndarray")
    if fixed_points.size == 0:
        return 0
    if fixed_points.ndim != 2 or fixed_points.shape[1] != 4:
        raise ValueError("fixed_points must have shape (N,4)")
    return int(fixed_points.shape[0])


def _build_mesh_transform_or_keep(
        alignment_records: AlignmentRecordList,
        *,
        prior_transform: nornir_imageregistration.ITransform,
        fixed_points: NDArray | None = None,
) -> tuple[
    nornir_imageregistration.ITransform,
    list[nornir_imageregistration.EnhancedAlignmentRecord],
    NDArray[np.floating]]:
    """Build a mesh transform, or keep *prior_transform* when too few points exist.

    Prevents hard failures when residual recovery / travel filters leave fewer
    than three control points for triangulation.
    """
    records = list(alignment_records)
    n_fixed = _fixed_point_count(fixed_points)
    if len(records) + n_fixed < 3:
        prettyoutput.Log(
            f'Insufficient points for mesh update '
            f'({len(records)} alignments + {n_fixed} fixed); keeping prior transform')
        if records:
            scores = _alignment_records_to_composite_scores(records)
        else:
            scores = np.zeros((0, 3), dtype=np.float64)
        return prior_transform, records, scores

    return _PeakListToTransform(
        records,
        WeightMethod.Registration,  # type: ignore[arg-type]
        fixed_points,
        percentile=None,
        cutoff=float('-inf'))


def _alignment_records_to_composite_scores(
        alignment_records: AlignmentRecordList,
        max_distance: float | None = None) -> NDArray[np.floating]:
    """
    Compute per-record registration, travel-distance, and composite inclusion scores.
    """

    if len(alignment_records) == 0:
        raise ValueError("No alignment records to calculate composite scores")

    weights_distance = np.asarray(list(map(lambda a: (a.weight, np.sqrt(a.peak.dot(a.peak))), alignment_records)))

    if max_distance is None:
        max_weight_distance = np.max(weights_distance, 0)
    else:
        max_weight_distance = np.asarray((1, max_distance))

    # I don't want a random near zero travel distance accidentally reducing a bad alignment score, so the
    # minimum travel distance is 1 for calculating the distance weight 
    floor_distances = np.maximum(1, weights_distance[:, 1])
    composite_weight = (max_weight_distance[0] - weights_distance[:, 0]) * np.sqrt(floor_distances)
    weights_distance = np.hstack((weights_distance, composite_weight.reshape((len(composite_weight), 1))))
    return weights_distance


def _PeakListToTransform(alignment_records: AlignmentRecordList,
                         weight_method: WeightMethod,
                         fixed_points: NDArray | None = None,
                         percentile: float | None = None,
                         cutoff: float | None = None) -> tuple[
    nornir_imageregistration.ITransform, list[nornir_imageregistration.EnhancedAlignmentRecord], NDArray[np.floating]]:
    """
    Build a mesh transform from cutoff-filtered alignment records and fixed points.
    """
    num_fixed = 0
    if fixed_points is not None:
        if not isinstance(fixed_points, np.ndarray):
            raise ValueError("fixed_points must be an ndarray")

        num_fixed = _fixed_point_count(fixed_points)

    num_alignments = len(alignment_records)
    if num_alignments == 0:
        raise ValueError("Need at least one new alignment_record to improve a transform")

    if num_alignments + num_fixed < 3:
        raise ValueError(
            f"Need at least three points to make a transform.  Got {len(alignment_records)} alignments and {num_fixed} fixed points")

    # TargetPoints = np.asarray(list(map(lambda a: a.TargetPoint, alignment_records)))
    OriginalSourcePoints = np.asarray(list(map(lambda a: a.SourcePoint, alignment_records)))
    AdjustedTargetPoints = np.asarray(list(map(lambda a: a.AdjustedTargetPoint, alignment_records)))
    # AdjustedWarpedPoints = np.asarray(list(map(lambda a: a.AdjustedWarpedPoint, alignment_records)))
    # CalculatedWarpedPoints = np.asarray(list(map(lambda a: a.CalculatedWarpedPoint, alignment_records)))

    # With Weights big numbers are good and with peak distance generally small numbers are good.
    # To merge these scores I invert the weights to subtract them from the max weight, then multiply by distance

    weights_distance = _alignment_records_to_composite_scores(alignment_records)
    composite_score = weights_distance[:, weight_method]
    # WarpedPeaks = AdjustedWarpedPoints - OriginalSourcePoints

    if cutoff is None:
        cutoff = float(np.max(composite_score))
        if percentile is not None:
            cutoff = float(np.percentile(composite_score, percentile))

    valid_indices = composite_score >= cutoff

    # Todo: Check that we have at least three points

    valid_target_points = AdjustedTargetPoints[valid_indices, :]
    valid_source_points = OriginalSourcePoints[valid_indices, :]

    if not np.array_equiv(valid_target_points.shape,
                          Triangulation.RemoveDuplicateControlPoints(valid_target_points).shape):
        raise Exception("Duplicate fixed points detected")

    if not np.array_equiv(valid_source_points.shape,
                          Triangulation.RemoveDuplicateControlPoints(valid_source_points).shape):
        raise Exception("Duplicate warped points detected")

    # See if we have enough points to build a transform.  If not include top scoring points until we have a transform
    if valid_target_points.shape[0] + num_fixed < 3:
        num_needed = 3 - num_fixed
        sorted_composite_indices = np.argsort(composite_score)
        # Registration scores: higher is better — take the strongest alignments.
        top_alignment_indices = sorted_composite_indices[-num_needed:]
        valid_target_points = AdjustedTargetPoints[top_alignment_indices, :]
        valid_source_points = OriginalSourcePoints[top_alignment_indices, :]
        prettyoutput.Log(
            f'Insufficient alignments found, expanding to use top {num_needed} alignments of {num_alignments} alignments')

    # PointPairs = np.hstack((TargetPoints, SourcePoints))
    point_pairs = np.hstack((valid_target_points, valid_source_points))

    if fixed_points is not None and fixed_points.shape[0] > 0:
        if fixed_points.shape[1] != 4:
            raise ValueError("fixed_points must have shape (N,4)")

        point_pairs = np.vstack((point_pairs, fixed_points))

    # PointPairs.append((ControlY, ControlX, mappedY, mappedX))

    prettyoutput.Log(
        f'Built transform with {point_pairs.shape[0]} of {num_alignments + num_fixed} points, including the {num_fixed} fixed points using cutoff {cutoff:g}')

    T = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(point_pairs)

    used_alignment_records = [alignment_records[valid_item[0]] for valid_item in
                              filter(lambda item: item[1], enumerate(valid_indices))]

    return T, used_alignment_records, weights_distance


def ConvertTransformToGridTransform(Transform: nornir_imageregistration.ITransform, source_image_shape: NDArray,
                                    cell_size: NDArray | None = None, grid_dims: NDArray | None = None,
                                    grid_spacing: NDArray | None = None,
                                    prefer_gpu: bool = False) -> nornir_imageregistration.ITransform:
    """Deprecated alias; use ``transforms.converters.ConvertTransformToGridTransform``."""
    return nornir_imageregistration.transforms.converters.ConvertTransformToGridTransform(
        Transform,
        source_image_shape,
        cell_size=cell_size,
        grid_dims=grid_dims,
        grid_spacing=grid_spacing,
        prefer_gpu=prefer_gpu)


# def AlignmentRecordsTo2DArray(alignment_records):
#     
#     def IsFinalFunc(record):
#         record.weight = 
#     
#     #Create a 2D array of 
#     Indices = np.hstack([np.asarray(a.ID,np.int32) for a in alignment_records])
#     
#     grid_dims = Indices.max()
#     
#     mask = np.zeros(grid_dims, np.bool)
#     
#     for a in alignment_records:
#         mask[a.ID] = 
#           


def AlignmentRecordsToDict(
        alignment_records: AlignmentRecordList) -> AlignmentRecordDict:
    """Index alignment records by their control-point grid key."""
    lookup = {}
    for a in alignment_records:
        lookup[a.ID] = a

    return lookup


def CalculateFinalizedAlignmentPointsMask(alignment_records: AlignmentRecordList,
                                          percentile: float = 0.5, max_travel_distance: float = 1.0,
                                          weight_cutoff: float | None = None) -> NDArray[np.bool_]:
    """
    Select points by weight and travel-distance cutoffs (legacy / tests).

    STOS ``RefineTransform`` uses ``refine_shared.finalize.evaluate_finalize_candidates``
    instead; keep this helper for travel+weight-only checks and legacy callers.
    """
    weights_distance = _alignment_records_to_composite_scores(alignment_records)
    # invert the weights to multiply by distance to promote low distance alignments 

    # composite_score = weights_distance[:,0] #np.prod(weights_distance,1)
    # composite_score = weights_distance[:,2]

    if weight_cutoff is None:
        if percentile is not None:
            weight_cutoff = float(np.percentile(weights_distance[:, 0], percentile))
        else:
            weight_cutoff = 0

    # weights = np.asarray(list(map(lambda a: a.weight, alignment_records)))
    # peak_distance = np.asarray(list(map(lambda a: np.linalg.norm(a.peak), alignment_records))) 

    # cutoff = np.percentile(weights, percentile)
    # if cutoff is None:  
    # if percentile is not None:
    # cutoff = np.percentile(composite_score, 100.0 - percentile)
    # else:
    # cutoff = np.max(composite_score) + 1

    valid_weight = weights_distance[:, 0] >= weight_cutoff
    valid_distance = weights_distance[:, 1] <= max_travel_distance

    finalize_mask = np.logical_and(valid_weight, valid_distance)

    return finalize_mask


def ApproximateRigidTransformByTargetPoints(input_transform: nornir_imageregistration.ITransform,
                                            target_points: NDArray,
                                            cell_size: NDArray[np.integer] | None = None,
                                            *,
                                            reference_pose: RingReferencePose | None = None,
                                            ring_scale_fraction_max: float | None = None,
                                            ring_angle_max_degrees: float | None = None,
                                            ring_allow_flip_change: bool | None = None) -> list[
    nornir_imageregistration.transforms.IRigidTransform] | list[nornir_imageregistration.transforms.Rigid]:
    """
    Estimate local rigid transforms at target points via inverse-mapped source points.
    """

    if isinstance(input_transform, nornir_imageregistration.transforms.IRigidTransform):
        return [input_transform] * target_points.shape[0]

    target_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(target_points)

    numPoints = target_points.shape[0]

    source_points = input_transform.InverseTransform(target_points)

    return ApproximateRigidTransformBySourcePoints(
        input_transform, source_points, cell_size,
        reference_pose=reference_pose,
        ring_scale_fraction_max=ring_scale_fraction_max,
        ring_angle_max_degrees=ring_angle_max_degrees,
        ring_allow_flip_change=ring_allow_flip_change)
    # translate the target points by 1, and find the angle between the source points
    # offset = np.array([0, 1])
    # offset_source_points = source_points + offset
    #
    # offsets = np.tile(offset, (numPoints, 1))
    # origins = np.tile(np.array([0, 0]), (numPoints, 1))
    #
    # recalculated_target_points = input_transform.Transform(source_points)
    # offset_target_points = input_transform.Transform(offset_source_points)
    #
    # target_delta = offset_target_points - recalculated_target_points
    #
    # angles = -np.round(nornir_imageregistration.ArcAngle(origins, offsets, target_delta), 3)
    #
    # target_offsets = target_points - source_points
    #
    # output_transforms = [nornir_imageregistration.transforms.Rigid(target_offset=target_offsets[i],
    #                                                                source_rotation_center=source_points[i],
    #                                                                angle=angles[i])
    #                      for i in range(0, len(angles))]
    #
    # return output_transforms


def calculate_offset(source_points: NDArray[np.floating],
                     cell_size: NDArray | None = None) -> NDArray[np.floating]:
    """
    Estimate a radial offset used to sample local orientation around source points.
    """
    # translate the target points a distance on the x-axis, and estimate the angle to determine the rotation
    xp = cp.get_array_module(source_points)

    offset = None
    if cell_size is None:
        # If we don't pass a cell_size, then make a reasonable guess by measuring how far away nearest points are from first point
        if source_points.shape[0] > 1:
            # 1-vs-rest nearest distance, not a pairwise matrix. CuVS/cdist
            # launch cost dominates; a vectorized norm stays on *xp*.
            estimated_cell_distance = float(
                xp.min(xp.linalg.norm(source_points[1:, :] - source_points[0:1, :], axis=1))
            ) / 2.0
            offset = xp.array((0, estimated_cell_distance))
        else:
            offset = xp.array((0, 1))
    else:
        offset = xp.array((0, cell_size[1] / 2.0))

    return offset


def _calculate_offset_ring(source_point: NDArray[np.floating],
                           offset: float,
                           nPoints: int = 8) -> NDArray[np.floating]:
    """
    Create a center-plus-ring sample pattern around one source point.
    """
    xp = cp.get_array_module(source_point)

    angles = xp.linspace(0, 2 * xp.pi, nPoints, endpoint=False)
    offsets = xp.vstack((xp.cos(angles), xp.sin(angles))).T * offset
    offsets += source_point
    offsets = xp.vstack((source_point, offsets))
    return offsets


def AdjustSourcePointsToIndexImage(source_points: NDArray[np.floating],
                                   source_image_shape: NDArray[np.integer]) -> NDArray[np.floating]:
    """
    Map source-space points to valid zero-based pixel indices for an image shape.
    """

    source_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_points)
    xp = cp.get_array_module(source_points)

    source_points_fraction = source_points / source_image_shape

    adjusted_source_points = source_points_fraction * (source_image_shape - 1)
    return adjusted_source_points


def ApproximateRigidTransformBySourcePoints(input_transform: nornir_imageregistration.ITransform,
                                            source_points: NDArray[np.floating],
                                            cell_size: NDArray | None = None,
                                            *,
                                            reference_pose: RingReferencePose | None = None,
                                            ring_scale_fraction_max: float | None = None,
                                            ring_angle_max_degrees: float | None = None,
                                            ring_allow_flip_change: bool | None = None) -> list[
    nornir_imageregistration.transforms.IRigidTransform]:
    """
    Estimate one local rigid transform per source point using transformed ring samples.

    Rings for all source points are transformed in one batched ``Transform`` call
    (avoids thousands of per-point GPU launches on CuPy), then rigid components
    are estimated with a vectorized host batch (avoids N× scipy align_vectors).

    Fitted scale, angle, and flip are clipped to ``reference_pose`` (derived from
    ``input_transform`` when omitted). Each result is rebuilt so
    ``Transform(source_point)`` still matches ``input_transform.Transform(source_point)``.
    """

    source_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_points)
    xp = cp.get_array_module(source_points)

    numPoints = int(source_points.shape[0])  # type: int

    # If the input transform is rigid, then we simply return that
    if isinstance(input_transform, nornir_imageregistration.transforms.IRigidTransform):
        return [input_transform] * numPoints

    if numPoints == 0:
        return []

    if reference_pose is None:
        reference_pose = reference_pose_from_transform(input_transform)
    if ring_scale_fraction_max is None:
        ring_scale_fraction_max = RING_SCALE_FRACTION_MAX
    if ring_angle_max_degrees is None:
        ring_angle_max_degrees = RING_ANGLE_MAX_DEGREES
    if ring_allow_flip_change is None:
        ring_allow_flip_change = RING_ALLOW_FLIP_CHANGE

    # Using the actual cell size can help avoid wildly incorrect scale values for the estimates rigid transforms
    offset = calculate_offset(source_points, cell_size)
    offset_distance = float(xp.linalg.norm(offset))

    n_ring_points = 8
    ring_angles = xp.linspace(0, 2 * xp.pi, n_ring_points, endpoint=False)
    ring_offsets = xp.vstack((xp.cos(ring_angles), xp.sin(ring_angles))).T * offset_distance
    centers = source_points[:, None, :]
    source_rings = xp.concatenate(
        (centers, centers + ring_offsets[None, :, :]),
        axis=1)  # (N, 1+8, 2)
    n_ring = int(source_rings.shape[1])
    flat_source = source_rings.reshape(numPoints * n_ring, 2)
    flat_target = input_transform.Transform(flat_source)
    flat_target = nornir_imageregistration.EnsurePointsAre2DNumpyArray(flat_target)
    target_rings = np.asarray(flat_target, dtype=np.float64).reshape(numPoints, n_ring, 2)
    source_rings_np = nornir_imageregistration.EnsurePointsAre2DNumpyArray(
        source_rings.reshape(numPoints * n_ring, 2)).reshape(numPoints, n_ring, 2)
    source_centers = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_points)

    components = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPointsBatched(
        source_rings=source_rings_np,
        target_rings=target_rings)

    scales = np.asarray([comp.scale for comp in components], dtype=np.float64)
    angles = np.asarray([comp.angle for comp in components], dtype=np.float64)
    flips = np.asarray([comp.reflected for comp in components], dtype=bool)
    clamped_scales, clamped_angles, clamped_flips = clamp_similarity_arrays(
        scales, angles, flips, reference_pose,
        scale_fraction_max=float(ring_scale_fraction_max),
        angle_max_degrees=float(ring_angle_max_degrees),
        allow_flip_change=bool(ring_allow_flip_change))

    desired_targets = nornir_imageregistration.EnsurePointsAre2DNumpyArray(
        input_transform.Transform(source_centers))

    output: list[nornir_imageregistration.transforms.IRigidTransform] = []
    zeros = np.zeros(2, dtype=np.float64)
    for i in range(numPoints):
        center = source_centers[i]
        rigid = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=zeros,
            source_rotation_center=center,
            angle=float(clamped_angles[i]),
            flip_ud=bool(clamped_flips[i]),
            scalar=float(clamped_scales[i]))
        mapped = np.asarray(rigid.Transform(center.reshape(1, 2)), dtype=np.float64).reshape(2)
        desired = np.asarray(desired_targets[i], dtype=np.float64).reshape(2)
        rigid.TranslateFixed(desired - mapped)
        output.append(rigid)
    return output


def BuildAlignmentROIs(transform: nornir_imageregistration.ITransform,
                       targetImage_param: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
                       sourceImage_param: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
                       target_image_stats: nornir_imageregistration.ImageStats | None,
                       source_image_stats: nornir_imageregistration.ImageStats | None,
                       target_controlpoint: NDArray | tuple[float, float],
                       alignmentArea: NDArray | tuple[float, float],
                       description: str | None = None,
                       defer_oob_check: bool = False) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray | None]:
    """
    Extract target/source ROIs in a common target-space frame for local registration.

    Accepts NumPy or CuPy control points; Rectangle/CropImage use a host snapshot.

    When ``defer_oob_check`` is False (default), an entirely-out-of-bounds source ROI
    raises ``ValueError`` immediately, which requires a ``bool(...)`` device sync on CuPy.
    When ``defer_oob_check`` is True, that sync is skipped: a 3-tuple is returned instead,
    with the (still un-synced) NaN mask as the third element (or ``None`` when
    ``source_image_stats`` is ``None``, i.e. no check was performed either way) so a caller
    processing many cells can batch the accept/reject decision into a single sync.
    """
    targetImage = nornir_imageregistration.ImageParamToImageArray(targetImage_param,  # type: ignore[arg-type]
                                                                  dtype=nornir_imageregistration.default_image_dtype())
    sourceImage = nornir_imageregistration.ImageParamToImageArray(sourceImage_param,  # type: ignore[arg-type]
                                                                  dtype=nornir_imageregistration.default_image_dtype())
    xp = cp.get_array_module(targetImage)
    sourceImage = _ensure_on_array_module(sourceImage, xp)

    # Adjust the point by 0.5 if it is an odd-sized area to ensure the output is centered on the desired pixel
    if hasattr(target_controlpoint, 'dtype') or hasattr(target_controlpoint, 'shape'):
        point_xp = cp.get_array_module(target_controlpoint)
    else:
        point_xp = xp
    point = point_xp.asarray(target_controlpoint, dtype=point_xp.float64).ravel()[:2].copy()
    area = point_xp.asarray(alignmentArea, dtype=point_xp.float64).ravel()[:2]
    adjust_mask = point_xp.mod(area, 2) > 0
    point[adjust_mask] += 0.5
    point_host = nornir_imageregistration.EnsureNumpyArray(point)
    area_host = nornir_imageregistration.EnsureNumpyArray(area)

    target_rectangle = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
        point=(float(point_host[0]) - (float(area_host[0]) / 2.0),
               float(point_host[1]) - (float(area_host[1]) / 2.0)),
        area=area_host)

    # Crop requires an integer for origin and area
    rounded_target_rectangle = nornir_imageregistration.Rectangle.SnapRound(target_rectangle)

    # Make sure the rectangle is the correct size, with an origin on an integer boundary
    rounded_target_rectangle = nornir_imageregistration.Rectangle.change_area(rounded_target_rectangle, alignmentArea,
                                                                              integer_origin=True)

    target_image_roi = nornir_imageregistration.CropImage(targetImage,
                                                          rounded_target_rectangle.BottomLeft[1],
                                                          rounded_target_rectangle.BottomLeft[0],
                                                          int(rounded_target_rectangle.Size[1]),
                                                          int(rounded_target_rectangle.Size[0]),
                                                          cval=False if target_image_stats is None else "random",
                                                          image_stats=target_image_stats)

    # Pull image subregions
    source_image_roi = nornir_imageregistration.assemble.SourceImageToTargetSpace(transform,
                                                                                  DataToTransform=sourceImage,
                                                                                  output_botleft=rounded_target_rectangle.BottomLeft,
                                                                                  output_area=rounded_target_rectangle.Size,
                                                                                  extrapolate=True,
                                                                                  cval=False if source_image_stats is None else np.nan)

    deferred_nan_mask: NDArray | None = None
    if source_image_stats is not None:
        roi_array = cast(Any, source_image_roi)
        roi_xp = cp.get_array_module(roi_array)
        nan_mask = roi_xp.isnan(roi_array)
        if defer_oob_check:
            # Leave the "entirely out of bounds" decision to the caller so it can batch
            # the sync across many cells instead of paying one bool(...) sync per cell here.
            deferred_nan_mask = nan_mask
        elif bool(nan_mask.all()):
            # The source ROI is entirely out of bounds — no usable image data for this cell.
            if isinstance(targetImage_param, nornir_imageregistration.Shared_Mem_Metadata):
                nornir_imageregistration.close_shared_memory(targetImage_param)
            if isinstance(sourceImage_param, nornir_imageregistration.Shared_Mem_Metadata):
                nornir_imageregistration.close_shared_memory(sourceImage_param)
            raise ValueError("Source image ROI is entirely out of bounds; skipping cell")
        source_image_roi = nornir_imageregistration.RandomNoiseMask(roi_array,  # type: ignore[arg-type]
                                                                    roi_xp.logical_not(nan_mask),
                                                                    imagestats=source_image_stats)
    elif 'DEBUG' in os.environ:
        roi_array = cast(Any, source_image_roi)
        roi_xp = cp.get_array_module(roi_array)
        if roi_xp.any(roi_xp.isnan(roi_array)):
            raise ValueError("Not handling NaN values in assembled image")

    if isinstance(targetImage_param, nornir_imageregistration.Shared_Mem_Metadata):
        nornir_imageregistration.close_shared_memory(targetImage_param)
    if isinstance(sourceImage_param, nornir_imageregistration.Shared_Mem_Metadata):
        nornir_imageregistration.close_shared_memory(sourceImage_param)

    if defer_oob_check:
        return target_image_roi, source_image_roi, deferred_nan_mask  # type: ignore[return-value]

    return target_image_roi, source_image_roi  # type: ignore[return-value]


def EnsureMaxContrast(image: NDArray) -> NDArray:
    """
    Normalize an image into the ``[0, 1]`` intensity range.
    """

    minval = image.min()
    maxval = image.max()

    if minval == 0 and maxval == 1:
        return image

    range = maxval - minval
    return (image - image.min()) / range


def StartAttemptAlignPoint(pool: nornir_pools.IPool,
                           taskname: str,
                           transform: nornir_imageregistration.ITransform,
                           targetImage: NDArray,
                           sourceImage: NDArray,
                           # targetMask: NDArray,
                           # sourceMask: NDArray,
                           target_image_stats: nornir_imageregistration.ImageStats | None,
                           source_image_stats: nornir_imageregistration.ImageStats | None,
                           target_controlpoint: NDArray | tuple[float, float],
                           alignmentArea: NDArray | tuple[float, float],
                           anglesToSearch: Iterable[float] | None = None,
                           min_alignment_overlap: float = 0.5,
                           *,
                           estimate_angle: bool = True,
                           search_scale: bool = True) -> nornir_pools.Task | None:
    """Create and enqueue an asynchronous rigid-registration task for one point.

    :param estimate_angle: See :func:`AttemptAlignPoint`.
    """
    if anglesToSearch is None:
        anglesToSearch = np.linspace(-7.5, 7.5, 11)
        # Ensure we check a non-rotated alignment
        anglesToSearch = np.union1d(anglesToSearch, [0])

    rigid_transform = ApproximateRigidTransformByTargetPoints(input_transform=transform,
                                                              target_points=target_controlpoint,  # type: ignore[arg-type]
                                                              cell_size=alignmentArea)  # type: ignore[arg-type]

    try:
        target_image_roi, source_image_roi = BuildAlignmentROIs(transform=rigid_transform[0],
                                                                targetImage_param=targetImage,
                                                                sourceImage_param=sourceImage,
                                                                target_image_stats=target_image_stats,
                                                                source_image_stats=source_image_stats,
                                                                target_controlpoint=target_controlpoint,
                                                                alignmentArea=alignmentArea,
                                                                description=taskname)
    except ValueError:
        # Entirely out-of-bounds source ROIs are not alignable (same as AttemptAlignPoint).
        return None

    target_image_roi = EnsureMaxContrast(target_image_roi)
    source_image_roi = EnsureMaxContrast(source_image_roi)

    # Just ignore pure color regions
    if not is_alignable_cell(target_image_roi):
        return None
    if not is_alignable_cell(source_image_roi):
        return None

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    # pool = Pools.GetGlobalMultithreadingPool()

    # task = pool.add_task("AttemptAlignPoint", nornir_imageregistration.FindOffset, targetImageROI, sourceImageROI, MinOverlap = 0.2)
    # apoint = task.wait_return()
    # apoint = nornir_imageregistration.FindOffset(targetImageROI, sourceImageROI, MinOverlap=0.2)
    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI], "Fixed <---> Warped")

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    #     nornir_imageregistration.stos_brute.SliceToSliceRigidRegistration(
    #                         targetImageROI,
    #                         sourceImageROI,
    #                         AngleSearchRange=anglesToSearch,
    #                         MinOverlap=min_alignment_overlap,
    #                         SingleThread=True,
    #                         Cluster=False,
    #                         TestFlip=False)
    #
    task = pool.add_task(taskname,
                         nornir_imageregistration.stos_brute.SliceToSliceRigidRegistration,
                         target_image=target_image_roi,
                         source_image=source_image_roi,
                         AngleSearchRange=anglesToSearch,
                         MinOverlap=min_alignment_overlap,
                         SingleThread=True,
                         TestFlip=False,
                         method=nornir_imageregistration.settings.SliceToSliceMethod.BruteForce,
                         estimate_angle=estimate_angle,
                         search_scale=search_scale)

    task.TargetROI = target_image_roi  # type: ignore[attr-defined]
    task.SourceROI = source_image_roi  # type: ignore[attr-defined]
    task.RigidTransform = rigid_transform  # type: ignore[attr-defined]

    return task


def AttemptAlignPoint(transform: nornir_imageregistration.ITransform,
                      targetImage: NDArray,
                      sourceImage: NDArray,
                      # targetMask: NDArray,
                      # sourceMask: NDArray,
                      target_image_stats: nornir_imageregistration.ImageStats | None,
                      source_image_stats: nornir_imageregistration.ImageStats | None,
                      target_controlpoint: NDArray | tuple[float, float],
                      alignmentArea: NDArray | tuple[float, float],
                      anglesToSearch: Iterable[float] | None = None,
                      min_alignment_overlap: float = 0.5,
                      *,
                      estimate_angle: bool = True,
                      search_scale: bool = True,
                      use_gpu: bool | None = None) -> nornir_imageregistration.AlignmentRecord | None:
    """Run synchronous rigid-registration for one control point.

    :param estimate_angle: If True, a log-polar estimate appends an angle to
        *anglesToSearch*. Pass False when the caller wants only the angles it
        supplied, for example a translation-only search.
    :param search_scale: If False, register at scale 1.0 with no scale probe or search.
    :param use_gpu: If False, score on the host even when the process backend is CuPy.
        None (default) follows ``GetActiveComputationLib``.
    """
    if anglesToSearch is None:
        anglesToSearch = np.linspace(-7.5, 7.5, 11)

    rigid_transform = ApproximateRigidTransformByTargetPoints(input_transform=transform,
                                                              target_points=target_controlpoint,  # type: ignore[arg-type]
                                                              cell_size=alignmentArea)  # type: ignore[arg-type]

    # #region agent log
    def _dbg_log(hypothesis_id, message, data):
        try:
            import json as _json, os as _os, time as _t
            with open(r'd:\src\git\nornir\debug-f7347d.log', 'a') as _f:
                _f.write(_json.dumps({'sessionId': 'f7347d', 'runId': 'point2',
                                      'hypothesisId': hypothesis_id,
                                      'location': 'local_distortion_correction.py:AttemptAlignPoint',
                                      'message': message, 'data': data, 'pid': _os.getpid(),
                                      'timestamp': int(_t.time() * 1000)}, default=str) + '\n')
        except Exception:
            pass

    def _dbg_arr(a):
        try:
            return nornir_imageregistration.EnsureNumpyArray(np.asarray(a)).ravel().tolist()[:8]
        except Exception:
            return str(a)

    _dbg_log('A', 'AttemptAlignPoint entry', {
        'estimate_angle': bool(estimate_angle),
        'anglesToSearch': _dbg_arr(list(anglesToSearch)),
        'target_controlpoint': _dbg_arr(target_controlpoint),
        'alignmentArea': _dbg_arr(alignmentArea),
        'rigid_transform_type': type(rigid_transform[0]).__name__,
        'rigid_target_botleft': _dbg_arr(getattr(rigid_transform[0], 'target_space_center_of_rotation', None)),
        'rigid_angle': str(getattr(rigid_transform[0], 'angle', None)),
        'transform_type': type(transform).__name__,
    })

    def _dbg_inverse(t):
        try:
            pt = np.asarray(nornir_imageregistration.EnsureNumpyArray(
                np.asarray(target_controlpoint, dtype=np.float64))).reshape(1, 2)
            return _dbg_arr(t.InverseTransform(pt))
        except Exception as exc:
            return f'error: {exc}'

    _dbg_log('G', 'where does each transform send the control point', {
        'mesh_inverse_of_target_point': _dbg_inverse(transform),
        'rigid_approx_inverse_of_target_point': _dbg_inverse(rigid_transform[0]),
    })
    # #endregion

    try:
        target_image_roi, source_image_roi = BuildAlignmentROIs(transform=rigid_transform[0],
                                                                targetImage_param=targetImage,
                                                                sourceImage_param=sourceImage,
                                                                target_image_stats=target_image_stats,
                                                                source_image_stats=source_image_stats,
                                                                target_controlpoint=target_controlpoint,
                                                                alignmentArea=alignmentArea,
                                                                description='')
    except ValueError:
        return None

    # #region agent log
    _dbg_log('D', 'alignment ROIs built', {
        'target_roi_shape': list(target_image_roi.shape),
        'source_roi_shape': list(source_image_roi.shape),
        'target_image_shape': list(np.shape(targetImage)),
        'source_image_shape': list(np.shape(sourceImage)),
    })
    # #endregion

    # Just ignore pure color regions
    if not is_alignable_cell(target_image_roi):
        return None
    if not is_alignable_cell(source_image_roi):
        return None

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    # pool = Pools.GetGlobalMultithreadingPool()

    # task = pool.add_task("AttemptAlignPoint", nornir_imageregistration.FindOffset, targetImageROI, sourceImageROI, MinOverlap = 0.2)
    # apoint = task.wait_return()
    # apoint = nornir_imageregistration.FindOffset(targetImageROI, sourceImageROI, MinOverlap=0.2)
    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI], "Fixed <---> Warped")

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    try:
        result = nornir_imageregistration.stos_brute.SliceToSliceRigidRegistration(
            target_image=target_image_roi,
            source_image=source_image_roi,
            AngleSearchRange=anglesToSearch,  # type: ignore[arg-type]
            MinOverlap=min_alignment_overlap,
            SingleThread=True,
            TestFlip=False,
            method=SliceToSliceMethod.BruteForce,
            estimate_angle=estimate_angle,
            search_scale=search_scale,
            use_gpu=use_gpu)
    except ValueError:
        # Empty / fully-extrema ROIs can fail ImagePermutationHelper stats; skip cell.
        return None

    # #region agent log
    try:
        import os as _os, time as _tm
        from PIL import Image as _Image
        _roi_dir = r'd:\src\git\nornir\_debug_rois'
        _os.makedirs(_roi_dir, exist_ok=True)
        _stamp = str(int(_tm.time() * 1000))

        def _save_roi(name, arr):
            a = nornir_imageregistration.EnsureNumpyArray(arr).astype(np.float64)
            a = np.nan_to_num(a, nan=0.0)
            lo, hi = float(a.min()), float(a.max())
            a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
            _Image.fromarray((a * 255).astype(np.uint8)).save(
                _os.path.join(_roi_dir, f'{_stamp}_{name}.png'))

        _pk = nornir_imageregistration.EnsureNumpyArray(np.asarray(result.peak)).ravel()
        _src_host = nornir_imageregistration.EnsureNumpyArray(source_image_roi)
        _save_roi('target', target_image_roi)
        _save_roi('source', source_image_roi)
        _save_roi('source_shifted_by_peak',
                  np.roll(np.roll(_src_host, int(round(float(_pk[0]))), axis=0),
                          int(round(float(_pk[1]))), axis=1))
        _dbg_log('F', 'ROI images saved', {'dir': _roi_dir, 'stamp': _stamp,
                                           'peak': _pk.tolist()})

        _tgt_host_img = nornir_imageregistration.EnsureNumpyArray(targetImage)
        _src_host_img = nornir_imageregistration.EnsureNumpyArray(sourceImage)
        _tgt_pt = nornir_imageregistration.EnsureNumpyArray(
            np.asarray(target_controlpoint, dtype=np.float64)).ravel()[:2]
        _src_pt = np.asarray(rigid_transform[0].InverseTransform(
            _tgt_pt.reshape(1, 2))).ravel()[:2]

        def _crop(a, centre_yx, half=256):
            cy, cx = int(round(float(centre_yx[0]))), int(round(float(centre_yx[1])))
            y0, x0 = max(0, cy - half), max(0, cx - half)
            return a[y0:y0 + 2 * half, x0:x0 + 2 * half]

        def _measure(a, b, label):
            try:
                r = nornir_imageregistration.stos_brute.SliceToSliceRigidRegistration(
                    target_image=a, source_image=b, AngleSearchRange=[0], MinOverlap=0.25,
                    SingleThread=True, TestFlip=False,
                    method=SliceToSliceMethod.BruteForce, estimate_angle=False,
                    search_scale=False, use_gpu=use_gpu)
                return {'label': label,
                        'peak': nornir_imageregistration.EnsureNumpyArray(
                            np.asarray(r.peak)).ravel().tolist(),
                        'weight': float(r.weight)}
            except Exception as exc:
                return {'label': label, 'error': str(exc)}

        # O: fingerprint the arrays the worker actually received, so they can be compared
        # against the images on disk that the stos transform was built from.
        def _fp(a, name):
            return {'name': name, 'shape': list(a.shape), 'dtype': str(a.dtype),
                    'mean': float(np.nanmean(a)), 'std': float(np.nanstd(a)),
                    'min': float(np.nanmin(a)), 'max': float(np.nanmax(a)),
                    'nan_fraction': float(np.mean(np.isnan(a))),
                    'corner_means': [float(np.nanmean(a[:64, :64])),
                                     float(np.nanmean(a[:64, -64:])),
                                     float(np.nanmean(a[-64:, :64])),
                                     float(np.nanmean(a[-64:, -64:]))]}
        _dbg_log('O', 'image arrays as received by the worker', {
            'target': _fp(_tgt_host_img, 'target'),
            'source': _fp(_src_host_img, 'source'),
        })

        # P: is the -173 degree local rigid angle real, or an artefact of the ring fit?
        # Sample where the mesh actually sends four points around the control point.
        _probe = _src_pt + np.array([[0., 0.], [100., 0.], [0., 100.], [-100., 0.]])
        _dbg_log('P', 'local orientation of the mesh around the point', {
            'rigid_angle_radians': float(getattr(rigid_transform[0], 'angle', float('nan'))),
            'rigid_angle_degrees': float(np.degrees(
                getattr(rigid_transform[0], 'angle', float('nan')))),
            'rigid_scale': float(getattr(rigid_transform[0], 'scalar', float('nan'))),
            'rigid_flip_ud': str(getattr(rigid_transform[0], 'flip_ud', None)),
            'source_probe_points': _probe.tolist(),
            'mesh_forward_of_probes': _dbg_arr(transform.Transform(_probe)),
            'rigid_forward_of_probes': _dbg_arr(rigid_transform[0].Transform(_probe)),
        })
        # Q: is the transform globally consistent with these two arrays at all? Warp the
        # whole source into target space on a coarse grid and compare against a coarse
        # target. A correct pairing matches strongly at ~zero offset regardless of local
        # distortion; noise here means the transform and the images do not belong together.
        import scipy.ndimage as _ndi
        _n = 512
        _tsh = np.asarray(_tgt_host_img.shape[:2], dtype=np.float64)
        _yy, _xx = np.meshgrid(np.linspace(0, _tsh[0] - 1, _n),
                               np.linspace(0, _tsh[1] - 1, _n), indexing='ij')
        _tflat = np.stack([_yy.ravel(), _xx.ravel()], axis=1)
        _sflat = np.asarray(nornir_imageregistration.EnsureNumpyArray(
            np.asarray(transform.InverseTransform(_tflat))), dtype=np.float64)
        _coarse_warped_source = _ndi.map_coordinates(
            _src_host_img.astype(np.float32), [_sflat[:, 0], _sflat[:, 1]],
            order=1, cval=0.0).reshape(_n, _n)
        _coarse_target = _ndi.map_coordinates(
            _tgt_host_img.astype(np.float32), [_tflat[:, 0], _tflat[:, 1]],
            order=1, cval=0.0).reshape(_n, _n)
        _coarse_raw_source = _ndi.map_coordinates(
            _src_host_img.astype(np.float32), [_yy.ravel(), _xx.ravel()],
            order=1, cval=0.0).reshape(_n, _n)
        _save_roi('global_target', _coarse_target)
        _save_roi('global_source_warped', _coarse_warped_source)
        _save_roi('global_source_raw', _coarse_raw_source)
        _inb = np.mean((_sflat[:, 0] >= 0) & (_sflat[:, 0] < _src_host_img.shape[0]) &
                       (_sflat[:, 1] >= 0) & (_sflat[:, 1] < _src_host_img.shape[1]))
        _dbg_log('Q', 'whole-section consistency of transform with these arrays', {
            'coarse_grid': _n,
            'fraction_of_target_mapping_inside_source': float(_inb),
            'warped_vs_target': _measure(_coarse_target, _coarse_warped_source, 'global_warped'),
            'raw_vs_target': _measure(_coarse_target, _coarse_raw_source, 'global_raw'),
            'target_extent': _tsh.tolist(),
            'source_corners_from_target_corners': [
                _sflat[0].tolist(), _sflat[_n - 1].tolist(),
                _sflat[_n * (_n - 1)].tolist(), _sflat[-1].tolist()],
        })
    except Exception as _roi_exc:
        import traceback as _tb
        _dbg_log('F', 'ROI diagnostics failed',
                 {'error': str(_roi_exc), 'trace': _tb.format_exc()[-1200:]})


    _dbg_log('B', 'SliceToSliceRigidRegistration result', {
        'peak': _dbg_arr(getattr(result, 'peak', None)),
        'angle': str(getattr(result, 'angle', None)),
        'weight': str(getattr(result, 'weight', None)),
        'scale': str(getattr(result, 'scale', None)),
        'flipped': str(getattr(result, 'flippedud', None)),
        'roi_half': [float(np.asarray(alignmentArea).ravel()[0]) / 2.0,
                     float(np.asarray(alignmentArea).ravel()[1]) / 2.0],
    })
    # #endregion

    if nornir_imageregistration.in_debug_mode():
        result.TargetROI = target_image_roi  # type: ignore[attr-defined]
        result.SourceROI = source_image_roi  # type: ignore[attr-defined]

    return result


def TryToImproveAlignments(transform: nornir_imageregistration.transforms.ITransform,
                           alignment_records: AlignmentRecordDict,
                           settings: nornir_imageregistration.settings.GridRefinement) \
        -> tuple[AlignmentRecordDict, list[AlignmentRecordKey]]:
    """
    Re-evaluate finalized alignments and keep only score-improving replacements.
    """

    items = alignment_records.items()

    if len(items) == 0:
        return dict(), list()

    SourcePoints = np.vstack([fp[1].SourcePoint for fp in items])
    TargetPoints = np.vstack([fp[1].TargetPoint for fp in items])
    # keys = [fp.ID for fp in alignment_records]
    keys = [fp[0] for fp in items]

    refined_alignments = _RefinePointsForTwoImages(transform, keys, SourcePoints, TargetPoints, settings)

    output = dict()  # type: AlignmentRecordDict
    improved_alignments = []  # type: list[tuple[int, int]]
    for refined_align_record in refined_alignments:
        key = refined_align_record.ID  # type: tuple[int, int]
        record = alignment_records[key]
        # task = task_tuple[0]
        # record = task_tuple[1]
        # refined_align_record = task.wait_return()
        chosen_record = record
        magnitude = np.sqrt(refined_align_record.peak.dot(refined_align_record.peak))
        if refined_align_record.weight > record.weight and magnitude < settings.max_travel_for_finalization:
            # oldPSDDelta = record.PSDDelta
            # record = nornir_imageregistration.EnhancedAlignmentRecord(ID=record.ID,
            #                                                              TargetPoint=record.TargetPoint,
            #                                                              SourcePoint=record.SourcePoint,
            #                                                              peak=refined_align_record.peak,
            #                                                              weight=refined_align_record.weight,
            #                                                              angle=refined_align_record.angle,
            #                                                              flipped_ud=refined_align_record.flippedud)
            # record.PSDDelta = oldPSDDelta
            # record.TargetROI = record.TargetROI
            # record.SourceROI = record.SourceROI
            chosen_record = refined_align_record
            improved_alignments.append(key)

        # Create a record that is unmoving
        output[key] = nornir_imageregistration.EnhancedAlignmentRecord(chosen_record.ID,
                                                                       TargetPoint=chosen_record.AdjustedTargetPoint,
                                                                       SourcePoint=chosen_record.SourcePoint,
                                                                       peak=np.asarray((0, 0), dtype=np.float32),
                                                                       weight=chosen_record.weight, angle=0,
                                                                       flipped_ud=chosen_record.flippedud,
                                                                       peak_ratio=getattr(chosen_record, 'peak_ratio',
                                                                                          None))

        # output[key].PSDDelta = chosen_record.PSDDelta

    # Close the pool to prevent threads from hanging around
    # pool.shutdown()
    return output, improved_alignments

