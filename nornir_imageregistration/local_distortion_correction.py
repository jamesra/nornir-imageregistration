"""
Created on Apr 7, 2015

@author: u0490822

This module performs local distortions of images to refine alignments of mosaics and sections
"""
import gc
import logging
import os
import enum
import copy
import time
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence, cast

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_imageregistration.assemble
import nornir_imageregistration.assemble_tiles
from nornir_imageregistration.spatial_distance import cdist as pairwise_cdist
from nornir_imageregistration.mathfuncs import EMA, estimate_cutoff
import nornir_imageregistration.phasecorrelation
import nornir_imageregistration.batched_phase_correlation
from nornir_imageregistration.settings import SliceToSliceMethod
import nornir_pools
from nornir_imageregistration.transforms.triangulation import Triangulation
from nornir_shared import prettyoutput

try:
    import cupy as cp
    import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

# import nornir_imageregistration.views.grid_data

# Summary type used for typing
AlignmentRecordDict = dict[tuple[int, int], nornir_imageregistration.EnhancedAlignmentRecord]
AlignmentRecordList = Sequence[nornir_imageregistration.EnhancedAlignmentRecord]
AlignmentRecordKey = tuple[int, int]


class _RefinePhaseTimer:
    """Accumulate wall time per named phase of grid refinement (opt-in).

    Enabled by setting ``NORNIR_REFINE_PHASE_TIMING`` to a truthy value. When
    disabled, ``section`` is a no-op context manager so default runs are
    unaffected (mirrors the ``_log_refinement_gpu_memory`` gating style).

    Under CuPy each section synchronizes the device on exit so the recorded
    wall time reflects actual kernel completion. The mosaic vertex loop already
    forces a per-vertex host sync (``.get()`` on each peak), so this adds no new
    serialization to the CuPy hot path while it is measured.
    """

    PHASES = ('prewarp', 'cell_extract', 'fft', 'host_sync', 'regularize', 'apply')

    def __init__(self) -> None:
        flag = os.environ.get('NORNIR_REFINE_PHASE_TIMING', '0').strip().lower()
        self.enabled = flag not in ('', '0', 'false', 'no', 'off')
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Clear all accumulated phase totals and counts."""
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def snapshot(self) -> dict[str, float]:
        """Return a copy of the current cumulative per-phase totals."""
        return dict(self.totals)

    @contextlib.contextmanager
    def section(self, name: str):
        """Time the wrapped block into the *name* bucket (no-op when disabled)."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            if nornir_imageregistration.UsingCupy() and cp is not None:
                try:
                    cp.cuda.Device().synchronize()
                except Exception:  # pragma: no cover - defensive against thunk/no-GPU
                    pass
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1


_PHASE_TIMER = _RefinePhaseTimer()


def _use_batched_gpu_vertex_measurement() -> bool:
    """Return True when the batched-GPU vertex path should be used.

    The batched path (one batched FFT + vectorized peak finder + single host
    sync) is the production default under CuPy: it is ~2.4x faster than the
    serial connected-component path and matches the CPU/golden registration
    within sub-pixel tolerance (see docs/mosaic_refine_grid_gpu_assessment.md).

    It is only meaningful under CuPy (the FFT/peak batching wins come from the
    GPU); the NumPy path always uses the serial measurement. Set
    ``NORNIR_REFINE_BATCHED_GPU=0`` (or false/no/off) to force the legacy serial
    path for A/B comparison or fallback. The env var is read per call so
    benchmarks/tests can toggle it within a process.
    """
    if not nornir_imageregistration.UsingCupy():
        return False
    flag = os.environ.get('NORNIR_REFINE_BATCHED_GPU', '').strip().lower()
    if flag in ('0', 'false', 'no', 'off'):
        return False
    return True


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
    """Compute the nominal target-space center for a refinement subregion."""
    target_a = A.Transform.Transform(np.asarray([full_source_a], dtype=np.float64))[0]
    target_b = B.Transform.Transform(np.asarray([full_source_b], dtype=np.float64))[0]
    return (target_a + target_b) * 0.5


def _filter_weighted_point_pair_updates(point_pair_updates: np.ndarray) -> np.ndarray:
    """
    Drop low-confidence overlap updates using the same estimate_cutoff heuristic as STOS refine.

    Not part of the mosaic refine path: legacy ir-refine-grid has no weight-percentile
    gating (it relies on regularize_displacements). Retained for STOS-style callers/tests.
    """
    if point_pair_updates.size == 0:
        return point_pair_updates

    positive_weight = point_pair_updates['Weight'] > 0
    if not np.any(positive_weight):
        return point_pair_updates[:0]

    weights = np.asarray(point_pair_updates['Weight'][positive_weight], dtype=np.float64)
    if weights.shape[0] < 3:
        return point_pair_updates[positive_weight]

    try:
        _, inflection_percentile, _, polyfit_weights = estimate_cutoff(weights)
        cutoff_value = float(polyfit_weights[inflection_percentile])  # type: ignore[index]
    except ValueError:
        return point_pair_updates[positive_weight]

    keep_mask = positive_weight.copy()
    positive_indices = np.flatnonzero(positive_weight)
    keep_mask[positive_indices] = weights >= cutoff_value
    return point_pair_updates[keep_mask]


def _phase_correlate_refinement_cell(
        cell_a: NDArray[np.floating],
        cell_b: NDArray[np.floating],
        subregion_shape: NDArray[np.integer],
        *,
        min_overlap: float = 0.25,
        max_overlap: float = 1.0) -> nornir_imageregistration.AlignmentRecord:
    """Preprocess and phase-correlate one refinement FFT cell (translate-path parity)."""
    xp = cp.get_array_module(cell_a)
    cell_a = xp.asarray(cell_a, dtype=np.float64)
    cell_b = xp.asarray(cell_b, dtype=np.float64)
    subregion_shape = np.asarray(subregion_shape, dtype=np.int64)

    if (cell_a.size == 0 or cell_b.size == 0
            or cell_a.min() == cell_a.max() or cell_b.min() == cell_b.max()
            or cell_a.max() == 0 or cell_b.max() == 0):
        return nornir_imageregistration.AlignmentRecord(peak=np.zeros(2, dtype=np.float64), weight=0.0)

    normalized_a = cell_a - cell_a.min()
    normalized_a /= normalized_a.max()
    normalized_b = cell_b - cell_b.min()
    normalized_b /= normalized_b.max()

    # Match legacy refine_one_point_fft: FFT the raw equal-size cells directly.
    # Random-noise padding would make refinement passes nondeterministic and is
    # not part of the C++ pipeline for grid refinement cells.
    return nornir_imageregistration.phasecorrelation.find_offset(
        normalized_a,
        normalized_b,
        min_overlap=min_overlap,
        max_overlap=max_overlap,
        target_shape=subregion_shape,
        source_shape=subregion_shape,
        fft_required=True)


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

    # Step 7 contingency: 'thread' mode overlaps host tile-load + coord compute
    # with warp kernels and is allowed even under CuPy. Default keeps the serial
    # CuPy path / multiprocess CPU path unchanged.
    use_thread_dispatch = 'thread' in _prewarp_dispatch_mode()
    if len(tiles_to_prewarp) <= 1 or (nornir_imageregistration.UsingCupy() and not use_thread_dispatch):
        for tile in tiles_to_prewarp:
            warped = _prewarp_tile_for_grid_refine(tile, target_space_scale)
            if use_cache:
                warped._revision = revision_cache.get(tile.ID, 0)  # type: ignore[attr-defined]
                _store_prewarped_tile_cache(prewarp_cache, tile.ID, warped)
            prewarped[tile.ID] = warped
        return prewarped

    pool = (nornir_pools.GetGlobalThreadPool()
            if use_thread_dispatch
            else nornir_pools.GetGlobalMultithreadingPool())
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
    # Legacy: origin = center - 0.5 * cell; pixel index i samples origin + i.
    start = np.floor(
        np.asarray(center_scaled, dtype=np.float64)
        - prewarped.origin
        - (cell_shape.astype(np.float64) / 2.0)).astype(np.int64)
    stop = start + cell_shape

    image_shape = np.asarray(prewarped.image.shape, dtype=np.int64)
    clipped_start = np.maximum(start, 0)
    clipped_stop = np.minimum(stop, image_shape)
    if np.any(clipped_start >= clipped_stop):
        return xp.zeros(tuple(int(v) for v in cell_shape), dtype=np.float32), 0.0

    cell = xp.zeros(tuple(int(v) for v in cell_shape), dtype=prewarped.image.dtype)
    write_start = clipped_start - start
    write_stop = write_start + (clipped_stop - clipped_start)
    cell[int(write_start[0]):int(write_stop[0]), int(write_start[1]):int(write_stop[1])] = \
        prewarped.image[int(clipped_start[0]):int(clipped_stop[0]), int(clipped_start[1]):int(clipped_stop[1])]

    valid_region = prewarped.valid_mask[
        int(clipped_start[0]):int(clipped_stop[0]), int(clipped_start[1]):int(clipped_stop[1])]
    valid_count = float(xp.count_nonzero(valid_region))
    return cell, valid_count / float(cell_shape.prod())


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
    centers_scaled = np.asarray(centers_scaled, dtype=np.float64)
    cell_shape = np.asarray(cell_shape, dtype=np.int64)
    num_vertices = centers_scaled.shape[0]
    shifts = np.zeros((num_vertices, 2), dtype=np.float64)
    measured = np.zeros(num_vertices, dtype=bool)

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
    centers_scaled = np.asarray(centers_scaled, dtype=np.float64)
    cell_shape = np.asarray(cell_shape, dtype=np.int64)
    num_vertices = centers_scaled.shape[0]
    shifts = np.zeros((num_vertices, 2), dtype=np.float64)
    measured = np.zeros(num_vertices, dtype=bool)

    fixed_shape = np.asarray(fixed.image.shape, dtype=np.float64)
    eligible_indices: list[int] = []
    fixed_cells: list[NDArray[np.floating]] = []
    moving_cells: list[NDArray[np.floating]] = []

    with _PHASE_TIMER.section('cell_extract'):
        for k in range(num_vertices):
            center = centers_scaled[k]
            local_fixed = center - fixed.origin
            if np.any(local_fixed < 0) or np.any(local_fixed >= fixed_shape):
                continue

            fixed_cell, fixed_fraction = _extract_refinement_cell(fixed, center, cell_shape)
            if fixed_fraction < cell_min_overlap:
                continue
            moving_cell, moving_fraction = _extract_refinement_cell(moving, center, cell_shape)
            if moving_fraction < cell_min_overlap:
                continue

            eligible_indices.append(k)
            fixed_cells.append(fixed_cell)
            moving_cells.append(moving_cell)

    if len(eligible_indices) == 0:
        return shifts, measured

    xp = cp.get_array_module(fixed_cells[0])
    fixed_stack = xp.stack(fixed_cells, axis=0)
    moving_stack = xp.stack(moving_cells, axis=0)

    with _PHASE_TIMER.section('fft'):
        peaks_dev, weights_dev = nornir_imageregistration.batched_phase_correlation.batched_find_offset(
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

    for batch_pos, k in enumerate(eligible_indices):
        peak = peaks[batch_pos]
        if weights[batch_pos] <= 0 or np.any(np.isnan(peak)):
            continue
        shifts[k, :] = peak
        measured[k] = True

    return shifts, measured


def _regularize_displacements(
        shifts: NDArray[np.floating],
        measured: NDArray[np.bool_],
        mesh_dims: tuple[int, int],
        median_radius: int = 1) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Port of legacy ``regularize_displacements`` (mosaic_refinement_common.cxx).

    Stages: median filter (radius ``median_radius``) on the measured displacement fields,
    radius-1 ring gap-fill for unmeasured vertices, then Gaussian blur (sigma=1) over the
    entire fields. Returns regularized per-vertex (y, x) shifts and the measured/filled
    flags (the legacy ``db`` image, accumulated into ``mass`` by the caller).
    """
    mesh_rows, mesh_cols = int(mesh_dims[0]), int(mesh_dims[1])
    dy = np.asarray(shifts[:, 0], dtype=np.float64).reshape(mesh_rows, mesh_cols)
    dx = np.asarray(shifts[:, 1], dtype=np.float64).reshape(mesh_rows, mesh_cols)
    db = np.asarray(measured, dtype=np.float64).reshape(mesh_rows, mesh_cols)

    # Stage 1: median denoise (ITK MedianImageFilter w/ zero-flux Neumann == 'nearest').
    if median_radius > 0:
        size = 2 * int(median_radius) + 1
        dy = scipy.ndimage.median_filter(dy, size=size, mode='nearest')
        dx = scipy.ndimage.median_filter(dx, size=size, mode='nearest')

    # Stage 2: gap-fill unmeasured vertices from the radius-1 ring (the legacy loop's
    # "expanding" radius is capped at 1 by `max_r = std::min(1, ...)`), using the legacy
    # offset pattern. Samples read the median-filtered fields; db is updated on success.
    dy_filled = dy.copy()
    dx_filled = dx.copy()
    db_filled = db.copy()
    unmeasured_rows, unmeasured_cols = np.nonzero(db == 0)
    for row, col in zip(unmeasured_rows.tolist(), unmeasured_cols.tolist()):
        py = 0.0
        px = 0.0
        w = 0.0
        r = 1
        x0, x1 = col - r, col + r
        y0, y1 = row - r, row + r
        d = 2 * r + 1
        for o in range(d):
            # Legacy ring pattern in (x=col, y=row) coordinates.
            for cx, cy in ((x0, y0 + o + 1), (x1, y0 + o), (x0 + o, y0), (x0 + o + 1, y1)):
                if 0 <= cx < mesh_cols and 0 <= cy < mesh_rows and db[cy, cx] != 0:
                    px += dx[cy, cx]
                    py += dy[cy, cx]
                    w += 1.0
        if w != 0.0:
            dy_filled[row, col] = py / w
            dx_filled[row, col] = px / w
            db_filled[row, col] = 1.0

    # Stage 3: Gaussian blur over the full fields (ITK DiscreteGaussianImageFilter with
    # variance=1, max_error=0.1 -> compact kernel; truncate=2.0 approximates it).
    dy_filled = scipy.ndimage.gaussian_filter(dy_filled, sigma=1.0, mode='nearest', truncate=2.0)
    dx_filled = scipy.ndimage.gaussian_filter(dx_filled, sigma=1.0, mode='nearest', truncate=2.0)

    out_shifts = np.column_stack((dy_filled.reshape(-1), dx_filled.reshape(-1)))
    return out_shifts, db_filled.reshape(-1)


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
        return nornir_pools.GetGlobalSerialPool()
    return nornir_pools.GetGlobalMultithreadingPool()


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
    """Return the opt-in prewarp dispatch mode (Step 7 contingency).

    Read from ``NORNIR_REFINE_PREWARP_MODE`` (default ``'serial'``). Tokens may
    be combined (for example ``'thread+cache'``):

    - ``thread``: dispatch the per-tile warp on the shared thread pool even
      under CuPy / on CPU instead of the default (serial under CuPy, multiprocess
      on CPU). Overlaps host tile-load + coordinate compute with warp kernels.
    - ``cache``: allow the cross-pass prewarp cache under CuPy (re-render only
      tiles whose lattice revision changed). Off by default because it was
      historically disabled under CuPy; gated by golden parity validation.

    The default keeps current behavior exactly.
    """
    return os.environ.get('NORNIR_REFINE_PREWARP_MODE', 'serial').strip().lower()


def _prewarp_cache_enabled() -> bool:
    """Return False when cross-pass prewarp caching is disabled for A/B testing or on CuPy."""
    if os.environ.get('NORNIR_DISABLE_PREWARP_CACHE', '').strip().lower() in (
            '1', 'true', 'yes', 'on'):
        return False
    # CuPy cross-pass cache changes mosaic outputs; keep CPU-only caching by
    # default, but allow opt-in via NORNIR_REFINE_PREWARP_MODE=cache (Step 7).
    if nornir_imageregistration.UsingCupy():
        return 'cache' in _prewarp_dispatch_mode()
    return True


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
            grid_dims=grid_dims_array)


def _resample_transform_to_output_grid(
        transform: nornir_imageregistration.ITransform,
        source_shape: NDArray[np.integer],
        resolved_cell_size: tuple[int, int],
        resolved_mesh_shape: tuple[int, int]) -> nornir_imageregistration.ITransform:
    """Resample a transform onto the legacy-compatible output grid lattice."""
    return nornir_imageregistration.transforms.converters.ConvertTransformToGridTransform(
        transform,
        source_image_shape=source_shape,
        cell_size=np.asarray(resolved_cell_size, dtype=np.int64),
        grid_dims=np.asarray(resolved_mesh_shape, dtype=np.int64))


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
    predicted_targets = np.asarray(grid_transform.Transform(source_points), dtype=np.float64)
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
    current_targets = np.asarray(grid_transform.TargetPoints, dtype=np.float64)
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

        for tile in list_tiles:
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

            measure_vertex_displacements = (
                _measure_grid_vertex_displacements_batched
                if _use_batched_gpu_vertex_measurement()
                else _measure_grid_vertex_displacements)
            for neighbor in neighbors[tile.ID]:
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
            pending_updates.append((tile, targets, applied_scaled))
            all_applied_components.append(np.abs(applied_scaled).reshape(-1))

            control_points_per_tile[tile.ID] = measured_count
            pass_vertex_diagnostics[tile.ID] = {
                'measured': measured_count,
                'gap_filled': filled_count,
                'updated': int(np.count_nonzero(np.any(applied_scaled != 0, axis=1))),
            }

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
    # transform. Resample only if a tile somehow is not on the output lattice.
    for tile in tiles.values():
        source_shape = _tile_source_shape_for_grid(tile)
        if _grid_transform_matches_lattice(
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

    for k, v in kwargs.items():
        prettyoutput.Log(f"\tUnused parameter to RefineStosFile: {k}:{v}\n")

    outputDir = os.path.dirname(OutputStosPath)

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
            min_alignment_overlap=min_alignment_overlap,  # type: ignore[arg-type]
            min_unmasked_area=min_unmasked_area,  # type: ignore[arg-type]
            single_thread_processing=False) as settings:

        output_transform = RefineTransform(stosTransform,
                                           settings,
                                           SaveImages=SaveImages,
                                           SavePlots=SavePlots,
                                           outputDir=outputDir)

        InputStos.Transform = nornir_imageregistration.transforms.ConvertTransformToGridTransform(output_transform,
                                                                                                  source_image_shape=settings.source_image.shape,  # type: ignore[arg-type]
                                                                                                  cell_size=settings.cell_size,
                                                                                                  grid_spacing=settings.grid_spacing)
        InputStos.Save(OutputStosPath)


def RefineTransform(stosTransform: nornir_imageregistration.ITransform,
                    settings: nornir_imageregistration.settings.GridRefinement,
                    SaveImages: bool = False,
                    SavePlots: bool = False,
                    outputDir: str | None = None) -> nornir_imageregistration.ITransform:
    """
    Iteratively refine a source-to-target transform from local alignment points.

    The routine alternates between generating candidate alignments, building an
    updated transform from cutoff-selected points, and finalizing stable points
    until convergence or pass limits are reached.
    """

    if (SavePlots or SaveImages) and outputDir is None:
        raise ValueError("outputDir must be specified if SavePlots or SaveImages is true.")

    # Convert inputs to numpy arrays

    final_pass = False  # True if this is the last iteration the loop will perform

    finalized_points = {}  # type: AlignmentRecordDict

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

    finalize_ema = EMA(settings.num_iterations // 2, 2)  # Track the cutoff values over the last three passes
    cutoff_ema = EMA(settings.num_iterations // 2, 2)
    first_cutoff = None  # The first cutoff value, we use this to decide which points make it into the final transform

    while i <= settings.num_iterations:
        if i == settings.num_iterations:
            final_pass = True

        alignment_points = _RefineGridPointsForTwoImages(stosTransform,
                                                         settings=settings,
                                                         finalized=finalized_points)

        if len(alignment_points) == 0:
            raise ValueError(f"No alignment points generated at pass #{i}")

        prettyoutput.Log(f"Pass {i} aligned {len(alignment_points)} points")

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
        cutoff_percentile_this_pass, inflection_percentile, cutoff_value_this_pass, polyfit_weights = estimate_cutoff(
            updated_and_finalized_weights_distance[:, WeightMethod.Registration])

        # cutoff_value = cutoff_ema.ema_value

        # transform_cutoff_percentile = (cutoff_percentile_this_pass + inflection_percentile) // 2
        transform_cutoff_percentile = inflection_percentile
        transform_cutoff_value = polyfit_weights[transform_cutoff_percentile]  # type: ignore[reportOptionalSubscript]
        cutoff_value = transform_cutoff_value
        cutoff_ema.add(transform_cutoff_value)
        # transform_cutoff_value = cutoff_ema.ema_value

        if first_cutoff is None:
            first_cutoff = float(cutoff_value_this_pass)

        if final_pass:
            prettyoutput.Log("FINAL PASS")
            transform_cutoff_value = first_cutoff

        prettyoutput.Log(
            f'#######\n' +
            f'Transform inclusion cutoff this pass: {transform_cutoff_percentile}% -> {transform_cutoff_value}\n' +
            f'Exponential Moving Average transform inclusion cutoff: {cutoff_ema.ema_value}\n')

        # finalize_percentile_this_pass = ((100 - cutoff_percentile) / 2.0) + cutoff_percentile

        (updatedTransform, included_alignment_records, weight_distance_composite_scores) = _PeakListToTransform(
            alignment_points,
            WeightMethod.Registration,  # type: ignore[arg-type]
            AlignRecordsToControlPoints(finalized_points.values()),  # type: ignore[arg-type]
            percentile=transform_cutoff_percentile,
            cutoff=transform_cutoff_value)

        prettyoutput.Log(f'{len(included_alignment_records)} points included in updated transform after cutoff')

        # if FirstPassCompositeScoreCutoff is None:
        #    FirstPassCompositeScoreCutoff = np.percentile(weight_distance_composite_scores[:, 2], 100.0 - percentile)
        #    FirstPassWeightScoreCutoff = np.percentile(weight_distance_composite_scores[:, 0], percentile)

        # if FirstPassFinalizeValue is not None:
        # cutoff_range = np.abs(FirstPassFinalizeValue - FirstPassWeightScoreCutoff)
        # fraction = i / (settings.num_iterations - 1)
        # FirstPassFinalizeValue - (cutoff_range * fraction)

        finalize_percentile_this_pass = cutoff_percentile_this_pass  # ((inflection_percentile + transform_cutoff_percentile) / 2.0) + transform_cutoff_value
        finalize_cutoff_this_pass = np.percentile(polyfit_weights,  # type: ignore[arg-type]
                                                  finalize_percentile_this_pass)
        finalize_ema.add(finalize_cutoff_this_pass)  # type: ignore[arg-type]

        if final_pass:
            finalize_cutoff_this_pass = first_cutoff

        # finalize_percentile_this_pass = cutoff_percentile
        # finalize_cutoff = cutoff_value_this_pass

        # finalize_percentile_this_pass = cutoff_percentile
        # finalize_cutoff = finalize_cutoff_this_pass
        finalize_cutoff_this_pass = 0  # Just use the distance measure to determine finalization
        finalize_cutoff = finalize_cutoff_this_pass

        if i != 0:
            prettyoutput.Log(
                f'Finalize cutoff this pass: {finalize_percentile_this_pass}% -> {finalize_cutoff_this_pass}\n' +
                f'Finalize Exponential Moving Average Cutoff calculated: {finalize_ema.ema_value}\n#####\n')

            new_finalized_points = CalculateFinalizedAlignmentPointsMask(alignment_points,
                                                                         percentile=finalize_percentile_this_pass,
                                                                         max_travel_distance=settings.max_travel_for_finalization,
                                                                         weight_cutoff=finalize_cutoff)
            new_finalized_alignments_list = list(
                filter(lambda index_item: new_finalized_points[index_item[0]], enumerate(alignment_points)))
        else:
            new_finalized_points = np.empty(())
            new_finalized_alignments_list = []

        # if FirstPassFinalizeValue is None:
        #     FirstPassFinalizeValue = np.percentile(weight_distance_composite_scores[:, 0],
        #                                            finalize_percentile_this_pass)

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

        finalized_points = {**finalized_points, **new_finalized_alignments_dict}

        prettyoutput.Log(
            f"Pass {i} has locked {new_finalization_count} new points, {len(finalized_points)} of {len(updated_and_finalized_alignment_points)} are locked")

        # (improved_finalized_dict, improved_alignments) = TryToImproveAlignments(updatedTransform,
        #                                                                         finalized_points,
        #                                                                         settings)

        # new_finalization_count = len(improved_finalized_dict)
        # finalized_points = {**finalized_points, **improved_finalized_dict}

        prettyoutput.Log(
            f"  Improved {len(improved_alignments)} finalized points using latest transform")

        if SavePlots:
            np.savez(os.path.join(outputDir,  # type: ignore[arg-type]
                                  f'weight_distance_composite_scores_pass{i}.npz'),
                     updated_and_finalized_weights_distance=updated_and_finalized_weights_distance,
                     weight_distance_composite_scores=weight_distance_composite_scores,
                     )
            percentile_filename = os.path.join(outputDir, f'percentile_pass{i}.svg')  # type: ignore[arg-type]
            nornir_imageregistration.views.plot_percentiles(weight_distance_composite_scores[:, 0],
                                                            percentile_filename,
                                                            title=f"Value at percentile",
                                                            horz_line_pos_list=[(transform_cutoff_value,
                                                                                 {'label': 'Transform Cutoff',
                                                                                  'color': 'green'}),
                                                                                # finalize_cutoff_this_pass, cutoff_value,
                                                                                (finalize_cutoff,
                                                                                 {'label': 'Finalize Cutoff',
                                                                                  'color': 'brown'})])

            histogram_filename = os.path.join(outputDir, f'weight_histogram_pass{i}.svg')  # type: ignore[arg-type]
            nornir_imageregistration.views.PlotWeightHistogram(alignment_points, filename=histogram_filename,
                                                               transform_cutoff=transform_cutoff_percentile / 100.0,
                                                               finalize_cutoff=finalize_percentile_this_pass / 100.0,
                                                               line_pos_list=[transform_cutoff_value,
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

        # Update the transform with the adjusted points
        combined_records_this_pass = {a.ID: a for a in included_alignment_records}
        if len(improved_alignments) > 0:
            for item in finalized_points.items():
                combined_records_this_pass[item[0]] = item[1]

            if len(combined_records_this_pass) > 2:
                print(
                    f'Building transform for next round with {len(included_alignment_records)} points and {len(finalized_points)} finalized points')
                updatedTransform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
                    AlignRecordsToControlPoints(combined_records_this_pass.values()))  # type: ignore[arg-type]

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

        if i == settings.num_iterations - 1:  # Check if the next pass is the final pass
            final_pass = True

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

    (nudged_final_points, nudged_point_keys) = TryToImproveAlignments(stosTransform, combined_records_this_pass,
                                                                      settings)
    prettyoutput.Log(
        f'Final tuning of points adjusted {len(improved_alignments)} of {len(combined_records_this_pass)} points')

    # Return a transform built from the finalized points
    if len(nudged_final_points) >= 3:
        final_transform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
            AlignRecordsToControlPoints(nudged_final_points.values()))  # type: ignore[arg-type]

    return final_transform


def _RefineGridPointsForTwoImages(transform: nornir_imageregistration.transforms.ITransform,
                                  finalized: AlignmentRecordDict,
                                  settings: nornir_imageregistration.settings.GridRefinement) -> list[
    nornir_imageregistration.EnhancedAlignmentRecord]:
    """
    Build a refinement grid, remove masked/finalized cells, and align remaining cells.
    """

    # Mark a grid along the fixed image, then find the points on the warped image

    grid_data = nornir_imageregistration.grid_subdivision.CenteredGridDivision(settings.source_image.shape,  # type: ignore[attr-defined]
                                                                               cell_size=settings.cell_size,
                                                                               grid_spacing=settings.grid_spacing,
                                                                               transform=transform)
    # grid_data = nornir_imageregistration.ITKGridDivision(settings.source_image.shape,
    #                                                                       cell_size=settings.cell_size,
    #                                                                       grid_spacing=settings.grid_spacing,
    #                                                                       transform=transform)

    # Remove finalized points from refinement consideration
    if finalized is not None and len(finalized) > 0:
        not_finalized = [tuple(grid_data.coords[i, :]) not in finalized for i in range(grid_data.coords.shape[0])]
        valid = np.asarray(not_finalized, bool)
        grid_data.RemoveMaskedPoints(valid)

    # grid_dims = nornir_imageregistration.TileGridShape(target_image.shape, grid_spacing)

    # Create target points from grid coordinates
    #    TargetPoints = coords * grid_spacing  # [np.asarray((iCol * grid_spacing[0], iRow * grid_spacing[1]), dtype=np.int32) for (iRow, iCol) in coords]

    # Grid dimensions round up, so if we are larger than image find out by how much and adjust the points so they are centered on the image
    #    overage = ((grid_dims * grid_spacing) - target_image.shape) / 2.0
    #    TargetPoints = np.round(TargetPoints - overage).astype(np.int64)
    # TODO, ensure fixedPoints are within the bounds of target_image
    grid_data.FilterOutofBoundsSourcePoints(settings.source_image.shape)
    grid_data.RemoveCellsUsingSourceImageMask(settings.source_mask, settings.min_unmasked_area)
    # nornir_imageregistration.views.grid_data.PlotGridPositionsAndMask(grid_data.SourcePoints, source_mask, OutputFilename=None)

    if grid_data.num_points == 0:
        # There is nothing to refine, perhaps the image is too small for the grid cell size?
        # prettyoutput.LogErr("No points meet criteria for grid refinement")
        raise ValueError("No points meet criteria for grid refinement")

    grid_data.PopulateTargetPoints(transform)
    grid_data.RemoveCellsUsingTargetImageMask(settings.target_mask, settings.min_unmasked_area)

    # nornir_imageregistration.views.grid_data.PlotGridPositionsAndMask(grid_data.TargetPoints, target_mask, OutputFilename=None)
    # grid_data.ApplyWarpedImageMask(source_mask)
    #     valid_inbounds = np.logical_and(np.all(FixedPoi4nts >= np.asarray((0, 0)), 1), np.all(TargetPoints < target_mask.shape, 1))
    #     TargetPoints = TargetPoints[valid_inbounds, :]
    #     coords = coords[valid_inbounds, :]
    #

    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]
    #
    #     # Filter Fixed points falling outside the mask
    #     if target_mask is not None:
    #         valid = nornir_imageregistration.index_with_array(target_mask, TargetPoints)
    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]
    #
    #     SourcePoints = transform.InverseTransform(TargetPoints).astype(np.int32)
    #     if source_mask is not None:
    #         valid = np.logical_and(np.all(SourcePoints >= np.asarray((0, 0)), 1), np.all(SourcePoints < source_mask.shape, 1))
    #         SourcePoints = SourcePoints[valid, :]
    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]
    #
    #         valid = nornir_imageregistration.index_with_array(source_mask, SourcePoints)
    #         SourcePoints = SourcePoints[valid, :]
    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]

    return _RefinePointsForTwoImages(transform, [tuple(row) for row in grid_data.coords], grid_data.SourcePoints,
                                     grid_data.TargetPoints, settings)


def _RefinePointsForTwoImages(transform: nornir_imageregistration.transforms.ITransform,
                              keys: list[tuple[int, int]],
                              sourcePoints: np.ndarray,
                              targetPoints: np.ndarray,
                              settings: nornir_imageregistration.settings.GridRefinement) -> list[
    nornir_imageregistration.EnhancedAlignmentRecord]:
    """
    Register corresponding source/target neighborhoods for each control-point key.
    """

    if len(keys) != targetPoints.shape[0]:
        raise ValueError("keys must have equal number of entries as points")

    nPoints = len(keys)

    pool = nornir_pools.GetGlobalSerialPool() if nornir_imageregistration.UsingCupy() else nornir_pools.GetGlobalMultithreadingPool()
    # pool = nornir_pools.GetGlobalThreadPool()
    tasks = list()
    alignment_records = list()

    rigid_transforms = ApproximateRigidTransformBySourcePoints(input_transform=transform, source_points=sourcePoints,
                                                               cell_size=settings.cell_size)

    if settings.single_thread_processing:
        target_image = settings.target_image
        source_image = settings.source_image
        for i in range(nPoints):
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
                flipped_ud=arecord.flippedud)

            if nornir_imageregistration.in_debug_mode():
                erec.TargetROI = arecord.TargetROI  # type: ignore[attr-defined]
                erec.SourceROI = arecord.SourceROI  # type: ignore[attr-defined]
                erec.TranslatedSourceROI = nornir_imageregistration.CropImage(
                    erec.SourceROI,  # type: ignore[attr-defined]
                    int(np.floor(-erec.peak[1])),
                    int(np.floor(-erec.peak[0])),
                    erec.SourceROI.shape[1],  # type: ignore[attr-defined]
                    erec.SourceROI.shape[0],  # type: ignore[attr-defined]
                    cval=float(np.median(erec.SourceROI.flat)))  # type: ignore[attr-defined]

            alignment_records.append(erec)

        return alignment_records

    for i in range(nPoints):
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

    for t in tasks:
        arecord = t.wait_return()
        if arecord is None:
            continue

        erec = nornir_imageregistration.EnhancedAlignmentRecord(ID=t.key,
                                                                TargetPoint=targetPoints[t.ID, :],
                                                                SourcePoint=sourcePoints[t.ID, :],
                                                                peak=arecord.peak,
                                                                weight=arecord.weight,
                                                                angle=arecord.angle,
                                                                flipped_ud=arecord.flippedud)

        if nornir_imageregistration.in_debug_mode():
            erec.TargetROI = arecord.TargetROI  # type: ignore[attr-defined]
            erec.SourceROI = arecord.SourceROI  # type: ignore[attr-defined]
            erec.TranslatedSourceROI = nornir_imageregistration.CropImage(erec.SourceROI, int(np.floor(-erec.peak[1])),  # type: ignore[attr-defined]
                                                                          int(np.floor(-erec.peak[0])),
                                                                          erec.SourceROI.shape[1],  # type: ignore[attr-defined]
                                                                          erec.SourceROI.shape[0],  # type: ignore[attr-defined]
                                                                          cval=float(np.median(erec.SourceROI.flat)))  # type: ignore[attr-defined]

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
    """

    SourcePoints = np.asarray(list(map(lambda a: a.SourcePoint, alignment_records)))
    TargetPoints = np.asarray(list(map(lambda a: a.AdjustedTargetPoint, alignment_records)))

    PointPairs = np.hstack((TargetPoints, SourcePoints))
    return PointPairs


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

        num_fixed = fixed_points.shape[0]

    num_alignments = len(alignment_records)
    if num_alignments == 0:
        raise ValueError("Need at one new alignment_record to improve a transform")

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
        top_alignment_indices = sorted_composite_indices[0:num_needed]
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
                                    grid_spacing: NDArray | None = None) -> nornir_imageregistration.transforms.triangulation.Triangulation:
    """
    Resample an arbitrary transform onto an ITK-style grid triangulation lattice.
    """

    grid_data = nornir_imageregistration.ITKGridDivision(source_image_shape, cell_size=cell_size,
                                                         grid_spacing=grid_spacing, grid_dims=grid_dims)
    grid_data.PopulateTargetPoints(Transform)

    point_pairs = np.hstack((grid_data.TargetPoints, grid_data.SourcePoints))

    # TODO, create a specific grid transform object that uses numpy's RegularGridInterpolator

    T = nornir_imageregistration.transforms.triangulation.Triangulation(point_pairs)
    T.gridWidth = grid_data.grid_dims[1]  # type: ignore[attr-defined]
    T.gridHeight = grid_data.grid_dims[0]  # type: ignore[attr-defined]

    return T


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
    Select points eligible for finalization by weight and travel-distance cutoffs.
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
                                            cell_size: NDArray[np.integer] | None = None) -> list[
    nornir_imageregistration.transforms.IRigidTransform] | list[nornir_imageregistration.transforms.Rigid]:
    """
    Estimate local rigid transforms at target points via inverse-mapped source points.
    """

    if isinstance(input_transform, nornir_imageregistration.transforms.IRigidTransform):
        return [input_transform] * target_points.shape[0]

    target_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(target_points)

    numPoints = target_points.shape[0]

    source_points = input_transform.InverseTransform(target_points)

    return ApproximateRigidTransformBySourcePoints(input_transform, source_points, cell_size)
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
            estimated_cell_distance = float(
                xp.min(pairwise_cdist(source_points[0:1, :], source_points[1:, :]))
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
                                            cell_size: NDArray | None = None) -> list[
    nornir_imageregistration.transforms.IRigidTransform]:
    """
    Estimate one local rigid transform per source point using transformed ring samples.
    """

    source_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_points)
    xp = cp.get_array_module(source_points)

    numPoints = int(source_points.shape[0])  # type: int

    # If the input transform is rigid, then we simply return that
    if isinstance(input_transform, nornir_imageregistration.transforms.IRigidTransform):
        return [input_transform] * numPoints

    output_transforms = []

    for iPoint in range(0, numPoints):
        source_point = source_points[iPoint, :]

        # Using the actual cell size can help avoid wildly incorrect scale values for the estimates rigid transforms
        offset = calculate_offset(source_points, cell_size)
        offset_distance = xp.linalg.norm(offset)

        source_point_ring = _calculate_offset_ring(source_point, offset_distance)  # type: ignore[arg-type]

        target_points = input_transform.Transform(source_point_ring)
        target_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(target_points)

        rigid_transform_components = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(
            source_points=source_point_ring,
            target_points=target_points)

        rigid_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=rigid_transform_components.translation,
            source_rotation_center=rigid_transform_components.source_rotation_center,
            angle=rigid_transform_components.angle,
            flip_ud=rigid_transform_components.reflected,
            scalar=rigid_transform_components.scale)

        # This debug check is here to warn if the rigid transform is not working correctly.  It can be removed,
        # but verify that the rigid transform returned is returning the correct ROI if used for alignment
        # test_target_point = rigid_transform.Transform(source_point)
        # if not np.allclose(test_target_point, target_points[0], atol=1):
        #     raise ValueError(
        #         f"Rigid transform failed to align point: Expected {target_points[0]} got {test_target_point}")

        output_transforms.append(rigid_transform)

    return output_transforms


def BuildAlignmentROIs(transform: nornir_imageregistration.ITransform,
                       targetImage_param: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
                       sourceImage_param: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
                       target_image_stats: nornir_imageregistration.ImageStats | None,
                       source_image_stats: nornir_imageregistration.ImageStats | None,
                       target_controlpoint: NDArray | tuple[float, float],
                       alignmentArea: NDArray | tuple[float, float],
                       description: str | None = None) -> tuple[NDArray, NDArray]:
    """
    Extract target/source ROIs in a common target-space frame for local registration.
    """
    xp = nornir_imageregistration.GetComputationModule()

    targetImage = nornir_imageregistration.ImageParamToImageArray(targetImage_param,  # type: ignore[arg-type]
                                                                  dtype=nornir_imageregistration.default_image_dtype())
    sourceImage = nornir_imageregistration.ImageParamToImageArray(sourceImage_param,  # type: ignore[arg-type]
                                                                  dtype=nornir_imageregistration.default_image_dtype())

    # Adjust the point by 0.5 if it is an odd-sized area to ensure the output is centered on the desired pixel
    target_controlpoint = target_controlpoint.astype(float, copy=False).flatten()  # type: ignore[union-attr]
    adjust_mask = np.mod(alignmentArea, 2) > 0
    target_controlpoint[adjust_mask] += 0.5

    target_rectangle = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
        point=(target_controlpoint[0] - (alignmentArea[0] / 2.0),
               target_controlpoint[1] - (alignmentArea[1] / 2.0)),
        area=alignmentArea)

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

    if source_image_stats is not None:
        roi_array = cast(Any, source_image_roi)
        roi_xp = cp.get_array_module(roi_array)
        nan_mask = roi_xp.isnan(roi_array)
        if bool(nan_mask.all()):
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
                           min_alignment_overlap: float = 0.5) -> nornir_pools.Task | None:
    """Create and enqueue an asynchronous rigid-registration task for one point."""
    if anglesToSearch is None:
        anglesToSearch = np.linspace(-7.5, 7.5, 11)
        # Ensure we check a non-rotated alignment
        anglesToSearch = np.union1d(anglesToSearch, [0])

    rigid_transform = ApproximateRigidTransformByTargetPoints(input_transform=transform,
                                                              target_points=target_controlpoint,  # type: ignore[arg-type]
                                                              cell_size=alignmentArea)  # type: ignore[arg-type]

    target_image_roi, source_image_roi = BuildAlignmentROIs(transform=rigid_transform[0],
                                                            targetImage_param=targetImage,
                                                            sourceImage_param=sourceImage,
                                                            target_image_stats=target_image_stats,
                                                            source_image_stats=source_image_stats,
                                                            target_controlpoint=target_controlpoint,
                                                            alignmentArea=alignmentArea,
                                                            description=taskname)

    target_image_roi = EnsureMaxContrast(target_image_roi)
    source_image_roi = EnsureMaxContrast(source_image_roi)

    # Just ignore pure color regions
    if not np.any(target_image_roi != target_image_roi[0][0]):
        return None
    if not np.any(source_image_roi != source_image_roi[0][0]):
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
                         method=nornir_imageregistration.settings.SliceToSliceMethod.BruteForce)

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
                      min_alignment_overlap: float = 0.5) -> nornir_imageregistration.AlignmentRecord | None:
    """Run synchronous rigid-registration for one control point."""
    if anglesToSearch is None:
        anglesToSearch = np.linspace(-7.5, 7.5, 11)

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
                                                                description='')
    except ValueError:
        return None

    # Just ignore pure color regions
    if not np.any(target_image_roi != target_image_roi[0][0]):
        return None
    if not np.any(source_image_roi != source_image_roi[0][0]):
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
    result = nornir_imageregistration.stos_brute.SliceToSliceRigidRegistration(
        target_image=target_image_roi,
        source_image=source_image_roi,
        AngleSearchRange=anglesToSearch,  # type: ignore[arg-type]
        MinOverlap=min_alignment_overlap,
        SingleThread=True,
        TestFlip=False,
        method=SliceToSliceMethod.BruteForce)

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
                                                                       flipped_ud=chosen_record.flippedud)

        # output[key].PSDDelta = chosen_record.PSDDelta

    # Close the pool to prevent threads from hanging around
    # pool.shutdown()
    return output, improved_alignments

