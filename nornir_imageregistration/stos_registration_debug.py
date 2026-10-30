"""Debug harness for a single STOS pair: inspect geometry and run one registration algorithm."""
from __future__ import annotations

import argparse
import enum
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

import nornir_imageregistration
from nornir_imageregistration.alignment_record import AlignmentRecord
from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.image_permutation_helper import ImagePermutationHelper
from nornir_imageregistration.local_distortion_correction import RefineTransform
from nornir_imageregistration.overlapmasking import GetOverlapMask
from nornir_imageregistration.settings.stos_brute import SliceToSliceMethod, StosBruteSettings
from nornir_imageregistration.spatial.rectangle import Rectangle
from nornir_imageregistration.stos_brute import (
    SliceToSliceRigidRegistrationWithPreprocessedImages,
    _fixed_correlation_shape,
)
from nornir_imageregistration.transforms.base import IRigidTransform, ITransform
from nornir_imageregistration.transforms.converters import ConvertTransformToRigidTransform
from nornir_imageregistration.transforms.factory import LoadTransform

PYRE_LOGPOLAR_LARGEST_DIMENSION: int = 818
PYRE_LOGPOLAR_MIN_OVERLAP: float = 0.75
LOCAL_RIGID_HALF_WIDTH_DEG: float = 5.0
LOCAL_RIGID_STEP_DEG: float = 1.0
DEFAULT_GRID_REFINE_ITERATIONS: int = 1
DEFAULT_INSPECT_MIN_OVERLAP: float = 0.75
DEFAULT_OVERLAP_LEVELS: tuple[float, ...] = (0.75, 0.5, 0.25)
RC2_GRID16_239_240_STOS: str = (
    r"Y:\Volumes\RC2\TEM\Grid16\239-240_ctrl-TEM_Leveled_map-TEM_Leveled.stos"
)
RC2_GRID16_239_240_MANUAL_STOS: str = (
    r"Y:\Volumes\RC2\TEM\Grid16\Manual\239-240_ctrl-TEM_Leveled_map-TEM_Leveled.stos"
)

_GRID_DOWNSAMPLE_RE = re.compile(r"(?:Grid|StosBrute)(\d+)", re.IGNORECASE)

_INSPECT_KWARGS = frozenset({
    "min_overlap",
    "larget_dimension",
    "largest_dimension",
})
_RIGID_KWARGS = frozenset({
    "min_overlap",
    "larget_dimension",
    "largest_dimension",
    "try_flipped",
    "angles",
    "initial_scale_hint",
    "estimated_scale_hint",
    "source_image_scale_factors",
    "SingleThread",
    "Cluster",
})
_LOCAL_RIGID_KWARGS = _RIGID_KWARGS | frozenset({
    "half_width_deg",
    "step_deg",
})
_GRID_REFINE_KWARGS = frozenset({
    "num_iterations",
    "cell_size",
    "grid_spacing",
    "angles_to_search",
    "final_pass_angles",
    "min_alignment_overlap",
    "min_unmasked_area",
    "single_thread_processing",
})


class StosDebugAlgorithm(enum.StrEnum):
    """Registration path to run for one STOS pair."""

    inspect = "inspect"
    logpolar = "logpolar"
    bruteforce = "bruteforce"
    local_rigid = "local_rigid"
    grid_refine = "grid_refine"


_ALGORITHM_KWARGS: dict[StosDebugAlgorithm, frozenset[str]] = {
    StosDebugAlgorithm.inspect: _INSPECT_KWARGS,
    StosDebugAlgorithm.logpolar: _RIGID_KWARGS,
    StosDebugAlgorithm.bruteforce: _RIGID_KWARGS,
    StosDebugAlgorithm.local_rigid: _LOCAL_RIGID_KWARGS,
    StosDebugAlgorithm.grid_refine: _GRID_REFINE_KWARGS,
}


@dataclass
class StosPairContext:
    """Loaded STOS pair with source (mapped) and target (control) images."""

    stos_path: str
    stos: StosFile
    source: ImagePermutationHelper
    target: ImagePermutationHelper
    stored_transform: ITransform | None
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]


@dataclass
class OverlapMaskStats:
    """Eligible-peak fraction of GetOverlapMask at one min_overlap and image scale."""

    min_overlap: float
    larget_dimension: int | None
    scalar: float
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    correlation_shape: tuple[int, int]
    eligible_fraction: float


@dataclass
class StosInspectReport:
    """Geometry and overlap diagnostics that do not run registration."""

    stos_path: str
    source_image_path: str
    target_image_path: str
    source_mask_path: str | None
    target_mask_path: str | None
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    downsample_hint: int | None
    stored_transform_type: str | None
    stored_angle_deg: float | None
    stored_offset_yx: tuple[float, float] | None
    stored_scale: float | None
    stored_flip_ud: bool | None
    source_valid_fraction: float
    target_valid_fraction: float
    stored_bbox_overlap: float | None
    overlap_masks: list[OverlapMaskStats]


@dataclass
class RigidDebugReport:
    """Structured fields from an AlignmentRecord plus bbox / no-peak flags."""

    weight: float
    peak_yx: tuple[float, float]
    angle_deg: float
    scale: float
    flip_ud: bool
    peak_ratio: float | None
    no_peak_fallback: bool
    bbox_overlap: float
    peak_exceeds_image_dimensions: bool
    alignment_repr: str


@dataclass
class GridRefineDebugReport:
    """Summary of a grid-refine debug run."""

    result_transform_type: str
    bbox_overlap: float | None
    num_iterations: int


@dataclass
class ReferencePoseReport:
    """Equivalent rigid pose from a known-good STOS, scaled into the pair's pixel space."""

    stos_path: str
    auto_discovered: bool
    downsample_hint: int | None
    transform_type: str | None
    used_equivalent_rigid: bool
    angle_deg: float | None
    peak_yx: tuple[float, float] | None
    peak_yx_in_pair: tuple[float, float] | None
    scale: float | None
    flip_ud: bool | None
    pixel_scale_to_pair: float


@dataclass
class PoseComparisonReport:
    """Result pose minus reference pose (angles wrapped to ±180°)."""

    angle_delta_deg: float | None
    peak_delta_yx: tuple[float, float] | None
    scale_ratio: float | None


@dataclass
class StosDebugResult:
    """Inspect report plus the outcome of one algorithm."""

    algorithm: StosDebugAlgorithm
    kwargs_used: dict[str, Any]
    inspect: StosInspectReport
    alignment: AlignmentRecord | None = None
    rigid: RigidDebugReport | None = None
    grid: GridRefineDebugReport | None = None
    reference: ReferencePoseReport | None = None
    comparison: PoseComparisonReport | None = None
    notes: list[str] = field(default_factory=list)


def pyre_logpolar_kwargs() -> dict[str, Any]:
    """Kwargs matching Pyre Operations→Log Polar (LimitImageSize=818, min_overlap=0.75)."""
    return {
        "min_overlap": PYRE_LOGPOLAR_MIN_OVERLAP,
        "larget_dimension": PYRE_LOGPOLAR_LARGEST_DIMENSION,
        "try_flipped": True,
    }


def is_no_peak_fallback(record: AlignmentRecord) -> bool:
    """True when find_peak reported no usable peak (weight 0)."""
    return float(record.weight) == 0.0


def peak_exceeds_image_dimensions(
        peak: NDArray[np.floating] | Sequence[float],
        source_shape: tuple[int, int],
        target_shape: tuple[int, int]) -> bool:
    """True if |peak| on each axis exceeds both source and target sizes (no possible overlap)."""
    peak_arr = nornir_imageregistration.EnsureNumpyArray(peak, dtype=np.float64).reshape(2)
    max_h = max(int(source_shape[0]), int(target_shape[0]))
    max_w = max(int(source_shape[1]), int(target_shape[1]))
    return abs(float(peak_arr[0])) > max_h and abs(float(peak_arr[1])) > max_w


def bbox_overlap_for_alignment(
        record: AlignmentRecord,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int]) -> float:
    """Axis-aligned bbox overlap of the aligned source vs the target image, in [0, 1]."""
    corners = record.GetTransformedCornerPoints(np.asarray(source_shape, dtype=np.int64))
    return _bbox_overlap_from_corners(corners, target_shape)


def bbox_overlap_for_transform(
        transform: ITransform,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int]) -> float:
    """Axis-aligned bbox overlap of source corners mapped by *transform* vs the target image."""
    corners = _source_corners(source_shape)
    # Rectangle overlap is host-only; four corners is a small explicit transfer.
    mapped = nornir_imageregistration.EnsureNumpyArray(
        transform.Transform(corners), dtype=np.float64)
    return _bbox_overlap_from_corners(mapped, target_shape)


def rigid_report_for_alignment(
        record: AlignmentRecord,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int]) -> RigidDebugReport:
    """Build a RigidDebugReport from an AlignmentRecord without running registration."""
    peak = nornir_imageregistration.EnsureNumpyArray(record.peak, dtype=np.float64).reshape(2)
    return RigidDebugReport(
        weight=float(record.weight),
        peak_yx=(float(peak[0]), float(peak[1])),
        angle_deg=float(record.angle),
        scale=float(record.scale),
        flip_ud=bool(record.flippedud),
        peak_ratio=None if record.peak_ratio is None else float(record.peak_ratio),
        no_peak_fallback=is_no_peak_fallback(record),
        bbox_overlap=bbox_overlap_for_alignment(record, source_shape, target_shape),
        peak_exceeds_image_dimensions=peak_exceeds_image_dimensions(
            peak, source_shape, target_shape),
        alignment_repr=str(record),
    )


def load_stos_pair(stos_path: str) -> StosPairContext:
    """Load a .stos, resolve image paths, and wrap source/target as ImagePermutationHelper."""
    if not os.path.isfile(stos_path):
        raise FileNotFoundError(stos_path)

    stos_dir = os.path.dirname(os.path.abspath(stos_path))
    stos = StosFile.Load(stos_path)
    stos.TryConvertRelativePathsToAbsolutePaths(stos_dir)

    source = ImagePermutationHelper(
        img=stos.MappedImageFullPath,
        mask=_existing_path(stos.MappedMaskFullPath),
        extrema_mask_size_cuttoff=None,
        dtype=nornir_imageregistration.default_image_dtype())
    target = ImagePermutationHelper(
        img=stos.ControlImageFullPath,
        mask=_existing_path(stos.ControlMaskFullPath),
        extrema_mask_size_cuttoff=None,
        dtype=nornir_imageregistration.default_image_dtype())

    stored: ITransform | None = None
    if stos.Transform:
        try:
            stored = LoadTransform(stos.Transform, pixelSpacing=1)
        except (ValueError, TypeError, AttributeError):
            stored = None

    return StosPairContext(
        stos_path=os.path.abspath(stos_path),
        stos=stos,
        source=source,
        target=target,
        stored_transform=stored,
        source_shape=(int(source.shape[0]), int(source.shape[1])),
        target_shape=(int(target.shape[0]), int(target.shape[1])),
    )


def find_manual_reference_stos(stos_path: str) -> str | None:
    """Return ``<stos_dir>/Manual/<basename>`` when that file exists and is not *stos_path*."""
    stos_abs = os.path.abspath(stos_path)
    candidate = os.path.join(os.path.dirname(stos_abs), "Manual", os.path.basename(stos_abs))
    if not os.path.isfile(candidate):
        return None
    if os.path.normcase(os.path.normpath(candidate)) == os.path.normcase(stos_abs):
        return None
    return os.path.abspath(candidate)


def load_reference_pose(
        reference_stos_path: str,
        pair: StosPairContext,
        *,
        auto_discovered: bool = False) -> ReferencePoseReport:
    """Load a known-good STOS transform (no images) and express its rigid pose in pair pixels."""
    if not os.path.isfile(reference_stos_path):
        raise FileNotFoundError(reference_stos_path)

    stos_dir = os.path.dirname(os.path.abspath(reference_stos_path))
    stos = StosFile.Load(reference_stos_path)
    stos.TryConvertRelativePathsToAbsolutePaths(stos_dir)
    transform: ITransform | None = None
    if stos.Transform:
        try:
            transform = LoadTransform(stos.Transform, pixelSpacing=1)
        except (ValueError, TypeError, AttributeError):
            transform = None

    source_shape = _stos_file_shape(getattr(stos, "MappedImageDim", None)) or pair.source_shape
    target_shape = _stos_file_shape(getattr(stos, "ControlImageDim", None)) or pair.target_shape
    equivalent = _equivalent_rigid(transform)
    pose_transform = equivalent if equivalent is not None else transform
    angle_deg, _, scale, flip_ud = _rigid_params(pose_transform)
    peak_yx = None if pose_transform is None else _peak_from_transform(
        pose_transform, source_shape, target_shape)
    ref_ds = _downsample_hint_from_path(reference_stos_path)
    pair_ds = _downsample_hint_from_path(pair.stos_path)
    pixel_scale = 1.0
    if ref_ds is not None and pair_ds is not None and pair_ds != 0:
        pixel_scale = float(ref_ds) / float(pair_ds)
    peak_in_pair = None if peak_yx is None else (peak_yx[0] * pixel_scale, peak_yx[1] * pixel_scale)
    return ReferencePoseReport(
        stos_path=os.path.abspath(reference_stos_path),
        auto_discovered=auto_discovered,
        downsample_hint=ref_ds,
        transform_type=None if transform is None else type(transform).__name__,
        used_equivalent_rigid=equivalent is not None and not isinstance(transform, IRigidTransform),
        angle_deg=angle_deg,
        peak_yx=peak_yx,
        peak_yx_in_pair=peak_in_pair,
        scale=scale,
        flip_ud=flip_ud,
        pixel_scale_to_pair=pixel_scale,
    )


def inspect_stos_pair(
        pair: StosPairContext,
        min_overlap: float = DEFAULT_INSPECT_MIN_OVERLAP,
        larget_dimension: int | None = None) -> StosInspectReport:
    """Compute load/geometry/overlap-mask diagnostics without running registration."""
    stored = pair.stored_transform
    angle_deg, offset_yx, scale, flip_ud = _rigid_params(stored)
    overlap_levels = _unique_overlap_levels(min_overlap)
    overlap_masks: list[OverlapMaskStats] = []
    for level in overlap_levels:
        overlap_masks.append(_overlap_mask_stats(
            pair.source_shape, pair.target_shape, level, None))
        if larget_dimension is not None:
            overlap_masks.append(_overlap_mask_stats(
                pair.source_shape, pair.target_shape, level, larget_dimension))

    stored_bbox: float | None = None
    if stored is not None:
        stored_bbox = bbox_overlap_for_transform(
            stored, pair.source_shape, pair.target_shape)

    return StosInspectReport(
        stos_path=pair.stos_path,
        source_image_path=pair.stos.MappedImageFullPath,
        target_image_path=pair.stos.ControlImageFullPath,
        source_mask_path=_existing_path(pair.stos.MappedMaskFullPath),
        target_mask_path=_existing_path(pair.stos.ControlMaskFullPath),
        source_shape=pair.source_shape,
        target_shape=pair.target_shape,
        downsample_hint=_downsample_hint_from_path(pair.stos_path),
        stored_transform_type=None if stored is None else type(stored).__name__,
        stored_angle_deg=angle_deg,
        stored_offset_yx=offset_yx,
        stored_scale=scale,
        stored_flip_ud=flip_ud,
        source_valid_fraction=_valid_fraction(pair.source),
        target_valid_fraction=_valid_fraction(pair.target),
        stored_bbox_overlap=stored_bbox,
        overlap_masks=overlap_masks,
    )


def run_stos_debug(
        stos_path: str | StosPairContext,
        algorithm: StosDebugAlgorithm | str,
        *,
        reference_stos: str | None = None,
        **kwargs: Any) -> StosDebugResult:
    """Load one STOS pair, inspect it, and run exactly one registration algorithm.

    Unknown kwargs raise TypeError listing the names accepted for *algorithm*.
    If *reference_stos* is omitted, ``<stos_dir>/Manual/<basename>`` is used when present.
    """
    algorithm = StosDebugAlgorithm(algorithm)
    kwargs = _normalize_kwargs(algorithm, kwargs)
    pair = stos_path if isinstance(stos_path, StosPairContext) else load_stos_pair(stos_path)

    notes: list[str] = []
    auto_discovered = False
    resolved_reference = reference_stos
    if resolved_reference is None:
        resolved_reference = find_manual_reference_stos(pair.stos_path)
        auto_discovered = resolved_reference is not None
    reference_report: ReferencePoseReport | None = None
    if resolved_reference is not None:
        reference_report = load_reference_pose(
            resolved_reference, pair, auto_discovered=auto_discovered)
        origin = "Manual auto" if auto_discovered else "explicit"
        notes.append(f"reference stos ({origin}): {reference_report.stos_path}")

    def _finish(
            inspect_report: StosInspectReport,
            *,
            alignment: AlignmentRecord | None = None,
            rigid: RigidDebugReport | None = None,
            grid: GridRefineDebugReport | None = None) -> StosDebugResult:
        comparison = _compare_to_reference(reference_report, rigid)
        if comparison is not None:
            notes.extend(_comparison_notes(comparison, reference_report))
        result = StosDebugResult(
            algorithm=algorithm,
            kwargs_used=kwargs,
            inspect=inspect_report,
            alignment=alignment,
            rigid=rigid,
            grid=grid,
            reference=reference_report,
            comparison=comparison,
            notes=list(notes),
        )
        return result

    if algorithm is StosDebugAlgorithm.inspect:
        inspect_report = inspect_stos_pair(
            pair,
            min_overlap=float(kwargs.get("min_overlap", DEFAULT_INSPECT_MIN_OVERLAP)),
            larget_dimension=_optional_int(kwargs.get("larget_dimension")),
        )
        _annotate_empty_overlap_masks(inspect_report, notes)
        _annotate_non_rigid_stored(inspect_report, notes)
        return _finish(inspect_report)

    if algorithm is StosDebugAlgorithm.grid_refine:
        inspect_report = inspect_stos_pair(pair)
        _annotate_empty_overlap_masks(inspect_report, notes)
        _annotate_non_rigid_stored(inspect_report, notes)
        grid_report, grid_notes = _run_grid_refine(pair, kwargs)
        notes.extend(grid_notes)
        return _finish(inspect_report, grid=grid_report)

    rigid_kwargs = dict(kwargs)
    method = SliceToSliceMethod.LogPolar
    if algorithm is StosDebugAlgorithm.bruteforce:
        method = SliceToSliceMethod.BruteForce
    elif algorithm is StosDebugAlgorithm.local_rigid:
        method = SliceToSliceMethod.BruteForce
        rigid_kwargs = _local_rigid_kwargs(pair, rigid_kwargs)

    settings = _stos_brute_settings(method, rigid_kwargs)
    inspect_report = inspect_stos_pair(
        pair,
        min_overlap=float(settings.min_overlap),
        larget_dimension=settings.larget_dimension,
    )
    _annotate_empty_overlap_masks(inspect_report, notes)
    _annotate_non_rigid_stored(inspect_report, notes)
    record = SliceToSliceRigidRegistrationWithPreprocessedImages(
        source_image_data=pair.source,
        target_image_data=pair.target,
        settings=settings,
        SingleThread=bool(rigid_kwargs.get("SingleThread", True)),
        Cluster=bool(rigid_kwargs.get("Cluster", False)),
    )
    rigid = rigid_report_for_alignment(record, pair.source_shape, pair.target_shape)
    notes.extend(_rigid_notes(rigid))
    return _finish(inspect_report, alignment=record, rigid=rigid)


def result_to_jsonable(result: StosDebugResult) -> dict[str, Any]:
    """Convert a StosDebugResult to JSON-serializable primitives."""
    payload = asdict(result)
    payload["algorithm"] = str(result.algorithm)
    payload.pop("alignment", None)
    if result.alignment is not None:
        payload["alignment"] = _alignment_to_dict(result.alignment)
    payload["kwargs_used"] = _jsonable(result.kwargs_used)
    return _jsonable(payload)


def format_human_summary(result: StosDebugResult) -> str:
    """Render a readable multi-line summary of a debug result."""
    inspect = result.inspect
    lines = [
        f"STOS: {inspect.stos_path}",
        f"algorithm: {result.algorithm}",
        f"source: {inspect.source_image_path} {inspect.source_shape}",
        f"target: {inspect.target_image_path} {inspect.target_shape}",
        f"downsample hint: {inspect.downsample_hint}",
        f"stored transform: {inspect.stored_transform_type}",
        f"stored angle deg: {inspect.stored_angle_deg}",
        f"stored offset yx: {inspect.stored_offset_yx}",
        f"stored scale: {inspect.stored_scale} flip: {inspect.stored_flip_ud}",
        f"source valid fraction: {inspect.source_valid_fraction:.4f}",
        f"target valid fraction: {inspect.target_valid_fraction:.4f}",
        f"stored bbox overlap: {inspect.stored_bbox_overlap}",
    ]
    for mask_stats in inspect.overlap_masks:
        dim = "native" if mask_stats.larget_dimension is None else f"dim={mask_stats.larget_dimension}"
        lines.append(
            f"overlap mask min_overlap={mask_stats.min_overlap} {dim}: "
            f"eligible={mask_stats.eligible_fraction:.4f} "
            f"corr={mask_stats.correlation_shape} scalar={mask_stats.scalar:.4f}"
        )
    if result.rigid is not None:
        rigid = result.rigid
        lines.extend([
            f"alignment: {rigid.alignment_repr}",
            f"weight: {rigid.weight:.4f} peak_yx: {rigid.peak_yx} angle: {rigid.angle_deg:.4f}",
            f"scale: {rigid.scale:.4f} flip: {rigid.flip_ud} peak_ratio: {rigid.peak_ratio}",
            f"no_peak_fallback: {rigid.no_peak_fallback}",
            f"result bbox overlap: {rigid.bbox_overlap:.4f}",
            f"peak exceeds image dimensions: {rigid.peak_exceeds_image_dimensions}",
        ])
    if result.reference is not None:
        ref = result.reference
        origin = "Manual auto" if ref.auto_discovered else "explicit"
        lines.extend([
            f"reference ({origin}): {ref.stos_path}",
            f"reference transform: {ref.transform_type} equivalent_rigid={ref.used_equivalent_rigid}",
            f"reference angle deg: {ref.angle_deg}",
            f"reference peak yx (pair px): {ref.peak_yx_in_pair}",
            f"reference scale: {ref.scale} flip: {ref.flip_ud} pixel_scale: {ref.pixel_scale_to_pair:.4f}",
        ])
    if result.comparison is not None:
        cmp = result.comparison
        lines.extend([
            f"vs reference angle delta deg: {cmp.angle_delta_deg}",
            f"vs reference peak delta yx: {cmp.peak_delta_yx}",
            f"vs reference scale ratio: {cmp.scale_ratio}",
        ])
    if result.grid is not None:
        lines.extend([
            f"grid transform: {result.grid.result_transform_type}",
            f"grid iterations: {result.grid.num_iterations}",
            f"grid bbox overlap: {result.grid.bbox_overlap}",
        ])
    if result.notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``python -m nornir_imageregistration.stos_registration_debug``."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    kwargs = _kwargs_from_cli(args)
    result = run_stos_debug(
        args.stos_path, args.algorithm, reference_stos=args.reference_stos, **kwargs)
    print(format_human_summary(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result_to_jsonable(result), handle, indent=2)
        print(f"wrote JSON: {args.json}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a STOS pair and run one registration algorithm with explicit kwargs.")
    parser.add_argument("stos_path", help="Path to a .stos file")
    parser.add_argument(
        "--algorithm",
        type=StosDebugAlgorithm,
        default=StosDebugAlgorithm.inspect,
        choices=list(StosDebugAlgorithm),
        help="Algorithm to run (default: inspect)")
    parser.add_argument(
        "--preset",
        choices=["pyre-logpolar"],
        default=None,
        help="Fill kwargs from a named preset; explicit flags override")
    parser.add_argument("--min-overlap", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--largest-dimension",
        type=_parse_optional_int,
        default=argparse.SUPPRESS,
        help="LimitImageSize equivalent (int), or 'none' for full resolution")
    parser.add_argument(
        "--try-flipped",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS)
    parser.add_argument("--num-iterations", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--cell-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--grid-spacing", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--half-width-deg", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--step-deg", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--reference-stos",
        default=None,
        help="Known-good .stos for pose comparison. Default: <stos_dir>/Manual/<basename> if present")
    parser.add_argument("--json", dest="json", default=None, help="Write structured result JSON")
    return parser


def _kwargs_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    preset_keys: set[str] = set()
    if args.preset == "pyre-logpolar":
        kwargs.update(pyre_logpolar_kwargs())
        preset_keys = set(kwargs)

    cli_map = {
        "min_overlap": "min_overlap",
        "largest_dimension": "larget_dimension",
        "try_flipped": "try_flipped",
        "num_iterations": "num_iterations",
        "cell_size": "cell_size",
        "grid_spacing": "grid_spacing",
        "half_width_deg": "half_width_deg",
        "step_deg": "step_deg",
    }
    cli_keys: set[str] = set()
    for attr, dest in cli_map.items():
        if hasattr(args, attr):
            kwargs[dest] = getattr(args, attr)
            cli_keys.add(dest)

    accepted = _ALGORITHM_KWARGS[StosDebugAlgorithm(args.algorithm)]
    for key in list(kwargs):
        if key not in accepted and key in preset_keys and key not in cli_keys:
            kwargs.pop(key)
    return kwargs


def _parse_optional_int(value: str) -> int | None:
    if value.lower() in ("none", "null", "-"):
        return None
    return int(value)


def _normalize_kwargs(algorithm: StosDebugAlgorithm, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    if "largest_dimension" in normalized and "larget_dimension" not in normalized:
        normalized["larget_dimension"] = normalized.pop("largest_dimension")
    elif "largest_dimension" in normalized:
        normalized.pop("largest_dimension")

    accepted = _ALGORITHM_KWARGS[algorithm]
    unknown = sorted(name for name in normalized if name not in accepted)
    if unknown:
        raise TypeError(
            f"Unknown kwargs for algorithm {algorithm!s}: {unknown}. "
            f"Accepted: {sorted(accepted)}"
        )
    return normalized


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _local_rigid_kwargs(pair: StosPairContext, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy Pyre RefineRigidTransformLocal defaults around the stored rigid angle."""
    local = dict(kwargs)
    half_width = float(local.pop("half_width_deg", LOCAL_RIGID_HALF_WIDTH_DEG))
    step = float(local.pop("step_deg", LOCAL_RIGID_STEP_DEG))
    angle_deg, _, scale, _ = _rigid_params(pair.stored_transform)
    center_deg = 0.0 if angle_deg is None else angle_deg
    if "angles" not in local:
        local["angles"] = _refine_angle_grid_deg(center_deg, half_width, step)
    local.setdefault("larget_dimension", PYRE_LOGPOLAR_LARGEST_DIMENSION)
    local.setdefault("try_flipped", False)
    if scale is not None:
        local.setdefault("initial_scale_hint", scale)
    return local


def _stos_brute_settings(method: SliceToSliceMethod, kwargs: dict[str, Any]) -> StosBruteSettings:
    settings_kwargs: dict[str, Any] = {"method": method}
    for name in (
            "angles",
            "min_overlap",
            "source_image_scale_factors",
            "larget_dimension",
            "try_flipped",
            "estimated_scale_hint",
            "initial_scale_hint",
    ):
        if name in kwargs:
            settings_kwargs[name] = kwargs[name]
    return StosBruteSettings(**settings_kwargs)


def _run_grid_refine(
        pair: StosPairContext,
        kwargs: dict[str, Any]) -> tuple[GridRefineDebugReport, list[str]]:
    if pair.stored_transform is None:
        raise ValueError("grid_refine requires a stored transform on the STOS file")

    num_iterations = int(kwargs.get("num_iterations", DEFAULT_GRID_REFINE_ITERATIONS))
    notes: list[str] = []
    with nornir_imageregistration.settings.GridRefinement.CreateWithPreprocessedImages(
            target_img_data=pair.target,
            source_img_data=pair.source,
            num_iterations=num_iterations,
            cell_size=kwargs.get("cell_size"),
            grid_spacing=kwargs.get("grid_spacing"),
            angles_to_search=kwargs.get("angles_to_search"),
            final_pass_angles=kwargs.get("final_pass_angles"),
            min_alignment_overlap=kwargs.get("min_alignment_overlap"),
            min_unmasked_area=kwargs.get("min_unmasked_area"),
            single_thread_processing=bool(kwargs.get("single_thread_processing", True)),
    ) as settings:
        result_transform = RefineTransform(pair.stored_transform, settings)

    bbox = bbox_overlap_for_transform(result_transform, pair.source_shape, pair.target_shape)
    if bbox <= 0.0:
        notes.append("grid_refine result bbox overlap is 0; refine may be starving on a bad rigid")
    return GridRefineDebugReport(
        result_transform_type=type(result_transform).__name__,
        bbox_overlap=bbox,
        num_iterations=num_iterations,
    ), notes


def _overlap_mask_stats(
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
        min_overlap: float,
        larget_dimension: int | None) -> OverlapMaskStats:
    scaled_source, scaled_target, scalar = _scaled_shapes(
        source_shape, target_shape, larget_dimension)
    correlation_shape = _fixed_correlation_shape(
        scaled_target, scaled_source, (0.0,), min_overlap)
    mask = GetOverlapMask(scaled_target, scaled_source, correlation_shape, MinOverlap=min_overlap)
    if mask is None:
        eligible = 1.0
    elif mask.size == 0:
        eligible = 0.0
    else:
        xp = cp.get_array_module(mask)
        eligible = float(xp.mean(mask))
    return OverlapMaskStats(
        min_overlap=min_overlap,
        larget_dimension=larget_dimension,
        scalar=scalar,
        source_shape=scaled_source,
        target_shape=scaled_target,
        correlation_shape=(int(correlation_shape[0]), int(correlation_shape[1])),
        eligible_fraction=eligible,
    )


def _scaled_shapes(
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
        larget_dimension: int | None) -> tuple[tuple[int, int], tuple[int, int], float]:
    if larget_dimension is None:
        return source_shape, target_shape, 1.0
    scalar = float(nornir_imageregistration.ScalarForMaxDimension(
        larget_dimension, [source_shape, target_shape]))
    if scalar > 1.0:
        scalar = 1.0
    if scalar == 1.0:
        return source_shape, target_shape, 1.0
    return _scale_shape(source_shape, scalar), _scale_shape(target_shape, scalar), scalar


def _scale_shape(shape: tuple[int, int], scalar: float) -> tuple[int, int]:
    return (max(1, int(round(shape[0] * scalar))), max(1, int(round(shape[1] * scalar))))


def _unique_overlap_levels(min_overlap: float) -> list[float]:
    levels = list(DEFAULT_OVERLAP_LEVELS)
    if min_overlap not in levels:
        levels.insert(0, min_overlap)
    return levels


def _annotate_empty_overlap_masks(inspect: StosInspectReport, notes: list[str]) -> None:
    for mask_stats in inspect.overlap_masks:
        if mask_stats.eligible_fraction > 0.0:
            continue
        dim = "native" if mask_stats.larget_dimension is None else f"dim={mask_stats.larget_dimension}"
        notes.append(
            f"overlap mask empty at min_overlap={mask_stats.min_overlap} ({dim}); "
            "find_peak will return weight 0"
        )


def _annotate_non_rigid_stored(inspect: StosInspectReport, notes: list[str]) -> None:
    if inspect.stored_transform_type is None or inspect.stored_angle_deg is not None:
        return
    notes.append(
        f"stored transform {inspect.stored_transform_type} is not rigid; "
        "angle/offset/scale/flip not extracted"
    )


def _equivalent_rigid(transform: ITransform | None) -> IRigidTransform | None:
    if transform is None:
        return None
    if isinstance(transform, IRigidTransform):
        return transform
    try:
        converted = ConvertTransformToRigidTransform(transform)
    except (NotImplementedError, ValueError, TypeError, AttributeError):
        return None
    if isinstance(converted, IRigidTransform):
        return converted
    return None


def _stos_file_shape(dim: Sequence[float] | None) -> tuple[int, int] | None:
    if dim is None:
        return None
    if len(dim) >= 4:
        return (int(dim[3]), int(dim[2]))
    if len(dim) == 2:
        return (int(dim[1]), int(dim[0]))
    return None


def _peak_from_transform(
        transform: ITransform,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int]) -> tuple[float, float]:
    """AlignmentRecord-style peak: mapped source center minus target center."""
    source_center = (np.array(source_shape, dtype=np.float64) - 1.0) / 2.0
    target_center = (np.array(target_shape, dtype=np.float64) - 1.0) / 2.0
    mapped = nornir_imageregistration.EnsureNumpyArray(
        transform.Transform(source_center.reshape(1, 2)), dtype=np.float64).reshape(2)
    return (float(mapped[0] - target_center[0]), float(mapped[1] - target_center[1]))


def _wrap_angle_delta_deg(delta: float) -> float:
    return float((delta + 180.0) % 360.0 - 180.0)


def _compare_to_reference(
        reference: ReferencePoseReport | None,
        rigid: RigidDebugReport | None) -> PoseComparisonReport | None:
    if reference is None or rigid is None:
        return None
    angle_delta = None
    if reference.angle_deg is not None:
        angle_delta = _wrap_angle_delta_deg(rigid.angle_deg - reference.angle_deg)
    peak_delta = None
    if reference.peak_yx_in_pair is not None:
        peak_delta = (
            rigid.peak_yx[0] - reference.peak_yx_in_pair[0],
            rigid.peak_yx[1] - reference.peak_yx_in_pair[1],
        )
    scale_ratio = None
    if reference.scale not in (None, 0.0):
        scale_ratio = rigid.scale / float(reference.scale)
    return PoseComparisonReport(
        angle_delta_deg=angle_delta,
        peak_delta_yx=peak_delta,
        scale_ratio=scale_ratio,
    )


def _comparison_notes(
        comparison: PoseComparisonReport,
        reference: ReferencePoseReport | None) -> list[str]:
    notes: list[str] = []
    if comparison.angle_delta_deg is not None:
        notes.append(
            f"result angle differs from reference by {comparison.angle_delta_deg:.2f} deg"
        )
    if comparison.peak_delta_yx is not None:
        dy, dx = comparison.peak_delta_yx
        notes.append(
            f"result peak differs from reference by ({dy:.2f}, {dx:.2f}) px"
        )
    if reference is not None and reference.used_equivalent_rigid:
        notes.append("reference pose is the equivalent rigid of a non-rigid stored transform")
    return notes


def _rigid_notes(rigid: RigidDebugReport) -> list[str]:
    notes: list[str] = []
    if rigid.no_peak_fallback:
        notes.append(
            "no_peak_fallback: weight is 0 (post-fix peak should be ~(0,0); "
            "pre-fix this was correlation-image shape/2)"
        )
    if rigid.peak_exceeds_image_dimensions:
        notes.append("peak magnitude exceeds both image dimensions (images cannot overlap)")
    if rigid.bbox_overlap <= 0.0:
        notes.append("result bbox overlap is 0")
    return notes


def _rigid_params(
        transform: ITransform | None,
) -> tuple[float | None, tuple[float, float] | None, float | None, bool | None]:
    if transform is None or not isinstance(transform, IRigidTransform):
        return None, None, None, None
    offset = nornir_imageregistration.EnsureNumpyArray(
        transform.target_offset, dtype=np.float64).reshape(2)
    return (
        float(math.degrees(float(transform.angle))),
        (float(offset[0]), float(offset[1])),
        float(transform.scalar),
        bool(transform.flip_ud),
    )


def _valid_fraction(image: ImagePermutationHelper) -> float:
    mask = image.BlendedMask
    if mask is None or mask.size == 0:
        return 1.0
    xp = cp.get_array_module(mask)
    return float(xp.mean(mask))


def _downsample_hint_from_path(path: str) -> int | None:
    match = _GRID_DOWNSAMPLE_RE.search(path.replace("\\", "/"))
    if match is None:
        return None
    return int(match.group(1))


def _existing_path(path: str | None) -> str | None:
    if path is None or not os.path.isfile(path):
        return None
    return path


def _source_corners(source_shape: tuple[int, int]) -> NDArray[np.floating]:
    height = float(source_shape[0])
    width = float(source_shape[1])
    return np.array(
        ((0.0, 0.0), (0.0, width), (height, 0.0), (height, width)),
        dtype=np.float64,
    )


def _bbox_overlap_from_corners(
        corners: NDArray[np.floating],
        target_shape: tuple[int, int]) -> float:
    corners = nornir_imageregistration.EnsureNumpyArray(corners, dtype=np.float64).reshape(-1, 2)
    source_rect = Rectangle.CreateFromBounds((
        float(np.min(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 0])),
        float(np.max(corners[:, 1])),
    ))
    if source_rect.Area <= 0.0:
        return 0.0
    target_rect = Rectangle.CreateFromPointAndArea((0.0, 0.0), target_shape)
    return float(Rectangle.overlap(source_rect, target_rect))


def _refine_angle_grid_deg(
        center_angle_deg: float,
        half_width_deg: float = LOCAL_RIGID_HALF_WIDTH_DEG,
        step_deg: float = LOCAL_RIGID_STEP_DEG) -> NDArray[np.floating]:
    return np.arange(
        center_angle_deg - half_width_deg,
        center_angle_deg + half_width_deg + step_deg * 0.5,
        step_deg,
        dtype=float,
    )


def _alignment_to_dict(record: AlignmentRecord) -> dict[str, Any]:
    peak = nornir_imageregistration.EnsureNumpyArray(record.peak, dtype=np.float64).reshape(2)
    return {
        "peak_yx": [float(peak[0]), float(peak[1])],
        "weight": float(record.weight),
        "angle_deg": float(record.angle),
        "scale": float(record.scale),
        "flip_ud": bool(record.flippedud),
        "peak_ratio": None if record.peak_ratio is None else float(record.peak_ratio),
        "repr": str(record),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
