"""Diagnostics for SliceToVolume transform-chain boundary discontinuities.

Measures whether adjacent sections agree at their shared interface after
SliceToVolume composition, isolates linear-blend vs RBF-fallback causes, and
compares SliceToVolume16 vs SliceToVolume1 residuals.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

import nornir_imageregistration.files.stosfile as stosfile
import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import IControlPoints, ITransform

logger = logging.getLogger(__name__)

DEFAULT_MIN_BLEND = 0.05
DEFAULT_TRAVEL_LIMIT = 512.0
DEFAULT_REBLEND_ITERATIONS = 8
DEFAULT_REBLEND_TOLERANCE = 0.5
DEFAULT_ENRICH_TOLERANCE = 4.0  # 64 full-res / downsample 16


@dataclass(frozen=True)
class StosFileInfo:
    """Resolved STOS path and filesystem metadata."""

    label: str
    path: str
    mtime: float
    size: int


@dataclass(frozen=True)
class BoundaryResidualSummary:
    """Aggregate boundary residual statistics for one STV group."""

    group: str
    sample_count: int
    mean_residual: float
    max_residual: float
    p95_residual: float
    worst_index: int
    worst_residual: float


@dataclass(frozen=True)
class CompositionExperiment:
    """Boundary residual after recomposing with alternate composition settings."""

    name: str
    min_blend: float | None
    travel_limit: float | None
    enrich_tolerance: float | None
    mean_residual: float
    max_residual: float
    delta_mean_from_stored: float
    delta_max_from_stored: float


@dataclass(frozen=True)
class DiagnosticReport:
    """Full diagnostic report for one section-pair interface."""

    volume_root: str
    mapped_section: int
    control_section: int
    center_section: int
    downsample: int
    files: dict[str, StosFileInfo]
    freshness_warnings: tuple[str, ...]
    hull: dict[str, float | int]
    stored_stv16: BoundaryResidualSummary
    stored_stv1: BoundaryResidualSummary | None
    level_comparison: dict[str, float | str] | None
    experiments: tuple[CompositionExperiment, ...]
    primary_cause: str
    recommendations: tuple[str, ...]


def _to_numpy(array: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert array inputs to float64 NumPy arrays."""
    if hasattr(array, "get"):
        return np.asarray(array.get(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def find_stos(volume_root: Path, group: str, pair: str) -> Path | None:
    """Return the first matching .stos path for a group and section-pair prefix."""
    pattern = str(volume_root / group / f"{pair}*.stos")
    matches = sorted(glob.glob(pattern))
    if not matches:
        simple = volume_root / group / f"{pair}.stos"
        if simple.is_file():
            return simple
        return None
    return Path(matches[0])


def stos_file_info(label: str, path: Path | None) -> StosFileInfo | None:
    """Collect filesystem metadata for a STOS path."""
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return StosFileInfo(label=label, path=str(path), mtime=stat.st_mtime, size=stat.st_size)


def load_transform(stos_path: Path) -> ITransform:
    """Load the transform object from a STOS file."""
    loaded = stosfile.StosFile.Load(str(stos_path))
    return nornir_imageregistration.transforms.LoadTransform(loaded.Transform)  # type: ignore[arg-type]


def _format_mtime(mtime: float) -> str:
    """Format a unix timestamp for reports."""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def check_freshness(files: Mapping[str, StosFileInfo | None]) -> tuple[str, ...]:
    """Warn when composed outputs appear older than their Grid16 inputs."""
    warnings: list[str] = []
    grid = files.get("grid_mapped_to_control")
    stv_mapped = files.get("stv_mapped_to_center")
    stv_control = files.get("stv_control_to_center")

    if grid is not None and stv_mapped is not None and stv_mapped.mtime < grid.mtime:
        warnings.append(
            f"{stv_mapped.label} is older than {grid.label} "
            f"({_format_mtime(stv_mapped.mtime)} < {_format_mtime(grid.mtime)})")

    if stv_control is not None and stv_mapped is not None and stv_mapped.mtime < stv_control.mtime:
        warnings.append(
            f"{stv_mapped.label} is older than {stv_control.label} "
            f"({_format_mtime(stv_mapped.mtime)} < {_format_mtime(stv_control.mtime)})")

    return tuple(warnings)


def sample_interface_points(
        grid_ab: ITransform,
        num_samples: int = 64) -> NDArray[np.float64]:
    """Sample points along the mapped-section edge that borders the control section."""
    if not hasattr(grid_ab, "SourcePoints"):
        raise ValueError("Grid mapped→control transform must expose SourcePoints")

    source_points = _to_numpy(grid_ab.SourcePoints)  # type: ignore[attr-defined]
    if source_points.shape[0] < 4:
        raise ValueError("Need at least four control points to sample an interface")

    y_min, x_min = np.min(source_points, axis=0)
    y_max, x_max = np.max(source_points, axis=0)
    # Interface between mapped (453) and control (452) is approximated by the source-space
    # edge with maximum Y (bottom of section 453 in Nornir coordinates).
    edge_mask = source_points[:, 0] >= (y_max - 1e-3)
    edge_points = source_points[edge_mask]
    if edge_points.shape[0] < 2:
        edge_points = source_points[np.argsort(source_points[:, 0])[-max(2, num_samples):]]

    order = np.argsort(edge_points[:, 1])
    edge_points = edge_points[order]
    if edge_points.shape[0] >= num_samples:
        indices = np.linspace(0, edge_points.shape[0] - 1, num_samples, dtype=int)
        return edge_points[indices]

    x_samples = np.linspace(x_min, x_max, num_samples)
    y_value = float(np.max(edge_points[:, 0]))
    return np.column_stack((np.full(num_samples, y_value), x_samples))


def hull_membership(
        control_to_center: IControlPoints,
        points_in_control_space: NDArray[np.float64]) -> tuple[NDArray[np.bool_], float]:
    """Return in-hull mask and fraction outside the control→center mesh hull."""
    source_points = _to_numpy(control_to_center.SourcePoints)
    delaunay = Delaunay(source_points)
    simplex_ids = delaunay.find_simplex(points_in_control_space)
    inside = simplex_ids >= 0
    outside_fraction = float(np.mean(~inside))
    return inside, outside_fraction


def boundary_residuals(
        points_on_mapped_section: NDArray[np.float64],
        grid_ab: ITransform,
        stv_ac: ITransform,
        stv_bc: ITransform,
        *,
        mapped_space_scale: float = 1.0,
        control_space_scale: float = 1.0) -> NDArray[np.float64]:
    """Compute per-point volume-space disagreement across a shared interface."""
    mapped_points = points_on_mapped_section * mapped_space_scale
    direct = _to_numpy(stv_ac.Transform(mapped_points))
    via_control = _to_numpy(grid_ab.Transform(points_on_mapped_section)) * control_space_scale
    chained = _to_numpy(stv_bc.Transform(via_control))
    return np.linalg.norm(direct - chained, axis=1)


def summarize_residuals(group: str, residuals: NDArray[np.float64]) -> BoundaryResidualSummary:
    """Aggregate residual statistics."""
    worst_index = int(np.argmax(residuals))
    return BoundaryResidualSummary(
        group=group,
        sample_count=int(residuals.shape[0]),
        mean_residual=float(np.mean(residuals)),
        max_residual=float(np.max(residuals)),
        p95_residual=float(np.percentile(residuals, 95)),
        worst_index=worst_index,
        worst_residual=float(residuals[worst_index]),
    )


def recompose_mapped_to_center(
        grid_ab_path: Path,
        stv_bc_path: Path,
        *,
        enrich_tolerance: float | None,
        min_blend: float | None,
        travel_limit: float | None,
        reblend_iterations: int = DEFAULT_REBLEND_ITERATIONS,
        reblend_tolerance: float = DEFAULT_REBLEND_TOLERANCE) -> ITransform:
    """Recompose mapped→center using AddStosTransforms with explicit blend settings."""
    composed = stosfile.AddStosTransforms(
        str(grid_ab_path),
        str(stv_bc_path),
        EnrichTolerance=enrich_tolerance,
        min_blend=min_blend,
        travel_limit=travel_limit,
        reblend_iterations=reblend_iterations,
        reblend_tolerance=reblend_tolerance,
    )
    return nornir_imageregistration.transforms.LoadTransform(composed.Transform)  # type: ignore[arg-type]


def recompose_with_terminal_blend(
        grid_ab_path: Path,
        control_to_volume_unblended_path: Path,
        *,
        enrich_tolerance: float | None,
        min_blend: float | None,
        travel_limit: float | None,
        reblend_iterations: int = DEFAULT_REBLEND_ITERATIONS,
        reblend_tolerance: float = DEFAULT_REBLEND_TOLERANCE) -> ITransform:
    """Mirror BuildSliceToVolume terminal-blend composition using an unblended parent chain."""
    unblended = stosfile.AddStosTransforms(
        str(grid_ab_path),
        str(control_to_volume_unblended_path),
        EnrichTolerance=enrich_tolerance,
        min_blend=None,
        travel_limit=None,
    )
    if min_blend is None and travel_limit is None:
        return nornir_imageregistration.transforms.LoadTransform(unblended.Transform)  # type: ignore[arg-type]

    rigid_ab = stosfile.RigidTransformFromStosPath(str(grid_ab_path))
    rigid_bc = stosfile.RigidTransformFromStosPath(str(control_to_volume_unblended_path))
    rigid_ac = nornir_imageregistration.transforms.addition.AddTransforms(rigid_bc, rigid_ab)  # type: ignore[arg-type]
    nonlinear = nornir_imageregistration.transforms.LoadTransform(unblended.Transform)  # type: ignore[arg-type]
    blended = nornir_imageregistration.transforms.utils.BlendTransformsIteratively(
        nonlinear,  # type: ignore[arg-type]
        rigid_ac,
        min_blend=min_blend,
        travel_limit=travel_limit,
        reblend_iterations=reblend_iterations,
        reblend_tolerance=reblend_tolerance,
    )
    return blended


def run_composition_experiments(
        interface_points: NDArray[np.float64],
        volume_root: Path,
        grid_ab_path: Path,
        stv_bc_path: Path,
        grid_ab: ITransform,
        stv_bc: ITransform,
        stored_summary: BoundaryResidualSummary,
        *,
        control_section: int,
        center_section: int,
        downsample: int,
        enrich_tolerance: float,
        travel_limit: float) -> tuple[CompositionExperiment, ...]:
    """Recompose with alternate settings and measure boundary residual changes."""
    experiments: list[CompositionExperiment] = []

    def _record(
            name: str,
            transform_ac: ITransform,
            *,
            min_blend: float | None,
            travel_limit_value: float | None,
            enrich_value: float | None) -> None:
        residuals = boundary_residuals(interface_points, grid_ab, transform_ac, stv_bc)
        summary = summarize_residuals(name, residuals)
        experiments.append(CompositionExperiment(
            name=name,
            min_blend=min_blend,
            travel_limit=travel_limit_value,
            enrich_tolerance=enrich_value,
            mean_residual=summary.mean_residual,
            max_residual=summary.max_residual,
            delta_mean_from_stored=summary.mean_residual - stored_summary.mean_residual,
            delta_max_from_stored=summary.max_residual - stored_summary.max_residual,
        ))

    no_blend = recompose_mapped_to_center(
        grid_ab_path,
        stv_bc_path,
        enrich_tolerance=enrich_tolerance,
        min_blend=None,
        travel_limit=None,
    )
    _record("no_linear_blend", no_blend, min_blend=None, travel_limit_value=None, enrich_value=enrich_tolerance)

    no_blend_no_enrich = recompose_mapped_to_center(
        grid_ab_path,
        stv_bc_path,
        enrich_tolerance=None,
        min_blend=None,
        travel_limit=None,
    )
    _record(
        "no_blend_no_enrich",
        no_blend_no_enrich,
        min_blend=None,
        travel_limit_value=None,
        enrich_value=None,
    )

    default_blend = recompose_mapped_to_center(
        grid_ab_path,
        stv_bc_path,
        enrich_tolerance=enrich_tolerance,
        min_blend=DEFAULT_MIN_BLEND,
        travel_limit=travel_limit,
    )
    _record(
        "default_pipeline_blend",
        default_blend,
        min_blend=DEFAULT_MIN_BLEND,
        travel_limit_value=travel_limit,
        enrich_value=enrich_tolerance,
    )

    unblended_bc_path = Path(f"{os.path.splitext(stv_bc_path)[0]}.unblended.stos")
    if not unblended_bc_path.is_file() and control_section != center_section:
        parent_section = control_section - 1
        grid_parent_path = find_stos(volume_root, "Grid16", f"{control_section}-{parent_section}")
        prior_stv_path = find_stos(volume_root, f"SliceToVolume{downsample}", f"{parent_section}-{center_section}")
        if grid_parent_path is not None and prior_stv_path is not None:
            unblended_bc = stosfile.AddStosTransforms(
                str(grid_parent_path),
                str(prior_stv_path),
                EnrichTolerance=enrich_tolerance,
                min_blend=None,
                travel_limit=None,
            )
            unblended_bc.Save(str(unblended_bc_path))

    if unblended_bc_path.is_file():
        terminal_blend = recompose_with_terminal_blend(
            grid_ab_path,
            unblended_bc_path,
            enrich_tolerance=enrich_tolerance,
            min_blend=DEFAULT_MIN_BLEND,
            travel_limit=travel_limit,
        )
        _record(
            "terminal_blend_unblended_chain",
            terminal_blend,
            min_blend=DEFAULT_MIN_BLEND,
            travel_limit_value=travel_limit,
            enrich_value=enrich_tolerance,
        )

    return tuple(experiments)


def infer_primary_cause(
        stored_summary: BoundaryResidualSummary,
        hull_outside_fraction: float,
        experiments: Sequence[CompositionExperiment]) -> tuple[str, tuple[str, ...]]:
    """Infer the most likely cause and recommended next steps."""
    no_blend = next((item for item in experiments if item.name == "no_linear_blend"), None)
    default_blend = next((item for item in experiments if item.name == "default_pipeline_blend"), None)

    recommendations: list[str] = []
    if stored_summary.max_residual < 1.0:
        return (
            "none_detected",
            ("Stored SliceToVolume transforms are consistent at the sampled interface.",),
        )

    if hull_outside_fraction > 0.05:
        recommendations.append(
            "Increase SliceToVolume enrichment tolerance or extend control coverage near the "
            f"interface ({hull_outside_fraction:.1%} of Grid16 target points fall outside the "
            "452→450 mesh hull and use RBF fallback).")
        primary = "rbf_hull_fallback"
    else:
        primary = "unknown"

    if no_blend is not None and no_blend.max_residual + 1.0 < stored_summary.max_residual:
        primary = "linear_blend_accumulation"
        recommendations.append(
            "Re-run SliceToVolume with `-min_blend 0` (or omit travel_limit/min_blend) so "
            "intermediate hops are composed without per-hop rigid blending.")
        recommendations.append(
            "If some warp regularization is still needed, use a smaller min_blend/travel_limit "
            "or apply LinearizeVolume only after SliceToVolume completes.")
    elif default_blend is not None and abs(default_blend.max_residual - stored_summary.max_residual) > 5.0:
        recommendations.append(
            "Stored transform does not match a fresh default-blend recomposition; rebuild "
            "SliceToVolume16 after Grid16 changes and verify VolumeData checksums.")

    if not recommendations:
        recommendations.append(
            "Inspect the residual plot for localized spikes; manual Grid16 edits or a missing "
            "intermediate hop are likely.")
        primary = "unknown"

    return primary, tuple(recommendations)


def plot_boundary_residuals(
        interface_points: NDArray[np.float64],
        residuals: NDArray[np.float64],
        hull_points: NDArray[np.float64] | None,
        hull_inside: NDArray[np.bool_] | None,
        output_path: str,
        *,
        title: str) -> None:
    """Save a quiver plot of interface residuals."""
    fig, axis = plt.subplots(figsize=(10, 8), dpi=150)
    axis.scatter(interface_points[:, 1], interface_points[:, 0], c=residuals, cmap="hot", s=20)
    if hull_points is not None and hull_inside is not None:
        axis.scatter(
            hull_points[~hull_inside, 1],
            hull_points[~hull_inside, 0],
            facecolors="none",
            edgecolors="cyan",
            s=60,
            linewidths=1.0,
            label="outside hull",
        )
    axis.set_title(title)
    axis.set_xlabel("X (mapped section source space)")
    axis.set_ylabel("Y (mapped section source space)")
    axis.invert_yaxis()
    axis.set_aspect("equal")
    if hull_points is not None and hull_inside is not None and np.any(~hull_inside):
        axis.legend(loc="upper right")
    fig.colorbar(axis.collections[0], ax=axis, label="boundary residual (pixels)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def diagnose_interface(
        volume_root: Path,
        *,
        mapped_section: int,
        control_section: int,
        center_section: int,
        downsample: int = 16,
        grid_group: str = "Grid16",
        num_samples: int = 64,
        output_dir: Path | None = None,
        enrich_tolerance: float = DEFAULT_ENRICH_TOLERANCE,
        travel_limit: float = DEFAULT_TRAVEL_LIMIT) -> DiagnosticReport:
    """Run the full interface diagnostic for one mapped/control section pair."""
    pair_ab = f"{mapped_section}-{control_section}"
    pair_ac = f"{mapped_section}-{center_section}"
    pair_bc = f"{control_section}-{center_section}"

    grid_ab_path = find_stos(volume_root, grid_group, pair_ab)
    stv_bc_path = find_stos(volume_root, f"SliceToVolume{downsample}", pair_bc)
    stv_ac16_path = find_stos(volume_root, f"SliceToVolume{downsample}", pair_ac)
    stv_ac1_path = find_stos(volume_root, "SliceToVolume1", pair_ac)

    required = {
        "grid_mapped_to_control": grid_ab_path,
        "stv_control_to_center": stv_bc_path,
        "stv_mapped_to_center": stv_ac16_path,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Missing required STOS inputs under {volume_root}: {', '.join(missing)}")

    file_infos = {
        key: stos_file_info(key, path)
        for key, path in {
            "grid_mapped_to_control": grid_ab_path,
            "stv_control_to_center": stv_bc_path,
            "stv_mapped_to_center": stv_ac16_path,
            "stv1_mapped_to_center": stv_ac1_path,
        }.items()
        if path is not None
    }
    freshness_warnings = check_freshness(file_infos)

    grid_ab = load_transform(grid_ab_path)  # type: ignore[arg-type]
    stv_bc = load_transform(stv_bc_path)  # type: ignore[arg-type]
    stv_ac16 = load_transform(stv_ac16_path)  # type: ignore[arg-type]

    interface_points = sample_interface_points(grid_ab, num_samples=num_samples)
    residuals16 = boundary_residuals(interface_points, grid_ab, stv_ac16, stv_bc)
    stored_stv16 = summarize_residuals(f"SliceToVolume{downsample}", residuals16)

    via_control = _to_numpy(grid_ab.Transform(interface_points))
    inside_hull, hull_outside_fraction = hull_membership(stv_bc, via_control)  # type: ignore[arg-type]

    experiments = run_composition_experiments(
        interface_points,
        volume_root,
        grid_ab_path,  # type: ignore[arg-type]
        stv_bc_path,  # type: ignore[arg-type]
        grid_ab,
        stv_bc,
        stored_stv16,
        control_section=control_section,
        center_section=center_section,
        downsample=downsample,
        enrich_tolerance=enrich_tolerance,
        travel_limit=travel_limit / float(downsample),
    )

    stored_stv1: BoundaryResidualSummary | None = None
    level_comparison: dict[str, float | str] | None = None
    if stv_ac1_path is not None:
        stv_ac1 = load_transform(stv_ac1_path)
        stv_bc1_path = find_stos(volume_root, "SliceToVolume1", pair_bc)
        if stv_bc1_path is not None:
            stv_bc1 = load_transform(stv_bc1_path)
            scale = float(downsample)
            residuals1 = boundary_residuals(
                interface_points,
                grid_ab,
                stv_ac1,
                stv_bc1,
                mapped_space_scale=scale,
                control_space_scale=scale,
            )
        else:
            residuals1 = boundary_residuals(
                interface_points * float(downsample),
                grid_ab,
                stv_ac1,
                stv_bc,
            )
        stored_stv1 = summarize_residuals("SliceToVolume1", residuals1)
        level_comparison = {
            "stv16_max_residual": stored_stv16.max_residual,
            "stv1_max_residual": stored_stv1.max_residual,
            "stv1_minus_stv16_max": stored_stv1.max_residual - stored_stv16.max_residual,
            "jump_present_in_stv16": stored_stv16.max_residual >= 1.0,
            "jump_only_in_stv1": stored_stv16.max_residual < 1.0 and stored_stv1.max_residual >= 1.0,
            "verdict": (
                "jump originates in SliceToVolume composition (present at downsample 16)"
                if stored_stv16.max_residual >= 1.0
                else (
                    "jump appears only after ScaleVolumeTransforms to SliceToVolume1"
                    if stored_stv1.max_residual >= 1.0
                    else "no large jump detected at sampled interface"
                )
            ),
        }

    primary_cause, recommendations = infer_primary_cause(
        stored_stv16,
        hull_outside_fraction,
        experiments,
    )

    report = DiagnosticReport(
        volume_root=str(volume_root),
        mapped_section=mapped_section,
        control_section=control_section,
        center_section=center_section,
        downsample=downsample,
        files={key: info for key, info in file_infos.items() if info is not None},
        freshness_warnings=freshness_warnings,
        hull={
            "sample_count": int(interface_points.shape[0]),
            "outside_fraction": hull_outside_fraction,
            "outside_count": int(np.sum(~inside_hull)),
        },
        stored_stv16=stored_stv16,
        stored_stv1=stored_stv1,
        level_comparison=level_comparison,
        experiments=experiments,
        primary_cause=primary_cause,
        recommendations=recommendations,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_boundary_residuals(
            interface_points,
            residuals16,
            via_control,
            inside_hull,
            str(output_dir / f"boundary_residual_{pair_ab}.png"),
            title=(
                f"{pair_ab} interface residual — max={stored_stv16.max_residual:.1f}px "
                f"({primary_cause})"),
        )
        report_path = output_dir / f"stos_chain_report_{pair_ab}.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(report), handle, indent=2)

    return report


def _print_report(report: DiagnosticReport) -> None:
    """Print a human-readable summary."""
    pair = f"{report.mapped_section}-{report.control_section}"
    print(f"Volume: {report.volume_root}")
    print(f"Interface: {pair} -> center {report.center_section}")
    print()
    print("Files:")
    for info in report.files.values():
        print(f"  {info.label:24} {_format_mtime(info.mtime)}  {info.path}")
    if report.freshness_warnings:
        print("\nFreshness warnings:")
        for warning in report.freshness_warnings:
            print(f"  - {warning}")
    print()
    print(
        f"SliceToVolume{report.downsample}: mean={report.stored_stv16.mean_residual:.2f}px "
        f"max={report.stored_stv16.max_residual:.2f}px "
        f"p95={report.stored_stv16.p95_residual:.2f}px")
    if report.stored_stv1 is not None:
        print(
            f"SliceToVolume1:          mean={report.stored_stv1.mean_residual:.2f}px "
            f"max={report.stored_stv1.max_residual:.2f}px "
            f"p95={report.stored_stv1.p95_residual:.2f}px")
    if report.level_comparison is not None:
        print(f"\nLevel verdict: {report.level_comparison['verdict']}")
    print(
        f"\nHull coverage: {report.hull['outside_fraction']:.1%} outside "
        f"({report.hull['outside_count']}/{report.hull['sample_count']} samples)")
    print("\nRecomposition experiments:")
    for experiment in report.experiments:
        print(
            f"  {experiment.name:24} mean={experiment.mean_residual:7.2f} "
            f"max={experiment.max_residual:7.2f}  "
            f"Δmean={experiment.delta_mean_from_stored:+7.2f} "
            f"Δmax={experiment.delta_max_from_stored:+7.2f}")
    print(f"\nPrimary cause: {report.primary_cause}")
    print("Recommendations:")
    for item in report.recommendations:
        print(f"  - {item}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume_root", type=Path, help="Volume TEM root, e.g. /storage4/RPC3/TEM")
    parser.add_argument("--mapped", type=int, default=453, help="Mapped section (default: 453)")
    parser.add_argument("--control", type=int, default=452, help="Control section (default: 452)")
    parser.add_argument("--center", type=int, default=450, help="Volume center section (default: 450)")
    parser.add_argument("--downsample", type=int, default=16, help="SliceToVolume downsample level")
    parser.add_argument("--grid-group", default="Grid16", help="Pairwise input STOS group")
    parser.add_argument("--samples", type=int, default=64, help="Interface sample count")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for JSON report and residual PNG",
    )
    parser.add_argument("--travel-limit", type=float, default=DEFAULT_TRAVEL_LIMIT)
    parser.add_argument("--enrich-tolerance", type=float, default=DEFAULT_ENRICH_TOLERANCE)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    output_dir = args.output_dir
    if output_dir is None:
        root = (
            os.environ.get("TEST_OUTPUT_DIR", "").strip()
            or os.environ.get("TESTOUTPUTPATH", "").strip()
        )
        if root:
            output_dir = Path(root) / "stos_chain_diagnostics"

    report = diagnose_interface(
        args.volume_root,
        mapped_section=args.mapped,
        control_section=args.control,
        center_section=args.center,
        downsample=args.downsample,
        grid_group=args.grid_group,
        num_samples=args.samples,
        output_dir=output_dir,
        enrich_tolerance=args.enrich_tolerance,
        travel_limit=args.travel_limit,
    )
    _print_report(report)
    if output_dir is not None:
        print(f"\nWrote artifacts under {output_dir}")
    return 0 if report.stored_stv16.max_residual < 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
