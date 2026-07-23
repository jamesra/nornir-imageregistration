#!/usr/bin/env python3
"""Report STOS transform orientation metrics and compare pipeline stages."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

import nornir_imageregistration.files
import nornir_imageregistration.transforms
from nornir_imageregistration.transforms.converters import ConvertTransformToRigidTransform
from nornir_imageregistration.transforms.pointrelations import ControlPointRelation, calculate_point_relation
from nornir_imageregistration.transforms.utils import (
    INVERSE_MAP_Y_CORRELATION_THRESHOLD,
    BlendWithLinear,
    estimate_inverse_map_y_correlation,
)


def _find_stos(volume_root: Path, group: str, pair: str) -> Path | None:
    """Return the first matching .stos path for a group and section pair prefix."""
    pattern = str(volume_root / group / f"{pair}*.stos")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[0])


def _rigid_translation(transform: nornir_imageregistration.transforms.ITransform) -> np.ndarray:
    """Return the fitted rigid translation vector for a transform."""
    rigid = ConvertTransformToRigidTransform(transform)
    return np.asarray(rigid._target_offset, dtype=np.float64)


def _describe_stos(stos_path: Path) -> dict[str, object]:
    """Load a STOS file and collect orientation diagnostics."""
    loaded = nornir_imageregistration.files.StosFile.Load(str(stos_path))
    transform = nornir_imageregistration.transforms.LoadTransform(loaded.Transform)  # type: ignore[arg-type]
    transform_type = type(transform).__name__
    flip_ud = bool(getattr(transform, 'flip_ud', False))
    inv_corr = estimate_inverse_map_y_correlation(transform)
    cross_product = 'n/a'
    if hasattr(transform, 'TargetPoints'):
        relation = calculate_point_relation(transform.TargetPoints)
        cross_product = ControlPointRelation(relation).name
    translation = _rigid_translation(transform)
    return {
        'path': stos_path,
        'type': transform_type,
        'flip_ud': flip_ud,
        'cross_product': cross_product,
        'inv_corr': inv_corr,
        'translation': translation,
    }


def _print_row(label: str, metrics: dict[str, object]) -> None:
    """Print one diagnostic row."""
    translation = metrics['translation']
    tx, ty = float(translation[0]), float(translation[1])  # type: ignore[index]
    print(
        f"{label:24} type={metrics['type']:<24} "
        f"flip_ud={str(metrics['flip_ud']):<5} "
        f"cross_product={metrics['cross_product']:<8} "
        f"inv_corr={metrics['inv_corr']:+.6f}  "
        f"rigid_t=({tx:+.3f}, {ty:+.3f})  "
        f"{metrics['path']}")


def _compare_pair(volume_root: Path, input_group: str, output_group: str, pair: str) -> bool:
    """Compare inv_corr sign between two groups for one section pair."""
    input_path = _find_stos(volume_root, input_group, pair)
    output_path = _find_stos(volume_root, output_group, pair)
    if input_path is None:
        print(f"MISSING  {input_group}/{pair}")
        return False
    if output_path is None:
        print(f"MISSING  {output_group}/{pair}")
        return False

    input_metrics = _describe_stos(input_path)
    output_metrics = _describe_stos(output_path)
    _print_row(f"{input_group}/{pair}", input_metrics)
    _print_row(f"{output_group}/{pair}", output_metrics)

    input_translation = np.asarray(input_metrics['translation'], dtype=np.float64)
    output_translation = np.asarray(output_metrics['translation'], dtype=np.float64)
    delta = output_translation - input_translation
    delta_norm = float(np.linalg.norm(delta))
    print(f"  rigid translation delta: ({delta[0]:+.3f}, {delta[1]:+.3f})  norm={delta_norm:.3f}")

    sign_flip = (
        abs(float(input_metrics['inv_corr'])) >= INVERSE_MAP_Y_CORRELATION_THRESHOLD
        and float(output_metrics['inv_corr']) != 0.0
        and (float(input_metrics['inv_corr']) > 0) != (float(output_metrics['inv_corr']) > 0))
    if sign_flip:
        print(f"  *** SIGN FLIP: {pair} inv_corr {input_metrics['inv_corr']:+.6f} -> "
              f"{output_metrics['inv_corr']:+.6f}")
    print()
    return sign_flip


def _simulate_subtle_linearize(stos_path: Path,
                               min_blend: float,
                               max_blend: float,
                               reblend_iterations: int) -> dict[str, object]:
    """Simulate a subtle LinearizeVolume pass on one STOS file in memory."""
    loaded = nornir_imageregistration.files.StosFile.Load(str(stos_path))
    transform = nornir_imageregistration.transforms.LoadTransform(loaded.Transform)  # type: ignore[arg-type]
    before = _describe_stos(stos_path)
    blended = BlendWithLinear(transform,
                              min_blend=min_blend,
                              max_blend=max_blend,
                              reblend_iterations=reblend_iterations)
    after_translation = _rigid_translation(blended)
    before_translation = np.asarray(before['translation'], dtype=np.float64)
    delta = after_translation - before_translation
    return {
        'before': before,
        'after_translation': after_translation,
        'delta': delta,
        'delta_norm': float(np.linalg.norm(delta)),
        'inv_corr_after': estimate_inverse_map_y_correlation(blended),
    }


def _print_simulated_chain(volume_root: Path, group: str, pairs: list[str], **blend_kwargs: float | int) -> None:
    """Report simulated subtle linearize translation deltas for section pairs."""
    print(f"Simulated subtle linearize on {group} ({blend_kwargs}):")
    for pair in pairs:
        stos_path = _find_stos(volume_root, group, pair)
        if stos_path is None:
            print(f"  {pair}: not found")
            continue
        result = _simulate_subtle_linearize(stos_path, **blend_kwargs)  # type: ignore[arg-type]
        before = result['before']
        delta = result['delta']
        print(
            f"  {pair}: inv_corr {before['inv_corr']:+.6f} -> {result['inv_corr_after']:+.6f}  "
            f"rigid_t delta=({delta[0]:+.3f}, {delta[1]:+.3f})  norm={result['delta_norm']:.3f}")
    print()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('volume_root', type=Path, help='Volume root, e.g. /storage4/RPC3/TEM')
    parser.add_argument('--grid', nargs='*', default=['453-452'],
                        help='Grid16 section pairs to inspect')
    parser.add_argument('--stv', nargs='*', default=['451-450', '453-450'],
                        help='SliceToVolume section pairs to inspect')
    parser.add_argument('--chain', nargs='*',
                        default=['452-450', '453-450', '454-450', '455-450', '456-450', '457-450'],
                        help='Section pairs for translation-delta chain report')
    parser.add_argument('--input-group', default='SliceToVolume1',
                        help='Input STOS group for sign-flip scan')
    parser.add_argument('--output-group', default='SliceToVolumeLinear1',
                        help='Output STOS group for sign-flip scan')
    parser.add_argument('--simulate-group', default=None,
                        help='STOS group to simulate subtle linearize on (e.g. SliceToVolume1)')
    parser.add_argument('--min-blend', type=float, default=0.01,
                        help='min_blend for --simulate-group')
    parser.add_argument('--max-blend', type=float, default=0.01,
                        help='max_blend for --simulate-group')
    parser.add_argument('--reblend-iterations', type=int, default=1,
                        help='reblend_iterations for --simulate-group')
    parser.add_argument('--scan-all', action='store_true',
                        help='Scan all section pairs in input/output groups')
    args = parser.parse_args()

    volume_root = args.volume_root
    print(f"Volume root: {volume_root}\n")

    for pair in args.grid:
        path = _find_stos(volume_root, 'Grid16', pair)
        if path is None:
            print(f"Grid16/{pair}: not found")
            continue
        _print_row(f"Grid16/{pair}", _describe_stos(path))
    print()

    sign_flips = 0
    for pair in args.stv:
        if _compare_pair(volume_root, args.input_group, args.output_group, pair):
            sign_flips += 1

    print(f"Translation delta chain ({args.input_group} -> {args.output_group}):")
    for pair in args.chain:
        _compare_pair(volume_root, args.input_group, args.output_group, pair)

    if args.simulate_group:
        _print_simulated_chain(volume_root,
                               args.simulate_group,
                               list(args.chain),
                               min_blend=args.min_blend,
                               max_blend=args.max_blend,
                               reblend_iterations=args.reblend_iterations)

    if args.scan_all:
        input_dir = volume_root / args.input_group
        if not input_dir.is_dir():
            print(f"Cannot scan: missing {input_dir}")
            return 1
        pairs: list[str] = []
        for path in sorted(input_dir.glob('*.stos')):
            stem = path.name.split('_', 1)[0]
            if stem not in pairs:
                pairs.append(stem)
        print(f"Scanning {len(pairs)} pairs in {args.input_group} -> {args.output_group}")
        for pair in pairs:
            if _compare_pair(volume_root, args.input_group, args.output_group, pair):
                sign_flips += 1
        print(f"Total sign flips: {sign_flips}")

    return 1 if sign_flips else 0


if __name__ == '__main__':
    raise SystemExit(main())
