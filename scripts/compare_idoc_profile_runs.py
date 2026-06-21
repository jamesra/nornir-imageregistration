#!/usr/bin/env python3
"""Archive and compare TestIDocBuild StageTimings and cProfile needle stats."""

from __future__ import annotations

import argparse
import json
import pstats
import shutil
from collections import defaultdict
from io import StringIO
from pathlib import Path

PROFILE_NEEDLES: tuple[str, ...] = (
    'find_peak',
    'AttemptAlignPoint',
    '_RefinePointsForTwoImages',
    '_prewarp_tile_for_grid_refine',
    '_prewarp_all_tiles_for_grid_refine',
    'GetOverlapMaskOnDevice',
    'SliceToSliceRigidRegistration',
    'BuildAlignmentROIs',
)

KEY_PIPELINES: tuple[str, ...] = (
    'RefineSectionAlignment',
    'AlignSections',
    'Mosaic',
    'Assemble',
    'AdjustContrast',
    'ImportIDoc',
)


def summarize_stage_timings(path: Path) -> dict[str, float]:
    """Sum stage seconds grouped by pipeline name."""
    data = json.loads(path.read_text(encoding='utf-8'))
    by_pipeline: dict[str, float] = defaultdict(float)
    total = 0.0
    for block in data:
        pipeline = block.get('pipeline', '?')
        for stage in block.get('stages', []):
            seconds = float(stage['seconds'])
            by_pipeline[pipeline] += seconds
            total += seconds
    by_pipeline['__TOTAL__'] = total
    return dict(by_pipeline)


def refine_block_totals(path: Path, block_index: int = 2) -> dict[str, float]:
    """Return per-stos timings for the Nth RefineSectionAlignment block."""
    data = json.loads(path.read_text(encoding='utf-8'))
    idx = 0
    for block in data:
        if block.get('pipeline') != 'RefineSectionAlignment':
            continue
        if idx == block_index:
            return {
                stage['stage'].split('@')[-1].strip(): float(stage['seconds'])
                for stage in block.get('stages', [])
            }
        idx += 1
    return {}


def profile_needles(profile_path: Path) -> dict[str, tuple[int, float]]:
    """Extract call count and cumulative seconds for each profile needle."""
    stats: dict[str, tuple[int, float]] = {}
    if not profile_path.is_file():
        return stats
    stream = StringIO()
    profile = pstats.Stats(str(profile_path), stream=stream)
    for needle in PROFILE_NEEDLES:
        stream.truncate(0)
        stream.seek(0)
        profile.print_stats(needle)
        text = stream.getvalue()
        for line in text.splitlines():
            if needle not in line or 'function calls' in line:
                continue
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                stats[needle] = (int(parts[0]), float(parts[3]))
                break
    return stats


def print_run_summary(label: str, run_dir: Path) -> None:
    """Print a human-readable summary for one run directory."""
    timings_path = run_dir / 'StageTimings.json'
    profile_path = run_dir / 'TestIDocBuild.profile'
    print(f'=== {label} ({run_dir}) ===')
    if not timings_path.is_file():
        print('  missing StageTimings.json')
        return
    timings = summarize_stage_timings(timings_path)
    print(f"  StageTimings SUM: {timings.get('__TOTAL__', 0.0):.1f}s")
    for pipeline in KEY_PIPELINES:
        if pipeline in timings:
            print(f'    {pipeline}: {timings[pipeline]:.1f}s')
    block2 = refine_block_totals(timings_path, block_index=2)
    if block2:
        print(f'    Grid8 block2 sum: {sum(block2.values()):.1f}s')
        for name, seconds in sorted(block2.items(), key=lambda item: -item[1]):
            print(f'      {seconds:6.1f}s  {name}')
    needles = profile_needles(profile_path)
    if needles:
        print('  cProfile needles (ncalls, cumtime):')
        for needle in PROFILE_NEEDLES:
            if needle in needles:
                ncalls, cumtime = needles[needle]
                print(f'    {needle}: {ncalls} calls, {cumtime:.1f}s cum')
    print()


def archive_run(source_dir: Path, dest_dir: Path) -> None:
    """Copy StageTimings, profile, and Timing.txt into a labeled archive directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ('StageTimings.json', 'TestIDocBuild.profile', 'Timing.txt'):
        src = source_dir / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)


def compare_runs(base_dir: Path, other_dir: Path) -> None:
    """Print timing deltas between two archived run directories."""
    base = summarize_stage_timings(base_dir / 'StageTimings.json')
    other = summarize_stage_timings(other_dir / 'StageTimings.json')
    print(f'=== Delta ({other_dir.name} vs {base_dir.name}) ===')
    print(f"  StageTimings SUM: {other.get('__TOTAL__', 0.0) - base.get('__TOTAL__', 0.0):+.1f}s")
    for pipeline in KEY_PIPELINES:
        if pipeline in base or pipeline in other:
            delta = other.get(pipeline, 0.0) - base.get(pipeline, 0.0)
            print(f'    {pipeline}: {delta:+.1f}s')


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize', type=Path, help='Summarize one run directory')
    parser.add_argument('--archive', type=Path, help='Archive source test output directory')
    parser.add_argument('--label', type=str, default='run', help='Label for archived run directory')
    parser.add_argument('--archive-root', type=Path, default=Path('/tmp/nornir-test-output'),
                        help='Root directory for archived runs')
    parser.add_argument('--compare', nargs=2, metavar=('BASE', 'OTHER'), type=Path,
                        help='Compare two archived run directories')
    args = parser.parse_args()

    if args.summarize is not None:
        print_run_summary(args.summarize.name, args.summarize)
    if args.archive is not None:
        dest = args.archive_root / f'TestIDocBuild-{args.label}'
        archive_run(args.archive, dest)
        print(f'Archived to {dest}')
        print_run_summary(args.label, dest)
    if args.compare is not None:
        compare_runs(args.compare[0], args.compare[1])
    if args.summarize is None and args.archive is None and args.compare is None:
        parser.error('Specify --summarize, --archive, or --compare')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
