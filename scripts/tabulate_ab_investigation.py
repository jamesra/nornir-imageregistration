#!/usr/bin/env python3
"""Tabulate archived TestIDocBuild A/B investigation runs."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from compare_idoc_profile_runs import KEY_PIPELINES, summarize_stage_timings, refine_block_totals


def mean(values: list[float]) -> float:
    """Return arithmetic mean or 0.0 for an empty list."""
    return statistics.mean(values) if values else 0.0


def collect_runs(archive_root: Path, prefix: str, repeats: int) -> list[Path]:
    """Return existing archived run directories for one case prefix."""
    runs: list[Path] = []
    for index in range(1, repeats + 1):
        path = archive_root / f'TestIDocBuild-{prefix}_run{index}'
        if path.is_dir():
            runs.append(path)
    return runs


def main() -> int:
    """Print a markdown table of mean timings per case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=Path('/tmp/nornir-test-output'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', type=Path,
                        default=Path('/tmp/nornir-test-output/TestIDocBuild-after-transfer-opt'))
    args = parser.parse_args()

    cases = (
        ('transfer-opt-rebaseline', 'Transfer-opt re-baseline (no P1)'),
        ('p1_fixed', 'Fixed Priority 1 (memory + revision cache)'),
        ('p1_no_cache', 'Prewarp cache disabled'),
        ('p1_head', 'Committed HEAD (no Priority 1)'),
    )
    metrics = ['__TOTAL__', *KEY_PIPELINES[:4], 'grid8_block2']
    rows: dict[str, dict[str, list[float]]] = {metric: {} for metric in metrics}

    if args.baseline.is_dir() and (args.baseline / 'StageTimings.json').is_file():
        base = summarize_stage_timings(args.baseline / 'StageTimings.json')
        block2 = refine_block_totals(args.baseline / 'StageTimings.json', block_index=2)
        rows['__TOTAL__']['after-transfer-opt baseline'] = [base.get('__TOTAL__', 0.0)]
        for pipeline in KEY_PIPELINES[:4]:
            rows[pipeline]['after-transfer-opt baseline'] = [base.get(pipeline, 0.0)]
        rows['grid8_block2']['after-transfer-opt baseline'] = [sum(block2.values())]

    for prefix, label in cases:
        for run_dir in collect_runs(args.archive_root, prefix, args.repeats):
            timings = summarize_stage_timings(run_dir / 'StageTimings.json')
            block2 = refine_block_totals(run_dir / 'StageTimings.json', block_index=2)
            rows['__TOTAL__'].setdefault(label, []).append(timings.get('__TOTAL__', 0.0))
            for pipeline in KEY_PIPELINES[:4]:
                rows[pipeline].setdefault(label, []).append(timings.get(pipeline, 0.0))
            rows['grid8_block2'].setdefault(label, []).append(sum(block2.values()))

    print('| Metric | ' + ' | '.join(sorted({label for metric in rows.values() for label in metric})) + ' |')
    print('| --- | ' + ' | '.join(['---'] * len({label for metric in rows.values() for label in metric})) + ' |')
    for metric in metrics:
        labels = sorted(rows[metric].keys())
        cells = []
        for label in labels:
            values = rows[metric][label]
            if len(values) == 1:
                cells.append(f'{values[0]:.1f}s')
            else:
                cells.append(f'{mean(values):.1f}s ± {statistics.pstdev(values):.1f}' if len(values) > 1 else f'{mean(values):.1f}s')
        print(f'| {metric} | ' + ' | '.join(cells) + ' |')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
