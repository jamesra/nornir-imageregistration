"""Scout a StosGroup for transforms that may benefit from a finer refine pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nornir_imageregistration.refine_assessment.group_scout import (
    DEFAULT_LIMIT,
    DEFAULT_SCHEDULES,
    RefineSchedule,
    ScoutConfig,
    default_output_dir,
    run_scout,
)


def __CreateArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Screen StosGroup transforms by cached pair ZNCC, then trial-refine '
            'the worst pairs with dense 256 and fine 128 schedules in scratch space.'
        ),
    )
    parser.add_argument(
        '--group',
        required=True,
        help='StosGroup directory containing group-root .stos files and stos_quality.json',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Scratch/report directory (default TESTOUTPUTPATH/refine_group_scout/<group>)',
    )
    parser.add_argument(
        '--limit',
        default=str(DEFAULT_LIMIT),
        help='Number of pairs to trial, or "all"',
    )
    parser.add_argument(
        '--include-manual',
        action='store_true',
        help='Do not exclude pairs that have a Manual/ override',
    )
    parser.add_argument(
        '--schedules',
        default='dense256,fine128',
        help='Comma-separated schedule names: dense256, fine128',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Override iteration count for every selected schedule',
    )
    return parser


def _parse_limit(text: str) -> int | None:
    if text.strip().lower() == 'all':
        return None
    value = int(text)
    if value < 0:
        raise ValueError('--limit must be >= 0 or "all"')
    return value


def _parse_schedules(text: str, iterations: int | None) -> tuple[RefineSchedule, ...]:
    known = {schedule.name: schedule for schedule in DEFAULT_SCHEDULES}
    selected: list[RefineSchedule] = []
    for name in text.split(','):
        key = name.strip()
        if not key:
            continue
        if key not in known:
            raise ValueError(f'Unknown schedule {key!r}; expected dense256 or fine128')
        schedule = known[key]
        if iterations is not None:
            schedule = RefineSchedule(
                schedule.name,
                schedule.cell_size,
                schedule.grid_spacing,
                int(iterations),
            )
        selected.append(schedule)
    if not selected:
        raise ValueError('At least one schedule is required')
    return tuple(selected)


def Execute(ExecArgs: list[str] | None = None) -> int:
    """Run the StosGroup scout and write a priority report."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
    parser = __CreateArgParser()
    args = parser.parse_args(ExecArgs)
    group_dir = str(Path(args.group).resolve())
    output_dir = args.output_dir or str(default_output_dir(group_dir))
    try:
        limit = _parse_limit(str(args.limit))
        schedules = _parse_schedules(str(args.schedules), args.iterations)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    config = ScoutConfig(
        group_dir=group_dir,
        output_dir=output_dir,
        limit=limit,
        exclude_manual=not bool(args.include_manual),
        schedules=schedules,
        command=['nornir-stos-group-scout', *ExecArgs],
    )
    _results, ranked, reports = run_scout(config)
    print(f'Wrote {reports["priority_csv"]}')
    print(f'Wrote {reports["priority_json"]}')
    print(f'Wrote {reports["config"]}')
    print(f'Ranked {len(ranked)} pairs')
    for index, row in enumerate(ranked[:20], start=1):
        print(
            f'{index:2d} {row.pair:12s} {row.schedule:8s} '
            f'accepted={row.accepted} repaired={row.repaired_clusters} '
            f'net={row.net_improved} dP05={row.delta_p05} '
            f'dPair={row.pair_zncc_delta}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(Execute())
