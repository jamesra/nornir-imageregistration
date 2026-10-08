"""Micro-benchmark: FilenameToTask iteration in Histogram (category 18).

Compares list(keys())+lookup vs dict.items() on realistic tile-path counts.
JSON output goes to TESTOUTPUTPATH; not installed with the package.
"""

from __future__ import annotations

import os
import tracemalloc
from typing import Any

import pyperf

# Realistic full-section tile counts; override with HIST_BENCH_TILE_COUNT.
DEFAULT_TILE_COUNT = 4096


class _InstantTask:
    def wait_return(self) -> Any:
        return None


def _build_filename_to_task(n: int) -> dict[str, _InstantTask]:
    return {f"/data/mosaic/tiles/section_001/tile_{i:05d}.png": _InstantTask() for i in range(n)}


def _collect_old(FilenameToTask: dict[str, _InstantTask]) -> int:
    count = 0
    for f in list(FilenameToTask.keys()):
        task = FilenameToTask[f]
        task.wait_return()
        count += 1
    return count


def _collect_new(FilenameToTask: dict[str, _InstantTask]) -> int:
    count = 0
    for f, task in FilenameToTask.items():
        task.wait_return()
        count += 1
    return count


def _bench_collect(runner: pyperf.Runner, name: str, func, n: int) -> None:
    FilenameToTask = _build_filename_to_task(n)

    def run() -> None:
        func(FilenameToTask)

    runner.bench_func(name, run)


def _peak_bytes(func, n: int, repeats: int = 5) -> int:
    peaks: list[int] = []
    for _ in range(repeats):
        FilenameToTask = _build_filename_to_task(n)
        tracemalloc.start()
        func(FilenameToTask)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    return max(peaks)


def _tile_count() -> int:
    raw = os.environ.get("HIST_BENCH_TILE_COUNT")
    if raw is None:
        return DEFAULT_TILE_COUNT
    return int(raw)


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--memory-only":
        n = _tile_count()
        old_peak = _peak_bytes(_collect_old, n)
        new_peak = _peak_bytes(_collect_new, n)
        print(f"n={n} tracemalloc peak old={old_peak} new={new_peak}")
        if old_peak:
            pct = (old_peak - new_peak) / old_peak * 100.0
            print(f"memory delta={pct:.1f}%")
        return

    n = _tile_count()
    runner = pyperf.Runner()
    runner.parse_args()
    _bench_collect(runner, "histogram_task_dict_old", _collect_old, n)
    _bench_collect(runner, "histogram_task_dict_new", _collect_new, n)


if __name__ == "__main__":
    main()
