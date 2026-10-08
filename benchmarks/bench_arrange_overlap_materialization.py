"""Peak memory for dict overlap iteration on mosaic arrange paths."""

from __future__ import annotations

import argparse
import tracemalloc
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _FakeTile:
    ID: int


@dataclass(frozen=True, slots=True)
class _FakeOverlap:
    a_id: int
    b_id: int

    @property
    def A(self) -> _FakeTile:
        return _FakeTile(self.a_id)

    @property
    def B(self) -> _FakeTile:
        return _FakeTile(self.b_id)

    @property
    def ID(self) -> tuple[int, int]:
        return self.a_id, self.b_id


def _build_overlap_dict(n: int) -> dict[Any, _FakeOverlap]:
    overlaps: dict[Any, _FakeOverlap] = {}
    for i in range(n):
        a = i * 2
        b = a + 1
        ov = _FakeOverlap(a, b)
        overlaps[ov.ID] = ov
    return overlaps


def _old_two_pass_layout_prep(overlaps: dict[Any, _FakeOverlap]) -> int:
    list_tile_overlaps = list(overlaps.values())
    count = 0
    for _t in list_tile_overlaps:
        count += 1
    for _tile_overlap in list_tile_overlaps:
        count += 1
    return count


def _new_two_pass_layout_prep(overlaps: dict[Any, _FakeOverlap]) -> int:
    count = 0
    for _t in overlaps.values():
        count += 1
    for _tile_overlap in overlaps.values():
        count += 1
    return count


def _peak_bytes(fn, overlaps: dict[Any, _FakeOverlap]) -> int:
    tracemalloc.start()
    fn(overlaps)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50_000)
    args = parser.parse_args()
    overlaps = _build_overlap_dict(args.n)
    old_peak = _peak_bytes(_old_two_pass_layout_prep, overlaps)
    new_peak = _peak_bytes(_new_two_pass_layout_prep, overlaps)
    print(f"n={args.n} old_peak={old_peak} new_peak={new_peak} saved={old_peak - new_peak}")


if __name__ == "__main__":
    main()
