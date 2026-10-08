"""Equivalence tests for mosaic overlap dict iteration (no list materialization)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypothesis import given, strategies as st

import nornir_imageregistration.arrange_mosaic as arrange_mosaic


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


def _normalize(
    tile_to_overlaps: dict[int, dict[Any, arrange_mosaic.TileToOverlap]],
) -> dict[int, dict[Any, tuple[int, _FakeOverlap]]]:
    return {
        tile_id: {
            overlap_id: (entry.iTile, entry.tile_overlap)
            for overlap_id, entry in overlaps.items()
        }
        for tile_id, overlaps in tile_to_overlaps.items()
    }


@st.composite
def overlap_dicts(draw: st.DrawFn) -> tuple[dict[Any, _FakeOverlap], list[_FakeOverlap]]:
    n = draw(st.integers(min_value=0, max_value=64))
    pairs = draw(
        st.lists(
            st.tuples(st.integers(0, 500), st.integers(0, 500)),
            min_size=n,
            max_size=n,
        )
    )
    overlaps: list[_FakeOverlap] = []
    seen: set[tuple[int, int]] = set()
    for a, b in pairs:
        lo, hi = (a, b) if a < b else (b, a)
        if lo == hi or (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        overlaps.append(_FakeOverlap(lo, hi))
    as_dict = {ov.ID: ov for ov in overlaps}
    return as_dict, overlaps


class TestCreateTileToOverlapsDictIterable:
    @given(overlap_dicts())
    def test_dict_matches_list_input(self, overlap_data: tuple[dict[Any, _FakeOverlap], list[_FakeOverlap]]) -> None:
        as_dict, as_list = overlap_data
        from_dict = arrange_mosaic.CreateTileToOverlapsDict(as_dict)  # type: ignore[arg-type]
        from_list = arrange_mosaic.CreateTileToOverlapsDict(as_list)  # type: ignore[arg-type]
        assert _normalize(from_dict) == _normalize(from_list)  # type: ignore[arg-type]
