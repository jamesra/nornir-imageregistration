"""Equality behaviour of TileOffset: ``!=`` must be the exact negation of ``==``."""

import operator

from hypothesis import given
from hypothesis import strategies as st

from nornir_imageregistration.settings.mosaic_tile_offset import TileOffset

_ids = st.integers(min_value=0, max_value=100000)
_coords = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_comments = st.one_of(st.none(), st.text(max_size=8))
_offsets = st.builds(TileOffset, _ids, _ids, _coords, _coords, _comments)


def test_equal_offsets_are_not_unequal():
    a = TileOffset(1, 2, 3.0, 4.0, "c")
    b = TileOffset(2, 1, 3.0, 4.0, "c")
    assert a == b
    assert operator.ne(a, b) is False


def test_different_offsets_are_unequal():
    a = TileOffset(1, 2, 3.0, 4.0)
    assert a != TileOffset(1, 3, 3.0, 4.0)
    assert a != TileOffset(1, 2, 3.5, 4.0)
    assert a != TileOffset(1, 2, 3.0, 4.5)
    assert a != TileOffset(1, 2, 3.0, 4.0, "note")


@given(_offsets, _offsets)
def test_ne_is_negation_of_eq(a, b):
    assert operator.ne(a, b) is operator.not_(operator.eq(a, b))
