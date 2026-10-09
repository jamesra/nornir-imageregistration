"""Unit tests for mosaic_tileset.tile_id_from_filename."""

from __future__ import annotations

import os
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from nornir_imageregistration.mosaic_tileset import tile_id_from_filename


def _stem_is_not_int(stem: str) -> bool:
    """True when int(stem) would raise, matching tile_id_from_filename."""
    try:
        int(stem)
    except (ValueError, TypeError):
        return True
    return False


class TestTileIdFromFilename(unittest.TestCase):
    """Parse numeric tile IDs from filename stems with enumerate fallbacks."""

    def test_numeric_stem(self) -> None:
        """Bare numeric stems become the tile ID."""
        self.assertEqual(tile_id_from_filename('123.png', 0), 123)
        self.assertEqual(tile_id_from_filename('0.tif', 9), 0)

    def test_non_numeric_stem_uses_default(self) -> None:
        """Non-numeric stems keep the caller-supplied default (enumerate index)."""
        self.assertEqual(tile_id_from_filename('tile_a.png', 7), 7)
        self.assertEqual(tile_id_from_filename('12x.png', 3), 3)
        self.assertEqual(tile_id_from_filename('', 1), 1)

    def test_directory_prefix_ignored(self) -> None:
        """Only the basename stem is parsed, matching Create's prior behavior."""
        self.assertEqual(tile_id_from_filename(os.path.join('tiles', '456.png'), 0), 456)
        self.assertEqual(tile_id_from_filename(os.path.join('tiles', 'other.png'), 2), 2)

    @given(
        tile_id=st.integers(min_value=0, max_value=10**9),
        default=st.integers(min_value=0, max_value=10**6),
        ext=st.sampled_from(('.png', '.tif', '.jpg', '')),
    )
    @settings(max_examples=50)
    def test_numeric_stem_property(self, tile_id: int, default: int, ext: str) -> None:
        """Any integer stem round-trips regardless of extension or default."""
        path = f'{tile_id}{ext}'
        self.assertEqual(tile_id_from_filename(path, default), tile_id)
        nested = os.path.join('a', 'b', path)
        self.assertEqual(tile_id_from_filename(nested, default), tile_id)

    @given(
        stem=st.text(
            alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='/\\'),
            min_size=1,
            max_size=32,
        ).filter(_stem_is_not_int),
        default=st.integers(min_value=0, max_value=10**6),
    )
    @settings(max_examples=50)
    def test_non_numeric_stem_property(self, stem: str, default: int) -> None:
        """Stems that are not integer literals always return default."""
        self.assertEqual(tile_id_from_filename(f'{stem}.png', default), default)


if __name__ == '__main__':
    unittest.main()
