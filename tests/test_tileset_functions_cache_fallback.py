"""A tile whose inputs are not in the local cache must still be built from source (#256).

``CreateOneTilesetTileWithPillowOverNetwork`` substituted the source paths only when the cache
*directory* was absent.  When the directory existed but held none of a tile's four inputs,
``use_temp_dir`` was ``False`` while ``temp_*`` still pointed into the cache, so the tile was
assembled from four paths that do not exist.  Nothing was written, a warning was logged, and the
build continued -- leaving a hole in the pyramid level with every source tile present on disk.

Measured on a 6x6 output level before the fix: 36 tiles with a full cache, 4 with a sparse cache,
and **0** with an existing-but-empty cache directory.  No exception in any case.

That state is normal after an interrupted run, because cached inputs are deleted as they are
consumed and the caller only falls back to ``None`` when the directory is gone entirely.
"""

import hashlib
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from nornir_imageregistration import tileset_functions

TILE = 32
GRID = 3  # output grid; the input level is twice this in each axis


def _write_tile(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.full((TILE, TILE), value % 251, dtype=np.uint8)).save(path)


def _digest(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _input_value(ix, iy):
    return ix * 7 + iy * 13


class _Level:
    """A source level, an output directory, and a shared cache populated to order."""

    def __init__(self, cached_output_tiles):
        self.root = tempfile.mkdtemp(prefix="tileset_cache_")
        self.source = os.path.join(self.root, "Level_002")
        self.dest = os.path.join(self.root, "Level_004")
        self.temp_in = os.path.join(self.root, "temp", "Level_002")
        self.temp_out = os.path.join(self.root, "temp", "Level_004")
        for path in (self.source, self.dest, self.temp_in, self.temp_out):
            os.makedirs(path, exist_ok=True)

        for iy in range(GRID * 2):
            for ix in range(GRID * 2):
                _write_tile(os.path.join(self.source, self._name(ix, iy)), _input_value(ix, iy))

        for (iy, ix) in cached_output_tiles:
            for dy in (0, 1):
                for dx in (0, 1):
                    sx, sy = ix * 2 + dx, iy * 2 + dy
                    _write_tile(os.path.join(self.temp_in, self._name(sx, sy)),
                                _input_value(sx, sy))

    @staticmethod
    def _name(ix, iy):
        return f"X{ix:03d}_Y{iy:03d}.png"

    def build(self, temp_input_dir='cache', workers=4):
        """Assemble every output tile the way BuildTilesetLevelWithPillow does."""
        cache = self.temp_in if temp_input_dir == 'cache' else temp_input_dir

        def one_tile(coords, executor):
            iy, ix = coords
            x1, y1 = ix * 2, iy * 2
            paths = [os.path.join(self.source, self._name(x1 + dx, y1 + dy))
                     for dy in (0, 1) for dx in (0, 1)]
            tileset_functions.CreateOneTilesetTileWithPillowOverNetwork(
                (TILE, TILE),
                TopLeft=paths[0], TopRight=paths[1],
                BottomLeft=paths[2], BottomRight=paths[3],
                temp_input_dir=cache,
                OutputFileFullPath=os.path.join(self.dest, self._name(ix, iy)),
                output_level_temp_dir=self.temp_out,
                executor=executor)

        coords = [(iy, ix) for iy in range(GRID) for ix in range(GRID)]
        with ThreadPoolExecutor(max_workers=workers) as outer:
            with ThreadPoolExecutor(max_workers=workers) as inner:
                for future in [outer.submit(one_tile, c, inner) for c in coords]:
                    future.result()

    def outputs(self):
        return {name: _digest(os.path.join(self.dest, name))
                for name in sorted(os.listdir(self.dest))}

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


ALL_TILES = [(iy, ix) for iy in range(GRID) for ix in range(GRID)]
EXPECTED = GRID * GRID


class TestEveryTileIsBuilt(unittest.TestCase):
    """The level must be complete regardless of what the cache happens to hold."""

    def _build_and_count(self, cached):
        level = _Level(cached)
        try:
            level.build()
            return level.outputs()
        finally:
            level.close()

    def test_a_full_cache_builds_every_tile(self):
        self.assertEqual(EXPECTED, len(self._build_and_count(ALL_TILES)))

    def test_an_empty_cache_directory_builds_every_tile(self):
        """Was 0 of 9: an existing but empty cache produced an entirely empty level."""
        self.assertEqual(EXPECTED, len(self._build_and_count([])))

    def test_a_sparse_cache_builds_every_tile(self):
        """Was 1 of 9: only the cached tile was produced."""
        self.assertEqual(EXPECTED, len(self._build_and_count(ALL_TILES[:1])))

    def test_no_cache_directory_builds_every_tile(self):
        """The path that already worked, because the source paths were substituted."""
        level = _Level([])
        try:
            level.build(temp_input_dir=None)
            self.assertEqual(EXPECTED, len(level.outputs()))
        finally:
            level.close()


class TestTheCacheDoesNotChangeTheOutput(unittest.TestCase):
    """Parity: the cache is an I/O optimisation and must not alter a single byte."""

    def setUp(self):
        self.results = {}
        for label, cached in (('full', ALL_TILES), ('sparse', ALL_TILES[:1]), ('empty', [])):
            level = _Level(cached)
            try:
                level.build()
                self.results[label] = level.outputs()
            finally:
                level.close()

        reference = _Level([])
        try:
            reference.build(temp_input_dir=None)
            self.results['no_cache_dir'] = reference.outputs()
        finally:
            reference.close()

    def test_all_cache_states_agree_byte_for_byte(self):
        reference = self.results['no_cache_dir']
        for label, outputs in self.results.items():
            with self.subTest(cache=label):
                self.assertEqual(reference, outputs)

    def test_the_tiles_are_not_blank(self):
        """Guard against 'identical' being satisfied by every output being empty."""
        level = _Level([])
        try:
            level.build()
            name = sorted(os.listdir(level.dest))[0]
            array = np.asarray(Image.open(os.path.join(level.dest, name)))
            self.assertGreater(len(np.unique(array)), 1)
        finally:
            level.close()


class TestTheCacheIsStillUsed(unittest.TestCase):
    """The fallback must not quietly disable the optimisation it falls back from."""

    def test_a_cached_tile_is_read_from_the_cache(self):
        level = _Level(ALL_TILES)
        opened = []
        real_open = Image.open

        def tracking_open(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                opened.append(str(path))
            return real_open(path, *args, **kwargs)

        try:
            import unittest.mock as mock
            with mock.patch.object(Image, 'open', tracking_open):
                level.build(workers=1)
            from_cache = [p for p in opened if os.path.normcase(level.temp_in) in os.path.normcase(p)]
            self.assertGreater(len(from_cache), 0)
        finally:
            level.close()

    def test_cached_inputs_are_consumed(self):
        """The existing behaviour of deleting cached inputs after use is preserved."""
        level = _Level(ALL_TILES)
        try:
            before = len(os.listdir(level.temp_in))
            self.assertEqual(EXPECTED * 4, before)
            level.build()
            after = len(os.listdir(level.temp_in)) if os.path.isdir(level.temp_in) else 0
            self.assertEqual(0, after)
        finally:
            level.close()


class TestMissingSources(unittest.TestCase):
    """A genuinely absent source must stay absent, not become an error."""

    def test_a_tile_with_no_sources_at_all_is_skipped_quietly(self):
        level = _Level([])
        try:
            for name in os.listdir(level.source):
                os.remove(os.path.join(level.source, name))
            level.build()
            self.assertEqual(0, len(level.outputs()))
        finally:
            level.close()

    def test_a_partially_present_tile_still_builds(self):
        """Edge tiles legitimately have fewer than four inputs."""
        level = _Level([])
        try:
            os.remove(os.path.join(level.source, _Level._name(1, 1)))
            level.build()
            self.assertIn(_Level._name(0, 0), level.outputs())
        finally:
            level.close()


if __name__ == '__main__':
    unittest.main()
