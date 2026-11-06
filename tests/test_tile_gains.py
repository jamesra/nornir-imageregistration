"""Per-tile brightness gains (nornir_imageregistration.tile_gains) on synthetic sections with known gains."""
import unittest

import numpy as np
from nornir_imageregistration import tile_gains
from nornir_imageregistration.tile_gains import TilePlacement
from scipy import ndimage

TILE = 128
PITCH = 112  # 16 px overlap


def make_section(rows: int, cols: int, gain_of, shading: float = 0.0, seed: int = 1):
    """Tiles cut from one textured field; tile counts = field * gain * vignetting. Capture order is column-major."""
    rng = np.random.default_rng(seed)
    height, width = PITCH * (rows - 1) + TILE, PITCH * (cols - 1) + TILE
    # Fine texture at about 20% contrast, like tissue at a coarse pyramid level.
    field = 1000.0 + 1000.0 * ndimage.gaussian_filter(rng.standard_normal((height, width)), 1.5)
    field = np.clip(field, 200.0, None)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    vignette = 1.0 - shading * (((yy - TILE / 2) / (TILE / 2)) ** 2 + ((xx - TILE / 2) / (TILE / 2)) ** 2)
    placements, tiles = {}, {}
    tile_id = 0
    for c in range(cols):
        for r in range(rows):
            y, x = r * PITCH, c * PITCH
            placements[tile_id] = TilePlacement(y, x, TILE, TILE)
            tiles[tile_id] = field[y:y + TILE, x:x + TILE] * gain_of(tile_id, r, c) * vignette
            tile_id += 1
    return placements, tiles


def detrend(placements, gains: dict[int, float]) -> dict[int, float]:
    """Remove the quadratic trend over tile centres from log gains: what solve_tile_gains returns by convention."""
    ids = sorted(gains)
    centre = np.array([[placements[t].y + placements[t].height / 2, placements[t].x + placements[t].width / 2]
                       for t in ids])
    centre -= centre.mean(axis=0)
    centre /= np.maximum(np.abs(centre).max(axis=0), 1.0)
    u, v = centre[:, 1], centre[:, 0]
    basis = np.stack([np.ones_like(u), u, v, u * u, v * v, u * v], axis=1)
    log_gain = np.log([gains[t] for t in ids])
    coef, *_ = np.linalg.lstsq(basis, log_gain, rcond=None)
    return dict(zip(ids, np.exp(log_gain - basis @ coef)))


class TestTileGains(unittest.TestCase):

    def assert_relative_gains(self, result, expected: dict[int, float], tolerance: float = 0.01):
        """Gains match the expected ones up to a common factor."""
        ids = sorted(expected)
        got = np.array([result.gains[t] for t in ids])
        want = np.array([expected[t] for t in ids])
        ratio = (got / want) / np.median(got / want)
        self.assertLess(float(np.max(np.abs(ratio - 1.0))), tolerance, f"worst {ratio.min():.4f}..{ratio.max():.4f}")

    def test_resume_step_is_recovered(self):
        """A pass that resumes 35% darker (RC2 white stripe) gives two gain levels, not a gradient."""
        rows, cols = 6, 8
        half = rows * cols // 2

        def gain(t, r, c):
            return 1.0 if t < half else 0.65

        placements, tiles = make_section(rows, cols, gain)
        result = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        # By convention the gains carry no quadratic trend over the section, so they are the true gains minus that
        # trend. On this 8-column toy the step projects strongly onto the trend; on a 30-column section much less.
        self.assert_relative_gains(result, detrend(placements, {t: gain(t, 0, 0) for t in tiles}), tolerance=0.01)
        self.assertEqual(result.pairs_rejected, 0)

    def test_resume_step_on_a_wide_section(self):
        """On a section as wide as RC2's (32 columns) the seam keeps nearly all of its step."""
        rows, cols = 4, 32
        half = rows * cols // 2

        def gain(t, r, c):
            return 1.0 if t < half else 0.65

        placements, tiles = make_section(rows, cols, gain)
        result = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        left = np.median([result.gains[t] for t in range(half - rows, half)])
        right = np.median([result.gains[t] for t in range(half, half + rows)])
        # About 2% of the step is left: its projection on the planar trend (see solve_tile_gains).
        self.assertAlmostEqual(right / left, 0.65, delta=0.65 * 0.03)

    def test_shading_is_not_read_as_gain(self):
        """Vignetting with equal gains must give gains near 1, not a compounding ramp along rows and columns."""
        placements, tiles = make_section(6, 8, lambda t, r, c: 1.0, shading=0.25)
        result = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        gains = np.array([result.gains[t] for t in sorted(result.gains)]).reshape(8, 6)  # columns of 6 tiles
        # Strong vignetting (25% at the corners) leaves ~1% noise per overlap; what must not happen is a ramp that
        # compounds along rows or columns (RC3 0800 trial without shading constants: gains 0.1-80).
        self.assertLess(float(gains.max() / gains.min()), 1.08)
        column_means = gains.mean(axis=1)
        self.assertLess(float(column_means.max() / column_means.min()), 1.05)

    def test_misplaced_tile_pairs_are_rejected(self):
        """A tile showing unrelated tissue gives disagreeing ratios; they are dropped, not averaged in."""
        placements, tiles = make_section(6, 8, lambda t, r, c: 1.0 + 0.01 * (t % 5))
        rng = np.random.default_rng(7)
        tiles[20] = np.clip(1000.0 + 4000.0 * ndimage.gaussian_filter(rng.standard_normal((TILE, TILE)), 2.0), 200, None) * 1.8
        result = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        self.assertGreater(result.pairs_rejected, 0)
        expected = detrend(placements, {t: 1.0 + 0.01 * (t % 5) for t in tiles})
        expected.pop(20)
        self.assert_relative_gains(result, expected, tolerance=0.01)

    def test_blank_tile_takes_capture_neighbours_gain(self):
        """A tile with no tissue has no usable overlap; the temporal prior gives it its capture neighbours' gain."""
        def gain(t, r, c):
            return 0.8 if t >= 24 else 1.0

        placements, tiles = make_section(6, 8, gain)
        tiles[30] = np.zeros((TILE, TILE))
        result = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        self.assertIn(30, result.tiles_without_pairs)
        neighbours = np.sqrt(result.gains[29] * result.gains[31])
        self.assertAlmostEqual(result.gains[30] / neighbours, 1.0, delta=0.03)

    def test_tile_without_overlaps_keeps_a_sane_gain(self):
        """Regression (RC2 1226): a tile without overlaps came out at gain 4e-16 while a hard gauge row pushed the
        sum of all log gains into it. It must stay near its capture neighbours."""
        placements, tiles = make_section(6, 8, lambda t, r, c: 0.5 + 0.004 * t)
        tiles[30] = np.zeros((TILE, TILE))
        result = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        self.assertIn(30, result.tiles_without_pairs)
        self.assertAlmostEqual(result.gains[30] / np.sqrt(result.gains[29] * result.gains[31]), 1.0, delta=0.03)
        self.assertGreater(min(result.gains.values()), 0.5)

    def test_neighbour_pairs_skip_diagonals(self):
        placements = {0: TilePlacement(0, 0, 100, 100), 1: TilePlacement(0, 90, 100, 100),
                      2: TilePlacement(90, 0, 100, 100), 3: TilePlacement(90, 90, 100, 100)}
        pairs = tile_gains.neighbour_pairs(placements)
        self.assertEqual(sorted(pairs), [(0, 1, 'LR'), (0, 2, 'TB'), (1, 3, 'TB'), (2, 3, 'LR')])

    def test_checksum_is_stable(self):
        placements, tiles = make_section(4, 4, lambda t, r, c: 1.0)
        a = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        b = tile_gains.compute_tile_gains(placements, tiles.__getitem__, 1, low=50, high=1e9)
        self.assertEqual(a.checksum(), b.checksum())



class TestStampGainPng(unittest.TestCase):

    def test_gain_goes_into_the_chunk_and_pixels_are_untouched(self):
        import json
        import os
        import tempfile

        from nornir_imageregistration import noise_crush
        from PIL import Image

        params = noise_crush.CrushParams(k=2.5, step=200, sigma=80.0, offset=0, bits=8, source_dtype='uint16')
        codes = (np.arange(64 * 64) % 251).reshape(64, 64).astype(np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, '000.png')
            noise_crush.save_crushed_png(path, codes, params)
            self.assertTrue(tile_gains.stamp_gain_png(path, 1.25))
            self.assertFalse(tile_gains.stamp_gain_png(path, 1.25))  # same gain: file untouched
            with Image.open(path) as im:
                values = json.loads(im.text[noise_crush.PNG_TEXT_KEY])
                pixels = np.asarray(im)
            np.testing.assert_array_equal(pixels, codes)
            self.assertAlmostEqual(values['gain'], 1.25)
            self.assertAlmostEqual(values['effective_step'], 160.0)
            self.assertAlmostEqual(values['effective_offset'], 80.0)
            self.assertEqual(noise_crush.read_crush_params(path), params)

    def test_unchanged_gain_reads_only_the_chunks_before_the_pixels(self):
        import os
        import tempfile

        from nornir_imageregistration import noise_crush

        import builtins
        from unittest import mock

        params = noise_crush.CrushParams(k=2.5, step=200, sigma=80.0, offset=0, bits=8, source_dtype='uint16')
        codes = np.random.default_rng(0).integers(0, 251, (512, 512)).astype(np.uint8)  # incompressible: ~256 KB PNG
        real_open = builtins.open
        bytes_read = []

        class CountingFile:
            def __init__(self, f):
                self._f = f

            def read(self, *args):
                data = self._f.read(*args)
                bytes_read.append(len(data))
                return data

            def __getattr__(self, name):
                return getattr(self._f, name)

            def __enter__(self):
                self._f.__enter__()
                return self

            def __exit__(self, *exc):
                return self._f.__exit__(*exc)

        def counting_open(file, mode='r', *args, **kwargs):
            f = real_open(file, mode, *args, **kwargs)
            return CountingFile(f) if 'b' in mode and 'r' in mode and str(file).endswith('.png') else f

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, '000.png')
            noise_crush.save_crushed_png(path, codes, params)
            self.assertTrue(tile_gains.stamp_gain_png(path, 1.25))
            size = os.path.getsize(path)
            self.assertGreater(size, 100_000)
            with mock.patch('builtins.open', counting_open):
                self.assertFalse(tile_gains.stamp_gain_png(path, 1.25))
            self.assertLess(sum(bytes_read), 4096, f'read {sum(bytes_read)} of {size} bytes to find the gain unchanged')


if __name__ == '__main__':
    unittest.main()
