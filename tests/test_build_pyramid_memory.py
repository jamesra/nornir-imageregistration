"""Tests for in-memory tile pyramid builders (CPU/GPU)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from nornir_imageregistration.core._core import (
    BuildTilePyramidsMemoryCpu,
    _build_pyramid_tile_cpu,
    _downsample_step,
)


class TestDownsampleStep(unittest.TestCase):
    def test_cpu_half_size(self) -> None:
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        out = _downsample_step(arr, 0.5)
        self.assertEqual(out.shape, (2, 2))


class TestBuildPyramidTileCpu(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_png(self, path: str, size: int) -> None:
        data = (np.arange(size * size, dtype=np.uint16) % 4096).reshape(size, size)
        Image.fromarray(data).save(path)

    def test_chained_levels_written(self) -> None:
        size = 32
        input_path = os.path.join(self.tmp, "tile.png")
        out2 = os.path.join(self.tmp, "l002.png")
        out4 = os.path.join(self.tmp, "l004.png")
        self._write_png(input_path, size)

        _build_pyramid_tile_cpu(
            input_path,
            [out2, out4],
            [0.5, 0.5],
        )

        self.assertTrue(os.path.isfile(out2))
        self.assertTrue(os.path.isfile(out4))
        im2 = Image.open(out2)
        im4 = Image.open(out4)
        self.assertEqual(im2.size, (16, 16))
        self.assertEqual(im4.size, (8, 8))

    def test_skip_intermediate_save(self) -> None:
        size = 16
        input_path = os.path.join(self.tmp, "tile.png")
        out4 = os.path.join(self.tmp, "l004.png")
        self._write_png(input_path, size)

        _build_pyramid_tile_cpu(
            input_path,
            [None, out4],
            [0.5, 0.5],
        )

        self.assertFalse(os.path.exists(os.path.join(self.tmp, "l002.png")))
        self.assertTrue(os.path.isfile(out4))
        self.assertEqual(Image.open(out4).size, (4, 4))


class TestBuildTilePyramidsMemoryCpu(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_batch_build(self) -> None:
        tiles: dict[str, list[str | None]] = {}
        for i in range(2):
            size = 16
            inp = os.path.join(self.tmp, f"in{i}.png")
            out = os.path.join(self.tmp, f"out{i}.png")
            data = np.full((size, size), i * 1000, dtype=np.uint16)
            Image.fromarray(data).save(inp)
            tiles[inp] = [out]

        BuildTilePyramidsMemoryCpu(tiles, [0.5], num_threads=2)
        for inp, outs in tiles.items():
            self.assertTrue(os.path.isfile(outs[0]))
            self.assertEqual(Image.open(outs[0]).size, (8, 8))


try:
    from nornir_buildmanager.operations.tile import BuildTilePyramidsCpu
    from nornir_buildmanager.volumemanager.levelnode import LevelNode
    from nornir_buildmanager.volumemanager.tilepyramidnode import TilePyramidNode
    _HAS_BUILDMANAGER = True
except ImportError:
    _HAS_BUILDMANAGER = False


@unittest.skipUnless(_HAS_BUILDMANAGER, "nornir-buildmanager not installed")
class TestBuildTilePyramidsCpuPipeline(unittest.TestCase):
    """Smoke test for the buildmanager pipeline entry point."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name
        pyramid_root = os.path.join(self.tmp, "TilePyramid")
        level1 = os.path.join(pyramid_root, "001")
        os.makedirs(level1)
        data = (np.arange(16 * 16, dtype=np.uint16) % 4096).reshape(16, 16)
        Image.fromarray(data).save(os.path.join(level1, "tile.png"))

        self.pyramid = TilePyramidNode.Create(NumberOfTiles=1)
        self.pyramid.Path = pyramid_root
        self.pyramid.UpdateOrAddChildByAttrib(LevelNode.Create(1), "Downsample")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_builds_coarser_levels(self) -> None:
        result = BuildTilePyramidsCpu(PyramidNode=self.pyramid, Levels=[2, 4])
        self.assertIsNotNone(result)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "TilePyramid", "002", "tile.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "TilePyramid", "004", "tile.png")))


if __name__ == "__main__":
    unittest.main()
