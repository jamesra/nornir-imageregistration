"""Bounded in-flight host memory for :func:`ConvertImagesInDictGpu`.

Loads were once submitted for every tile at once, so each completed task
retained a decoded host array until its chunk was consumed and resident memory
tracked the section size: a 419 MB section measured a 415 MB peak, and an 8 GB
section would have held 8 GB.  Dispatch now follows the same hybrid rule as
:func:`ConvertImagesInDictGpuPyramid`.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.core import _core

_MB = 1024 * 1024


class TestGpuLoadWindowSize(unittest.TestCase):
    """The dispatch rule, which needs no GPU."""

    def test_section_inside_budget_is_submitted_upfront(self):
        """Sections that fit keep the old all-upfront behaviour exactly."""
        window = _core._gpu_load_window_size(n_tiles=200,
                                             tile_host_bytes=2 * _MB,
                                             chunk_size=16,
                                             num_io_workers=64,
                                             budget_bytes=2048 * _MB)
        self.assertEqual(window, 200)

    def test_section_exactly_at_budget_is_submitted_upfront(self):
        window = _core._gpu_load_window_size(n_tiles=100,
                                             tile_host_bytes=1 * _MB,
                                             chunk_size=4,
                                             num_io_workers=8,
                                             budget_bytes=100 * _MB)
        self.assertEqual(window, 100)

    def test_oversized_section_is_bounded_by_the_budget(self):
        """The regression: in-flight tiles must not track the tile count."""
        window = _core._gpu_load_window_size(n_tiles=10000,
                                             tile_host_bytes=32 * _MB,
                                             chunk_size=2,
                                             num_io_workers=4,
                                             budget_bytes=1024 * _MB)
        self.assertEqual(window, 32)
        self.assertLess(window, 10000)

    def test_window_never_exceeds_the_tile_count(self):
        window = _core._gpu_load_window_size(n_tiles=5,
                                             tile_host_bytes=512 * _MB,
                                             chunk_size=64,
                                             num_io_workers=64,
                                             budget_bytes=1 * _MB)
        self.assertEqual(window, 5)

    def test_window_is_floored_at_one_chunk(self):
        """A window under one chunk would stall the GPU every chunk."""
        window = _core._gpu_load_window_size(n_tiles=1000,
                                             tile_host_bytes=64 * _MB,
                                             chunk_size=8,
                                             num_io_workers=2,
                                             budget_bytes=64 * _MB)
        self.assertEqual(window, 8)

    def test_window_is_floored_at_one_task_per_worker(self):
        """Fewer tasks than workers would idle the load pool."""
        window = _core._gpu_load_window_size(n_tiles=1000,
                                             tile_host_bytes=64 * _MB,
                                             chunk_size=1,
                                             num_io_workers=16,
                                             budget_bytes=64 * _MB)
        self.assertEqual(window, 16)

    def test_memory_scales_with_workers_not_section_size(self):
        """Growing the section must not grow the window once past the budget."""
        common = dict(tile_host_bytes=16 * _MB, chunk_size=4,
                      num_io_workers=8, budget_bytes=256 * _MB)
        small = _core._gpu_load_window_size(n_tiles=1000, **common)
        huge = _core._gpu_load_window_size(n_tiles=100000, **common)
        self.assertEqual(small, huge)

    def test_empty_section(self):
        self.assertEqual(_core._gpu_load_window_size(n_tiles=0,
                                                     tile_host_bytes=1,
                                                     chunk_size=1,
                                                     num_io_workers=1,
                                                     budget_bytes=1), 0)


@unittest.skipUnless(nornir_imageregistration.HasCupy() and nornir_imageregistration.UsingCupy(),
                     "ConvertImagesInDictGpu requires an active CuPy backend")
class TestBoundedWindowPreservesOutput(unittest.TestCase):
    """A bounded window must convert every tile, identically to all-upfront."""

    TILE = 64
    N = 40

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.src = os.path.join(root, 'src')
        os.makedirs(self.src)

        rng = np.random.default_rng(4)
        self.tiles = [(rng.random((self.TILE, self.TILE)) * 65535).astype(np.uint16)
                      for _ in range(self.N)]
        self.inputs = []
        for i, tile in enumerate(self.tiles):
            p = os.path.join(self.src, f't{i:03d}.png')
            nornir_imageregistration.SaveImage(p, tile, bpp=16)
            self.inputs.append(p)

        self._original_budget = _core.CONVERT_IMAGES_GPU_LOAD_BUDGET_BYTES
        self.addCleanup(self._restore_budget)
        self.addCleanup(self._tmp.cleanup)

    def _restore_budget(self):
        _core.CONVERT_IMAGES_GPU_LOAD_BUDGET_BYTES = self._original_budget

    def _convert_with_budget(self, budget_bytes: int, tag: str) -> list[np.ndarray]:
        out_dir = os.path.join(self._tmp.name, f'out_{tag}')
        os.makedirs(out_dir, exist_ok=True)
        mapping = {p: os.path.join(out_dir, os.path.basename(p)) for p in self.inputs}

        _core.CONVERT_IMAGES_GPU_LOAD_BUDGET_BYTES = budget_bytes
        result = _core.ConvertImagesInDictGpu(mapping, InputBpp=16, OutputBpp=16,
                                             MinMax=(0, 65535), Gamma=1.0)
        self.assertTrue(result)

        outputs = []
        for out_path in mapping.values():
            self.assertTrue(os.path.exists(out_path),
                            f"{tag}: {out_path} was never written")
            # LoadImage returns device arrays under the CuPy backend; compare on the host.
            outputs.append(nornir_imageregistration.EnsureNumpyArray(
                nornir_imageregistration.LoadImage(out_path)))
        return outputs

    def test_bounded_window_matches_all_upfront(self):
        """One tile of budget forces the window; output must be unchanged."""
        upfront = self._convert_with_budget(2048 * _MB, 'upfront')

        tile_bytes = self.TILE * self.TILE * 2
        bounded = self._convert_with_budget(tile_bytes, 'bounded')

        self.assertEqual(len(bounded), len(upfront))
        for i, (a, b) in enumerate(zip(upfront, bounded)):
            np.testing.assert_array_equal(a, b, err_msg=f'tile {i} differs')

    def test_every_tile_is_converted_under_a_tiny_budget(self):
        tile_bytes = self.TILE * self.TILE * 2
        outputs = self._convert_with_budget(tile_bytes, 'tiny')
        self.assertEqual(len(outputs), self.N)


if __name__ == '__main__':
    unittest.main()
