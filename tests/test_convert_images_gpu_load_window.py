"""Bounded in-flight host memory for :func:`ConvertImagesInDictGpu`.

Loads were once submitted for every tile at once, so each completed task
retained a decoded host array until its chunk was consumed and resident memory
tracked the section size: a 419 MB section measured a 415 MB peak, and an 8 GB
section would have held 8 GB.  Since 4815403 loads are a sliding window of
``_GPU_CONTRAST_IO_TASKS`` tasks (the byte-budget sizing helper and its tests are
gone); this checks that any window width converts every tile identically.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.core import _core


@unittest.skipUnless(nornir_imageregistration.HasCupy() and nornir_imageregistration.UsingCupy(),
                     "ConvertImagesInDictGpu requires an active CuPy backend")
class TestBoundedWindowPreservesOutput(unittest.TestCase):
    """A narrow load window must convert every tile, identically to the default window."""

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

        self._original_window = _core._GPU_CONTRAST_IO_TASKS
        self.addCleanup(self._restore_window)
        self.addCleanup(self._tmp.cleanup)

    def _restore_window(self):
        _core._GPU_CONTRAST_IO_TASKS = self._original_window

    def _convert_with_window(self, io_tasks: int, tag: str) -> list[np.ndarray]:
        out_dir = os.path.join(self._tmp.name, f'out_{tag}')
        os.makedirs(out_dir, exist_ok=True)
        mapping = {p: os.path.join(out_dir, os.path.basename(p)) for p in self.inputs}

        _core._GPU_CONTRAST_IO_TASKS = io_tasks
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
        """A one-task window gives the same output as the default window."""
        upfront = self._convert_with_window(self._original_window, 'default')
        bounded = self._convert_with_window(1, 'bounded')

        self.assertEqual(len(bounded), len(upfront))
        for i, (a, b) in enumerate(zip(upfront, bounded)):
            np.testing.assert_array_equal(a, b, err_msg=f'tile {i} differs')

    def test_every_tile_is_converted_under_a_tiny_budget(self):
        outputs = self._convert_with_window(1, 'tiny')
        self.assertEqual(len(outputs), self.N)


if __name__ == '__main__':
    unittest.main()
