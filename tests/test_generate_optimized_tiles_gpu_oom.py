"""GenerateOptimizedTiles retries a CuPy strip that ran out of GPU memory instead of failing the section.

Several AssembleTiles processes can share one GPU, and each CuPy pool keeps its peak, so a strip can fail
to allocate while the memory is free a moment later (RC3, 2-3 lanes on one 24 GB card). The strip is
retried after this process drops its decoded tile images, stops prefetching and returns its cached
blocks; any other error, or an out-of-memory error after the last retry, still stops the section.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import mosaic_tileset
from nornir_imageregistration.transforms.rigid import Rigid


class TestStripOutOfMemoryRetry(unittest.TestCase):

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        paths, transforms = [], []
        for i in range(4):
            path = os.path.join(self._tmp.name, f"{i}.png")
            nornir_imageregistration.SaveImage(path, np.full((32, 32), 40.0 + 20.0 * i, dtype=np.float32), bpp=8)
            paths.append(path)
            transforms.append(Rigid(target_offset=(0.0, 32.0 * i)))
        self.tileset = mosaic_tileset.Create(transforms, paths, image_to_source_space_scale=1.0)
        self.tileset.TranslateToZeroOrigin()
        self.tiles = [self.tileset[key] for key in sorted(self.tileset)]

    def _run(self, failures_by_call: dict[int, BaseException], prefetch_strips: int = 2):
        """Run with each strip assembly call numbered; a call listed in *failures_by_call* raises that error."""
        calls: list[int] = []
        resident_at_retry: list[list[bool]] = []

        def assemble(FixedRegion=None, target_space_scale=None):
            call = len(calls)
            calls.append(call)
            for tile in self.tileset.TargetSpaceIntersections(FixedRegion):
                _ = tile.Image
            if call in failures_by_call:
                raise failures_by_call[call]
            shape = (int(FixedRegion.Height), int(FixedRegion.Width))
            return np.full(shape, 0.5, dtype=np.float32), np.ones(shape, dtype=bool)

        def released() -> None:
            resident_at_retry.append([t._image is not None for t in self.tiles])

        self.tileset.AssembleImage = assemble
        with mock.patch.object(nornir_imageregistration, 'GetActiveComputationLib',
                               return_value=nornir_imageregistration.ComputationLib.cupy), \
                mock.patch.object(mosaic_tileset, '_GPU_OOM_RETRY_WAITS_S', (0.0, 0.0)), \
                mock.patch.object(mosaic_tileset, '_release_gpu_cache', side_effect=released), \
                mock.patch.object(mosaic_tileset.time, 'sleep'):
            yielded = list(self.tileset.GenerateOptimizedTiles(tile_dims=(32, 32), max_temp_image_area=32 * 32,
                                                               target_space_scale=1.0,
                                                               prefetch_strips=prefetch_strips))
        return yielded, calls, resident_at_retry

    @staticmethod
    def _oom() -> RuntimeError:
        return RuntimeError("cudaErrorMemoryAllocation: out of memory")

    def test_an_out_of_memory_strip_is_retried_and_every_tile_is_written(self) -> None:
        expected, _, _ = self._run({})
        yielded, calls, resident_at_retry = self._run({1: self._oom()})
        self.assertEqual(len(calls), 5, "the failed strip is assembled once more")
        self.assertEqual([(t[0], t[1]) for t in yielded], [(t[0], t[1]) for t in expected])
        for got, want in zip(yielded, expected):
            np.testing.assert_array_equal(got[2], want[2])
        # Before the cached blocks are returned, no decoded tile image is left in this process.
        self.assertEqual(resident_at_retry, [[False] * 4])

    def test_out_of_memory_after_the_last_retry_stops_the_section(self) -> None:
        oom = self._oom()
        with self.assertRaises(RuntimeError) as raised:
            self._run({1: oom, 2: oom, 3: oom})
        self.assertIs(raised.exception, oom)

    def test_another_error_is_not_retried(self) -> None:
        with self.assertRaises(ValueError):
            self._run({1: ValueError("not a memory error")})

    def test_cupy_out_of_memory_error_is_recognised(self) -> None:
        self.assertTrue(mosaic_tileset._is_gpu_out_of_memory(RuntimeError("CUFFT_ALLOC_FAILED")))
        self.assertFalse(mosaic_tileset._is_gpu_out_of_memory(ValueError("shape mismatch")))
        out_of_memory_error = getattr(getattr(getattr(nornir_imageregistration.cp, 'cuda', None), 'memory', None),
                                      'OutOfMemoryError', None)
        if not isinstance(out_of_memory_error, type):
            self.skipTest("CuPy is not installed")
        self.assertTrue(mosaic_tileset._is_gpu_out_of_memory(out_of_memory_error(1024, 0)))


if __name__ == '__main__':
    unittest.main()
