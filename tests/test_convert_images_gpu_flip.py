"""Smoke tests for ConvertImagesInDictGpu Flip/Flop vs CPU convert, plus VRAM bound."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.core import _ConvertSingleImage

try:
    import setup_imagetest
except (ImportError, ModuleNotFoundError):
    from . import setup_imagetest


@unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
class TestConvertImagesInDictGpuFlip(setup_imagetest.ImageTestBase):
    """GPU convert with Flip/Flop must match CPU and keep VRAM near the batch budget."""

    def setUp(self) -> None:
        super().setUp()
        self._previous_lib = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)

    def tearDown(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(self._previous_lib)
        super().tearDown()

    def _write_gradient_tiles(self, root: str, count: int = 4,
                              shape: tuple[int, int] = (128, 96)) -> dict[str, str]:
        """Write asymmetric gradient tiles so Flip/Flop change pixel layout."""
        h, w = shape
        mapping: dict[str, str] = {}
        rng = np.random.default_rng(42)
        for i in range(count):
            # Row ramp + column ramp + noise so flips are unambiguous.
            rows = np.linspace(20, 200, h, dtype=np.float32)[:, None]
            cols = np.linspace(10, 180, w, dtype=np.float32)[None, :]
            img = np.clip(rows + cols + rng.integers(0, 20, size=(h, w)), 0, 255).astype(np.uint8)
            src = os.path.join(root, f"src_{i:03d}.png")
            dst = os.path.join(root, f"dst_{i:03d}.png")
            nornir_imageregistration.SaveImage(src, img, bpp=8)
            mapping[src] = dst
        return mapping

    def _assert_gpu_matches_cpu(self, *, Flip: bool, Flop: bool) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mapping = self._write_gradient_tiles(tmp)
            minmax = (30.0, 220.0)
            gamma = 1.2

            nornir_imageregistration.ConvertImagesInDictGpu(
                mapping, Flip=Flip, Flop=Flop, InputBpp=8, OutputBpp=8,
                MinMax=minmax, Gamma=gamma, batch_bytes=4 * 1024 * 1024)

            for src, dst in mapping.items():
                self.assertTrue(os.path.exists(dst), f"missing GPU output {dst}")
                expected = _ConvertSingleImage(
                    src, Flip=Flip, Flop=Flop, MinMax=minmax, Gamma=gamma, Bpp=8)
                actual = nornir_imageregistration.LoadImage(dst, backend="numpy")
                np.testing.assert_allclose(
                    actual.astype(np.float32), expected.astype(np.float32),
                    atol=1.0,
                    err_msg=f"Flip={Flip} Flop={Flop} mismatch for {os.path.basename(src)}")

    def test_gpu_matches_cpu_no_flip(self) -> None:
        self._assert_gpu_matches_cpu(Flip=False, Flop=False)

    def test_gpu_matches_cpu_flip(self) -> None:
        self._assert_gpu_matches_cpu(Flip=True, Flop=False)

    def test_gpu_matches_cpu_flop(self) -> None:
        self._assert_gpu_matches_cpu(Flip=False, Flop=True)

    def test_gpu_matches_cpu_flip_and_flop(self) -> None:
        self._assert_gpu_matches_cpu(Flip=True, Flop=True)

    def test_vram_stays_near_batch_budget(self) -> None:
        """Production convert must free CuPy pools so multi-call VRAM stays near batch budget."""
        import cupy as cp

        batch_bytes = 8 * 1024 * 1024  # 8 MB
        # Larger tiles so a naive per-worker upload of many tiles would blow past budget.
        shape = (512, 512)
        tile_count = 16
        tile_bytes = shape[0] * shape[1] * 4  # float32

        # Clean slate for a stable baseline only — assert below must not free again.
        cp.get_default_memory_pool().free_all_blocks()
        pinned_pool = cp.get_default_pinned_memory_pool()
        pinned_pool.free_all_blocks()

        with tempfile.TemporaryDirectory() as tmp:
            mapping_a = self._write_gradient_tiles(tmp, count=tile_count, shape=shape)
            nornir_imageregistration.ConvertImagesInDictGpu(
                mapping_a, Flip=True, InputBpp=8, OutputBpp=8,
                MinMax=(10.0, 245.0), Gamma=1.0, batch_bytes=batch_bytes)
            after_first = cp.get_default_memory_pool().used_bytes()

            # Second call (new outputs) — pool must not ratchet upward across converts.
            mapping_b = {
                src: os.path.join(tmp, f"dst2_{i:03d}.png")
                for i, src in enumerate(mapping_a.keys())
            }
            nornir_imageregistration.ConvertImagesInDictGpu(
                mapping_b, Flip=False, InputBpp=8, OutputBpp=8,
                MinMax=(10.0, 245.0), Gamma=1.0, batch_bytes=batch_bytes)
            after_second = cp.get_default_memory_pool().used_bytes()

        # No free_all_blocks() here: production ConvertImagesInDictGpu must have freed.
        self.assertLess(
            after_first, batch_bytes * 4,
            f"VRAM used {after_first} bytes after first convert; expected near batch "
            f"budget ({batch_bytes} bytes) from production free path")
        self.assertLess(
            after_second, batch_bytes * 4,
            f"VRAM used {after_second} bytes after second convert; pool should not "
            f"ratchet across calls")
        self.assertLess(
            after_second, tile_bytes * tile_count,
            f"VRAM used {after_second} looks like all {tile_count} tiles still resident")


if __name__ == '__main__':
    unittest.main()
