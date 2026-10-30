"""Unit tests for WindowFilterCache lock, memory, and read-only semantics."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import mock

import numpy as np
from numpy.typing import DTypeLike, NDArray

from nornir_imageregistration.image_filter_cache import (
    FilterWindowCreationFunction,
    WindowFilterCache,
)
from nornir_imageregistration.type_info import ShapeLike


def _stub_creation(
    calls: list[tuple[tuple[int, int], np.dtype]],
) -> FilterWindowCreationFunction:
    """Build a creation function that records calls and returns distinct arrays."""

    def create(
        shape: ShapeLike,
        dtype: DTypeLike | None,
    ) -> NDArray[np.floating]:
        row, col = int(shape[0]), int(shape[1])
        resolved = np.dtype(dtype)
        calls.append(((row, col), resolved))
        return np.arange(row * col, dtype=resolved).reshape((row, col))

    return create


class TestWindowFilterCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="window_filter_cache_")
        self._creation_calls: list[tuple[tuple[int, int], np.dtype]] = []
        patcher = mock.patch(
            "nornir_imageregistration.gettempdir",
            return_value=self._tmpdir,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.cache = WindowFilterCache(
            "unit-test-cache",
            _stub_creation(self._creation_calls),
            dtype=np.float64,
        )

    def tearDown(self) -> None:
        self.cache._loaded_images.clear()
        if os.path.isdir(self.cache.cache_dir):
            for name in os.listdir(self.cache.cache_dir):
                os.remove(os.path.join(self.cache.cache_dir, name))
            os.rmdir(self.cache.cache_dir)

    def test_keep_get_or_create_returns_input_when_shape_matches(self) -> None:
        image = np.zeros((4, 5), dtype=np.float64)
        out = self.cache.KeepGetOrCreate(image, (4, 5))
        self.assertIs(out, image)
        self.assertEqual(self._creation_calls, [])

    def test_get_or_create_uses_memory_after_first_build(self) -> None:
        shape = (3, 3)
        first = self.cache.GetOrCreate(shape)
        second = self.cache.GetOrCreate(shape)
        self.assertIs(first, second)
        self.assertEqual(len(self._creation_calls), 1)

    def test_cached_arrays_are_read_only(self) -> None:
        arr = self.cache.GetOrCreate((2, 2))
        self.assertFalse(arr.flags.writeable)

    def test_concurrent_get_or_create_builds_once_per_shape(self) -> None:
        shape = (8, 8)
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait(timeout=5)
            self.cache.GetOrCreate(shape)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(8)]
            for future in as_completed(futures):
                future.result()
        self.assertEqual(len(self._creation_calls), 1)

    def test_keep_get_or_create_fetches_cache_when_shape_differs(self) -> None:
        image = np.zeros((2, 2), dtype=np.float64)
        cached = self.cache.KeepGetOrCreate(image, (5, 5))
        self.assertEqual(cached.shape, (5, 5))
        self.assertEqual(len(self._creation_calls), 1)

    def test_keep_get_or_create_rejects_non_2d_shape(self) -> None:
        with self.assertRaises(ValueError):
            self.cache.KeepGetOrCreate(None, (3, 4, 5))

    def test_loads_from_disk_when_not_in_memory(self) -> None:
        shape = (6, 6)
        first = self.cache.GetOrCreate(shape)
        del self.cache._loaded_images[shape]
        second = self.cache.GetOrCreate(shape)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(self._creation_calls), 1)
        self.assertFalse(second.flags.writeable)


if __name__ == "__main__":
    unittest.main()
