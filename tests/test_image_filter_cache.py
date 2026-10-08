"""Unit tests for WindowFilterCache."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import DTypeLike
from hypothesis import given, settings
from hypothesis import strategies as st

from nornir_imageregistration.image_filter_cache import (
    CreateWindowFilterCache,
    WindowFilterCache,
)


def _ones_filter(shape, dtype=None):
    dtype = np.dtype(dtype if dtype is not None else np.float32)
    return np.ones(shape, dtype=dtype)


class TestWindowFilterCache(unittest.TestCase):
    """Behavior of the disk-backed filter window cache."""

    def _cache(
        self,
        creation_function=_ones_filter,
        dtype: DTypeLike = np.float32,
    ) -> WindowFilterCache:
        name = f"test_wfc_{uuid.uuid4().hex}"
        return WindowFilterCache(name, creation_function, dtype=dtype)

    def test_keep_get_or_create_rejects_non_2d_shape(self) -> None:
        cache = self._cache()
        with self.assertRaises(ValueError):
            cache.KeepGetOrCreate(None, (4, 4, 4))

    def test_keep_get_or_create_returns_matching_image_without_cache_lookup(self) -> None:
        cache = self._cache()
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        kept = cache.KeepGetOrCreate(image, (4, 4))
        self.assertIs(kept, image)

    def test_get_or_create_marks_arrays_read_only(self) -> None:
        cache = self._cache()
        arr = cache.GetOrCreate((8, 8))
        self.assertFalse(arr.flags.writeable)
        self.assertEqual(arr.shape, (8, 8))

    def test_on_disk_wrong_dtype_is_removed_and_rebuilt(self) -> None:
        cache = self._cache(dtype=np.float32)
        shape = (6, 7)
        path = os.path.join(cache.cache_dir, f"{shape[0]}x{shape[1]}.npy")
        np.save(path, np.zeros(shape, dtype=np.float64))

        arr = cache.GetOrCreate(shape)
        self.assertEqual(arr.dtype, np.float32)
        self.assertFalse(arr.flags.writeable)
        reloaded = np.load(path)
        self.assertEqual(reloaded.dtype, np.float32)

    def test_concurrent_get_or_create_single_build_and_shared_array(self) -> None:
        create_count = 0
        count_lock = threading.Lock()

        def counting_create(shape, dtype=None):
            nonlocal create_count
            with count_lock:
                create_count += 1
                tag = create_count
            dtype = np.dtype(dtype if dtype is not None else np.float32)
            return np.full(shape, tag, dtype=dtype)

        cache = self._cache(creation_function=counting_create)
        shape = (32, 32)
        barrier = threading.Barrier(16)
        results: list[np.ndarray | None] = [None] * 16

        def worker(index: int) -> None:
            barrier.wait()
            results[index] = cache.GetOrCreate(shape)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker, i) for i in range(16)]
            for future in as_completed(futures):
                future.result()

        self.assertEqual(create_count, 1)
        first = results[0]
        assert first is not None
        for arr in results:
            assert arr is not None
            self.assertIs(arr, first)

        npy_path = os.path.join(cache.cache_dir, "32x32.npy")
        self.assertTrue(os.path.isfile(npy_path))

    @given(
        height=st.integers(min_value=1, max_value=64),
        width=st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=25, deadline=None)
    def test_get_or_create_output_shape_matches_request(self, height: int, width: int) -> None:
        cache = self._cache()
        shape = (height, width)
        arr = cache.GetOrCreate(shape)
        self.assertEqual(tuple(arr.shape), shape)
        self.assertFalse(arr.flags.writeable)


class TestCreateWindowFilterCache(unittest.TestCase):
    def test_factory_builds_hann_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            original = os.environ.get("NORNIR_TEMP_DIR")
            os.environ["NORNIR_TEMP_DIR"] = temp_root
            try:
                cache = CreateWindowFilterCache("hann", dtype=np.float32)
                arr = cache.GetOrCreate((16, 16))
            finally:
                if original is None:
                    os.environ.pop("NORNIR_TEMP_DIR", None)
                else:
                    os.environ["NORNIR_TEMP_DIR"] = original

        self.assertEqual(arr.shape, (16, 16))
        self.assertEqual(arr.dtype, np.float32)
        self.assertFalse(arr.flags.writeable)


if __name__ == "__main__":
    unittest.main()
