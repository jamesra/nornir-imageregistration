"""WindowFilterCache must be safe to reach from worker threads (#104).

``__GetOrCreateCachedImage`` did a dict lookup, then a disk load, then a build-and-save, with no
lock.  Every thread that arrived before the first one finished missed the dict and built its own
copy: measured at **one build per worker thread** for a single shape -- 16 of 16 with 16 workers --
so the cache deduplicated nothing under concurrency.  Those threads also raced to ``np.save`` the
same path.

``assemble_tiles`` held an external ``_distance_cache_lock`` at one of its three call sites and not
at the other two, and the unguarded one is the path ``TilesToImageParallel`` takes for every tile:
it submits ``TransformTile`` with ``distanceImage=None`` to ``GetGlobalMultithreadingPool``, so the
in-``TransformTile`` lookup always misses the short-circuit and enters the cache concurrently.

The duplicated work is cheap (``CreateDistanceImage`` is 1.8 ms at 1024x1024) and bounded by the
worker count per distinct shape, so this is a correctness and ownership fix rather than a
throughput one.  The lock now lives in the class that holds the invariant.
"""

import os
import shutil
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.image_filter_cache import WindowFilterCache


class _CountingCreator:
    """Records every build, and stalls so concurrent callers pile up behind the first."""

    def __init__(self, barrier_parties=0, stall=0.02):
        self.shapes_built = []
        self._lock = threading.Lock()
        self._stall = stall
        self._entered = threading.Event()

    @property
    def build_count(self):
        with self._lock:
            return len(self.shapes_built)

    def __call__(self, shape, dtype):
        with self._lock:
            self.shapes_built.append(tuple(shape))
        self._entered.set()
        # A real build is not instantaneous; without a stall the race is easy to miss.
        threading.Event().wait(self._stall)
        return np.fromfunction(lambda y, x: (y + x).astype(dtype), tuple(shape), dtype=np.int32)


class CacheTestBase(unittest.TestCase):

    def setUp(self):
        self.caches = []

    def tearDown(self):
        for cache in self.caches:
            shutil.rmtree(cache.cache_dir, ignore_errors=True)

    def make_cache(self, creator, name=None):
        name = name or f"test_{self.id().rsplit('.', 1)[-1]}_{len(self.caches)}"
        cache = WindowFilterCache(name, creator)
        shutil.rmtree(cache.cache_dir, ignore_errors=True)
        os.makedirs(cache.cache_dir, exist_ok=True)
        self.caches.append(cache)
        return cache


class TestConcurrentMissesBuildOnce(CacheTestBase):
    """The property that was broken: the cache must actually deduplicate."""

    def test_one_build_regardless_of_thread_count(self):
        for workers in (2, 8, 16):
            with self.subTest(workers=workers):
                creator = _CountingCreator()
                cache = self.make_cache(creator)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(cache.GetOrCreate, (64, 64))
                               for _ in range(workers)]
                    results = [f.result() for f in futures]

                self.assertEqual(1, creator.build_count)
                self.assertEqual(workers, len(results))

    def test_every_caller_gets_the_same_object(self):
        creator = _CountingCreator()
        cache = self.make_cache(creator)
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = [f.result() for f in
                       [pool.submit(cache.GetOrCreate, (48, 48)) for _ in range(12)]]
        self.assertEqual(1, len({id(r) for r in results}))

    def test_only_one_writer_touches_the_cache_file(self):
        creator = _CountingCreator()
        cache = self.make_cache(creator)
        shape = (72, 72)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for future in [pool.submit(cache.GetOrCreate, shape) for _ in range(16)]:
                future.result()

        path = os.path.join(cache.cache_dir, f"{shape[0]}x{shape[1]}.npy")
        self.assertTrue(os.path.exists(path))
        reloaded = np.load(path)  # would raise if a concurrent write had torn the file
        self.assertEqual(shape, tuple(reloaded.shape))
        self.assertEqual(1, creator.build_count)


class TestTheImagesAreReadOnly(CacheTestBase):
    """The class documents cache images as read-only; there must be no window otherwise."""

    def test_a_freshly_built_image_is_read_only(self):
        cache = self.make_cache(_CountingCreator())
        self.assertFalse(cache.GetOrCreate((40, 40)).flags.writeable)

    def test_an_image_reloaded_from_disk_is_read_only(self):
        creator = _CountingCreator()
        first = self.make_cache(creator, name="readonly_shared")
        first.GetOrCreate((40, 40))
        # A second cache over the same directory takes the on-disk load path.
        second = WindowFilterCache("readonly_shared", creator)
        self.caches.append(second)
        self.assertFalse(second.GetOrCreate((40, 40)).flags.writeable)

    def test_no_caller_in_a_concurrent_burst_sees_a_writeable_array(self):
        cache = self.make_cache(_CountingCreator())
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = [f.result() for f in
                       [pool.submit(cache.GetOrCreate, (56, 56)) for _ in range(16)]]
        self.assertEqual([], [r for r in results if r.flags.writeable])


class TestDistinctShapesStillWork(CacheTestBase):
    """Serialising creation must not lose or confuse concurrent requests."""

    def test_each_shape_is_built_once_and_returned_correctly(self):
        creator = _CountingCreator(stall=0.005)
        cache = self.make_cache(creator)
        shapes = [(32 + 4 * i, 32 + 4 * i) for i in range(10)]
        requests = shapes * 3

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = [f.result() for f in
                       [pool.submit(cache.GetOrCreate, s) for s in requests]]

        self.assertEqual(len(requests), len(results))
        for shape, result in zip(requests, results):
            with self.subTest(shape=shape):
                self.assertEqual(shape, tuple(result.shape))
        self.assertEqual(len(shapes), creator.build_count)
        self.assertEqual(set(shapes), set(creator.shapes_built))


class TestTheResultsAreUnchanged(CacheTestBase):
    """Locking must not alter what the cache produces."""

    def test_the_cached_image_matches_a_direct_build(self):
        creator = _CountingCreator(stall=0)
        cache = self.make_cache(creator)
        dtype = nornir_imageregistration.default_depth_image_dtype()
        for shape in ((16, 16), (33, 17)):
            with self.subTest(shape=shape):
                cached = np.asarray(cache.GetOrCreate(shape))
                direct = np.asarray(creator(shape, dtype))
                self.assertTrue(np.array_equal(cached, direct))

    def test_a_second_call_returns_the_cached_object(self):
        creator = _CountingCreator(stall=0)
        cache = self.make_cache(creator)
        first = cache.GetOrCreate((24, 24))
        second = cache.GetOrCreate((24, 24))
        self.assertIs(first, second)
        self.assertEqual(1, creator.build_count)

    def test_keep_get_or_create_still_short_circuits(self):
        """A correctly shaped image is returned untouched, without consulting the cache."""
        creator = _CountingCreator(stall=0)
        cache = self.make_cache(creator)
        supplied = np.zeros((28, 28),
                            dtype=nornir_imageregistration.default_depth_image_dtype())
        returned = cache.KeepGetOrCreate(supplied, (28, 28))
        self.assertIs(supplied, returned)
        self.assertEqual(0, creator.build_count)

    def test_keep_get_or_create_rejects_a_bad_shape_argument(self):
        cache = self.make_cache(_CountingCreator(stall=0))
        with self.assertRaises(ValueError):
            cache.KeepGetOrCreate(None, (1, 2, 3))


class TestTheLockIsReentrant(CacheTestBase):
    """The creation function is caller supplied and nothing stops it consulting the cache."""

    def test_a_creator_that_reenters_does_not_deadlock(self):
        cache = None
        depth = []

        def reentrant(shape, dtype):
            depth.append(shape)
            if len(depth) == 1:
                cache.GetOrCreate((8, 8))
            return np.zeros(tuple(shape), dtype=dtype)

        cache = self.make_cache(reentrant)
        result = cache.GetOrCreate((16, 16))
        self.assertEqual((16, 16), tuple(result.shape))

    def test_the_lock_is_not_left_held(self):
        cache = self.make_cache(_CountingCreator(stall=0))
        cache.GetOrCreate((12, 12))
        self.assertTrue(cache._lock.acquire(blocking=False))
        cache._lock.release()


if __name__ == '__main__':
    unittest.main()
