"""Concurrent first calls must create exactly one shared temp folder (#107).

``_EnsureSharedTempFolder`` tested ``_temp_folder_created`` and set it as separate steps, so
every thread that reached the check before any of them set the flag created its own root.  With
8 threads that was 8 roots, and 7 of 8 saved files landed in a root other than the one finally
published in ``_sharedTempRoot``.

Each root *was* registered for cleanup, so a clean exit still removed them all.  What the race
cost was a directory per caller for the whole process lifetime, files scattered across roots
instead of in the shared one, and N roots rather than 1 left behind when a process dies without
running its atexit handlers.

The tests below drive the race deterministically with a barrier inside ``mkdtemp``.  The barrier
uses a timeout so it cannot hang once the fix is in place -- after the fix only one thread ever
reaches ``mkdtemp``, so a barrier sized for N threads would never fill.
"""

import atexit
import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from nornir_imageregistration.transformed_image_data_temp_files import (
    TransformedImageDataViaTempFile as TempFileData)

NUM_THREADS = 8
# Only needs to outlast the spread in thread startup, which is milliseconds. Kept small because
# every test in this file waits it out: once creation is serialized the barrier never fills.
BARRIER_TIMEOUT = 0.25


class _RaceHarness:
    """Force concurrent entry into the folder-creation path and record what happened."""

    def __init__(self, num_threads=NUM_THREADS):
        self.num_threads = num_threads
        self.roots_created = []
        self.registrations = []
        self.files_written = []
        self.reached_mkdtemp = 0
        self.errors = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(num_threads)

    def __enter__(self):
        self._saved_created = TempFileData._temp_folder_created
        self._saved_root = TempFileData._sharedTempRoot
        TempFileData._temp_folder_created = False
        TempFileData._sharedTempRoot = None

        real_mkdtemp = tempfile.mkdtemp

        def barriered_mkdtemp(*args, **kwargs):
            with self._lock:
                self.reached_mkdtemp += 1
            try:
                self._barrier.wait(timeout=BARRIER_TIMEOUT)
            except threading.BrokenBarrierError:
                pass  # Expected once creation is serialized: only one thread arrives.
            path = real_mkdtemp(*args, **kwargs)
            with self._lock:
                self.roots_created.append(path)
            return path

        def recording_register(fn, *args, **kwargs):
            with self._lock:
                self.registrations.append((fn, args[0] if args else None))

        self._patches = [mock.patch.object(tempfile, 'mkdtemp', barriered_mkdtemp),
                         mock.patch.object(atexit, 'register', recording_register)]
        for patch in self._patches:
            patch.start()
        return self

    def run(self, target):
        threads = [threading.Thread(target=self._guard(target)) for _ in range(self.num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive(), "worker thread did not finish"

    def _guard(self, target):
        def wrapped():
            try:
                target(self)
            except Exception as exc:  # noqa: BLE001 - recorded and asserted on
                with self._lock:
                    self.errors.append(exc)

        return wrapped

    @property
    def cleanup_registrations(self):
        """Only the rmtree registrations.

        Folder creation also registers the deferred-deletion flush (#108), which is not a
        per-root cleanup and would otherwise be counted as one.
        """
        return [(fn, path) for fn, path in self.registrations if fn is shutil.rmtree]

    @property
    def registered_paths(self):
        return [path for _, path in self.cleanup_registrations]

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        for path in set(self.roots_created):
            shutil.rmtree(path, ignore_errors=True)
        TempFileData._temp_folder_created = self._saved_created
        TempFileData._sharedTempRoot = self._saved_root


def _ensure(harness):
    TempFileData._EnsureSharedTempFolder()


def _save(harness):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = TempFileData.SaveArrayToTemporaryFile("Image", image)
    with harness._lock:
        harness.files_written.append(path)


class TestOnlyOneFolderIsCreated(unittest.TestCase):

    def test_concurrent_ensure_creates_one_root(self):
        with _RaceHarness() as harness:
            harness.run(_ensure)
            self.assertEqual([], harness.errors)
            self.assertEqual(1, len(harness.roots_created))

    def test_only_one_thread_reaches_mkdtemp(self):
        with _RaceHarness() as harness:
            harness.run(_ensure)
            self.assertEqual(1, harness.reached_mkdtemp)

    def test_every_caller_gets_the_same_path(self):
        with _RaceHarness() as harness:
            returned = []

            def collect(h):
                path = TempFileData._EnsureSharedTempFolder()
                with h._lock:
                    returned.append(path)

            harness.run(collect)
            self.assertEqual([], harness.errors)
            self.assertEqual(NUM_THREADS, len(returned))
            self.assertEqual(1, len(set(returned)))
            self.assertEqual(harness.roots_created[0], returned[0])

    def test_the_published_root_is_the_created_one(self):
        with _RaceHarness() as harness:
            harness.run(_ensure)
            self.assertEqual(harness.roots_created[0], TempFileData._sharedTempRoot)
            self.assertTrue(TempFileData._temp_folder_created)


class TestCleanupCoversWhatWasCreated(unittest.TestCase):

    def test_one_registration(self):
        with _RaceHarness() as harness:
            harness.run(_ensure)
            self.assertEqual(1, len(harness.cleanup_registrations))

    def test_the_registration_matches_the_created_root(self):
        """The original built this argument by re-reading the class attribute."""
        with _RaceHarness() as harness:
            harness.run(_ensure)
            self.assertEqual(set(harness.roots_created), set(harness.registered_paths))

    def test_it_registers_rmtree(self):
        with _RaceHarness() as harness:
            harness.run(_ensure)
            handler, path = harness.cleanup_registrations[0]
            self.assertIs(shutil.rmtree, handler)
            self.assertEqual(TempFileData._sharedTempRoot, path)

    def test_no_created_root_is_left_unregistered(self):
        with _RaceHarness() as harness:
            harness.run(_ensure)
            unregistered = set(harness.roots_created) - set(harness.registered_paths)
            self.assertEqual(set(), unregistered)


class TestConcurrentSavesShareTheFolder(unittest.TestCase):

    def test_all_files_land_in_the_shared_root(self):
        with _RaceHarness() as harness:
            harness.run(_save)
            self.assertEqual([], harness.errors)
            self.assertEqual(NUM_THREADS, len(harness.files_written))
            shared = os.path.normcase(os.path.normpath(str(TempFileData._sharedTempRoot)))
            for path in harness.files_written:
                with self.subTest(path=path):
                    self.assertEqual(
                        shared, os.path.normcase(os.path.normpath(os.path.dirname(path))))

    def test_one_root_for_many_saves(self):
        with _RaceHarness() as harness:
            harness.run(_save)
            self.assertEqual(1, len(harness.roots_created))

    def test_the_files_do_not_collide(self):
        with _RaceHarness() as harness:
            harness.run(_save)
            self.assertEqual(NUM_THREADS, len(set(harness.files_written)))

    def test_every_file_is_readable(self):
        with _RaceHarness() as harness:
            harness.run(_save)
            expected = np.arange(64, dtype=np.float32).reshape(8, 8)
            for path in harness.files_written:
                with self.subTest(path=path):
                    np.testing.assert_array_equal(expected, np.load(path))


class TestTheLockExists(unittest.TestCase):

    def test_the_class_holds_a_lock(self):
        self.assertIsInstance(TempFileData._temp_folder_lock,
                              type(threading.Lock()))

    def test_the_lock_is_shared_not_per_instance(self):
        self.assertIn('_temp_folder_lock', vars(TempFileData))

    def test_the_lock_is_not_left_held(self):
        with _RaceHarness() as harness:
            harness.run(_ensure)
        self.assertTrue(TempFileData._temp_folder_lock.acquire(blocking=False))
        TempFileData._temp_folder_lock.release()


class TestSequentialBehaviourIsUnchanged(unittest.TestCase):

    def setUp(self):
        self._created = TempFileData._temp_folder_created
        self._root = TempFileData._sharedTempRoot
        TempFileData._temp_folder_created = False
        TempFileData._sharedTempRoot = None
        self._made = None

    def tearDown(self):
        if self._made is not None and self._made != self._root:
            shutil.rmtree(self._made, ignore_errors=True)
        TempFileData._temp_folder_created = self._created
        TempFileData._sharedTempRoot = self._root

    def test_it_is_idempotent(self):
        first = TempFileData._EnsureSharedTempFolder()
        self._made = first
        for _ in range(3):
            self.assertEqual(first, TempFileData._EnsureSharedTempFolder())

    def test_it_creates_a_real_directory(self):
        self._made = TempFileData._EnsureSharedTempFolder()
        self.assertTrue(os.path.isdir(self._made))

    def test_saving_still_works(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = TempFileData.SaveArrayToTemporaryFile("Image", image)
        self._made = os.path.dirname(path)
        np.testing.assert_array_equal(image, np.load(path))

    def test_a_none_image_still_raises_before_creating_anything(self):
        with self.assertRaises(ValueError):
            TempFileData.SaveArrayToTemporaryFile("Image", None)
        self.assertIsNone(TempFileData._sharedTempRoot)


if __name__ == '__main__':
    unittest.main()
