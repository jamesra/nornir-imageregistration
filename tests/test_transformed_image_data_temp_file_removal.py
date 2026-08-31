"""A temp file whose memmap is still open must be retried, not abandoned (#108).

``image`` and ``centerDistanceImage`` hand out ``np.load(..., mmap_mode='r')`` arrays.  While a
caller holds one -- or a view of one, which keeps it alive through ``.base`` -- the file cannot be
deleted on Windows.  ``Clear`` dropping the instance's own reference does not help in that case.

``_RemoveTempFiles`` logged the failure and dropped it, so the file stayed for the lifetime of the
process even though it becomes deletable the moment the caller lets go.  Failures are now
re-queued and retried on later calls, with a bounded slice per call and a final attempt at exit.

The live assemble path drops its arrays before calling ``Clear`` and was already deleting both
files successfully, so ``TestTheLivePatternStillDeletesImmediately`` guards that no-regression
case explicitly.
"""

import gc
import os
import shutil
import threading
import unittest
from unittest import mock

import numpy as np

import nornir_pools
from nornir_imageregistration.transformed_image_data import TransformedImageDataState
from nornir_imageregistration.transformed_image_data_temp_files import (
    TransformedImageDataViaTempFile as TempFileData)


class _Isolated:
    """Give a test its own temp root and pending queue, and clean up afterwards."""

    def __enter__(self):
        self._saved = (TempFileData._temp_folder_created,
                       TempFileData._sharedTempRoot,
                       TempFileData._pending_deletions)
        TempFileData._temp_folder_created = False
        TempFileData._sharedTempRoot = None
        TempFileData._pending_deletions = type(TempFileData._pending_deletions)()
        self.root = TempFileData._EnsureSharedTempFolder()
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)
        (TempFileData._temp_folder_created,
         TempFileData._sharedTempRoot,
         TempFileData._pending_deletions) = self._saved


def _populated(shape=(64, 64)):
    """An instance whose arrays live in temp files, as the worker path produces."""
    instance = TempFileData()
    instance._image = np.random.default_rng(1).random(shape).astype(np.float32)
    instance._centerDistanceImage = np.random.default_rng(2).random(shape).astype(np.float32)
    instance._image_state = TransformedImageDataState.IN_MEMORY
    instance._center_distance_image_state = TransformedImageDataState.IN_MEMORY
    # Force the temp-file path regardless of the size threshold.
    with mock.patch.object(TempFileData, 'tempfile_threshold', 0):
        instance.ConvertToTempFileIfLarge()
    return instance


def _touch(root, name):
    path = os.path.join(root, name)
    with open(path, 'wb') as handle:
        handle.write(b'x')
    return path


class TestRemovingASingleFile(unittest.TestCase):
    """_TryRemoveTempFile reports whether the path is gone."""

    def test_none_is_reported_gone(self):
        self.assertTrue(TempFileData._TryRemoveTempFile(None))

    def test_an_existing_file_is_removed(self):
        with _Isolated() as env:
            path = _touch(env.root, "present.npy")
            self.assertTrue(TempFileData._TryRemoveTempFile(path))
            self.assertFalse(os.path.exists(path))

    def test_an_absent_file_is_reported_gone(self):
        with _Isolated() as env:
            path = os.path.join(env.root, "absent.npy")
            self.assertTrue(TempFileData._TryRemoveTempFile(path))

    def test_a_failure_is_reported_not_swallowed(self):
        with _Isolated() as env:
            path = _touch(env.root, "locked.npy")
            with mock.patch.object(os, 'remove', side_effect=PermissionError(13, "in use")):
                self.assertFalse(TempFileData._TryRemoveTempFile(path))

    def test_permission_error_is_an_oserror(self):
        """The original 'except IOError' did catch this; it just discarded the outcome."""
        self.assertTrue(issubclass(PermissionError, OSError))
        self.assertIs(IOError, OSError)


class TestTheLivePatternStillDeletesImmediately(unittest.TestCase):
    """No regression: the assemble path drops its arrays before Clear."""

    def test_both_files_go_and_nothing_is_pending(self):
        with _Isolated():
            instance = _populated()
            paths = [instance._image_path, instance._centerDistanceImage_path]
            self.assertTrue(all(os.path.exists(p) for p in paths))

            # Stands in for _composite_transformed_tile_onto_canvas: read, then let go.
            def consume(tid):
                return float(np.sum(tid.image[:4, :4]) + np.sum(tid.centerDistanceImage[:4, :4]))

            self.assertIsInstance(consume(instance), float)
            instance.Clear()
            nornir_pools.WaitOnAllPools()
            self.assertEqual([], [p for p in paths if os.path.exists(p)])
            self.assertEqual(0, len(TempFileData._pending_deletions))


class TestAHeldMemmapIsDeferredThenRetried(unittest.TestCase):

    def test_a_held_array_defers_deletion(self):
        with _Isolated():
            instance = _populated()
            paths = [instance._image_path, instance._centerDistanceImage_path]
            held = instance.image
            self.assertIsInstance(held, np.memmap)
            instance._image = None
            instance._centerDistanceImage = None
            TempFileData._RemoveTempFiles(None, paths[0])
            self.assertTrue(os.path.exists(paths[0]))
            self.assertIn(paths[0], TempFileData._pending_deletions)
            del held

    def test_a_later_call_retries_and_succeeds(self):
        with _Isolated():
            instance = _populated()
            image_path = instance._image_path
            held = instance.image
            instance._image = None
            instance._centerDistanceImage = None
            TempFileData._RemoveTempFiles(None, image_path)
            self.assertEqual(1, len(TempFileData._pending_deletions))

            del held
            gc.collect()
            TempFileData._RemoveTempFiles(None, None)
            self.assertFalse(os.path.exists(image_path))
            self.assertEqual(0, len(TempFileData._pending_deletions))

    def test_a_view_also_defers_and_is_retried(self):
        """A slice keeps the memmap alive through .base."""
        with _Isolated():
            instance = _populated()
            image_path = instance._image_path
            view = instance.image[4:8, 4:8]
            self.assertIsInstance(view.base, np.memmap)
            instance._image = None
            instance._centerDistanceImage = None
            TempFileData._RemoveTempFiles(None, image_path)
            self.assertTrue(os.path.exists(image_path))

            del view
            gc.collect()
            TempFileData._RemoveTempFiles(None, None)
            self.assertFalse(os.path.exists(image_path))

    def test_the_instance_forgets_the_paths_either_way(self):
        """Clear must not keep handing the same paths back on a later call."""
        with _Isolated():
            instance = _populated()
            held = instance.image
            instance.Clear()
            self.assertIsNone(instance._image_path)
            self.assertIsNone(instance._centerDistanceImage_path)
            del held


class TestTheRetrySliceIsBounded(unittest.TestCase):
    """A run that accumulates undeletable files must not sweep them all on every Clear."""

    def test_at_most_the_cap_is_retried_per_call(self):
        with _Isolated() as env:
            cap = TempFileData._max_deletion_retries_per_call
            paths = [_touch(env.root, f"pending_{i}.npy") for i in range(cap * 3)]
            TempFileData._pending_deletions.extend(paths)

            TempFileData._RemoveTempFiles(None, None)

            self.assertEqual(len(paths) - cap, len(TempFileData._pending_deletions))
            self.assertEqual(cap, sum(0 if os.path.exists(p) else 1 for p in paths))

    def test_repeated_calls_drain_the_queue(self):
        with _Isolated() as env:
            cap = TempFileData._max_deletion_retries_per_call
            paths = [_touch(env.root, f"drain_{i}.npy") for i in range(cap * 3)]
            TempFileData._pending_deletions.extend(paths)

            calls = 0
            while len(TempFileData._pending_deletions) > 0 and calls < 50:
                TempFileData._RemoveTempFiles(None, None)
                calls += 1

            self.assertEqual(0, len(TempFileData._pending_deletions))
            self.assertEqual([], [p for p in paths if os.path.exists(p)])

    def test_the_queue_is_fair(self):
        """Rotating rather than re-scanning the front, so nothing is starved."""
        with _Isolated() as env:
            cap = TempFileData._max_deletion_retries_per_call
            paths = [os.path.join(env.root, f"fair_{i}.npy") for i in range(cap * 2)]
            TempFileData._pending_deletions.extend(paths)
            # None must keep reporting gone, or the absent current pair joins the queue.
            with mock.patch.object(TempFileData, '_TryRemoveTempFile',
                                   side_effect=lambda path: path is None):
                TempFileData._RemoveTempFiles(None, None)
            # The first cap entries were tried and re-queued at the back.
            self.assertEqual(paths[cap:] + paths[:cap],
                             list(TempFileData._pending_deletions))

    def test_an_absent_pending_path_is_dropped(self):
        with _Isolated() as env:
            TempFileData._pending_deletions.append(os.path.join(env.root, "gone.npy"))
            TempFileData._RemoveTempFiles(None, None)
            self.assertEqual(0, len(TempFileData._pending_deletions))


class TestTheExitFlush(unittest.TestCase):

    def test_it_deletes_what_it_can(self):
        with _Isolated() as env:
            paths = [_touch(env.root, f"flush_{i}.npy") for i in range(3)]
            TempFileData._pending_deletions.extend(paths)
            TempFileData._FlushPendingDeletions()
            self.assertEqual([], [p for p in paths if os.path.exists(p)])
            self.assertEqual(0, len(TempFileData._pending_deletions))

    def test_it_warns_about_genuine_residue(self):
        with _Isolated() as env:
            TempFileData._pending_deletions.append(_touch(env.root, "stuck.npy"))
            with mock.patch.object(TempFileData, '_TryRemoveTempFile', return_value=False):
                with self.assertLogs(
                        'nornir_imageregistration.transformed_image_data_temp_files',
                        level='WARNING') as captured:
                    TempFileData._FlushPendingDeletions()
            self.assertIn('could not be deleted', captured.output[0])

    def test_it_is_silent_when_there_is_nothing_stuck(self):
        with _Isolated() as env:
            TempFileData._pending_deletions.append(_touch(env.root, "fine.npy"))
            logger = 'nornir_imageregistration.transformed_image_data_temp_files'
            with mock.patch.object(TempFileData, '_TryRemoveTempFile', return_value=True):
                with mock.patch(f'{logger}.logging.getLogger') as get_logger:
                    TempFileData._FlushPendingDeletions()
                    get_logger.return_value.warning.assert_not_called()

    def test_it_clears_the_queue_either_way(self):
        with _Isolated() as env:
            TempFileData._pending_deletions.append(_touch(env.root, "either.npy"))
            with mock.patch.object(TempFileData, '_TryRemoveTempFile', return_value=False):
                TempFileData._FlushPendingDeletions()
            self.assertEqual(0, len(TempFileData._pending_deletions))

    def test_it_is_registered_when_the_folder_is_created(self):
        saved = (TempFileData._temp_folder_created, TempFileData._sharedTempRoot)
        TempFileData._temp_folder_created = False
        TempFileData._sharedTempRoot = None
        try:
            with mock.patch('atexit.register') as registered:
                TempFileData._EnsureSharedTempFolder()
            handlers = [call[0][0] for call in registered.call_args_list]
            self.assertIn(TempFileData._FlushPendingDeletions, handlers)
            self.assertIn(shutil.rmtree, handlers)
            # LIFO: the flush must run before the directory is removed, so it is registered last.
            self.assertGreater(handlers.index(TempFileData._FlushPendingDeletions),
                               handlers.index(shutil.rmtree))
        finally:
            shutil.rmtree(TempFileData._sharedTempRoot or '', ignore_errors=True)
            TempFileData._temp_folder_created, TempFileData._sharedTempRoot = saved


class TestConcurrentRemoval(unittest.TestCase):
    """Clear runs the removal on a thread pool, so the queue is touched concurrently."""

    def test_the_queue_survives_concurrent_calls(self):
        with _Isolated() as env:
            paths = [_touch(env.root, f"conc_{i}.npy") for i in range(40)]
            TempFileData._pending_deletions.extend(paths)

            errors = []

            def worker():
                try:
                    for _ in range(10):
                        TempFileData._RemoveTempFiles(None, None)
                except Exception as exc:  # noqa: BLE001 - recorded and asserted
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
                self.assertFalse(thread.is_alive())

            self.assertEqual([], errors)
            self.assertEqual(0, len(TempFileData._pending_deletions))
            self.assertEqual([], [p for p in paths if os.path.exists(p)])

    def test_the_lock_is_not_left_held(self):
        with _Isolated() as env:
            TempFileData._pending_deletions.append(_touch(env.root, "lock.npy"))
            TempFileData._RemoveTempFiles(None, None)
        self.assertTrue(TempFileData._pending_deletions_lock.acquire(blocking=False))
        TempFileData._pending_deletions_lock.release()


if __name__ == '__main__':
    unittest.main()
