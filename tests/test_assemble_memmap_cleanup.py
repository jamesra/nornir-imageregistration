"""Memmap assemble buffers must clean up without raising at shutdown.

__CreateOutputBufferForArea registers weakref.finalize(buffer, os.remove, path)
for its memmap backing file. weakref._exitfunc can run that finalizer before
NumPy releases the mapping, and on Windows os.remove then raises WinError 32
from inside interpreter shutdown, where no caller can handle it:

    PermissionError: [WinError 32] The process cannot access the file because
    it is being used by another process: '...\\Temp\\image_64x48_....npy'

The memmap path is currently dead code (_use_memmap() returns False), so these
tests force it on. They are the groundwork for re-enabling it; see issue #13.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration.assemble_tiles as assemble_tiles


class TestRemoveMemmapBackingFile(unittest.TestCase):

    def test_removes_an_unmapped_file(self) -> None:
        fd, path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        self.assertTrue(os.path.exists(path))

        assemble_tiles._remove_memmap_backing_file(path)

        self.assertFalse(os.path.exists(path), "Backing file should be deleted when unmapped")

    def test_missing_file_is_not_an_error(self) -> None:
        path = os.path.join(tempfile.gettempdir(), 'nornir-does-not-exist-12345.npy')
        self.assertFalse(os.path.exists(path))

        assemble_tiles._remove_memmap_backing_file(path)  # must not raise

    def test_open_mapping_does_not_raise(self) -> None:
        """The case that broke shutdown: mapping still held when cleanup runs."""
        path = os.path.join(tempfile.gettempdir(), 'nornir-memmap-cleanup-test.npy')
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        mapped = np.memmap(path, dtype=np.float32, mode='w+', shape=(16, 16))
        mapped.fill(0)
        try:
            # Plain os.remove is what the code used to register; on Windows this
            # is the call that raised out of weakref._exitfunc.
            assemble_tiles._remove_memmap_backing_file(path)
        finally:
            del mapped

    def test_permission_error_is_swallowed_on_every_platform(self) -> None:
        """Pin the behavior even where the OS would allow the unlink."""
        path = os.path.join(tempfile.gettempdir(), 'nornir-memmap-perm-test.npy')

        with mock.patch('os.remove', side_effect=PermissionError(32, 'in use')) as removed:
            assemble_tiles._remove_memmap_backing_file(path)

        removed.assert_called_once_with(path)


class TestCreateOutputBufferForAreaWithMemmap(unittest.TestCase):
    """End-to-end: forcing the memmap path must not leave a raising finalizer."""

    def _create_buffer_fn(self):
        return getattr(assemble_tiles, '__CreateOutputBufferForArea')

    def test_memmap_buffer_finalizer_tolerates_open_mapping(self) -> None:
        create = self._create_buffer_fn()

        with mock.patch.object(assemble_tiles, '_use_memmap', return_value=True):
            full_image, z_buffer = create(64, 48, np.float32)

            self.assertIsInstance(full_image, np.memmap)
            self.assertEqual(full_image.shape, (64, 48))
            self.assertEqual(full_image.dtype, np.float32)
            self.assertEqual(z_buffer.shape, (64, 48))

            path = full_image.filename
            self.assertIsNotNone(path)

            # Invoke cleanup while the mapping is still open -- exactly the
            # shutdown ordering that raised before.
            assemble_tiles._remove_memmap_backing_file(path)

            del full_image, z_buffer


class TestCreateOutputBufferMemmapErrorPath(unittest.TestCase):
    """#188: memmap open failure must not UnboundLocalError on the path name."""

    def test_path_build_failure_does_not_unbound_local(self) -> None:
        """If join fails inside the old try, LogErr referenced an unbound name."""
        create = getattr(assemble_tiles, '__CreateOutputBufferForArea')

        with mock.patch.object(assemble_tiles, '_use_memmap', return_value=True):
            with mock.patch('os.path.join', side_effect=OSError('simulated join failure')):
                with self.assertRaises(OSError) as raised:
                    create(8, 8, np.float32)

        self.assertIsInstance(raised.exception, OSError)
        self.assertNotIsInstance(raised.exception, UnboundLocalError)

    def test_memmap_failure_logs_bound_path(self) -> None:
        create = getattr(assemble_tiles, '__CreateOutputBufferForArea')

        with mock.patch.object(assemble_tiles, '_use_memmap', return_value=True):
            with mock.patch.object(
                    assemble_tiles.nornir_imageregistration, 'gettempdir',
                    return_value=tempfile.gettempdir()):
                with mock.patch.object(
                        assemble_tiles, 'GetProcessAndThreadUniqueString',
                        return_value='failpath'):
                    with mock.patch(
                            'numpy.memmap', side_effect=OSError('simulated memmap failure')):
                        with mock.patch.object(assemble_tiles.prettyoutput, 'LogErr') as log_err:
                            with self.assertRaises(OSError):
                                create(8, 8, np.float32)

        log_err.assert_called_once()
        self.assertIn('image_8x8_failpath.npy', log_err.call_args[0][0])


class TestExitSweeper(unittest.TestCase):
    """The sweeper retries deletions the per-buffer finalizers could not make."""

    def setUp(self) -> None:
        self._saved = set(assemble_tiles._memmap_temp_files)
        assemble_tiles._memmap_temp_files.clear()

    def tearDown(self) -> None:
        assemble_tiles._memmap_temp_files.clear()
        assemble_tiles._memmap_temp_files.update(self._saved)

    def test_sweeper_deletes_a_registered_file(self) -> None:
        fd, path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        assemble_tiles._register_memmap_temp_file(path)

        assemble_tiles._sweep_memmap_temp_files()

        self.assertFalse(os.path.exists(path))
        self.assertEqual(assemble_tiles._memmap_temp_files, set())

    def test_sweeper_deletes_a_file_that_was_mapped_when_first_attempted(self) -> None:
        """The Windows case: finalizer fails, sweeper succeeds after gc."""
        path = os.path.join(tempfile.gettempdir(), 'nornir-sweeper-mapped.npy')
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        assemble_tiles._register_memmap_temp_file(path)
        mapped = np.memmap(path, dtype=np.float32, mode='w+', shape=(8, 8))
        mapped.fill(0)

        # Mapping open: the finalizer's attempt cannot remove the file on
        # Windows, and must leave it registered for the sweeper.
        assemble_tiles._remove_memmap_backing_file(path)

        del mapped
        assemble_tiles._sweep_memmap_temp_files()

        self.assertFalse(
            os.path.exists(path),
            "Sweeper should delete the backing file once the mapping is released")

    def test_successful_removal_deregisters_the_path(self) -> None:
        fd, path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        assemble_tiles._register_memmap_temp_file(path)

        assemble_tiles._remove_memmap_backing_file(path)

        self.assertNotIn(path, assemble_tiles._memmap_temp_files)

    def test_sweeper_is_safe_when_nothing_is_registered(self) -> None:
        assemble_tiles._sweep_memmap_temp_files()  # must not raise

    def test_sweeper_tolerates_an_undeletable_file(self) -> None:
        path = os.path.join(tempfile.gettempdir(), 'nornir-sweeper-locked.npy')
        assemble_tiles._register_memmap_temp_file(path)

        with mock.patch('os.remove', side_effect=PermissionError(32, 'in use')):
            assemble_tiles._sweep_memmap_temp_files()  # must not raise

class TestRealInterpreterShutdown(unittest.TestCase):
    """End-to-end behavior at a real interpreter exit, not a simulation."""

    def _run_child(self, keep_alive: bool) -> tuple[subprocess.CompletedProcess, str]:
        marker = os.path.join(tempfile.gettempdir(), 'nornir-shutdown-probe.txt')
        if os.path.exists(marker):
            os.remove(marker)
        self.addCleanup(lambda: os.path.exists(marker) and os.remove(marker))

        release = '' if keep_alive else 'del full_image, z_buffer'

        script = textwrap.dedent(
            f"""
            import numpy as np
            import nornir_imageregistration.assemble_tiles as at

            at._use_memmap = lambda: True
            create = getattr(at, '__CreateOutputBufferForArea')
            full_image, z_buffer = create(64, 48, np.float32)

            with open({marker!r}, 'w') as handle:
                handle.write(full_image.filename or '')

            {release}
            """
        )

        completed = subprocess.run(
            [sys.executable, '-c', script], capture_output=True, text=True)

        self.assertTrue(os.path.exists(marker), "Child never created the buffer")
        with open(marker) as handle:
            backing_path = handle.read().strip()
        self.assertTrue(backing_path, "Child did not report a backing file path")
        return completed, backing_path

    def test_no_error_escapes_when_mapping_is_held_until_exit(self) -> None:
        """The regression this work fixes: shutdown must stay clean."""
        completed, backing_path = self._run_child(keep_alive=True)
        self.addCleanup(lambda: os.path.exists(backing_path) and os.remove(backing_path))

        self.assertEqual(
            completed.returncode, 0,
            f"Child exited {completed.returncode}\nstderr={completed.stderr}")
        self.assertNotIn('PermissionError', completed.stderr)
        self.assertNotIn('Exception ignored', completed.stderr)

    def test_file_is_deleted_when_the_buffer_is_released_before_exit(self) -> None:
        """The case the sweeper can actually win."""
        completed, backing_path = self._run_child(keep_alive=False)
        self.addCleanup(lambda: os.path.exists(backing_path) and os.remove(backing_path))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(
            os.path.exists(backing_path),
            f"Backing file {backing_path} should be gone once the buffer was released")

    @unittest.skipUnless(sys.platform == 'win32', 'POSIX unlinks mapped files fine')
    def test_known_limitation_mapping_held_at_exit_leaks_on_windows(self) -> None:
        """Documents the residual leak rather than pretending it is fixed.

        At atexit the holder's module globals are still alive, so the mapping is
        still open and gc cannot release it. Windows refuses to unlink a mapped
        file, so the backing file survives. Deleting it would require forcibly
        closing a mapping that live code may still hold, trading a leaked temp
        file for a possible access violation.

        If that tradeoff is ever taken, this test should start failing and be
        replaced by the stronger assertion.
        """
        completed, backing_path = self._run_child(keep_alive=True)
        self.addCleanup(lambda: os.path.exists(backing_path) and os.remove(backing_path))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(
            os.path.exists(backing_path),
            "Mapping held at exit is now deletable on Windows; revisit the "
            "sweeper's documented limitation")


if __name__ == "__main__":
    unittest.main()
