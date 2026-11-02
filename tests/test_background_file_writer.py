"""Background writes of STOS previews: same bytes, complete files only, bounded queue.

``AssembleStosOverlays`` hands each pair's three PNGs to a ``BackgroundFileWriter``
so the NAS writes overlap the next pair's warp (srv, 2026-11-02: 11-12 s of a
13.6 s pair was writing previews to /storage2).
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from nornir_imageregistration import assemble
from nornir_imageregistration.background_file_writer import (
    BackgroundFileWriter,
    write_via_local_temp,
)

_FIXTURE = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'idoc_690_691', 'StosBrute16',
    '690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos')


def _read(path: str) -> bytes:
    with open(path, 'rb') as handle:
        return handle.read()


class TestWriteViaLocalTemp(unittest.TestCase):

    def test_writes_the_saved_bytes_and_leaves_no_partial(self):
        with tempfile.TemporaryDirectory() as out_dir:
            dest = os.path.join(out_dir, 'sub', 'a.png')

            def save(path: str) -> None:
                with open(path, 'wb') as handle:
                    handle.write(b'x' * 100_000)

            write_via_local_temp(dest, save)

            self.assertEqual(_read(dest), b'x' * 100_000)
            self.assertEqual(os.listdir(os.path.dirname(dest)), ['a.png'])

    def test_a_failed_save_leaves_nothing_under_the_final_name(self):
        with tempfile.TemporaryDirectory() as out_dir:
            dest = os.path.join(out_dir, 'a.png')

            def save(path: str) -> None:
                with open(path, 'wb') as handle:
                    handle.write(b'half')
                raise OSError('disk went away')

            with self.assertRaises(OSError):
                write_via_local_temp(dest, save)

            self.assertEqual(os.listdir(out_dir), [])


class TestBackgroundFileWriter(unittest.TestCase):

    def test_close_waits_for_every_write(self):
        with tempfile.TemporaryDirectory() as out_dir:
            paths = [os.path.join(out_dir, f'{i}.bin') for i in range(8)]

            def slow_save(path: str) -> None:
                time.sleep(0.05)
                with open(path, 'wb') as handle:
                    handle.write(b'done')

            with BackgroundFileWriter(max_workers=2, max_pending=3) as writer:
                for path in paths:
                    writer.submit(path, slow_save)

            for path in paths:
                self.assertEqual(_read(path), b'done')

    def test_submit_blocks_while_the_queue_is_full(self):
        """Queued images are held in memory, so the queue must not grow past max_pending."""
        release = threading.Event()

        def blocked_save(path: str) -> None:
            release.wait(10)
            with open(path, 'wb') as handle:
                handle.write(b'.')

        with tempfile.TemporaryDirectory() as out_dir:
            writer = BackgroundFileWriter(max_workers=1, max_pending=2)
            writer.submit(os.path.join(out_dir, '0.bin'), blocked_save)
            writer.submit(os.path.join(out_dir, '1.bin'), blocked_save)
            third = threading.Thread(
                target=writer.submit, args=(os.path.join(out_dir, '2.bin'), blocked_save))
            third.start()
            third.join(0.3)
            self.assertTrue(third.is_alive(), 'third submit should wait for a free slot')

            release.set()
            third.join(10)
            self.assertFalse(third.is_alive())
            writer.close()
            self.assertEqual(sorted(os.listdir(out_dir)), ['0.bin', '1.bin', '2.bin'])

    def test_close_raises_the_first_write_error(self):
        def failing_save(_path: str) -> None:
            raise ValueError('cannot encode')

        with tempfile.TemporaryDirectory() as out_dir:
            writer = BackgroundFileWriter()
            writer.submit(os.path.join(out_dir, 'a.png'), failing_save)
            with self.assertRaises(ValueError):
                writer.close()

    def test_caller_exception_wins_over_write_error(self):
        def failing_save(_path: str) -> None:
            raise ValueError('cannot encode')

        with tempfile.TemporaryDirectory() as out_dir, self.assertRaises(KeyError),                 BackgroundFileWriter() as writer:
            writer.submit(os.path.join(out_dir, 'a.png'), failing_save)
            raise KeyError('caller failed first')


class TestStosPreviewsUnchanged(unittest.TestCase):

    def test_background_previews_match_direct_writes_byte_for_byte(self):
        names = ('overlay.png', 'diff.png', 'warped.png')
        with tempfile.TemporaryDirectory() as out_dir:
            direct = os.path.join(out_dir, 'direct')
            queued = os.path.join(out_dir, 'queued')

            def paths(root: str) -> dict[str, str]:
                return {'overlay_path': os.path.join(root, names[0]),
                        'diff_path': os.path.join(root, names[1]),
                        'warped_path': os.path.join(root, names[2])}

            assemble.WriteStosPreviewImages(_FIXTURE, **paths(direct))
            with BackgroundFileWriter() as writer:
                assemble.WriteStosPreviewImages(_FIXTURE, writer=writer, **paths(queued))

            for name in names:
                with self.subTest(name=name):
                    self.assertEqual(_read(os.path.join(direct, name)), _read(os.path.join(queued, name)))
            self.assertEqual(sorted(os.listdir(queued)), sorted(names))


if __name__ == '__main__':
    unittest.main()
