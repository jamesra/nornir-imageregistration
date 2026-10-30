"""ConvertImagesInDict bDeleteOriginal must wait delete tasks and log OSError (#208)."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import nornir_imageregistration.core._core as core


class _FakeTask:
    def __init__(self, name: str, *, error: BaseException | None = None) -> None:
        self.name = name
        self._error = error
        self.waited = False

    def wait(self) -> None:
        self.waited = True
        if self._error is not None:
            raise self._error


class _FakePool:
    def __init__(self) -> None:
        self.tasks: list[_FakeTask] = []
        self.shutdown_called = False
        self.wait_completion_called = False

    def add_task(self, name, func, *args, **kwargs):
        del func, args, kwargs
        if name.startswith('Delete '):
            path = name[len('Delete '):]
            err = OSError('locked') if path.endswith('bad.png') else None
            task = _FakeTask(name, error=err)
        else:
            task = _FakeTask(name)
        self.tasks.append(task)
        return task

    def wait_completion(self) -> None:
        self.wait_completion_called = True

    def shutdown(self) -> None:
        self.shutdown_called = True


class TestConvertImagesInDictDeleteWait(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.mkdtemp()
        self._src_ok = os.path.join(self._directory, 'ok.png')
        self._src_bad = os.path.join(self._directory, 'bad.png')
        self._dst_ok = os.path.join(self._directory, 'ok_out.png')
        self._dst_bad = os.path.join(self._directory, 'bad_out.png')
        for path in (self._src_ok, self._src_bad):
            with open(path, 'wb') as handle:
                handle.write(b'\x00')

    def test_delete_failures_are_waited_and_logged(self) -> None:
        pool = _FakePool()
        mapping = {
            self._src_ok: self._dst_ok,
            self._src_bad: self._dst_bad,
        }
        with mock.patch.object(core.nornir_pools, 'GetMultithreadingPool', return_value=pool), \
                mock.patch.object(core, '_ConvertSingleImageToFile'), \
                mock.patch.object(core.prettyoutput, 'LogErr') as log_err, \
                mock.patch.object(core.prettyoutput, 'CurseString'), \
                mock.patch.object(core, '_TaskProgressReporter') as reporter_cls, \
                mock.patch.object(core.nornir_shared.images, 'GetImageBpp', return_value=8):
            reporter = mock.MagicMock()
            reporter_cls.return_value = reporter
            core.ConvertImagesInDict(mapping, bDeleteOriginal=True)

        delete_tasks = [t for t in pool.tasks if t.name.startswith('Delete ')]
        self.assertEqual(2, len(delete_tasks))
        self.assertTrue(all(t.waited for t in delete_tasks))
        self.assertTrue(log_err.called)
        logged = ' '.join(str(call) for call in log_err.call_args_list)
        self.assertIn('bad.png', logged)
        self.assertTrue(pool.wait_completion_called)


if __name__ == '__main__':
    unittest.main()
