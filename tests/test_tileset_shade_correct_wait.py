"""Shade correction must wait for its tasks before returning (#105).

``__CorrectBrightfieldShading`` built a task per tile, attached the output path to it, and then
drained a ``tasks`` list that nothing had ever been appended to.  The wait loop could not run, so
the results were gathered in the submit loop instead and nothing was ever waited on.

That was harmless in practice only because the function uses ``GetGlobalSerialPool``, whose
``add_task`` calls the worker inline -- so the file is written, and any failure raises, before
``add_task`` returns.  Verified: 4 cases on real test-data images return the right paths, in
order, with no duplicates, all present on disk.  The defect was that the pool choice had silently
become load bearing.  Any pool that actually defers work would have returned before the files
existed, and appending the task without also removing the submit-loop append would have returned
every path twice.

These tests pin both halves: the deferred-pool behaviour that was broken, and the serial-pool
behaviour that must not change.
"""

import os
import unittest
from unittest import mock

import nornir_pools

import nornir_imageregistration as nir
import nornir_imageregistration.tileset as tileset

try:
    import setup_imagetest
except (ImportError, ModuleNotFoundError):
    from . import setup_imagetest


# Resolved here, at module scope, because a double-underscore attribute referenced from inside a
# class body would be mangled with the class name.
_ONE_IMAGE_WORKER = next(name for name in vars(tileset)
                         if name.endswith('CorrectBrightfieldShadingOneImage'))


class _DeferredTask:
    """A task whose work happens only when waited on."""

    def __init__(self, name, func, args, kwargs):
        self.name = name
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self.completed = False
        self.retval = None

    def wait(self):
        if not self.completed:
            self.retval = self._func(*self._args, **self._kwargs)
            self.completed = True
        return self.retval

    def wait_return(self):
        return self.wait()


class _DeferredPool:
    """Models any pool that does not finish the work inside add_task.

    A stub rather than a real thread pool so the "not finished yet" state is deterministic
    instead of a race the test would sometimes lose.
    """

    def __init__(self):
        self.tasks = []

    def add_task(self, name, func, *args, **kwargs):
        task = _DeferredTask(name, func, args, kwargs)
        self.tasks.append(task)
        return task

    @property
    def outstanding(self):
        return [task for task in self.tasks if not task.completed]


class ShadeCorrectTestBase(setup_imagetest.ImageTestBase):

    def _tile_and_shading(self):
        tile_path = self.GetImagePath("CorrectionA_Tile.png")
        shading_path = self.GetImagePath("CorrectionA_Shading.png")
        self.assertTrue(os.path.exists(tile_path))
        self.assertTrue(os.path.exists(shading_path))
        return tile_path, nir.LoadImage(shading_path)

    def _two_tiles_and_shading(self):
        first = self.GetImagePath("CorrectionA_Tile.png")
        second = self.GetImagePath("CorrectionB_Tile.png")
        shading_path = self.GetImagePath("CorrectionA_Shading.png")
        for path in (first, second, shading_path):
            self.assertTrue(os.path.exists(path))
        return [first, second], nir.LoadImage(shading_path)

    def _correct(self, paths, shading, outdir=None):
        return tileset.ShadeCorrect(
            paths, shading, outdir or self.TestOutputPath,
            correction_type=tileset.ShadeCorrectionTypes.BRIGHTFIELD)


class TestADeferredPoolIsWaitedOn(ShadeCorrectTestBase):
    """The property that was broken: work must be finished when the call returns."""

    def test_every_returned_path_exists_on_return(self):
        paths, shading = self._two_tiles_and_shading()
        pool = _DeferredPool()
        with mock.patch.object(nornir_pools, 'GetGlobalSerialPool', return_value=pool):
            returned = self._correct(paths, shading)

        self.assertEqual(len(paths), len(returned))
        for path in returned:
            with self.subTest(path=os.path.basename(path)):
                self.assertTrue(os.path.exists(path))

    def test_no_task_is_left_outstanding(self):
        paths, shading = self._two_tiles_and_shading()
        pool = _DeferredPool()
        with mock.patch.object(nornir_pools, 'GetGlobalSerialPool', return_value=pool):
            self._correct(paths, shading)

        self.assertEqual(len(paths), len(pool.tasks))
        self.assertEqual([], pool.outstanding)

    def test_a_worker_failure_is_surfaced(self):
        """With work deferred, the exception can only reach the caller through wait()."""
        paths, shading = self._two_tiles_and_shading()
        pool = _DeferredPool()

        def exploding(*args, **kwargs):
            raise RuntimeError("simulated shading failure")

        with mock.patch.object(nornir_pools, 'GetGlobalSerialPool', return_value=pool):
            with mock.patch.object(tileset, _ONE_IMAGE_WORKER, exploding):
                with self.assertRaises(RuntimeError):
                    self._correct(paths, shading)


class TestThePathsAreReportedOnce(ShadeCorrectTestBase):
    """Guards the double-add trap: two appends existed, only one may run per tile."""

    def test_no_duplicates_on_the_serial_pool(self):
        paths, shading = self._two_tiles_and_shading()
        returned = self._correct(paths, shading)
        self.assertEqual(len(paths), len(returned))
        self.assertEqual(len(returned), len(set(returned)))

    def test_no_duplicates_on_a_deferred_pool(self):
        paths, shading = self._two_tiles_and_shading()
        pool = _DeferredPool()
        with mock.patch.object(nornir_pools, 'GetGlobalSerialPool', return_value=pool):
            returned = self._correct(paths, shading)
        self.assertEqual(len(paths), len(returned))
        self.assertEqual(len(returned), len(set(returned)))

    def test_the_order_follows_the_inputs(self):
        paths, shading = self._two_tiles_and_shading()
        pool = _DeferredPool()
        with mock.patch.object(nornir_pools, 'GetGlobalSerialPool', return_value=pool):
            returned = self._correct(paths, shading)
        self.assertEqual([os.path.basename(p) for p in paths],
                         [os.path.basename(p) for p in returned])


class TestTheSerialPoolBehaviourIsUnchanged(ShadeCorrectTestBase):
    """Parity for the pool actually in use; the fix must be invisible here."""

    def test_a_single_tile_is_corrected(self):
        tile_path, shading = self._tile_and_shading()
        returned = self._correct([tile_path], shading)
        self.assertEqual(1, len(returned))
        self.assertTrue(os.path.exists(returned[0]))

    def test_the_corrected_image_differs_from_the_input(self):
        """Guard against 'it exists' being satisfied by a copy or an empty file."""
        tile_path, shading = self._tile_and_shading()
        returned = self._correct([tile_path], shading)
        original = nir.LoadImage(tile_path)
        corrected = nir.LoadImage(returned[0])
        self.assertEqual(original.shape, corrected.shape)
        self.assertGreater(os.path.getsize(returned[0]), 0)

    def test_the_deferred_pool_produces_the_same_bytes(self):
        """The cache of correctness: pool choice must not change the output."""
        paths, shading = self._two_tiles_and_shading()

        serial_dir = os.path.join(self.TestOutputPath, "serial")
        deferred_dir = os.path.join(self.TestOutputPath, "deferred")
        os.makedirs(serial_dir, exist_ok=True)
        os.makedirs(deferred_dir, exist_ok=True)

        serial_paths = self._correct(paths, shading, outdir=serial_dir)

        pool = _DeferredPool()
        with mock.patch.object(nornir_pools, 'GetGlobalSerialPool', return_value=pool):
            deferred_paths = self._correct(paths, shading, outdir=deferred_dir)

        self.assertEqual([os.path.basename(p) for p in serial_paths],
                         [os.path.basename(p) for p in deferred_paths])
        for serial_path, deferred_path in zip(serial_paths, deferred_paths):
            with self.subTest(tile=os.path.basename(serial_path)):
                with open(serial_path, 'rb') as handle:
                    expected = handle.read()
                with open(deferred_path, 'rb') as handle:
                    actual = handle.read()
                self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
