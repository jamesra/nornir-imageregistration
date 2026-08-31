"""create_shared_memory_array must accept any shape spelling (#102).

The helper read ``shape.prod()`` and handed *shape* straight to ``np.ndarray``, so it silently
required a NumPy array.  Both of its callers got that wrong, in different ways:

* ``_TransformImageUsingCoords`` line 621 (the normal path) passed ``output_area_shape``, a plain
  tuple, which has no ``.prod()`` -- ``AttributeError`` on every backend.
* line 521 (the empty-subroi branch) passed ``output_area``, which under CuPy has been promoted to
  a device array -- ``TypeError: 'ndarray' object cannot be interpreted as an integer``.

So every branch of ``return_shared_memory=True`` raised.  Normalising the shape in the helper fixes
both, and is where the contract belongs: the buffer is always host memory, so a device-resident
shape should be brought across once here rather than at each call site.

Note the parameter still cannot deliver a usable result end to end, for reasons outside this fix:
the returned metadata names a segment that is already destroyed, because the only strong reference
to the buffer is dropped when the function returns only the metadata.  That is tracked separately.
"""

import unittest

import numpy as np

import nornir_imageregistration


class TestEveryShapeSpellingIsAccepted(unittest.TestCase):

    def _check(self, shape, expected=(4, 5)):
        meta, array = nornir_imageregistration.create_shared_memory_array(
            shape, dtype=np.float32)
        self.assertEqual(expected, tuple(array.shape))
        self.assertEqual(expected, tuple(meta.shape))
        self.assertEqual(np.float32, array.dtype)
        # The buffer must be host memory whatever the shape was expressed as.
        self.assertIsInstance(array, np.ndarray)
        return array

    def test_a_tuple_is_accepted(self):
        """The normal-path caller passes a tuple; this used to be AttributeError."""
        self._check((4, 5))

    def test_a_list_is_accepted(self):
        self._check([4, 5])

    def test_a_numpy_array_is_accepted(self):
        self._check(np.array([4, 5], dtype=np.int32))

    def test_a_numpy_array_of_another_integer_width_is_accepted(self):
        self._check(np.array([4, 5], dtype=np.int64))

    def test_a_cupy_array_is_accepted(self):
        """The empty-subroi caller passes a device array when the CuPy backend is active."""
        try:
            import cupy
        except ImportError:
            self.skipTest("cupy not installed")
        if cupy.cuda.runtime.getDeviceCount() == 0:
            self.skipTest("no CUDA device")
        self._check(cupy.asarray(np.array([4, 5], dtype=np.int32)))

    def test_the_buffer_is_writable_and_holds_what_is_written(self):
        array = self._check((4, 5))
        array.fill(3.5)
        self.assertTrue(np.array_equal(np.full((4, 5), 3.5, dtype=np.float32), array))

    def test_a_one_dimensional_shape_still_works(self):
        meta, array = nornir_imageregistration.create_shared_memory_array(
            (7,), dtype=np.uint8)
        self.assertEqual((7,), tuple(array.shape))
        self.assertEqual((7,), tuple(meta.shape))

    def test_the_buffer_is_large_enough_for_the_requested_dtype(self):
        """byte_size used shape.prod(); make sure the size still tracks itemsize."""
        for dtype in (np.uint8, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                _, array = nornir_imageregistration.create_shared_memory_array(
                    (16, 16), dtype=dtype)
                array.fill(1)  # would fault or truncate if the segment were undersized
                self.assertEqual(256 * np.dtype(dtype).itemsize, array.nbytes)


class TestTheEmptySubroiBranchUsesTheHostShape(unittest.TestCase):
    """The specific line the finding named."""

    def test_it_passes_output_area_shape_not_output_area(self):
        import inspect
        from nornir_imageregistration import assemble

        source = inspect.getsource(assemble._TransformImageUsingCoords)
        calls = [line.strip() for line in source.splitlines()
                 if 'create_shared_memory_array' in line]
        self.assertTrue(calls, "expected the shared-memory branches to still exist")
        for call in calls:
            with self.subTest(call=call):
                self.assertNotIn('(output_area,', call.replace(' ', ''),
                                 "a device-resident output_area is being used as a host shape")


if __name__ == '__main__':
    unittest.main()
