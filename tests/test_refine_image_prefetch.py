"""The next STOS refine pair's images load in the background, and give identical results.

``image_prefetch.prefetch_stos_images`` decodes a pair's images and masks on a
background thread while the pipeline refines the previous pair, and
``RefineStosFile`` collects them with ``take_image``. A prefetched array must be
exactly what ``LoadImage`` returns, and anything stale or missing must fall back to
the path so the caller loads it as before.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import tempfile
import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.refine_shared import image_prefetch

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'idoc_690_691')
_STOS = os.path.join(_FIXTURE_DIR, 'StosBrute16', '690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos')


def _host(array) -> np.ndarray:
    return nornir_imageregistration.EnsureNumpyArray(array)


class _PrefetchCase(unittest.TestCase):

    def setUp(self) -> None:
        image_prefetch.clear_prefetched()
        self._env = os.environ.pop('NORNIR_REFINE_PREFETCH', None)

    def tearDown(self) -> None:
        image_prefetch.clear_prefetched()
        if self._env is not None:
            os.environ['NORNIR_REFINE_PREFETCH'] = self._env
        else:
            os.environ.pop('NORNIR_REFINE_PREFETCH', None)


class TestTakeImage(_PrefetchCase):

    def test_requests_cover_both_images_and_masks(self):
        requests = image_prefetch.stos_image_requests(_STOS)

        self.assertEqual(len(requests), 4)
        self.assertEqual(sum(1 for _path, dtype in requests if dtype is bool), 2)

    def test_without_prefetch_returns_the_path(self):
        path, dtype = image_prefetch.stos_image_requests(_STOS)[0]

        self.assertEqual(image_prefetch.take_image(path, dtype), path)
        self.assertIsNone(image_prefetch.take_image(None, dtype))

    def test_prefetched_arrays_match_load_image(self):
        image_prefetch.prefetch_stos_images(_STOS)
        for path, dtype in image_prefetch.stos_image_requests(_STOS):
            with self.subTest(path=os.path.basename(path)):
                taken = image_prefetch.take_image(path, dtype)
                self.assertNotIsInstance(taken, str)
                expected = nornir_imageregistration.LoadImage(path, dtype=dtype)
                self.assertIs(type(taken), type(expected), 'backend must match LoadImage')
                self.assertEqual(taken.dtype, expected.dtype)
                np.testing.assert_array_equal(_host(taken), _host(expected))

    def test_each_prefetch_is_taken_once(self):
        image_prefetch.prefetch_stos_images(_STOS)
        path, dtype = image_prefetch.stos_image_requests(_STOS)[0]

        self.assertNotIsInstance(image_prefetch.take_image(path, dtype), str)
        self.assertEqual(image_prefetch.take_image(path, dtype), path)

    def test_dtype_is_part_of_the_key(self):
        image_prefetch.prefetch_stos_images(_STOS)
        path, _dtype = image_prefetch.stos_image_requests(_STOS)[0]

        self.assertEqual(image_prefetch.take_image(path, np.float64), path)

    def test_disabled_by_environment(self):
        os.environ['NORNIR_REFINE_PREFETCH'] = '0'
        image_prefetch.prefetch_stos_images(_STOS)
        path, dtype = image_prefetch.stos_image_requests(_STOS)[0]

        self.assertEqual(image_prefetch.take_image(path, dtype), path)

    def test_missing_stos_is_ignored(self):
        image_prefetch.prefetch_stos_images(os.path.join(_FIXTURE_DIR, 'missing.stos'))
        image_prefetch.prefetch_stos_images(None)


class TestStaleAndBounded(_PrefetchCase):
    """Work on a copy of the fixture so files can be changed."""

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.mkdtemp()
        shutil.copytree(_FIXTURE_DIR, os.path.join(self._tmp, 'idoc'))
        self.stos = os.path.join(self._tmp, 'idoc', 'StosBrute16', os.path.basename(_STOS))

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_file_changed_after_prefetch_returns_the_path(self):
        image_prefetch.prefetch_stos_images(self.stos)
        path, dtype = image_prefetch.stos_image_requests(self.stos)[0]
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))

        self.assertEqual(image_prefetch.take_image(path, dtype), path)

    def _second_pair(self, name: str) -> str:
        """Another stos over different copies of the images, so no keys are shared."""
        other_dir = os.path.join(self._tmp, name)
        shutil.copytree(os.path.join(self._tmp, 'idoc'), other_dir)
        return os.path.join(other_dir, 'StosBrute16', os.path.basename(_STOS))

    def test_pipeline_order_current_pair_survives_the_next_prefetch(self):
        """The pipeline prefetches pair i+1 just before refining pair i."""
        image_prefetch.prefetch_stos_images(self.stos)
        image_prefetch.prefetch_stos_images(self._second_pair('next'))

        for path, dtype in image_prefetch.stos_image_requests(self.stos):
            self.assertNotIsInstance(image_prefetch.take_image(path, dtype), str)

    def test_the_current_pair_survives_two_prefetches_ahead(self):
        """The pipeline prefetches the next mapping's images and a mapping's next pair."""
        image_prefetch.prefetch_stos_images(self.stos)
        image_prefetch.prefetch_stos_images(self._second_pair('second'))
        image_prefetch.prefetch_stos_images(self._second_pair('third'))

        path, dtype = image_prefetch.stos_image_requests(self.stos)[0]
        self.assertNotIsInstance(image_prefetch.take_image(path, dtype), str)

    def test_a_fourth_prefetch_discards_the_oldest_unclaimed(self):
        image_prefetch.prefetch_stos_images(self.stos)
        image_prefetch.prefetch_stos_images(self._second_pair('second'))
        image_prefetch.prefetch_stos_images(self._second_pair('third'))
        fourth = self._second_pair('fourth')
        image_prefetch.prefetch_stos_images(fourth)

        path, dtype = image_prefetch.stos_image_requests(self.stos)[0]
        self.assertEqual(image_prefetch.take_image(path, dtype), path)
        self.assertNotIn(image_prefetch._key(path, dtype), image_prefetch._pending)
        fourth_path, fourth_dtype = image_prefetch.stos_image_requests(fourth)[0]
        self.assertNotIsInstance(image_prefetch.take_image(fourth_path, fourth_dtype), str)

    def test_image_files_prefetched_without_a_stos_are_taken_by_path(self):
        """The next mapping's stos does not exist yet, so its image files are named directly."""
        requests = [(path, image_prefetch.refine_image_dtype() if dtype is not bool else bool)
                    for path, dtype in image_prefetch.stos_image_requests(self.stos)]
        image_prefetch.prefetch_image_files(requests)

        for path, dtype in image_prefetch.stos_image_requests(self.stos):
            self.assertNotIsInstance(image_prefetch.take_image(path, dtype), str)

    def test_file_shared_by_consecutive_pairs_is_loaded_once_and_given_to_both(self):
        """1503-1502 then 1502-1501 both read section 1502."""
        same_images = os.path.join(os.path.dirname(self.stos), 'next_pair.stos')
        shutil.copy(self.stos, same_images)
        image_prefetch.prefetch_stos_images(self.stos)
        image_prefetch.prefetch_stos_images(same_images)
        path, dtype = image_prefetch.stos_image_requests(self.stos)[0]
        self.assertEqual(len(image_prefetch._pending), 4, 'shared files load once')

        first = image_prefetch.take_image(path, dtype)
        second = image_prefetch.take_image(path, dtype)

        self.assertNotIsInstance(first, str)
        self.assertNotIsInstance(second, str)
        self.assertIsNot(first, second, 'each pair gets its own array')
        np.testing.assert_array_equal(_host(first), _host(second))
        self.assertEqual(image_prefetch.take_image(path, dtype), path)

    def test_shared_file_survives_eviction_of_the_older_pair(self):
        same_images = os.path.join(os.path.dirname(self.stos), 'next_pair.stos')
        shutil.copy(self.stos, same_images)
        image_prefetch.prefetch_stos_images(self.stos)
        image_prefetch.prefetch_stos_images(same_images)
        # The first pair never claims its images; a fourth prefetch evicts it.
        image_prefetch.prefetch_stos_images(self._second_pair('third'))
        image_prefetch.prefetch_stos_images(self._second_pair('fourth'))

        for path, dtype in image_prefetch.stos_image_requests(same_images):
            self.assertNotIsInstance(image_prefetch.take_image(path, dtype), str)


class TestRefineOutputUnchanged(_PrefetchCase):

    def test_refine_with_prefetch_writes_the_same_stos(self):
        with tempfile.TemporaryDirectory() as output_dir:
            plain = os.path.join(output_dir, 'plain.stos')
            prefetched = os.path.join(output_dir, 'prefetched.stos')
            kwargs = {'num_iterations': 2, 'cell_size': (256, 256), 'grid_spacing': (192, 192)}

            ldc.RefineStosFile(InputStos=_STOS, OutputStosPath=plain, **kwargs)

            loads: list[str] = []
            real_load = nornir_imageregistration.LoadImage

            def counting_load(path, *args, **load_kwargs):
                loads.append(path)
                return real_load(path, *args, **load_kwargs)

            image_prefetch.prefetch_stos_images(_STOS)
            # Wait for the background loads so only RefineStosFile's own loads are counted.
            for future in list(image_prefetch._pending.values()):
                future.result()
            nornir_imageregistration.LoadImage = counting_load
            try:
                ldc.RefineStosFile(InputStos=_STOS, OutputStosPath=prefetched, **kwargs)
            finally:
                nornir_imageregistration.LoadImage = real_load

            self.assertEqual(loads, [], 'RefineStosFile should have used the prefetched images')
            self.assertTrue(filecmp.cmp(plain, prefetched, shallow=False))


if __name__ == '__main__':
    unittest.main()
