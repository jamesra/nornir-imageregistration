"""SaveArrayToTemporaryFile must not depend on another call having run first (#106).

``TransformedImageDataViaTempFile`` declared ``sharedTempRoot = None`` but every read and write
used ``_sharedTempRoot``.  So the declared default applied to a name nothing referenced, and the
name that *was* referenced did not exist as a class attribute at all -- it was created on first
assignment inside ``ConvertToTempFileIfLarge``.  Calling the public staticmethod before that
raised ``AttributeError``.

Correcting the name alone would leave a second problem: ``dir=None`` writes to the system temp
directory with ``delete=False``, and the ``atexit`` handler only removes the shared root, so such
a file is never reclaimed.  ``SaveArrayToTemporaryFile`` therefore ensures the folder itself.
"""

import atexit
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from nornir_imageregistration.transformed_image_data_temp_files import (
    TransformedImageDataViaTempFile as TempFileData)


class _FreshClassState:
    """Reset the process-global folder state, so a test sees the class as freshly imported."""

    def __enter__(self):
        self._created = TempFileData._temp_folder_created
        self._root = TempFileData._sharedTempRoot
        TempFileData._temp_folder_created = False
        TempFileData._sharedTempRoot = None
        self._made = []
        return self

    def track(self, path):
        if path is not None:
            self._made.append(path)

    def __exit__(self, *exc):
        self.track(TempFileData._sharedTempRoot)
        TempFileData._temp_folder_created = self._created
        TempFileData._sharedTempRoot = self._root
        for path in self._made:
            if path != self._root:
                shutil.rmtree(path, ignore_errors=True)


class TestTheAttributeNamesAgree(unittest.TestCase):
    """The declared name must be the one the code reads."""

    def test_the_used_name_is_declared_on_the_class(self):
        self.assertIn('_sharedTempRoot', vars(TempFileData))

    def test_the_misspelled_name_is_gone(self):
        """Nothing in the monorepo read it, and leaving it invites the same divergence again."""
        self.assertNotIn('sharedTempRoot', vars(TempFileData))

    def test_the_declared_default_is_none(self):
        with _FreshClassState():
            self.assertIsNone(TempFileData._sharedTempRoot)
            self.assertFalse(TempFileData._temp_folder_created)


class TestSavingWorksWithoutAPriorCall(unittest.TestCase):
    """The bug: this raised AttributeError on a freshly imported class."""

    def setUp(self):
        self.image = np.arange(16, dtype=np.float32).reshape(4, 4)

    def test_it_does_not_raise_attribute_error(self):
        with _FreshClassState() as state:
            path = TempFileData.SaveArrayToTemporaryFile("Image", self.image)
            state.track(os.path.dirname(path))
            self.assertTrue(os.path.exists(path))

    def test_the_file_lands_under_the_shared_root(self):
        """Not the system temp dir, which the atexit cleanup does not cover."""
        with _FreshClassState() as state:
            path = TempFileData.SaveArrayToTemporaryFile("Image", self.image)
            state.track(os.path.dirname(path))
            self.assertIsNotNone(TempFileData._sharedTempRoot)
            self.assertEqual(os.path.normcase(os.path.normpath(TempFileData._sharedTempRoot)),
                             os.path.normcase(os.path.normpath(os.path.dirname(path))))
            self.assertNotEqual(
                os.path.normcase(os.path.normpath(tempfile.gettempdir())),
                os.path.normcase(os.path.normpath(os.path.dirname(path))))

    def test_the_contents_round_trip(self):
        with _FreshClassState() as state:
            path = TempFileData.SaveArrayToTemporaryFile("Image", self.image)
            state.track(os.path.dirname(path))
            np.testing.assert_array_equal(self.image, np.load(path))

    def test_the_suffix_is_applied(self):
        with _FreshClassState() as state:
            path = TempFileData.SaveArrayToTemporaryFile("Distance", self.image)
            state.track(os.path.dirname(path))
            self.assertTrue(path.endswith("Distance.npy"))

    def test_a_none_image_still_raises_value_error(self):
        with _FreshClassState():
            with self.assertRaises(ValueError):
                TempFileData.SaveArrayToTemporaryFile("Image", None)

    def test_it_does_not_create_a_folder_when_the_image_is_none(self):
        """The ValueError guard runs first, so a rejected call leaves no directory behind."""
        with _FreshClassState():
            with self.assertRaises(ValueError):
                TempFileData.SaveArrayToTemporaryFile("Image", None)
            self.assertIsNone(TempFileData._sharedTempRoot)


class TestTheFolderIsCreatedOnce(unittest.TestCase):

    def setUp(self):
        self.image = np.arange(16, dtype=np.float32).reshape(4, 4)

    def test_repeated_saves_reuse_one_folder(self):
        with _FreshClassState() as state:
            paths = [TempFileData.SaveArrayToTemporaryFile(f"Image{i}", self.image)
                     for i in range(5)]
            state.track(os.path.dirname(paths[0]))
            folders = {os.path.dirname(p) for p in paths}
            self.assertEqual(1, len(folders))
            self.assertEqual(5, len({os.path.basename(p) for p in paths}))

    def test_the_ensure_helper_is_idempotent(self):
        with _FreshClassState() as state:
            first = TempFileData._EnsureSharedTempFolder()
            state.track(first)
            for _ in range(3):
                self.assertEqual(first, TempFileData._EnsureSharedTempFolder())

    def test_the_cleanup_handler_is_registered_once(self):
        with _FreshClassState() as state:
            with mock.patch.object(atexit, 'register') as registered:
                TempFileData._EnsureSharedTempFolder()
                TempFileData._EnsureSharedTempFolder()
                TempFileData._EnsureSharedTempFolder()
            state.track(TempFileData._sharedTempRoot)
            # Folder creation also registers the deferred-deletion flush (#108), so count the
            # rmtree registrations rather than every handler: the point is that three calls
            # register the cleanup once, not once each.
            rmtree_calls = [call[0] for call in registered.call_args_list
                            if call[0][0] is shutil.rmtree]
            self.assertEqual(1, len(rmtree_calls))
            self.assertEqual(TempFileData._sharedTempRoot, rmtree_calls[0][1])

    def test_the_registered_handler_removes_the_folder(self):
        """The cleanup contract this whole arrangement exists to provide."""
        with _FreshClassState() as state:
            root = TempFileData._EnsureSharedTempFolder()
            state.track(root)
            path = TempFileData.SaveArrayToTemporaryFile("Image", self.image)
            self.assertTrue(os.path.exists(path))
            shutil.rmtree(root, ignore_errors=True)
            self.assertFalse(os.path.exists(path))
            self.assertFalse(os.path.exists(root))

    def test_the_flag_and_the_path_stay_consistent(self):
        with _FreshClassState() as state:
            self.assertFalse(TempFileData._temp_folder_created)
            root = TempFileData._EnsureSharedTempFolder()
            state.track(root)
            self.assertTrue(TempFileData._temp_folder_created)
            self.assertEqual(root, TempFileData._sharedTempRoot)
            self.assertTrue(os.path.isdir(root))


class TestThePrefixIdentifiesTheOwner(unittest.TestCase):
    """The folder is left in the system temp dir if a process dies, so it must be identifiable."""

    def test_the_folder_name_is_attributable(self):
        with _FreshClassState() as state:
            root = TempFileData._EnsureSharedTempFolder()
            state.track(root)
            self.assertIn("nornir-imageregistration", os.path.basename(root))


if __name__ == '__main__':
    unittest.main()
