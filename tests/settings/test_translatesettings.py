'''
Created on Sep 1, 2022

@author: u0490822
'''
import json
import os
import unittest

import nornir_imageregistration

try:
    import setup_imagetest
except (ImportError, ModuleNotFoundError):
    from . import setup_imagetest

import nornir_imageregistration


class TestTranslateSettings(setup_imagetest.TestBase):

    def testSaveLoadTranslateSettings(self):
        settings = nornir_imageregistration.settings.TranslateSettings()

        min_overlap_value = 1.0
        settings.min_overlap = min_overlap_value

        settings_path = os.path.join(self.TestOutputPath, "translatesettings.json")
        self.assertFalse(os.path.exists(settings_path), f"{settings_path} file should not exist at start of test")

        nornir_imageregistration.settings.GetOrSaveTranslateSettings(settings, settings_path)
        self.assertTrue(os.path.exists(settings_path), f"{settings_path} File should exist after saving")

        settings_reload = nornir_imageregistration.settings.GetOrSaveTranslateSettings(None, settings_path)
        self.assertTrue(settings_reload.min_overlap == min_overlap_value,
                        "Settings loaded from disk should match value saved")

    def testCorruptTranslateSettingsJsonPreservedWhenNoDefaults(self):
        settings_path = os.path.join(self.TestOutputPath, "corrupt_translatesettings.json")
        corrupt_body = "{ not valid json\n"
        with open(settings_path, "w", encoding="utf-8") as handle:
            handle.write(corrupt_body)

        with self.assertRaises(json.JSONDecodeError):
            nornir_imageregistration.settings.GetOrSaveTranslateSettings(None, settings_path)

        with open(settings_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), corrupt_body)

    def testNonObjectTranslateSettingsJsonPreservedWhenNoDefaults(self):
        settings_path = os.path.join(self.TestOutputPath, "non_object_translatesettings.json")
        non_object_body = "[1, 2, 3]\n"
        with open(settings_path, "w", encoding="utf-8") as handle:
            handle.write(non_object_body)

        with self.assertRaises(TypeError):
            nornir_imageregistration.settings.GetOrSaveTranslateSettings(None, settings_path)

        with open(settings_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), non_object_body)

    def testCorruptTranslateSettingsJsonRecoveredWithDefaults(self):
        settings = nornir_imageregistration.settings.TranslateSettings()
        settings.min_overlap = 0.75
        settings_path = os.path.join(self.TestOutputPath, "recover_translatesettings.json")
        with open(settings_path, "w", encoding="utf-8") as handle:
            handle.write("{ bad json")

        reloaded = nornir_imageregistration.settings.GetOrSaveTranslateSettings(settings, settings_path)
        self.assertEqual(reloaded.min_overlap, 0.75)
        reloaded_from_disk = nornir_imageregistration.settings.GetOrSaveTranslateSettings(None, settings_path)
        self.assertEqual(reloaded_from_disk.min_overlap, 0.75)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSaveLoadTranslateSettings']
    unittest.main()
