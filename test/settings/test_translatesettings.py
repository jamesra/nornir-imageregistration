'''
Created on Sep 1, 2022

@author: u0490822
'''
import unittest
import os
import nornir_imageregistration

from test.setup_imagetest import TestBase


class TestTranslateSettings(TestBase):

    def testSaveLoadTranslateSettings(self):
        
        settings = nornir_imageregistration.settings.TranslateSettings()
        
        min_overlap_value = 1.0
        settings.min_overlap = min_overlap_value
        
        settings_path = os.path.join(self.TestOutputPath, "translatesettings.json")
        self.assertFalse(os.path.exists(settings_path), f"{settings_path} file should not exist at start of test")
        
        nornir_imageregistration.settings.GetOrSaveTranslateSettings(settings, settings_path)
        self.assertTrue(os.path.exists(settings_path), f"{settings_path} File should exist after saving")
        
        settings_reload = nornir_imageregistration.settings.GetOrSaveTranslateSettings(None, settings_path)
        self.assertTrue(settings_reload.min_overlap == min_overlap_value, "Settings loaded from disk should match value saved")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSaveLoadTranslateSettings']
    unittest.main()
