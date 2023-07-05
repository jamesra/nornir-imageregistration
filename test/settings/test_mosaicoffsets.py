import os
import unittest

import nornir_imageregistration
from test.setup_imagetest import TestBase


class TestMosaicOffsetSettings(TestBase):

    def testSaveLoadMosaicOffsets(self):
        offsets = [nornir_imageregistration.settings.TileOffset(1, 2, 342.0, -23.9, None),
                   nornir_imageregistration.settings.TileOffset(43, 44, -231.2, 34.1, "#Science should be funded"),
                   nornir_imageregistration.settings.TileOffset(1, 0, -131.2, 54.2, "Yes"),
                   nornir_imageregistration.settings.TileOffset(1, 3, -1, 355.2),
                   nornir_imageregistration.settings.TileOffset(998, 1001, -3848.4, 4352.000, "Yes")]

        offsets_path = os.path.join(self.TestOutputPath, "mosaic_offsets.json")
        self.assertFalse(os.path.exists(offsets_path), f"{offsets_path} file should not exist at start of test")

        nornir_imageregistration.settings.SaveMosaicOffsets(offsets, offsets_path)
        self.assertTrue(os.path.exists(offsets_path), f"{offsets_path} File should exist after saving")

        offsets_reload = nornir_imageregistration.settings.LoadMosaicOffsets(offsets_path)
        self.assertTrue(len(offsets_reload) == len(offsets), "Settings loaded from disk should match value saved")

        sorted_offsets = sorted(offsets, key=lambda k: k.ID)

        for (i, _) in enumerate(sorted_offsets):
            self.assertTrue(sorted_offsets[i] == offsets_reload[i],
                            f'Item {i} does not match:\n\t{sorted_offsets[i]}\n\t{offsets_reload[i]}')


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testSaveLoadTranslateSettings']
    unittest.main()
