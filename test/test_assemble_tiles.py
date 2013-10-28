'''
Created on Oct 28, 2013

@author: u0490822
'''
import unittest
import setup_imagetest
import glob
import nornir_imageregistration.assemble_tiles as at
from nornir_imageregistration.io.mosaicfile import MosaicFile
import os
import nornir_imageregistration.transforms.factory as tfactory

from nornir_imageregistration.mosaic  import Mosaic

class Test(setup_imagetest.MosaicTestBase):

    @property
    def MosaicFiles(self, testName=None):
        if testName is None:
            testName = "Test1"

        return glob.glob(os.path.join(self.TestDataSource, testName, "*.mosaic"))


    def testAssembleEachMosaicType(self):
        
        for m in self.MosaicFiles:

            mosaic = Mosaic.LoadFromMosaicFile(m)

            self.assertIsNotNone(mosaic.MappedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " mapped bounding box: " + str(mosaic.MappedBoundingBox))

            self.assertIsNotNone(mosaic.FixedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " fixed bounding box: " + str(mosaic.FixedBoundingBox))
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()