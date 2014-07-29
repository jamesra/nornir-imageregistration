'''
Created on Mar 21, 2013

@author: u0490822
'''
import unittest
import os
import glob
import logging
from nornir_shared.misc import SetupLogging
import shutil
import cProfile


class TestBase(unittest.TestCase):

    @property
    def classname(self):
        clsstr = str(self.__class__.__name__)
        return clsstr


    @property
    def TestInputPath(self):
        if 'TESTINPUTPATH' in os.environ:
            TestInputDir = os.environ["TESTINPUTPATH"]
            self.assertTrue(os.path.exists(TestInputDir), "Test input directory specified by TESTINPUTPATH environment variable does not exist")
            return TestInputDir
        else:
            self.fail("TESTINPUTPATH environment variable should specfify input data directory")

        return None

    @property
    def TestOutputPath(self):
        if 'TESTOUTPUTPATH' in os.environ:
            TestOutputDir = os.environ["TESTOUTPUTPATH"]
            return os.path.join(TestOutputDir, self.classname)
        else:
            self.fail("TESTOUTPUTPATH environment variable should specfify input data directory")

        return None

    @property
    def TestLogPath(self):
        if 'TESTOUTPUTPATH' in os.environ:
            TestOutputDir = os.environ["TESTOUTPUTPATH"]
            return os.path.join(TestOutputDir, "Logs", self.classname)
        else:
            self.fail("TESTOUTPUTPATH environment variable should specfify input data directory")

        return None

    @property
    def TestProfilerOutputPath(self):
        return os.path.join(self.TestOutputPath, self.classname + '.profile')

    def setUp(self):
        self.VolumeDir = self.TestOutputPath

        # Remove output of earlier tests

        try:
            if os.path.exists(self.VolumeDir):
                shutil.rmtree(self.VolumeDir)

            os.makedirs(self.VolumeDir)
        except:
            pass

        self.profiler = None

        if 'PROFILE' in os.environ:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        SetupLogging(Level=logging.INFO)
        self.Logger = logging.getLogger(self.classname)

    def tearDown(self):
        if not self.profiler is None:
            self.profiler.dump_stats(self.TestProfilerOutputPath)

        unittest.TestCase.tearDown(self)


class ImageTestBase(TestBase):

    def GetImagePath(self, ImageFilename):
        return os.path.join(self.ImportedDataPath, ImageFilename)

    def setUp(self):
        self.ImportedDataPath = os.path.join(self.TestInputPath, "Images")

        super(ImageTestBase, self).setUp()


class MosaicTestBase(TestBase):

    @property
    def TestName(self):
        raise NotImplementedError("Test should override TestName property")
    
    @property
    def TestOutputPath(self):
        return os.path.join(super(MosaicTestBase,self).TestOutputPath, self.id())

    def GetMosaicFiles(self):
        return glob.glob(os.path.join(self.ImportedDataPath, self.TestName, "*.mosaic"))

    def GetTileFullPath(self, downsamplePath=None):
        if downsamplePath is None:
            downsamplePath = "001"

        return os.path.join(self.ImportedDataPath, self.TestName, "Leveled", "TilePyramid", downsamplePath)

    def setUp(self):
        self.ImportedDataPath = os.path.join(self.TestInputPath, "Transforms", "Mosaics")

        super(MosaicTestBase, self).setUp()


if __name__ == "__main__":
    # import syssys.argv = ['', 'Test.testName']
    unittest.main()