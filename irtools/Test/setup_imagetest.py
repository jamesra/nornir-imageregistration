'''
Created on Mar 21, 2013

@author: u0490822
'''
import unittest
import os
import logging
import utils.misc
import shutil


class ImageTestBase(unittest.TestCase):

    @property
    def classname(self):
        clsstr = str(self.__class__.__name__)
        return clsstr

    def setUp(self):
        TestBaseDir = os.getcwd()
        if 'TESTDIR' in os.environ:
            TestBaseDir = os.environ["TESTDIR"]

        self.TestDataSource = os.path.join(os.getcwd(), "Test", "Data", "Images")
        self.VolumeDir = os.path.join(TestBaseDir, "Test", "Data", "TestOutput", self.classname)

        # Remove output of earlier tests
        if os.path.exists(self.VolumeDir):
            shutil.rmtree(self.VolumeDir)

        os.makedirs(self.VolumeDir)

        utils.misc.SetupLogging(os.path.join(TestBaseDir, 'Logs', self.classname))
        self.Logger = logging.getLogger(self.classname)


if __name__ == "__main__":
    # import syssys.argv = ['', 'Test.testName']
    unittest.main()