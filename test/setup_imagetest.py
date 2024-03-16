'''
Created on Mar 21, 2013

@author: u0490822
'''
from abc import ABC, abstractmethod
import cProfile
import glob
import logging
import os
import shutil
from typing import AnyStr
import unittest

import numpy as np
from numpy.typing import NDArray
import six

import nornir_imageregistration
import nornir_pools
from nornir_shared.misc import SetupLogging

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp


class TestBase(unittest.TestCase, ABC):

    @staticmethod
    def use_cp() -> bool:
        return nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    @property
    def classname(self) -> str:
        clsstr = str(self.__class__.__name__)
        return clsstr

    @property
    def TestInputPath(self) -> str:
        if 'TESTINPUTPATH' in os.environ:
            TestInputDir = os.environ["TESTINPUTPATH"]
            self.assertTrue(os.path.exists(TestInputDir),
                            "Test input directory specified by TESTINPUTPATH environment variable does not exist")
            return TestInputDir
        else:
            raise EnvironmentError("TESTINPUTPATH environment variable should specfify input data directory")

    @property
    def TestOutputPath(self) -> str:
        if 'TESTOUTPUTPATH' in os.environ:
            TestOutputDir = os.environ["TESTOUTPUTPATH"]
            return os.path.join(TestOutputDir, self.classname, self._testMethodName)
        else:
            raise EnvironmentError("TESTOUTPUTPATH environment variable should specify output data directory")

    @property
    def TestLogPath(self) -> str:
        # if 'TESTOUTPUTPATH' in os.environ:
        # TestOutputDir = os.environ["TESTOUTPUTPATH"]
        return os.path.join(self.TestOutputPath, "Logs")

    # else:
    # self.fail("TESTOUTPUTPATH environment variable should specfify input data directory")

    # return None

    @property
    def TestProfilerOutputPath(self) -> str:
        return os.path.join(self.TestOutputPath, self._testMethodName + '.profile')

    def setUp(self):
        super(TestBase, self).setUp()
        self.VolumeDir = self.TestOutputPath

        # Remove output of earlier tests

        try:
            if os.path.exists(self.VolumeDir):
                shutil.rmtree(self.VolumeDir)
        except:
            pass

        try:
            os.makedirs(self.VolumeDir, exist_ok=True)
        except PermissionError as e:
            print(str(e))
            pass

        self.profiler = None

        if 'PROFILE' in os.environ:
            os.environ[
                'PROFILE'] = self.TestOutputPath  # Overwrite the value with the directory we want the profile data saved in
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        SetupLogging(Level=logging.INFO)
        self.Logger = logging.getLogger(self.classname)

    def tearDown(self):

        nornir_pools.ClosePools()

        if not self.profiler is None:
            self.profiler.dump_stats(self.TestProfilerOutputPath)

        super(TestBase, self).tearDown()


class ImageTestBase(TestBase):

    def GetImagePath(self, ImageFilename) -> str:
        return os.path.join(self.ImportedDataPath, ImageFilename)

    @property
    def TestOutputPath(self) -> str:
        return os.path.join(super(ImageTestBase, self).TestOutputPath, self.id().split('.')[-1])

    def setUp(self):
        self.ImportedDataPath = os.path.join(self.TestInputPath, "Images")

        super(ImageTestBase, self).setUp()


class TransformTestBase(TestBase):

    @property
    @abstractmethod
    def TestName(self) -> str:
        raise NotImplementedError("Test should override TestName property")

    @property
    def TestInputDataPath(self) -> str:
        return os.path.join(self.TestInputPath, 'Transforms', self.TestName)

    @property
    def TestOutputPath(self) -> str:
        return os.path.join(super(TransformTestBase, self).TestOutputPath, self.id().split('.')[-1])

    def GetMosaicFiles(self) -> list[AnyStr]:
        return glob.glob(os.path.join(self.ImportedDataPath, self.TestName, "*.mosaic"))

    def GetMosaicFile(self, filenamebase: str):
        (base, ext) = os.path.splitext(filenamebase)
        if ext is None or len(ext) == 0:
            filenamebase += '.mosaic'

        return glob.glob(os.path.join(self.TestInputDataPath, filenamebase + ".mosaic"))[0]

    def GetStosFiles(self, *args) -> list[AnyStr]:
        return glob.glob(os.path.join(self.TestInputDataPath, *args, "*.stos"))

    def GetStosFilePath(self, *args) -> str:
        '''Return a .stos file at a specific path'''
        filenamebase = args[-1]
        (base, ext) = os.path.splitext(filenamebase)
        if ext is None or len(ext) == 0:
            filenamebase += '.stos'

        path = os.path.join(self.TestInputDataPath, *args[0:-1], filenamebase)
        self.assertTrue(os.path.exists(path), f'{path} is missing')

        return path

    def GetTileFullPath(self, downsamplePath=None) -> str:
        if downsamplePath is None:
            downsamplePath = "001"

        return os.path.join(self.ImportedDataPath, self.TestName, "Leveled", "TilePyramid", downsamplePath)

    def setUp(self):
        self.ImportedDataPath = os.path.join(self.TestInputPath, "Transforms", "Mosaics")

        if not os.path.exists(self.TestOutputPath):
            if six.PY3:
                os.makedirs(self.TestOutputPath, exist_ok=True)
            else:
                if not os.path.exists(self.TestOutputPath):
                    os.makedirs(self.TestOutputPath)

        super(TransformTestBase, self).setUp()


def array_distance(array: NDArray[np.floating]) -> NDArray[np.floating]:
    '''Convert an Mx2 array into a Mx1 array of euclidean distances'''
    if array.ndim == 1:
        return np.sqrt(np.sum(array ** 2))

    return np.sqrt(np.sum(array ** 2, 1))


if __name__ == "__main__":
    # import syssys.argv = ['', 'Test.testName']
    unittest.main()
