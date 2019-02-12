import glob
import os
import unittest

from nornir_imageregistration.files.mosaicfile import MosaicFile
from nornir_imageregistration.mosaic  import Mosaic
from scipy import stats
from scipy.misc import imsave

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.core as core
import nornir_imageregistration.tileset as tiles
import nornir_imageregistration.transforms.factory as tfactory
from nornir_shared.tasktimer import TaskTimer
import numpy as np

import test_assemble_tiles

  
class BasicTests(test_assemble_tiles.TestMosaicAssemble):

    @property
    def TestName(self):
        return "PMG1"

    def test_CreateDistanceBuffer(self):

        firstShape = (10, 10)
        dMatrix = at.CreateDistanceImage(firstShape)
        self.assertAlmostEqual(dMatrix[0, 0], 7.07, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 9], 7.07, 2, "Distance matrix incorrect")

        secondShape = (11, 11)
        dMatrix = at.CreateDistanceImage(secondShape)

        self.assertAlmostEqual(dMatrix[0, 0], 7.78, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[10, 10], 7.78, 2, "Distance matrix incorrect")

        thirdShape = (10, 11)
        dMatrix = at.CreateDistanceImage(thirdShape)

        self.assertAlmostEqual(dMatrix[0, 0], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 0], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 10], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 10], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 5], 5, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[4, 0], 5.53, 2, "Distance matrix incorrect")
        
    def test_CreateDistanceBuffer2(self):

#         zeroEvenShape = (2, 2)
#         dMatrix = at.CreateDistanceImage2(zeroEvenShape)
# 
#         zeroOddShape = (3, 3)
#         dMatrix = at.CreateDistanceImage2(zeroOddShape)

        zeroOddShape = (5, 5)
        dMatrix = at.CreateDistanceImage2(zeroOddShape)
        
        
        firstShape = (10, 10)
        dMatrixReference = at.CreateDistanceImage(firstShape)
        dMatrix = at.CreateDistanceImage2(firstShape)
        self.assertAlmostEqual(dMatrix[0, 0], 7.78, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 9], 7.78, 2, "Distance matrix incorrect")
        
        secondShape = (11, 11)
        dMatrix = at.CreateDistanceImage2(secondShape)

        self.assertAlmostEqual(dMatrix[0, 0], 7.07, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[10, 10], 7.07, 2, "Distance matrix incorrect")

        thirdShape = (10, 11)
        dMatrix = at.CreateDistanceImage2(thirdShape)

        self.assertAlmostEqual(dMatrix[0, 0], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 0], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 10], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 10], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 5], 5.5, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[4, 0], 5.0249, 2, "Distance matrix incorrect")


    def test_MosaicBoundsEachMosaicType(self):

        for m in self.GetMosaicFiles():

            mosaic = Mosaic.LoadFromMosaicFile(m)

            self.assertIsNotNone(mosaic.MappedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " mapped bounding box: " + str(mosaic.MappedBoundingBox))

            self.assertIsNotNone(mosaic.FixedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " fixed bounding box: " + str(mosaic.FixedBoundingBox))
