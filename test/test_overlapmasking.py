'''
Created on Sep 14, 2018

@author: u0490822
'''
import unittest

import logging
import os 

import numpy as np

import nornir_imageregistration.core as core
import nornir_imageregistration
import nornir_imageregistration.overlapmasking
import nornir_imageregistration.stos_brute as stos_brute
from nornir_imageregistration.spatial import Rectangle


from . import setup_imagetest

    
class TestOverlapMask(setup_imagetest.ImageTestBase):
    
    def testOverlapMask(self):

        FixedImageSize = np.asarray((128,128),dtype=np.int32)
        MovingImageSize = np.asarray((128,128),dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((64,64), dtype=np.int32)

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize, MinOverlap=0.25, MaxOverlap=0.75)
        core.ShowGrayscale([mask],title="Overlap Mask: 25% Min overlap, 75% max overlap", PassFail=True)


        FixedImageSize = np.asarray((128,256),dtype=np.int32)
        MovingImageSize = np.asarray((128,64),dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((64,64), dtype=np.int32)

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize, MinOverlap=0.25, MaxOverlap=0.75)
        core.ShowGrayscale([mask],title="Overlap Mask: 25% Min overlap, 75% max overlap", PassFail=True)

        return

    def testOverlapMaskPopulation(self):

        FixedImageSize = np.asarray((128,128),dtype=np.int32)
        MovingImageSize = np.asarray((128,128),dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((64,64), dtype=np.int32)

        QuadrantSize = CorrelationImageSize // 2

        BruteForceMask = np.zeros(QuadrantSize, dtype=np.bool)
        BruteForceMaskOptimized = np.zeros(QuadrantSize, dtype=np.bool)

        nornir_imageregistration.overlapmasking._PopulateMaskQuadrantBruteForce(BruteForceMask, FixedImageSize, MovingImageSize, MinOverlap=0.25, MaxOverlap=0.75)
        nornir_imageregistration.overlapmasking._PopulateMaskQuadrantBruteForceOptimized(BruteForceMaskOptimized, FixedImageSize, MovingImageSize, MinOverlap=0.25, MaxOverlap=0.75)

        self.assertTrue(core.ShowGrayscale([BruteForceMask, BruteForceMaskOptimized], title="Two equal masks", PassFail=True))

        self.assertTrue(np.array_equal(BruteForceMask, BruteForceMaskOptimized), "Masks should be equal regardless of how they are made")

        return

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()