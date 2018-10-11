'''
Created on Sep 26, 2018

@author: u0490822
'''
import unittest

import os
import pickle
import nornir_pools
import nornir_shared.plot
import nornir_shared.plot as plot
import numpy as np
import nornir_imageregistration.files
from nornir_imageregistration.local_distortion_correction import RefineTwoImages
from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
import nornir_imageregistration.assemble

from nornir_shared.files import try_locate_file

from . import setup_imagetest
from . import test_arrange
import local_distortion_correction

# class TestLocalDistortion(setup_imagetest.TransformTestBase):
# 
#     @property
#     def TestName(self):
#         return "GridRefinement"
#     
#     def setUp(self):
#         super(TestLocalDistortion, self).setUp()
#         return 
# 
# 
#     def tearDown(self):
#         super(TestLocalDistortion, self).tearDown()
#         return
# 
# 
#     def testRefineMosaic(self):
#         '''
#         This is a test for the refine mosaic feature which is not fully implemented
#         '''
#         tilesDir = self.GetTileFullPath()
#         mosaicFile = self.GetMosaicFile("Translated")
#         mosaic = nornir_imageregistration.mosaic.Mosaic.LoadFromMosaicFile(mosaicFile)
#         mosaic.RefineMosaic(tilesDir, usecluster=False)
#         pass
    
class TestSliceToSliceRefinement(setup_imagetest.TransformTestBase, setup_imagetest.PickleHelper):

    @property
    def TestName(self):
        return "StosRefinement"
    
    def setUp(self):
        super(TestSliceToSliceRefinement, self).setUp()
        return 

    def tearDown(self):
        super(TestSliceToSliceRefinement, self).tearDown()
        return
    
    @property
    def ImageDir(self):
        return os.path.join(self.ImportedDataPath, '..\\..\\Images\\')
    
    def testStosRefinement(self):
        '''
        This is a test for the refine mosaic feature which is not fully implemented
        '''
        tilesDir = self.GetTileFullPath()
        stosFile = self.GetStosFile("0164-0162_brute_32")
        stosObj = nornir_imageregistration.files.StosFile.Load(stosFile)
        
        fixedImage = os.path.join(self.ImageDir, stosObj.ControlImageFullPath)
        warpedImage = os.path.join(self.ImageDir, stosObj.MappedImageFullPath)
        
        stosTransform = nornir_imageregistration.transforms.factory.LoadTransform(stosObj.Transform, 1)
        
        alignment_points = self.ReadOrCreateVariable('alignment_points')
        
        if alignment_points is None:
            alignment_points = RefineTwoImages(stosTransform,
                        os.path.join(self.ImageDir, stosObj.ControlImageFullPath),
                        os.path.join(self.ImageDir, stosObj.MappedImageFullPath),
                        os.path.join(self.ImageDir, stosObj.ControlMaskFullPath),
                        os.path.join(self.ImageDir, stosObj.MappedMaskFullPath))
        
            self.SaveVariable(alignment_points, 'alignment_points')
            
        updatedTransform = local_distortion_correction._PeakListToTransform(alignment_points)
        
        stosObj.Transform = updatedTransform
        stosObj.Save(os.path.join(self.TestOutputPath, "UpdatedTransform.stos"))
                
        warpedToFixedImage = nornir_imageregistration.assemble.TransformStos(updatedTransform, fixedImage=fixedImage, warpedImage=warpedImage)
        
        nornir_imageregistration.core.ShowGrayscale([fixedImage, warpedImage, warpedToFixedImage])
        
        pass
    
    def testAlignmentRecordsToTransforms(self):
        '''
        Converts a set of alignment records into a transform
        '''
        
        a = EnhancedAlignmentRecord((0,0), (0,0), (10,10), (10,100), 1.5)
        b = EnhancedAlignmentRecord((1,0), (10,0), (20,10), (10,100), 2.5)
        c = EnhancedAlignmentRecord((0,1), (0,10), (10,20), (10,100), 3)
        d = EnhancedAlignmentRecord((1,1), (10,10), (20,20), (10,100), 2)
        
        points = [a,b,c,d]
        
        transform = local_distortion_correction._PeakListToTransform(points)
        
        test1 = np.asarray(((0,0),(5,5),(10,10)))
        expected1 = np.asarray(((20,110),(25,115),(30,120))) 
        actual2 = transform.Transform(test1)
        
        self.assertTrue(np.array_equal(expected1, actual2))
        
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()