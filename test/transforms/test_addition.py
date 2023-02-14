'''
Created on Jan 31, 2020

@author: u0490822

Tests transform addition
'''
import unittest 
import os
import numpy

from test.setup_imagetest import TransformTestBase
import nornir_imageregistration 

from . import TransformCheck, ForwardTransformCheck, NearestFixedCheck, NearestWarpedCheck, \
              IdentityTransformPoints, TranslateTransformPoints, MirrorTransformPoints, OffsetTransformPoints, \
              __transform_tolerance, TranslateRotateTransformPoints, TranslateRotateScaleTransformPoints 

class TestTranslationTransformAddition(TransformTestBase):
    '''
    Add a set of transforms that each translate a point by a random amount and ensure points can pass through the transforms correctly without scaling issues
    '''
    
    @property
    def TemplateIdentityTransformFullPath(self):
        return os.path.join(self.TestInputDataPath,"0-1.stos")
     
    def CreateTransformFromTemplate(self, control, mapped):
        stos = nornir_imageregistration.StosFile.Load(self.TemplateIdentityTransformFullPath)
        stos.ControlSectionNumber = control
        stos.MappedSectionNumber = mapped
        stos.ControlImageFullPath = f'{control}.png'
        stos.MappedImageFullPath = f'{mapped}.png'
        stos.ControlMaskFullPath = f'{control}_Mask.png'
        stos.MappedMaskFullPath = f'{mapped}_Mask.png'
        return stos
    
    @property
    def TestName(self):
        return "TranslationTransformAddition"

    def setUp(self):
        
        self.assertIsNotNone(self.CreateTransformFromTemplate(0,1))
        super(TestTranslationTransformAddition, self).setUp()
        pass


    def tearDown(self):
        super(TestTranslationTransformAddition, self).tearDown()
        pass
 
    def test_transformSimpleIdentityAddition(self):
        '''
        Adds a bunch of identical transforms and ensures the output transforms matches the original 
        '''
        control = 0
        mapped = 10
        
        transforms_stos = []
        transforms = []
        for i in range(control,mapped):
            transforms_stos.append( self.CreateTransformFromTemplate(i,i+1) )
            transforms.append(nornir_imageregistration.transforms.LoadTransform(transforms_stos[i].Transform))
        
        ####################################################################
        #Part 1: Copy the original transform and add the transforms directly
         
        original = nornir_imageregistration.transforms.LoadTransform(transforms_stos[0].Transform)
         
        while len(transforms) > 1:
            A_to_B = transforms[0]
            B_to_C = transforms[1]
            
            A_to_C = B_to_C.AddTransform(A_to_B, EnrichTolerance=True, create_copy=False)
            transforms[0] = A_to_C
            del transforms[1]
            
        addedTransform = transforms[0]

        #All transforms were identical so the added transform should equal the original
        self.assertEqual(addedTransform.MappedBoundingBox, original.MappedBoundingBox)
        self.assertEqual(addedTransform.FixedBoundingBox, original.FixedBoundingBox)
        
        ###################################################################
        #Part 2: Add the stos transforms as stosfile objects
        
        originalStos = self.CreateTransformFromTemplate(0,1)
         
        while len(transforms_stos) > 1:
            A_to_B = transforms_stos[1]
            B_to_C = transforms_stos[0]
            
            A_to_C = nornir_imageregistration.AddStosTransforms(A_to_B, B_to_C, EnrichTolerance=True)
            transforms_stos[0] = A_to_C
            del transforms_stos[1]
            
        addedStos = transforms_stos[0]
        
        #self.assertEqual(addedStos.Transform, originalStos.Transform)
        self.assertEqual(addedStos.ControlSectionNumber, 0)
        self.assertEqual(addedStos.MappedSectionNumber, 10)
         
        self.assertEqual(addedStos.ControlImageFullPath, f'{control}.png')
        self.assertEqual(addedStos.MappedImageFullPath, f'{mapped}.png')
        self.assertEqual(addedStos.ControlMaskFullPath, f'{control}_Mask.png')
        self.assertEqual(addedStos.MappedMaskFullPath, f'{mapped}_Mask.png')
        
        return
    
    def test_transformAdditionChangingSize(self):
        '''Add a stack of transforms together where each transform is a different size'''
        
        control = 0
        mapped = 10
        
        #Create a set of control points for the transform
        warpedPoint = numpy.array([[0, 0],
                                [10, 0],
                                [0, 10],
                                [10, 10],
                                [5,5]])
        
        fixedPoint = numpy.array([[0, 0],
                                [20, 0],
                                [0, 20],
                                [20, 20],
                                [10,10]])
        
        nPoints = warpedPoint.shape[0]
        pointpairs_A_to_B = numpy.hstack((fixedPoint, warpedPoint))
        pointpairs_B_to_C = numpy.hstack((warpedPoint, fixedPoint))
        
        A_to_B = nornir_imageregistration.transforms.MeshWithRBFFallback(pointpairs_A_to_B)
        B_to_C = nornir_imageregistration.transforms.MeshWithRBFFallback(pointpairs_B_to_C)
        
        TransformCheck(self, A_to_B, warpedPoint, fixedPoint)
        TransformCheck(self, B_to_C, fixedPoint, warpedPoint)
        
        testPoint = numpy.array([[5,5]])
        
        testResultAB = A_to_B.Transform(testPoint) #Should be double the testpoint
        
        A_to_C = B_to_C.AddTransform(A_to_B, EnrichTolerance=True, create_copy=False)
        
        testResultBC = B_to_C.Transform(testPoint) #should be half of the testpoint
        
        TransformCheck(self, A_to_C, warpedPoint, warpedPoint)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()