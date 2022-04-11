'''
Created on Jan 31, 2020

@author: u0490822

Tests transform addition
'''
import unittest 

from test.setup_imagetest import TransformTestBase
import nornir_imageregistration.core as core

class TestTranslationTransformAddition(TransformTestBase):
    '''
    Add a set of transforms that each translate a point by a random amount and ensure points can pass through the transforms correctly without scaling issues
    '''
    @property
    def TestName(self):
        return "TranslationTransformAddition"

    def setUp(self):
        TemplateIdentityTransformFullPath = self.TestInputDataPath("0-1.stos")
        template_transform = core.files.StosFile.Load(TemplateIdentityTransformFullPath)
        
        self.assertIsNotNone(template_transform)
        
        #Clone the transform a few times, shift the translation for each clone
        
        
        pass


    def tearDown(self):
        pass
 
    def test_transformAddition(self):
        return


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()