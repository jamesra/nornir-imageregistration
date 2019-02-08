'''
Created on Feb 5, 2019

@author: u0490822
'''
import unittest
import numpy as np
import nornir_imageregistration


class Test(unittest.TestCase):


    def setUp(self):
        self.imageA = np.random.rand(64, 64)
        self.imageB = np.random.rand(64, 64)
        self.imageC = np.random.rand(64, 64)
        self.imageD = np.random.rand(64, 64)
        self.imageE = np.random.rand(64, 64)
        self.imageF = np.random.rand(64, 64)
        
        self.grid_5x1 = [self.imageA, self.imageB, self.imageC, self.imageD, self.imageE]
        self.grid_2x3 = [[self.imageA, self.imageB], [self.imageC, self.imageD], [self.imageE, self.imageF]]

    def tearDown(self):
        pass
    
    def testShowGrayscaleFailButton(self): 
        self.assertFalse(nornir_imageregistration.ShowGrayscale(self.imageA, title="Ensure the FAIL button works by clicking it now, select Pass button for all other tests", PassFail=True))
        
    def testShowGrayscale(self): 
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_2x3, title="2x3 images in a grid with a title\nFollowed by no title", PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_2x3, title=None, PassFail=True))

        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.imageA, title="A single image with a title followed by no title", PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.imageA, title=None, PassFail=True))

        self.assertTrue(nornir_imageregistration.ShowGrayscale([self.imageA], title="A single image in a list with a title followed by no title", PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale([self.imageA], title=None, PassFail=True))

        self.assertTrue(nornir_imageregistration.ShowGrayscale([self.imageA, self.imageB], title="Two images in a list with a title\nFollowed by no title", PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale([self.imageA, self.imageB], title=None, PassFail=True))
        
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_5x1, title="Five images in a list with a title\nFollowed by no title", PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_5x1, title=None, PassFail=True))
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()