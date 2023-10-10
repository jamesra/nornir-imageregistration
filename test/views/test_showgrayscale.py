'''
Created on Feb 5, 2019

@author: u0490822
'''
import unittest

import numpy as np

import nornir_imageregistration


class Test(unittest.TestCase):

    def setUp(self):
        super(Test, self).setUp()

        self.imageA = np.random.rand(64, 64)
        self.imageB = np.random.rand(64, 64)
        self.imageC = np.random.rand(64, 64)
        self.imageD = np.random.rand(64, 64)
        self.imageE = np.random.rand(64, 64)
        self.imageF = np.random.rand(64, 64)

        roi = nornir_imageregistration.Rectangle.CreateFromBounds(
            (1.5, 1.5, self.imageA.shape[0] - 1.5, self.imageA.shape[1] - 1.5))

        self.imageWideA = np.random.rand(64, 128)
        self.imageWideB = np.random.rand(64, 128)
        self.imageWideC = np.random.rand(64, 128)
        self.imageWideD = np.random.rand(64, 128)
        self.imageWideE = np.random.rand(64, 128)
        self.imageWideF = np.random.rand(64, 128)

        self.imageTallA = np.random.rand(128, 64)
        self.imageTallB = np.random.rand(128, 64)
        self.imageTallC = np.random.rand(128, 64)
        self.imageTallD = np.random.rand(128, 64)
        self.imageTallE = np.random.rand(128, 64)
        self.imageTallF = np.random.rand(128, 64)

        self.grid_5x1 = [self.imageA, self.imageB, self.imageC, self.imageD, self.imageE]
        self.grid_2x3 = [[self.imageA, self.imageB], [self.imageC, self.imageD], [self.imageE, self.imageF]]
        self.grid_2x3_titles = [['A', 'B'], ['C', 'D'], ['E', 'F']]
        self.grid_2x3_roi = [(roi, None), [None, roi], (roi, None)]

        self.wide_grid_5x1 = [self.imageWideA, self.imageWideB, self.imageWideC, self.imageWideD, self.imageWideE]
        self.tall_grid_5x1 = [self.imageTallA, self.imageTallB, self.imageTallC, self.imageTallD, self.imageTallE]
        self.list_6x1 = [self.imageA, self.imageB, self.imageC, self.imageD, self.imageE, self.imageF]

        self.list_6x1_titles = ['A', 'B', 'C', 'D', 'E', 'F']
        self.list_6x1_roi = [roi, None, None, roi, roi, None]

    def tearDown(self):
        super(Test, self).tearDown()
        pass

    def testShowGrayscaleFailButton(self):
        self.assertFalse(nornir_imageregistration.ShowGrayscale(self.imageA,
                                                                title="Ensure the FAIL button works by clicking it now\nSelect Pass button for all other tests",
                                                                PassFail=True))

    def testShowGrayscale_single_image(self):
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.imageA,
                                                               title="A single image with a title and image title",
                                                               image_titles='A',
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.imageA,
                                                               title="A single image with a title and image title and roi",
                                                               image_titles='A',
                                                               rois=nornir_imageregistration.Rectangle.CreateFromBounds(
                                                                   (1.5, 1.5, self.imageA.shape[0] - 1.5,
                                                                    self.imageA.shape[1] - 1.5)),
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.imageA,
                                                               title="A single image with a title followed by no title",
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.imageA, title=None, PassFail=True))

        self.assertTrue(nornir_imageregistration.ShowGrayscale([self.imageA],
                                                               title="A single image in a list with a title and image title",
                                                               image_titles='A',
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale([self.imageA],
                                                               title="A single image in a list with a title followed by no title",
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale([self.imageA], title=None, PassFail=True))

    def testShowGrayscale_basics(self):
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_2x3, title="2x3 images in a grid with a title",
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_2x3,
                                                               title="2x3 images in a grid with image titles",
                                                               image_titles=self.grid_2x3_titles,
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_2x3,
                                                               title="2x3 images in a grid with image titles and ROIs\nFollowed by no title",
                                                               image_titles=self.grid_2x3_titles,
                                                               rois=self.grid_2x3_roi,
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_2x3, title=None, PassFail=True))

        self.assertTrue(nornir_imageregistration.ShowGrayscale((self.imageA, self.imageB),
                                                               title="Two images in a tuple with a title\nFollowed by no title",
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale((self.imageA, self.imageB), title=None, PassFail=True))

        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_5x1,
                                                               title="Five images in a list with a title\nFollowed by no title",
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.grid_5x1, title=None, PassFail=True))

    def testShowGrayscale_autolayout(self):
        self.assertTrue(
            nornir_imageregistration.ShowGrayscale(self.list_6x1, title="Six images with reasonable auto-layout",
                                                   PassFail=True))
        self.assertTrue(
            nornir_imageregistration.ShowGrayscale(self.list_6x1,
                                                   title="Six images with reasonable auto-layout and image titles",
                                                   image_titles=self.list_6x1_titles, PassFail=True))
        self.assertTrue(
            nornir_imageregistration.ShowGrayscale(self.list_6x1,
                                                   title="Six images with reasonable auto-layout, image titles, and ROIs marked",
                                                   image_titles=self.list_6x1_titles, rois=self.list_6x1_roi,
                                                   PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.wide_grid_5x1,
                                                               title="Five wide images with reasonable auto-layout",
                                                               PassFail=True))
        self.assertTrue(nornir_imageregistration.ShowGrayscale(self.tall_grid_5x1,
                                                               title="Five tall images with reasonable auto-layout",
                                                               PassFail=True))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
