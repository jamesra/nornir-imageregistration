'''
Created on Apr 1, 2013

@author: u0490822
'''
import unittest
import math
import os

import numpy as np
import numpy.testing

import nornir_imageregistration.transforms.factory as factory
import nornir_imageregistration.spatial as spatial
import nornir_imageregistration.mosaic as mosaic
import nornir_imageregistration.core as core

import test.setup_imagetest


tau = math.pi * 2.0

class TestMath(unittest.TestCase):

    def ValidateCornersMatchRectangle(self, points, rectangle):

        # self.assertTrue((rectangle.BottomLeft == points[0, :]).all(), "Bottom Left coordinate incorrect")
       # self.assertTrue((rectangle.TopLeft == points[1, :]).all(), "Top Left coordinate incorrect")
        # self.assertTrue((rectangle.BottomRight == points[2, :]).all(), "Bottom Right coordinate incorrect")
       # self.assertTrue((rectangle.TopRight == points[3, :]).all(), "Top Right coordinate incorrect")

        (minY, minX, maxY, maxX) = spatial.PointBoundingBox(points)

        numpy.testing.assert_allclose(rectangle.BottomLeft, np.array([minY, minX]), atol=0.01, err_msg="Bottom Left coordinate incorrect", verbose=True)
        numpy.testing.assert_allclose(rectangle.TopLeft, np.array([maxY, minX]), atol=0.01, err_msg="Top Left coordinate incorrect", verbose=True)
        numpy.testing.assert_allclose(rectangle.BottomRight, np.array([minY, maxX]), atol=0.01, err_msg="Bottom Right coordinate incorrect", verbose=True)
        numpy.testing.assert_allclose(rectangle.TopRight, np.array([maxY, maxX]), atol=0.01, err_msg="Top Right coordinate incorrect", verbose=True)


    def testGetTransformedRigidCornerPointsNoTranslateNoRotate(self):

        '''Ensure that we can correctly translate and rotate points correctly'''

        Height = 128
        Width = 256

        CornerPoints = factory.GetTransformedRigidCornerPoints((Height, Width), 0, (0, 0))
        ExpectedRectangle = spatial.Rectangle.CreateFromPointAndArea((0, 0), (Height - 1, Width - 1))

        self.ValidateCornersMatchRectangle(CornerPoints, ExpectedRectangle)

        return

    def testGetTransformedRigidCornerPointsNoTranslateRotate(self):

        '''Ensure that we can correctly translate and rotate points correctly'''

        Height = 128
        Width = 256

        # Rotate a half-turn
        CornerPoints = factory.GetTransformedRigidCornerPoints((Height, Width), tau / 2.0, (0, 0))
        ExpectedRectangle = spatial.Rectangle.CreateFromPointAndArea((0, 0), (Height - 1, Width - 1))
        self.ValidateCornersMatchRectangle(CornerPoints, ExpectedRectangle)

        # Rotate a quarter-turn
        CornerPoints = factory.GetTransformedRigidCornerPoints((Height, Width), tau / 4.0, (0, 0))
        p = ((-Width / 2.0) + (Height / 2.0), (-Height / 2.0) + (Width / 2.0))
        ExpectedRectangle = spatial.Rectangle.CreateFromPointAndArea(p, (Width - 1, Height - 1))
        self.ValidateCornersMatchRectangle(CornerPoints, ExpectedRectangle)

        return

class TestIO(test.setup_imagetest.MosaicTestBase):

    @property
    def TestName(self):
        return "PMG1"

    def testTransformIO(self):
        mfiles = self.GetMosaicFiles()

        for mfile in mfiles:
            mosaicObj = mosaic.Mosaic.LoadFromMosaicFile(mfile)

            (imagePath, transform) = mosaicObj.ImageToTransform.items()[0]

            imageFullPath = os.path.join(self.GetTileFullPath(), imagePath)

            # Load the first image and transform
            (height, width) = core.GetImageSize(imageFullPath)

            self.LoadSaveTransform(transform)

            MappedBounds = transform.MappedBoundingBox

            imageBoundRect = spatial.Rectangle.CreateFromPointAndArea((0, 0), (height, width))

            mappedBoundRect = spatial.Rectangle.CreateFromBounds(MappedBounds)

            self.assertTrue(spatial.Rectangle.contains(MappedBounds, imageBoundRect), "Mapped points should fall inside the image mapped for mosaic files")

            self.assertGreaterEqual(imageBoundRect.Width, mappedBoundRect.Width)
            self.assertGreaterEqual(imageBoundRect.Height, mappedBoundRect.Height)


    def LoadSaveTransform(self, transform):

        transformString = factory.TransformToIRToolsString(transform)

        loadedTransform = factory.LoadTransform(transformString)

        pointMatch = numpy.allclose(transform.points, loadedTransform.points, atol=0.5)

        self.assertTrue(pointMatch, "Converting transform to string and back alters transform")

        self.assertTrue(numpy.allclose(transform.FixedBoundingBox, loadedTransform.FixedBoundingBox), "Fixed bounding box should match after converting transform to string and back")
        self.assertTrue(numpy.allclose(transform.MappedBoundingBox, loadedTransform.MappedBoundingBox), "Mapped bounding box should match after converting transform to string and back")

        secondString = factory.TransformToIRToolsString(loadedTransform)
        self.assertTrue(secondString == transformString, "Converting transform to string twice should produce identical string")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()