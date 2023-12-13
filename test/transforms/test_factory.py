'''
Created on Apr 1, 2013

@author: u0490822
'''
import math
import os
import unittest

import numpy as np
import numpy.testing

import nornir_imageregistration.core as core
import nornir_imageregistration.mosaic as mosaic
import nornir_imageregistration.spatial as spatial
from nornir_imageregistration.transforms.base import IControlPoints, \
    IDiscreteTransform
import nornir_imageregistration.transforms.factory as factory
from nornir_imageregistration.transforms.rigid import Rigid, RigidNoRotation
import test.setup_imagetest

tau = math.pi * 2.0


class TestMath(unittest.TestCase):

    def ValidateCornersMatchRectangle(self, points, rectangle):
        # self.assertTrue((rectangle.BottomLeft == points[0, :]).all(), "Bottom Left coordinate incorrect")
        # self.assertTrue((rectangle.TopLeft == points[1, :]).all(), "Top Left coordinate incorrect")
        # self.assertTrue((rectangle.BottomRight == points[2, :]).all(), "Bottom Right coordinate incorrect")
        # self.assertTrue((rectangle.TopRight == points[3, :]).all(), "Top Right coordinate incorrect")

        (minY, minX, maxY, maxX) = spatial.BoundsArrayFromPoints(points)

        numpy.testing.assert_allclose(rectangle.BottomLeft, np.array([minY, minX]), atol=0.01,
                                      err_msg="Bottom Left coordinate incorrect", verbose=True)
        numpy.testing.assert_allclose(rectangle.TopLeft, np.array([maxY, minX]), atol=0.01,
                                      err_msg="Top Left coordinate incorrect", verbose=True)
        numpy.testing.assert_allclose(rectangle.BottomRight, np.array([minY, maxX]), atol=0.01,
                                      err_msg="Bottom Right coordinate incorrect", verbose=True)
        numpy.testing.assert_allclose(rectangle.TopRight, np.array([maxY, maxX]), atol=0.01,
                                      err_msg="Top Right coordinate incorrect", verbose=True)

    def testGetTransformedRigidCornerPointsNoTranslateNoRotate(self):
        '''Ensure that we can correctly translate and rotate points correctly'''

        Height = 128
        Width = 256

        CornerPoints = factory.GetTransformedRigidCornerPoints((Height, Width), 0, (0, 0))
        ExpectedRectangle = spatial.Rectangle.CreateFromPointAndArea((0, 0), (Height, Width))

        self.ValidateCornersMatchRectangle(CornerPoints, ExpectedRectangle)

        return

    def testGetTransformedRigidCornerPointsNoTranslateRotate(self):
        '''Ensure that we can correctly translate and rotate points correctly'''

        Height = 128
        Width = 256

        # Rotate a half-turn
        CornerPoints = factory.GetTransformedRigidCornerPoints((Height, Width), tau / 2.0, (0, 0))
        ExpectedRectangle = spatial.Rectangle.CreateFromPointAndArea((0, 0), (Height, Width))
        self.ValidateCornersMatchRectangle(CornerPoints, ExpectedRectangle)

        # Rotate a quarter-turn
        CornerPoints = factory.GetTransformedRigidCornerPoints((Height, Width), tau / 4.0, (0, 0))
        p = ((-Width / 2.0) + (Height / 2.0), (-Height / 2.0) + (Width / 2.0))
        ExpectedRectangle = spatial.Rectangle.CreateFromPointAndArea(p, (Width, Height))
        self.ValidateCornersMatchRectangle(CornerPoints, ExpectedRectangle)

        return


class TestIO(test.setup_imagetest.TransformTestBase):

    @property
    def TestName(self):
        return "PMG1"

    def testTransformIO(self):
        mfiles = self.GetMosaicFiles()

        for mfile in mfiles:
            print(f'*** Testing {mfile} ***')
            mosaicObj = mosaic.Mosaic.LoadFromMosaicFile(mfile)

            (imagePath, transform) = list(mosaicObj.ImageToTransform.items())[0]

            imageFullPath = os.path.join(self.GetTileFullPath(), imagePath)

            # Load the first image and transform
            (height, width) = core.GetImageSize(imageFullPath)

            self.LoadSaveTransform(transform)

            if isinstance(transform, IDiscreteTransform):
                MappedBounds = transform.MappedBoundingBox

                imageBoundRect = spatial.Rectangle.CreateFromPointAndArea((0, 0), (height, width))

                mappedBoundRect = spatial.Rectangle.CreateFromBounds(MappedBounds)

                self.assertTrue(spatial.Rectangle.contains(MappedBounds, imageBoundRect),
                                "Mapped points should fall inside the image mapped for mosaic files")

                self.assertGreaterEqual(imageBoundRect.Width, mappedBoundRect.Width)
                self.assertGreaterEqual(imageBoundRect.Height, mappedBoundRect.Height)

    def LoadSaveTransform(self, transform):

        transformString = factory.TransformToIRToolsString(transform)

        loadedTransform = factory.LoadTransform(transformString)

        if isinstance(transform, IControlPoints):
            self.assertTrue(isinstance(loadedTransform, IControlPoints),
                            "Loaded transform must have same interface as saved transform")
            pointMatch = numpy.allclose(transform.points, loadedTransform.points, atol=0.1)
            self.assertTrue(pointMatch, f"Converting transform to string and back alters transform: {transformString}")

        if isinstance(transform, IDiscreteTransform):
            self.assertTrue(isinstance(loadedTransform, IDiscreteTransform),
                            "Loaded transform must have same interface as saved transform")
            self.assertTrue(
                numpy.allclose(transform.FixedBoundingBox.ToArray(), loadedTransform.FixedBoundingBox.ToArray(),
                               rtol=1e-04),
                "Fixed bounding box should match after converting transform to string and back")
            self.assertTrue(
                numpy.allclose(transform.MappedBoundingBox.ToArray(), loadedTransform.MappedBoundingBox.ToArray(),
                               rtol=1e-04),
                "Mapped bounding box should match after converting transform to string and back")

        if isinstance(transform, RigidNoRotation):
            self.assertTrue(isinstance(loadedTransform, RigidNoRotation),
                            "Loaded transform must have same interface as saved transform")
            self.assertTrue(numpy.allclose(transform._target_offset, loadedTransform._target_offset))

        if isinstance(transform, Rigid):
            self.assertTrue(isinstance(loadedTransform, Rigid),
                            "Loaded transform must have same interface as saved transform")
            self.assertTrue(numpy.allclose(transform.angle, loadedTransform.angle))
            self.assertTrue(numpy.allclose(transform.source_space_center_of_rotation,
                                           loadedTransform._source_space_center_of_rotation))

        secondString = factory.TransformToIRToolsString(loadedTransform)
        self.assertTrue(secondString == transformString,
                        "Converting transform to string twice should produce identical string")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
