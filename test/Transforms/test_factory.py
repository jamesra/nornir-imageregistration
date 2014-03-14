'''
Created on Apr 1, 2013

@author: u0490822
'''
import unittest
import math

import numpy as np
import numpy.testing

import nornir_imageregistration.transforms.factory as factory
import nornir_imageregistration.spatial as spatial


tau = math.pi * 2.0

class Test(unittest.TestCase):

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

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()