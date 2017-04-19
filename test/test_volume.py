'''
Created on Jan 27, 2014

@author: u0490822
'''
import unittest

from nornir_imageregistration.transforms import *

import numpy as np
import volume


IdentityTransformPoints = np.array([[0, 0, 0, 0],
                              [0, 10, 0, 10],
                              [10, 0, 10, 0],
                              [10, 10, 10, 10]])

MirrorTransformPoints = np.array([[0, 0, 0, 0],
                                  [0, -10, 0, -10],
                                  [-10, 0, 10, 0],
                                  [-10, -10, 10, 10]])



class TestVolume(unittest.TestCase):


    def testBoundsZeroAlign(self):

        OneToTwo = meshwithrbffallback.MeshWithRBFFallback(IdentityTransformPoints)
        OneToThree = meshwithrbffallback.MeshWithRBFFallback(MirrorTransformPoints)

        vol = volume.Volume()

        vol.AddSection(2, OneToTwo)
        vol.AddSection(3, OneToThree)

        volBounds = vol.VolumeBounds

        self.assertEqual(volBounds.ToTuple(), (-10, -10, 10, 10), "Volume bounds are not correct")
        vol.TranslateToZeroOrigin()

        zeroedVolBounds = vol.VolumeBounds
        self.assertEqual(zeroedVolBounds.ToTuple(), (0, 0, 20, 20), "Volume bounds are not correct")
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
