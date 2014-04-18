'''
Created on Mar 3, 2014

@author: u0490822
'''
import unittest
import itertools

import nornir_imageregistration.spatial as spatial


class Test(unittest.TestCase):

    def testBoundingBox(self):
        '''Test convention is that all rectangles with the same first letter overlap but do not overlap with other letters'''

        ABoxs = {"A1" : spatial.BoundingBox.CreateFromPointAndVolume((0, 0, 0), (1, 10, 5)),
                  "A1Duplicate" : spatial.BoundingBox.CreateFromPointAndVolume((0, 0, 0), (1, 10, 5)),
                  "A1BoundsDefined" : spatial.BoundingBox.CreateFromBounds((0, 0, 0, 1, 10, 5)),
                  "A2" : spatial.BoundingBox.CreateFromPointAndVolume((0, 5, 0), (1, 3, 5)),
                  "A3" : spatial.BoundingBox.CreateFromBounds((0, -5, -5, 1, 5, 5)),
                  "A4" : spatial.BoundingBox.CreateFromBounds((0, 4, 0, 1, 5, 1)),
                  }

        BBoxs = {"B1" : spatial.BoundingBox.CreateFromPointAndVolume((0, 10, 10), (1, 5, 10)),
                  "B2" : spatial.BoundingBox.CreateFromBounds((0, 10, 10, 1, 15, 20))}

        AboveBBoxs = {"B1" : spatial.BoundingBox.CreateFromPointAndVolume((-2, 10, 10), (-0.1, 5, 10)),
                  "B2" : spatial.BoundingBox.CreateFromBounds((-2, 10, 10, -0.1, 15, 20))}

        BelowBBoxs = {"B1" : spatial.BoundingBox.CreateFromPointAndVolume((1.1, 10, 10), (2, 5, 10)),
                  "B2" : spatial.BoundingBox.CreateFromBounds((1.1, 10, 10, 2, 15, 20))}

        self.CompareGroups(ABoxs, BBoxs)
        self.CompareGroups(ABoxs, AboveBBoxs)
        self.CompareGroups(ABoxs, BelowBBoxs)
        self.CompareGroups(BBoxs, AboveBBoxs)
        self.CompareGroups(BBoxs, BelowBBoxs)

        self.CompareCombinations(ABoxs)

    def CompareGroups(self, DictA, DictB):
        '''Compare two dictionaries of non-overlapping rectangles'''

        for KeyA, A in list(DictA.items()):
            for keyB, B in list(DictB.items()):
                self.assertFalse(spatial.BoundingBox.contains(A, B), "Non overlapping boxes %s - %s should not contain each other" % (KeyA, keyB))

    def CompareCombinations(self, DictA):

        for (iOne, iTwo) in itertools.combinations(list(DictA.keys()), 2):
            RectOne = DictA[iOne]
            RectTwo = DictA[iTwo]

            self.assertTrue(spatial.BoundingBox.contains(RectOne, RectTwo), "Boxes %s - %s from same group should overlap" % (iOne, iTwo))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()