'''
Created on Mar 3, 2014

@author: u0490822
'''
import unittest
import itertools

import nornir_imageregistration.spatial as spatial


class Test(unittest.TestCase):

    def testRectangle(self):
        '''Test convention is that all rectangles with the same first letter overlap but do not overlap with other letters'''

        ARects = {"A1" : spatial.Rectangle.CreateFromPointAndArea((0, 0), (10, 5)),
                  "A1Duplicate" : spatial.Rectangle.CreateFromPointAndArea((0, 0), (10, 5)),
                  "A1BoundsDefined" : spatial.Rectangle.CreateFromBounds((0, 0, 10, 5)),
                  "A2" : spatial.Rectangle.CreateFromPointAndArea((5, 0), (3, 5)),
                  "A3" : spatial.Rectangle.CreateFromBounds((-5, -5, 5, 5)),
                  "A4" : spatial.Rectangle.CreateFromBounds((4, 0, 5, 1)),
                  }

        BRects = {"B1" : spatial.Rectangle.CreateFromPointAndArea((10, 10), (5, 10)),
                  "B2" : spatial.Rectangle.CreateFromBounds((10, 10, 15, 20))}

        self.CompareGroups(ARects, BRects)
        self.CompareCombinations(ARects)

    def CompareGroups(self, DictA, DictB):
        '''Compare two dictionaries of non-overlapping rectangles'''

        for KeyA, A in DictA.items():
            for keyB, B in DictB.items():
                self.assertFalse(spatial.Rectangle.contains(A, B), "Non overlapping rectangles %s - %s should not contain each other" % (KeyA, keyB))

    def CompareCombinations(self, DictA):

        for (iOne, iTwo) in itertools.combinations(DictA.keys(), 2):
            RectOne = DictA[iOne]
            RectTwo = DictA[iTwo]

            self.assertTrue(spatial.Rectangle.contains(RectOne, RectTwo), "Rectangles %s - %s from same group should overlap" % (iOne, iTwo))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()