'''
Created on Mar 3, 2014

@author: u0490822
'''
import unittest
import itertools
import numpy as np

import nornir_imageregistration.spatial as spatial


class Test(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        self.ARects = {"A1" : spatial.Rectangle.CreateFromPointAndArea((0, 0), (10, 5)),
                  "A1Duplicate" : spatial.Rectangle.CreateFromPointAndArea((0, 0), (10, 5)),
                  "A1BoundsDefined" : spatial.Rectangle.CreateFromBounds((0, 0, 10, 5)),
                  "A2" : spatial.Rectangle.CreateFromPointAndArea((4.99, 0), (3, 5)),
                  "A3" : spatial.Rectangle.CreateFromBounds((-5, -5, 5, 5)),
                  "A4" : spatial.Rectangle.CreateFromBounds((4, 0, 5, 1))}
        

        self.BRects = {"B1" : spatial.Rectangle.CreateFromPointAndArea((10, 10), (5, 10)),
                  "B2" : spatial.Rectangle.CreateFromBounds((10, 10, 15, 20))}
        
        self.OverlapRects = {          "BottomHalfYOverlapA1" : spatial.Rectangle.CreateFromPointAndArea((0, 0), (5, 5)),
                  "TopHalfYOverlapA1" : spatial.Rectangle.CreateFromPointAndArea((-5, 0), (10, 5)),
                  "LeftHalfXOverlapA1" : spatial.Rectangle.CreateFromPointAndArea((0, -2.5), (10, 5)),
                  "RightHalfXOverlapA1" : spatial.Rectangle.CreateFromPointAndArea((0, 2.5), (10, 5)),
                  
                  "TopRightQuarterOverlapA1" : spatial.Rectangle.CreateFromPointAndArea((5, 2.5), (10, 5)),
                  "BottomLeftQuarterOverlapA1" : spatial.Rectangle.CreateFromPointAndArea((-5, -2.5), (10, 5)),
                  }

    def testRectangle(self):
        '''Test convention is that all rectangles with the same first letter overlap but do not overlap with other letters'''      

        self.CompareGroups(self.ARects, self.BRects)
        self.CompareCombinations(self.ARects)
        
    def testScale(self):
        
        rect = self.ARects["A1"]
        scaled_rect = spatial.Rectangle.scale(rect, 2)
        
        self.assertTrue(np.allclose(rect.Center, scaled_rect.Center), "Scaled rectangle should have the same center")
        self.assertTrue(np.allclose(rect.Area, scaled_rect.Area / 4), "Scaled rectangle should have quadruple the area")
        self.assertTrue(np.allclose(rect.Size, scaled_rect.Size / 2), "Scaled rectangle should have double the size")
        
    def testOverlaps(self):
        
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.ARects["A1"]), 1.0, "Identical rectangles should have 1.0 overlap")
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.ARects["A1BoundsDefined"]), 1.0, "Identical rectangles should have 1.0 overlap")
        
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.BRects["B1"]), 0, "Non-overlapping rectangles should have 0 overlap")
        
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.OverlapRects["BottomHalfYOverlapA1"]), 0.5, "Expected 50% overlap with BottomHalfYOverlapA1")
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.OverlapRects["TopHalfYOverlapA1"]), 0.5, "Expected 50% overlap with TopHalfYOverlapA1")

        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.OverlapRects["LeftHalfXOverlapA1"]), 0.5, "Expected 50% overlap with LeftHalfXOverlapA1")
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.OverlapRects["RightHalfXOverlapA1"]), 0.5, "Expected 50% overlap with RightHalfXOverlapA1")
        
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.OverlapRects["TopRightQuarterOverlapA1"]), 0.25, "Expected 50% overlap with LeftHalfXOverlapA1")
        self.assertEqual(spatial.Rectangle.overlap(self.ARects["A1"], self.OverlapRects["BottomLeftQuarterOverlapA1"]), 0.25, "Expected 50% overlap with RightHalfXOverlapA1")
        
    def testRectangleSet(self):
        A_rect_list = list(self.ARects.values())
        self.EnumerateOverlappingRectangles(A_rect_list)
        
        B_rect_list = list(self.BRects.values())
        ABRect_list = A_rect_list + B_rect_list
        self.EnumerateOverlappingRectangles(ABRect_list)
        
    def EnumerateOverlappingRectangles(self, rect_list):
         
        rset = spatial.RectangleSet.Create(rect_list)
        
        print("Rectangle List:")
        for i in range(0, len(rect_list)):
            print("%d: %s" % (i,str(rect_list[i])))
        
        OverlapSets = {}
        for i in range(0,len(rect_list)):
            OverlapSets[i] = set()
            
        print("Validate overlapping rectangles")
        for (A,B) in rset.EnumerateOverlapping():
            OverlapSets[A].add(B)
            OverlapSets[B].add(A)
            self.assertTrue(spatial.Rectangle.contains(rect_list[A], rect_list[B]), "Overlapping rectangles do not overlap")
        
        for (A, overlap_set) in OverlapSets.items():
            print("%d: %s" % (A, overlap_set))
            
        print("Validate non-overlapping rectangles")
        for (A, overlap_set) in OverlapSets.items():
            non_overlap_set = overlap_set.copy()
            non_overlap_set ^= set(range(0,len(rect_list)))
            print("%d: %s" % (A, non_overlap_set))
            for B in non_overlap_set:
                if A != B:
                    self.assertFalse(spatial.Rectangle.contains(rect_list[A], rect_list[B]), "%d - %d: Non-overlapping rectangles overlap" % (A,B))
        
        print("Done")
        print("")
        
    def CompareGroups(self, DictA, DictB):
        '''Compare two dictionaries of non-overlapping rectangles'''

        for KeyA, A in list(DictA.items()):
            for keyB, B in list(DictB.items()):
                self.assertFalse(spatial.Rectangle.contains(A, B), "Non overlapping rectangles %s - %s should not contain each other" % (KeyA, keyB))

    def CompareCombinations(self, DictA):

        for (iOne, iTwo) in itertools.combinations(list(DictA.keys()), 2):
            RectOne = DictA[iOne]
            RectTwo = DictA[iTwo]

            self.assertTrue(spatial.Rectangle.contains(RectOne, RectTwo), "Rectangles %s - %s from same group should overlap" % (iOne, iTwo))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()