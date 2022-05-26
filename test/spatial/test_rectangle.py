'''
Created on Mar 3, 2014

@author: u0490822
'''
import itertools
import unittest
import math

import nornir_imageregistration.spatial as spatial
import numpy as np 
import hypothesis
import hypothesis.strategies
import hypothesis.extra.numpy
from . import rectangles


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
        
    def testConstructors(self):
        r = spatial.Rectangle.CreateFromPointAndArea((1, 2), (10, 5))
        self.assertTrue((r.BottomLeft == np.asarray((1, 2))).all())
        self.assertTrue((r.TopRight == np.asarray((11, 7))).all())
        
        r = spatial.Rectangle.CreateFromCenterPointAndArea((1, 2), (10, 20))
        self.assertTrue((r.BottomLeft == np.asarray((-4, -8))).all())
        self.assertTrue((r.TopRight == np.asarray((6, 12))).all())

    def testRectangle(self):
        '''Test convention is that all rectangles with the same first letter overlap but do not overlap with other letters'''      

        self.CompareGroups(self.ARects, self.BRects)
        self.CompareCombinations(self.ARects)
        
    def testScale(self):
        
        rect = self.ARects["A1"]
        scaled_rect = spatial.Rectangle.scale_on_center(rect, 2)
        
        self.assertTrue(np.allclose(rect.Center, scaled_rect.Center), "Scaled rectangle should have the same center")
        self.assertTrue(np.allclose(rect.Area, scaled_rect.Area / 4), "Scaled rectangle should have quadruple the area")
        self.assertTrue(np.allclose(rect.Size, scaled_rect.Size / 2), "Scaled rectangle should have double the size")
        
    
    
    @hypothesis.given(points=hypothesis.extra.numpy.arrays(np.float32, shape=(16,2), elements=hypothesis.strategies.floats(-100.0, 100.0, width=16), fill=hypothesis.strategies.nothing()),
                      shapes=hypothesis.extra.numpy.arrays(np.uint8, shape=(16,2),elements=hypothesis.strategies.integers(0, 255), fill=hypothesis.strategies.nothing()))
    def testUnion(self, points, shapes): 
        self.runUnionTest(points=points, shapes=shapes)
    
    def runUnionTest(self, points, shapes):
        rect_list = []
        for i in range(points.shape[0]):
            origin = points[i,:]
            shape = shapes[i,:]
            
            r = spatial.Rectangle.CreateFromPointAndArea(origin, shape)
            rect_list.append(r)
            
        self.runUnionTestOnList(rect_list)
        
        # bounds = spatial.Rectangle.Union(*rect_list)
        #
        # top_right_points = points + shapes
        #
        # expected_mins = np.min(points,0)
        # expected_maxs = np.max(top_right_points,0)
        # actual_mins = bounds[0:2]
        # actual_maxs = bounds[2:]
        #
        # mins_match = np.allclose(expected_mins, actual_mins)
        # maxs_match = np.allclose(expected_maxs, actual_maxs)
        #
        # if mins_match == False or maxs_match == False:
        #     print(f'Points:\n{points}')
        #     print(f'Shapes:\n{shapes}')
        #     print(f'Bounds:\n{bounds}')
        #
        # self.assertTrue(mins_match)
        # self.assertTrue(maxs_match)
    
    @hypothesis.settings(deadline=None)
    @hypothesis.given(rects=rectangles(0,32))
    def testUnionRectangles(self, rects): 
        if len(rects) == 0:
            try:
                self.runUnionTestOnList(rect_list=rects)
                self.fail("Empty rectangle list did not throw proper exception")
            except ValueError:
                hypothesis.event("Empty rectangle list threw proper exception")
                return        
        else:
            self.runUnionTestOnList(rect_list=rects)
    
    def runUnionTestOnList(self, rect_list):
        bounds = spatial.Rectangle.Union(*rect_list)
        
        points = np.array([r.BottomLeft for r in rect_list])
        shapes = np.array([r.Dimensions for r in rect_list])
        
        top_right_points = points + shapes
        
        expected_mins = np.min(points,0)
        expected_maxs = np.max(top_right_points,0)
        actual_mins = bounds[0:2]
        actual_maxs = bounds[2:]

        mins_match = np.allclose(expected_mins, actual_mins)
        maxs_match = np.allclose(expected_maxs, actual_maxs)
        
        if mins_match == False or maxs_match == False:
            print(f'Points:\n{points}')
            print(f'Shapes:\n{shapes}')
            print(f'Bounds:\n{bounds}')
        
        self.assertTrue(mins_match)
        self.assertTrue(maxs_match)
        
    def testUnionRepro(self):
        points= np.zeros((16,2))
        shapes= np.zeros((16,2))
        points[0,1] = 1
        self.runUnionTest(points=points, shapes=shapes)
        
        
        
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
        
        overlap_rect_list = list(self.OverlapRects.values())
        self.EnumerateOverlappingRectangles(overlap_rect_list)
        
    def EnumerateOverlappingRectangles(self, rect_list):
         
        rset = spatial.RectangleSet.Create(rect_list)
        
        print("Rectangle List:")
        for i in range(0, len(rect_list)):
            print("%d: %s" % (i, str(rect_list[i])))
        
        OverlapSets = {}
        for i in range(0, len(rect_list)):
            OverlapSets[i] = set()
            
        print("Validate overlapping rectangles")
        for (A, B) in rset.EnumerateOverlapping():
            #Make sure it is not a duplicate
            self.assertFalse(B in OverlapSets[A])
            self.assertFalse(A in OverlapSets[B])
            print ("{0},{1}".format(A,B))
            
            OverlapSets[A].add(B)
            OverlapSets[B].add(A)
            self.assertTrue(spatial.Rectangle.Intersect(rect_list[A], rect_list[B]) is not None, "Overlapping rectangles do not overlap")
        
        for (A, overlap_set) in OverlapSets.items():
            print("%d: %s" % (A, overlap_set))
            
        print("Validate non-overlapping rectangles")
        for (A, overlap_set) in OverlapSets.items():
            non_overlap_set = overlap_set.copy()
            non_overlap_set ^= set(range(0, len(rect_list)))
            print("%d: %s" % (A, non_overlap_set))
            for B in non_overlap_set:
                if A != B:
                    self.assertFalse(spatial.Rectangle.contains(rect_list[A], rect_list[B]), "%d - %d: Non-overlapping rectangles overlap" % (A, B))
        
        
        print("Done")
        print("")
        
    def testLongVerticalRectangleSet(self):
        y_range = np.linspace(0, 90, 10)
        x_lims = (0,10)
        area = (20,20)
        origins = [(y, x_lims[0]) for y in y_range]
        
        rectangles = [spatial.Rectangle.CreateFromPointAndArea(origin, area) for origin in origins]
        rset = spatial.RectangleSet.Create(rectangles)
        for (A,B) in rset.EnumerateOverlapping():
            print("{0},{1}".format(A,B))
            self.assertTrue(abs(B - A) < 2)    
            
    def testLongVerticalRectangleSet_slightoffset(self):
        y_range = np.linspace(0, 90, 10)
        x_origins = np.linspace(0, 9, 10)
        area = (20,20)
        origins = [(y, x_origins[i]) for (i,y) in enumerate(y_range)]
        
        rectangles = [spatial.Rectangle.CreateFromPointAndArea(origin, area) for origin in origins]
        rset = spatial.RectangleSet.Create(rectangles)
        for (A,B) in rset.EnumerateOverlapping():
            print("{0},{1}".format(A,B))
            self.assertTrue(abs(B - A) < 2) 
            
    def testGridRectangleSet(self):
        y_range = np.linspace(0, 40, 5)
        x_range = np.linspace(10, 90, 10)
        area = (12,12)
        origins = []
        for (i_y,y) in enumerate(y_range):
            for x in x_range:
                origins.append((y,x+i_y))
        
        rectangles = [spatial.Rectangle.CreateFromPointAndArea(origin, area) for origin in origins]
        rset = spatial.RectangleSet.Create(rectangles)
        matches = []
        for (A,B) in rset.EnumerateOverlapping():
            print("{0},{1}".format(A,B))
            matches.append((A,B))
            #self.assertTrue(abs(B - A) < 2)
            
        sorted_matches = sorted(matches)
        
        self.assertTrue((24,35) in sorted_matches)
        self.assertTrue((26,35) in sorted_matches)
        self.assertTrue((25,35) in sorted_matches)
        print("Done!") 
            
        
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
