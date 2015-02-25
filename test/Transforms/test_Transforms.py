'''
Created on Mar 18, 2013

@author: u0490822
'''
import unittest

from nornir_imageregistration.transforms import *
import os
import numpy as np

### MirrorTransformPoints###
### A simple four control point mapping on two 20x20 grids centered on 0,0###
###               Fixed Space                                    WarpedSpace           ###
# . . . . . . . . . . 2 . . . . . . . . . 3      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . 0 . . . . . . . . . 1      1 . . . . . . . . . 0 . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      . . . . . . . . . . . . . . . . . . . . .
# . . . . . . . . . . . . . . . . . . . . .      3 . . . . . . . . . 2 . . . . . . . . . .


# Coordinates are CY, CX, MY, MX
MirrorTransformPoints = np.array([[0, 0, 0, 0],
                              [0, 10, 0, -10],
                              [10, 0, -10, 0],
                              [10, 10, -10, -10]])

IdentityTransformPoints = np.array([[0, 0, 0, 0],
                              [1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [1, 1, 1, 1]])

# Translate points by (1,2)
TranslateTransformPoints = np.array([[0, 0, 1, 2],
                              [1, 0, 2, 2],
                              [0, 1, 1, 3],
                              [1, 1, 2, 3]])

#Used to test IsOffsetAtZero
OffsetTransformPoints = np.array([[1, 1, 0, 0],
                              [2, 1, 1, 0],
                              [1, 2, 0, 1],
                              [2, 2, 1, 1]])

def TransformCheck(test, transform, warpedPoint, fixedPoint):
        '''Ensures that a point can map to its expected transformed position and back again'''
        fp = transform.Transform(warpedPoint)
        test.assertTrue(np.array_equal(np.around(fp, 2), fixedPoint))
        wp = transform.InverseTransform(fp)
        test.assertTrue(np.array_equal(np.around(wp, 2), warpedPoint))

class Test(unittest.TestCase):


    def testIdentity(self):
        T = meshwithrbffallback.MeshWithRBFFallback(IdentityTransformPoints)

        warpedPoint = np.array([[0, 0],
                                [0.25, 0.25],
                                [1, 1],
                                [-1, -1]])
        TransformCheck(self, T, warpedPoint, warpedPoint)

    def testTranslate(self):
        T = meshwithrbffallback.MeshWithRBFFallback(TranslateTransformPoints)

        warpedPoint = np.array([[1, 2],
                                [1.25, 2.25],
                                [2, 3],
                                [0, 1]])

        controlPoint = np.array([[0, 0],
                                [0.25, 0.25],
                                [1, 1],
                                [-1, -1]])

        TransformCheck(self, T, warpedPoint, controlPoint)


    def testTriangulation(self):
#        os.chdir('C:\\Buildscript\\Test\\Stos')
#        MToCStos = IrTools.IO.stosfile.StosFile.Load('27-26.stos')
#        CToVStos = IrTools.IO.stosfile.StosFile.Load('26-25.stos')
#
#        # I'll need to make sure I remember to set the downsample factor when I warp the .mosaic files
#        (CToV, cw, ch) = IrTools.Transforms.factory.TransformFactory.LoadTransform(CToVStos.Transform)
#        (MToC, mw, mh) = IrTools.Transforms.factory.TransformFactory.LoadTransform(MToCStos.Transform)
#
#        MToV = CToV.AddTransform(MToC)
#
#        MToCStos.Transform = IrTools.Transforms.factory.TransformFactory.TransformToIRToolsGridString(MToC, mw, mh)
#        MToCStos.Save("27-26_Test.stos")
#
#        MToVStos = copy.deepcopy(MToCStos)
#        MToVStos.ControlImageFullPath = CToVStos.ControlImageFullPath
#        MToVStos.Transform = IrTools.Transforms.factory.TransformFactory.TransformToIRToolsGridString(MToV, mw, mh)
#        MToVStos.ControlImageDim = CToVStos.ControlImageDim
#        MToVStos.MappedImageDim = MToCStos.MappedImageDim
#
#        MToVStos.Save("27-25.stos")

        global MirrorTransformPoints
        T = triangulation.Triangulation(MirrorTransformPoints)
        self.assertEqual(len(T.FixedTriangles), 2)
        self.assertEqual(len(T.WarpedTriangles), 2)

        warpedPoint = np.array([[-5, -5]])
        TransformCheck(self, T, warpedPoint, -warpedPoint)


        # Add a point to the mirror transform, make sure it still works
        T.AddPoint([5.0, 5.0, -5.0, -5.0])

        # We should have a new triangulation if we added a point
        self.assertTrue(len(T.FixedTriangles) > 2)
        self.assertTrue(len(T.WarpedTriangles) > 2)

        TransformCheck(self, T, warpedPoint, -warpedPoint)

        # Try points not on the transform points
        warpedPoints = np.array([[-2.0, -4.0],
                                [-4.0, -2.0],
                                [0.0, -9.0],
                                [-9.0, 0.0]])
        TransformCheck(self, T, warpedPoints, -warpedPoints)


    def test_OriginAtZero(self):
        global IdentityTransformPoints
        global OffsetTransformPoints
        
        IdentityTransform = triangulation.Triangulation(IdentityTransformPoints)
        OffsetTransform =  triangulation.Triangulation(OffsetTransformPoints)
        self.assertTrue(utils.IsOriginAtZero([IdentityTransform]), "Origin of identity transform is at zero")
        self.assertFalse(utils.IsOriginAtZero([OffsetTransform]), "Origin of Offset Transform is not at zero")
        
        self.assertTrue(utils.IsOriginAtZero([IdentityTransform, OffsetTransform]), "Origin of identity transform and offset transform is at zero")
        
    def test_bounds(self):
        
        global IdentityTransformPoints
        IdentityTransform = triangulation.Triangulation(IdentityTransformPoints)
        
        

#        print "Fixed Verts"
#        print T.FixedTriangles
#        print "\nWarped Verts"
#        print T.WarpedTriangles
#
#        T.AddPoint([5, 5, -5, -5])
#        print "\nPoint added"
#        print "Fixed Verts"
#        print T.FixedTriangles
#        print "\nWarped Verts"
#        print T.WarpedTriangles
#
#        T.AddPoint([5, 5, 5, 5])
#        print "\nDuplicate Point added"
#        print "Fixed Verts"
#        print T.FixedTriangles
#        print "\nWarped Verts"
#        print T.WarpedTriangles
#
#        warpedPoint = [[-5, -5]]
#        fp = T.ViewTransform(warpedPoint)
#        print("__Transform " + str(warpedPoint) + " to " + str(fp))
#        wp = T.InverseTransform(fp)
#
#        T.UpdatePoint(3, [10, 15, -10, -15])
#        print "\nPoint updated"
#        print "Fixed Verts"
#        print T.FixedTriangles
#        print "\nWarped Verts"
#        print T.WarpedTriangles
#
#        warpedPoint = [[-9, -14]]
#        fp = T.ViewTransform(warpedPoint)
#        print("__Transform " + str(warpedPoint) + " to " + str(fp))
#        wp = T.InverseTransform(fp)
#
#        T.RemovePoint(1)
#        print "\nPoint removed"
#        print "Fixed Verts"
#        print T.FixedTriangles
#        print "\nWarped Verts"
#        print T.WarpedTriangles
#
#        print "\nFixedPointsInRect"
#        print T.GetFixedPointsRect([-1, -1, 14, 4])



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
