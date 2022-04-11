'''
Created on Mar 18, 2013

@author: u0490822
'''
import os
import unittest

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import *

from . import TransformCheck, ForwardTransformCheck, NearestFixedCheck, NearestWarpedCheck, \
              IdentityTransformPoints, TranslateTransformPoints, MirrorTransformPoints, OffsetTransformPoints, \
              __transform_tolerance, TranslateRotateTransformPoints, TranslateRotateScaleTransformPoints

import numpy as np
from nornir_imageregistration.transforms import rbftransform
from nornir_imageregistration.transforms.factory import CreateRigidTransform


class TestTransforms(unittest.TestCase):

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
    
    def testRBFLinearFallbackWithRotation(self):
        T = rbftransform.RBFWithLinearCorrection(TranslateRotateTransformPoints[:, 2:], TranslateRotateTransformPoints[:, 0:2])
        warpedPoint = np.array([[1, 2],
                                [1.25, 2.25],
                                [2, 3],
                                [0, 1]])
        fixedPoint = np.array([[1, -3],
                                [1.25, -3.25],
                                [2, -4],
                                [0, -2]])

        nPoints = TranslateRotateTransformPoints.shape[0]
        rotate_y_component = T.Weights[nPoints]
        scale_y_component = T.Weights[nPoints + 1]
        translate_y_component = T.Weights[nPoints + 2]
        
        axis_offset = nPoints + 3
        rotate_x_component = T.Weights[axis_offset + nPoints]
        scale_x_component = T.Weights[axis_offset + nPoints + 1]
        translate_x_component = T.Weights[axis_offset + nPoints + 2]
        
        RT = nornir_imageregistration.transforms.Rigid([translate_y_component, translate_x_component], source_rotation_center=[0,0],angle=np.radians(-90))
        #TransformCheck(RT, warpedPoint, fixedPoint)
        print("Rotation weights", T.Weights)
        fp = T.Transform(warpedPoint)
        np.testing.assert_allclose(fp, fixedPoint, atol=1e-5, rtol=0)
        fp2 = RT.Transform(warpedPoint)
        np.testing.assert_allclose(fp2, fixedPoint, atol=1e-5, rtol=0)

    def testRBFLinearFallbackWithRotationAndScaling(self):
        T = rbftransform.RBFWithLinearCorrection(TranslateRotateScaleTransformPoints[:, 2:], TranslateRotateScaleTransformPoints[:, 0:2])
        warpedPoint = np.array([[1, 2],
                                [1.25, 2.25],
                                [2, 3],
                                [0, 1]])
        fixedPoint = np.array([[3, -4],
                                [3.5, -4.5],
                                [5, -6],
                                [1, -2]])

        nPoints = TranslateRotateScaleTransformPoints.shape[0]
        rotate_y_component = T.Weights[nPoints]
        scale_y_component = T.Weights[nPoints + 1]
        translate_y_component = T.Weights[nPoints + 2]
        
        axis_offset = nPoints + 3
        rotate_x_component = T.Weights[axis_offset + nPoints]
        scale_x_component = T.Weights[axis_offset + nPoints + 1]
        translate_x_component = T.Weights[axis_offset + nPoints + 2]
        
        RT = nornir_imageregistration.transforms.CenteredSimilarity2DTransform([translate_y_component, translate_x_component], source_rotation_center=[0,0],angle=np.radians(-90),scalar=-scale_x_component)
        print("Scaling also", T.Weights)
        fp = T.Transform(warpedPoint)
        np.testing.assert_allclose(fp, fixedPoint, atol=1e-5, rtol=0)
        fp2 = RT.Transform(warpedPoint)
        np.testing.assert_allclose(fp2, fixedPoint, atol=1e-5, rtol=0)
        
    def testRBFLinearFallback(self):
        T = rbftransform.RBFWithLinearCorrection(TranslateTransformPoints[:, 2:], TranslateTransformPoints[:, 0:2])
        
        warpedPoint = np.array([[1, 2],
                                [1.25, 2.25],
                                [2, 3],
                                [0, 1]])

        fixedPoint = np.array([[0, 0],
                                [0.25, 0.25],
                                [1, 1],
                                [-1, -1]])

        nPoints = TranslateTransformPoints.shape[0]
        rotate_y_component = T.Weights[nPoints]
        scale_y_component = T.Weights[nPoints + 1]
        translate_y_component = T.Weights[nPoints + 2]
        
        axis_offset = nPoints + 3
        rotate_x_component = T.Weights[axis_offset + nPoints]
        scale_x_component = T.Weights[axis_offset + nPoints + 1]
        translate_x_component = T.Weights[axis_offset + nPoints + 2]

        RT = nornir_imageregistration.transforms.Rigid([translate_y_component, translate_x_component], angle=rotate_y_component)

        fp = T.Transform(warpedPoint)
        np.testing.assert_allclose(fp, fixedPoint, atol=1e-5, rtol=0)

        fp2 = RT.Transform(warpedPoint)
        np.testing.assert_allclose(fp2, fixedPoint, atol=1e-5, rtol=0)

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

        NearestFixedCheck(self, T, MirrorTransformPoints[:, 0:2], MirrorTransformPoints[:, 0:2] - 1)
        NearestWarpedCheck(self, T, MirrorTransformPoints[:, 2:4], MirrorTransformPoints[:, 2:4] - 1)
 
        # Add a point to the mirror transform, make sure it still works
        T.AddPoint([5.0, 5.0, -5.0, -5.0])

        # Make sure the new point can be found correctly
        NearestFixedCheck(self, T, T.TargetPoints, T.TargetPoints - 1)
        NearestWarpedCheck(self, T, T.SourcePoints, T.SourcePoints - 1)
        
        # Add a duplicate and see what happens
        NumBefore = T.NumControlPoints
        T.AddPoint([5.0, 5.0, -5.0, -5.0])
        NumAfter = T.NumControlPoints
        
        self.assertEqual(NumBefore, NumAfter)

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
        
    def testRBFTriangulation(self):
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
        T = nornir_imageregistration.transforms.RBFWithLinearCorrection(MirrorTransformPoints[:, 2:4], MirrorTransformPoints[:, 0:2])
        self.assertEqual(len(T.FixedTriangles), 2)
        self.assertEqual(len(T.WarpedTriangles), 2)

        warpedPoint = np.array([[-5, -5]])
        ForwardTransformCheck(self, T, warpedPoint, -warpedPoint)

        NearestFixedCheck(self, T, T.TargetPoints, T.TargetPoints - 1)
        NearestWarpedCheck(self, T, T.SourcePoints, T.SourcePoints - 1)

        # Add a point to the mirror transform, make sure it still works
        T.AddPoint([5.0, 5.0, -5.0, -5.0])

        NearestFixedCheck(self, T, T.TargetPoints, T.TargetPoints - 1)
        NearestWarpedCheck(self, T, T.SourcePoints, T.SourcePoints - 1)

        # Add a duplicate and see what happens
        NumBefore = T.NumControlPoints
        T.AddPoint([5.0, 5.0, -5.0, -5.0])
        NumAfter = T.NumControlPoints

        self.assertEqual(NumBefore, NumAfter)

        # We should have a new triangulation if we added a point
        self.assertTrue(len(T.FixedTriangles) > 2)
        self.assertTrue(len(T.WarpedTriangles) > 2)

        ForwardTransformCheck(self, T, warpedPoint, -warpedPoint)
        
        # Try removing a point

        # Try points not on the transform points
        warpedPoints = np.array([[-2.0, -4.0],
                                [-4.0, -2.0],
                                [0.0, -9.0],
                                [-9.0, 0.0]])
        ForwardTransformCheck(self, T, warpedPoints, -warpedPoints)
        
        # Try points outside the transform points
        # There is no inverse transform for the RBFTransform, so only 
        # check the forward transform
        warpedPoints = np.array([[-15.0, 0.0],
                                [11.0, 0.0],
                                [11.0, 11.0],
                                [-11.0, 11.0]])
        ForwardTransformCheck(self, T, warpedPoints, -warpedPoints) 

        T.AddPoints([[2.5, 2.5, -2.5, -2.5],
                     [7.5, 7.5, -7.5, -7.5]])

        ForwardTransformCheck(self, T, warpedPoints, -warpedPoints)
    
    def testMeshWithRBFFallback(self):
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
        T = nornir_imageregistration.transforms.MeshWithRBFFallback(MirrorTransformPoints)
        self.assertEqual(len(T.FixedTriangles), 2)
        self.assertEqual(len(T.WarpedTriangles), 2)

        warpedPoint = np.array([[-5, -5]])
        TransformCheck(self, T, warpedPoint, -warpedPoint)

        NearestFixedCheck(self, T, T.TargetPoints, T.TargetPoints - 1)
        NearestWarpedCheck(self, T, T.SourcePoints, T.SourcePoints - 1)

        # Add a point to the mirror transform, make sure it still works
        T.AddPoint([5.0, 5.0, -5.0, -5.0])

        NearestFixedCheck(self, T, T.TargetPoints, T.TargetPoints - 1)
        NearestWarpedCheck(self, T, T.SourcePoints, T.SourcePoints - 1)

        # Add a duplicate and see what happens
        NumBefore = T.NumControlPoints
        T.AddPoint([5.0, 5.0, -5.0, -5.0])
        NumAfter = T.NumControlPoints

        self.assertEqual(NumBefore, NumAfter)

        # We should have a new triangulation if we added a point
        self.assertTrue(len(T.FixedTriangles) > 2)
        self.assertTrue(len(T.WarpedTriangles) > 2)

        TransformCheck(self, T, warpedPoint, -warpedPoint)
        
        # Try removing a point

        # Try points not on the transform points
        warpedPoints = np.array([[-2.0, -4.0],
                                [-4.0, -2.0],
                                [0.0, -9.0],
                                [-9.0, 0.0]])
        TransformCheck(self, T, warpedPoints, -warpedPoints)
        
        # Try points outside the transform points
        # There is no inverse transform for the RBFTransform, so only 
        # check the forward transform
        warpedPoints = np.array([[-15.0, 0.0],
                                [11.0, 0.0],
                                [11.0, 11.0],
                                [-11.0, 11.0]])
        TransformCheck(self, T, warpedPoints, -warpedPoints) 

        T.AddPoints([[2.5, 2.5, -2.5, -2.5],
                     [7.5, 7.5, -7.5, -7.5]])

        TransformCheck(self, T, warpedPoints, -warpedPoints)

    def test_OriginAtZero(self):
        global IdentityTransformPoints
        global OffsetTransformPoints
        
        IdentityTransform = triangulation.Triangulation(IdentityTransformPoints)
        OffsetTransform = triangulation.Triangulation(OffsetTransformPoints)
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