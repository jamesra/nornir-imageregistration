import unittest
import nornir_imageregistration
import nornir_imageregistration.transforms

from nornir_imageregistration.transforms import *
from . import TransformCheck, ForwardTransformCheck, NearestFixedCheck, NearestWarpedCheck, \
              IdentityTransformPoints, TranslateTransformPoints, MirrorTransformPoints, OffsetTransformPoints
import numpy as np
import os
from test.setup_imagetest import ImageTestBase


class TestTransforms(unittest.TestCase):
    
    def test_transform_boundingboxes(self):
        A_fixed_center = (0, 75)
        B_fixed_center = (0,-75) 
        A_shape = (100,100)
        B_shape = (100,100)
        A_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(A_fixed_center, A_shape)
        B_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(B_fixed_center, B_shape)
        shared_mapped_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea((0,0), A_shape)
        transform_A = nornir_imageregistration.transforms.Rigid(A_fixed_center, MappedBoundingBox=shared_mapped_bbox)
        transform_B = nornir_imageregistration.transforms.Rigid(B_fixed_center, MappedBoundingBox=shared_mapped_bbox)
        
        np.testing.assert_array_equal(transform_A.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
        np.testing.assert_array_equal(transform_B.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
        
        np.testing.assert_array_equal(transform_A.FixedBoundingBox.BoundingBox, A_target_bbox.BoundingBox)
        np.testing.assert_array_equal(transform_B.FixedBoundingBox.BoundingBox, B_target_bbox.BoundingBox)
        return 
 
    def testIdentity(self):
        T = nornir_imageregistration.transforms.Rigid([0,0], [0,0], 0)

        warpedPoint = np.array([[0, 0],
                                [0.25, 0.25],
                                [1, 1],
                                [-1, -1]])
        TransformCheck(self, T, warpedPoint, warpedPoint)

    def testTranslate(self):
        T = nornir_imageregistration.transforms.Rigid([1,1], [0,0], 0)

        warpedPoint = OffsetTransformPoints[:,2:4]

        controlPoint = OffsetTransformPoints[:,0:2]

        TransformCheck(self, T, warpedPoint, controlPoint)
        
    def testRotate(self):
        
        angle = np.pi / 4.0
        T = nornir_imageregistration.transforms.Rigid([0,0], [0,0], np.pi / 4.0)

        sourcePoint = [[0,0],
                       [10,0]]

        targetPoint = [[0,0],
                       [np.cos(angle) * 10, np.sin(angle) * 10]]
        
        sourcePoint = np.asarray(sourcePoint)
        targetPoint = np.asarray(targetPoint)

        TransformCheck(self, T, sourcePoint, targetPoint)
        
    def testOffsetRotate(self):
        
        angle = np.pi / 4.0
        T = nornir_imageregistration.transforms.Rigid([0,0], source_rotation_center=[1,0], angle=np.pi / 4.0)

        sourcePoint = [[0,0],
                       [10,0]]

        targetPoint = [[-np.cos(angle), -np.sin(angle)],
                       [np.cos(angle) * 9, np.sin(angle) * 9]]
        targetPoint = np.asarray(targetPoint) + np.asarray([1,0])
        
        sourcePoint = np.asarray(sourcePoint) 

        TransformCheck(self, T, sourcePoint, targetPoint)
        
class TestRigidImageAssembly(ImageTestBase):
    
    def testRigidTransformAssembly(self):
        angle=-132.0
        X=-4
        Y=22
        
        angle = (angle / 180) * np.pi
        
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        
        warped_size = nornir_imageregistration.GetImageSize(WarpedImagePath)
        half_warped_size = np.asarray(warped_size) / 2.0
        
        fixed_size = nornir_imageregistration.GetImageSize(FixedImagePath)
        half_fixed_size = np.asarray(fixed_size) / 2.0
        
        transform = nornir_imageregistration.transforms.Rigid([Y,X], half_fixed_size, angle)
        transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(transform, None, WarpedImagePath)
        
        self.assertTrue(nornir_imageregistration.ShowGrayscale([FixedImagePath, transformedImageData, WarpedImagePath],
                                               title="Second image should be perfectly aligned with the first",  
                                               PassFail=True))