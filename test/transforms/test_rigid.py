import unittest
import nornir_imageregistration
import nornir_imageregistration.transforms

from nornir_imageregistration.transforms import *
from . import TransformCheck, ForwardTransformCheck, NearestFixedCheck, NearestWarpedCheck, \
              IdentityTransformPoints, TranslateTransformPoints, MirrorTransformPoints, OffsetTransformPoints
import numpy as np
import os
from test.setup_imagetest import ImageTestBase
from test.transforms import TransformAgreementCheck


class TestTransforms(unittest.TestCase):
    
    # def test_rigid_transform_boundingboxes(self):
    #     A_fixed_center = (0, 75)
    #     B_fixed_center = (0,-75) 
    #     A_shape = (100,100)
    #     B_shape = (100,100)
    #     A_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(A_fixed_center, A_shape)
    #     B_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(B_fixed_center, B_shape)
    #     shared_mapped_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea((0,0), A_shape)
    #     transform_A = nornir_imageregistration.transforms.Rigid(A_fixed_center, MappedBoundingBox=shared_mapped_bbox)
    #     transform_B = nornir_imageregistration.transforms.Rigid(B_fixed_center, MappedBoundingBox=shared_mapped_bbox)
    #
    #     np.testing.assert_array_equal(transform_A.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
    #     np.testing.assert_array_equal(transform_B.MappedBoundingBox.BoundingBox, shared_mapped_bbox.BoundingBox)
    #
    #     np.testing.assert_array_equal(transform_A.FixedBoundingBox.BoundingBox, A_target_bbox.BoundingBox)
    #     np.testing.assert_array_equal(transform_B.FixedBoundingBox.BoundingBox, B_target_bbox.BoundingBox)
    #     return 
 
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
        
class TestRigidFactory(ImageTestBase):
    
    def testRigidvsMeshFactory(self):
        r = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=[10,10],
                                                                             source_image_shape=[10,10],
                                                                             rangle=0,
                                                                             warped_offset=[-10,5],
                                                                             flip_ud=False)
        
        m = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=[10,10],
                                                                             source_image_shape=[10,10],
                                                                             rangle=0,
                                                                             warped_offset=[-10,5],
                                                                             flip_ud=False)
        
        p1 = [[0,0],
              [10,10]]
        
        r1 = r.Transform(p1)
        m1 = m.Transform(p1)
        
        np.testing.assert_allclose(r1, m1, err_msg="Mesh and Rigid Transform do not agree")
        
        ir1 = r.InverseTransform(r1)
        im1 = m.InverseTransform(m1)
        
        np.testing.assert_allclose(ir1, p1, err_msg="Mesh and Rigid Transform do not agree")
        np.testing.assert_allclose(im1, p1, err_msg="Mesh and Rigid Transform do not agree")
        np.testing.assert_allclose(im1, ir1, err_msg="Mesh and Rigid Transform do not agree")
        
        TransformAgreementCheck(r,m, [10,-3])
        
    def testRigidvsMeshFactoryRotation(self):
        d_angle = 90
        r_angle = (d_angle / 180.0) * np.pi
        r = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=[10,10],
                                                                             source_image_shape=[10,10],
                                                                             rangle=r_angle,
                                                                             warped_offset=[0,0],
                                                                             flip_ud=False)
        
        m = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=[10,10],
                                                                             source_image_shape=[10,10],
                                                                             rangle=r_angle,
                                                                             warped_offset=[0,0],
                                                                             flip_ud=False)
        
        p1 = [[0,0],
              [9,0],
              [0,9],
              [9,9]]
        
        TransformAgreementCheck(r, m, p1)
        
    def testRigidvsMeshFactoryRotationTranslate(self):
        d_angle = 90
        r_angle = (d_angle / 180.0) * np.pi
        r = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=[10,10],
                                                                             source_image_shape=[10,10],
                                                                             rangle=r_angle,
                                                                             warped_offset=[-10,5],
                                                                             flip_ud=False)
        
        m = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=[10,10],
                                                                             source_image_shape=[10,10],
                                                                             rangle=r_angle,
                                                                             warped_offset=[-10,5],
                                                                             flip_ud=False)
        
        p1 = [[0,0],
              [9,9]]
        
        TransformAgreementCheck(r, m, p1)

        
class TestRigidImageAssembly(ImageTestBase):
    
    def testRigidTransformAssemblyDirect(self):
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
        
        offset = [Y,X]
        corrected_offset = [offset[0] + ((fixed_size[0] - warped_size[0]) / 2),
                            offset[1] + ((fixed_size[1] - warped_size[1]) / 2)]
        
        transform = nornir_imageregistration.transforms.Rigid(corrected_offset, half_warped_size, angle)
        transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(transform, None, WarpedImagePath)
        
        self.assertTrue(nornir_imageregistration.ShowGrayscale([FixedImagePath, transformedImageData, WarpedImagePath],
                                               title="Second image should be perfectly aligned with the first\nBottom image is the original",  
                                               PassFail=True))
        
    def testRigidTransformAssemblyFactory(self):
        angle=-132.0
        X=-4
        Y=22
        
        __transform_tolerance = 1e-5
        
        angle = (angle / 180) * np.pi
        
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        
        warped_size = nornir_imageregistration.GetImageSize(WarpedImagePath)
        half_warped_size = np.asarray(warped_size) / 2.0
        
        fixed_size = nornir_imageregistration.GetImageSize(FixedImagePath)
        half_fixed_size = np.asarray(fixed_size) / 2.0
        
        offset = [Y,X]
        corrected_offset = [offset[0] + ((fixed_size[0] - warped_size[0]) / 2),
                            offset[1] + ((fixed_size[1] - warped_size[1]) / 2)]
        
        reference_transform = nornir_imageregistration.transforms.Rigid(corrected_offset, half_warped_size, angle)
        reference_ImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(reference_transform, None, WarpedImagePath)
        
        r_transform = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=fixed_size,
                                                                             source_image_shape=warped_size,
                                                                             rangle=angle,
                                                                             warped_offset=[Y,X],
                                                                             flip_ud=False)
        
        m_transform = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=fixed_size,
                                                                             source_image_shape=warped_size,
                                                                        rangle=angle,
                                                                                  warped_offset=[Y,X],
                                                                             flip_ud=False)
        
        #np.testing.assert_allclose(r_transform.MappedBoundingBox.Corners, m_transform.MappedBoundingBox.Corners, atol=__transform_tolerance)
        #np.testing.assert_allclose(r_transform.FixedBoundingBox.Corners, m_transform.FixedBoundingBox.Corners, atol=__transform_tolerance)
        
        r_transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(r_transform, None, WarpedImagePath)
        m_transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(m_transform, None, WarpedImagePath)
        
        self.assertTrue(nornir_imageregistration.ShowGrayscale([[FixedImagePath, WarpedImagePath],
                                                                [reference_ImageData],
                                                                [r_transformedImageData, m_transformedImageData]],
                                                                title="Middle row image should be perfectly aligned with the top left\nBottom row should match middle row",  
                                                                PassFail=True))
        
    def testRigidTransformAssemblyFactoryTranslateOnly(self):
        angle=0
        X=-6
        Y=-434.6
        
        __transform_tolerance = 1e-5
        
        angle = (angle / 180) * np.pi
        
        WarpedImagePath = os.path.join(self.ImportedDataPath, "test_rigid", "891.png")
        FixedImagePath = os.path.join(self.ImportedDataPath, "test_rigid", "890.png")
        
        warped_size = nornir_imageregistration.GetImageSize(WarpedImagePath)
        half_warped_size = np.asarray(warped_size) / 2.0
        
        fixed_size = nornir_imageregistration.GetImageSize(FixedImagePath)
        half_fixed_size = np.asarray(fixed_size) / 2.0
        
        offset = [Y,X]
        corrected_offset = [offset[0] + ((fixed_size[0] - warped_size[0]) / 2),
                            offset[1] + ((fixed_size[1] - warped_size[1]) / 2)]
        reference_transform = nornir_imageregistration.transforms.Rigid(corrected_offset, half_warped_size, angle)
        reference_ImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(reference_transform, None, WarpedImagePath)
        
        r_transform = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=fixed_size,
                                                                             source_image_shape=warped_size,
                                                                             rangle=angle,
                                                                             warped_offset=[Y,X],
                                                                             flip_ud=False)
        
        m_transform = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=fixed_size,
                                                                             source_image_shape=warped_size,
                                                                             rangle=angle,
                                                                             warped_offset=[Y,X],
                                                                             flip_ud=False)
        
        #np.testing.assert_allclose(r_transform.MappedBoundingBox.Corners, m_transform.MappedBoundingBox.Corners, atol=__transform_tolerance)
        #np.testing.assert_allclose(r_transform.FixedBoundingBox.Corners, m_transform.FixedBoundingBox.Corners, atol=__transform_tolerance)
        
        r_transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(r_transform, None, WarpedImagePath)
        m_transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(m_transform, None, WarpedImagePath)
        
        self.assertTrue(nornir_imageregistration.ShowGrayscale([[FixedImagePath, WarpedImagePath],
                                                                [reference_ImageData],
                                                                [r_transformedImageData, m_transformedImageData]],
                                                                title="Middle row image should be perfectly aligned with the top left\nBottom row should match middle row",  
                                                                PassFail=True))
        