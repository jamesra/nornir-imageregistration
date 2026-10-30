import os

import numpy as np

import nornir_imageregistration
from setup_imagetest import ImageTestBase


class TestRigidImageAssembly(ImageTestBase):

    def testRigidTransformAssemblyDirect(self):
        angle = 132.0
        X = -4
        Y = 22

        angle = (angle / 180) * np.pi

        WarpedImagePath = os.path.join(self.ImportedDataPath,
                                       "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        FixedImagePath = os.path.join(self.ImportedDataPath,
                                      "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")

        print(f'Warped/Source: {WarpedImagePath}')
        print(f'Fixed/Target: {FixedImagePath}')

        WarpedImage = nornir_imageregistration.LoadImage(WarpedImagePath)
        FixedImage = nornir_imageregistration.LoadImage(FixedImagePath)

        warped_size = WarpedImage.shape
        half_warped_size = np.asarray(warped_size) / 2.0

        fixed_size = FixedImage.shape
        half_fixed_size = np.asarray(fixed_size) / 2.0

        offset = [Y, X]
        corrected_offset = [offset[0] + ((fixed_size[0] - warped_size[0]) / 2),
                            offset[1] + ((fixed_size[1] - warped_size[1]) / 2)]

        transform = nornir_imageregistration.transforms.Rigid(corrected_offset, half_warped_size, angle)
        transformedImageData = nornir_imageregistration.assemble.SourceImageToTargetSpace(transform, WarpedImagePath,
                                                                                          output_botleft=(0, 0),
                                                                                          output_area=fixed_size)

        # delta = transformedImageData
        delta = np.abs(FixedImage - transformedImageData)

        self.assertTrue(
            nornir_imageregistration.ShowGrayscale((FixedImage, transformedImageData, delta, WarpedImagePath),
                                                   title="Second image should be perfectly aligned with the first",
                                                   image_titles=(
                                                       "Expected Output", "Transform Output", "Difference", "Input"),
                                                   PassFail=True))

    def testRigidTransformAssemblyFactory(self):
        angle = 132.0
        X = -4
        Y = 22

        __transform_tolerance = 1e-5

        angle = (angle / 180) * np.pi

        WarpedImagePath = os.path.join(self.ImportedDataPath,
                                       "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        FixedImagePath = os.path.join(self.ImportedDataPath,
                                      "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")

        warped_size = nornir_imageregistration.GetImageSize(WarpedImagePath)
        half_warped_size = np.asarray(warped_size) / 2.0

        fixed_size = nornir_imageregistration.GetImageSize(FixedImagePath)
        half_fixed_size = np.asarray(fixed_size) / 2.0

        offset = [Y, X]
        corrected_offset = [offset[0] + ((fixed_size[0] - warped_size[0]) / 2),
                            offset[1] + ((fixed_size[1] - warped_size[1]) / 2)]

        reference_transform = nornir_imageregistration.transforms.Rigid(corrected_offset, half_warped_size, angle)
        reference_ImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(reference_transform,
                                                                                        WarpedImagePath)

        r_transform = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=fixed_size,
                                                                                       source_image_shape=warped_size,
                                                                                       rangle=angle,
                                                                                       warped_offset=[Y, X],
                                                                                       flip_ud=False)

        m_transform = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(
            target_image_shape=fixed_size,
            source_image_shape=warped_size,
            rangle=angle,
            warped_offset=[Y, X],
            flip_ud=False)

        # np.testing.assert_allclose(r_transform.MappedBoundingBox.Corners, m_transform.MappedBoundingBox.Corners, atol=__transform_tolerance)
        # np.testing.assert_allclose(r_transform.FixedBoundingBox.Corners, m_transform.FixedBoundingBox.Corners, atol=__transform_tolerance)

        r_transformedImageData = nornir_imageregistration.assemble.SourceImageToTargetSpace(r_transform,
                                                                                            WarpedImagePath)
        m_transformedImageData = nornir_imageregistration.assemble.SourceImageToTargetSpace(m_transform,
                                                                                            WarpedImagePath)

        self.assertTrue(nornir_imageregistration.ShowGrayscale([[FixedImagePath, WarpedImagePath],
                                                                [reference_ImageData],
                                                                [r_transformedImageData, m_transformedImageData]],
                                                               title="Reference image should be perfectly aligned with the Target\nBottom row should match middle row",
                                                               image_titles=(('Target', 'Source'), 'Reference',
                                                                             ('Rigid Transform', 'Mesh Transform')),
                                                               PassFail=True))

    def testRigidTransformAssemblyFactoryTranslateOnly(self):
        angle = 0
        X = -6
        Y = -434.6

        __transform_tolerance = 1e-5

        angle = (angle / 180) * np.pi

        WarpedImagePath = os.path.join(self.ImportedDataPath, "test_rigid", "891.png")
        FixedImagePath = os.path.join(self.ImportedDataPath, "test_rigid", "890.png")

        warped_size = nornir_imageregistration.GetImageSize(WarpedImagePath)
        half_warped_size = np.asarray(warped_size) / 2.0

        fixed_size = nornir_imageregistration.GetImageSize(FixedImagePath)
        half_fixed_size = np.asarray(fixed_size) / 2.0

        offset = [Y, X]
        corrected_offset = [offset[0] + ((fixed_size[0] - warped_size[0]) / 2),
                            offset[1] + ((fixed_size[1] - warped_size[1]) / 2)]
        reference_transform = nornir_imageregistration.transforms.Rigid(corrected_offset, half_warped_size, angle)
        reference_ImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(reference_transform,
                                                                                        WarpedImagePath)

        r_transform = nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=fixed_size,
                                                                                       source_image_shape=warped_size,
                                                                                       rangle=angle,
                                                                                       warped_offset=[Y, X],
                                                                                       flip_ud=False)

        m_transform = nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(
            target_image_shape=fixed_size,
            source_image_shape=warped_size,
            rangle=angle,
            warped_offset=[Y, X],
            flip_ud=False)

        # np.testing.assert_allclose(r_transform.MappedBoundingBox.Corners, m_transform.MappedBoundingBox.Corners, atol=__transform_tolerance)
        # np.testing.assert_allclose(r_transform.FixedBoundingBox.Corners, m_transform.FixedBoundingBox.Corners, atol=__transform_tolerance)

        r_transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(r_transform, WarpedImagePath)
        m_transformedImageData = nornir_imageregistration.assemble.WarpedImageToFixedSpace(m_transform, WarpedImagePath)

        self.assertTrue(nornir_imageregistration.ShowGrayscale([[FixedImagePath, WarpedImagePath],
                                                                [reference_ImageData],
                                                                [r_transformedImageData, m_transformedImageData]],
                                                               title="Reference image should be perfectly aligned with the Target\nBottom row should match middle row",
                                                               image_titles=[('Target', 'Source'), ('Reference',),
                                                                             ('Rigid Transform', 'Mesh Transform')],
                                                               PassFail=True))
