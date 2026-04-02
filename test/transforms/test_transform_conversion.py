import unittest

import nornir_imageregistration.transforms as transforms
from nornir_imageregistration.transforms.transform_type import TransformType

class testTransformConversion(unittest.TestCase):
    def _create_rigid_transform(self):
        return transforms.CenteredSimilarity2DTransform(
            target_offset=(0.0, 0.0),
            source_rotation_center=(32.0, 32.0),
            angle=0.0,
            scalar=1.0
        )

    def test_rigid_to_mesh(self):
        rigid = self._create_rigid_transform()
        output = transforms.ConvertTransform(
            rigid,
            TransformType.MESH,
            source_image_shape=(64, 64)
        )
        self.assertEqual(output.type, TransformType.MESH)

    def test_rigid_to_grid(self):
        rigid = self._create_rigid_transform()
        output = transforms.ConvertTransform(
            rigid,
            TransformType.GRID,
            source_image_shape=(64, 64)
        )
        self.assertEqual(output.type, TransformType.GRID)

    def test_rigid_to_rbf(self):
        rigid = self._create_rigid_transform()
        output = transforms.ConvertTransform(
            rigid,
            TransformType.RBF,
            source_image_shape=(64, 64)
        )
        self.assertEqual(output.type, TransformType.RBF)

    def test_mesh_to_rbf(self):
        rigid = self._create_rigid_transform()
        mesh_transform = transforms.ConvertTransform(
            rigid,
            TransformType.MESH,
            source_image_shape=(64, 64)
        )
        output = transforms.ConvertTransform(mesh_transform, TransformType.RBF)
        self.assertEqual(output.type, TransformType.RBF)

    def test_grid_to_rbf(self):
        rigid = self._create_rigid_transform()
        grid_transform = transforms.ConvertTransform(
            rigid,
            TransformType.GRID,
            source_image_shape=(64, 64)
        )
        output = transforms.ConvertTransform(grid_transform, TransformType.RBF)
        self.assertEqual(output.type, TransformType.RBF)
