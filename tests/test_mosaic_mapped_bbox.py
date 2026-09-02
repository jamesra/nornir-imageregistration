"""Regression for #193: mapped_bbox_shape when all_same_dims=False."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import nornir_imageregistration as nir
from nornir_imageregistration.mosaic import Mosaic


class TestEnsureMappedBoundingBoxes(unittest.TestCase):
    def test_all_same_dims_false_loads_each_tile(self) -> None:
        """#193: documented all_same_dims=False must not UnboundLocalError."""
        t_a = MagicMock()
        t_a.MappedBoundingBox = None
        t_b = MagicMock()
        t_b.MappedBoundingBox = None
        mosaic = Mosaic.__new__(Mosaic)
        mosaic._ImageToTransform = {'a.png': t_a, 'b.png': t_b}

        sizes = {
            'a.png': (100, 80),
            'b.png': (50, 40),
        }

        def _size(path: str):
            return sizes[path.replace('\\', '/').split('/')[-1]]

        with patch.object(nir, 'GetImageSize', side_effect=_size) as get_size:
            mosaic.EnsureTransformsHaveMappedBoundingBoxes(1.0, r'D:\tiles', all_same_dims=False)

        self.assertEqual(get_size.call_count, 2)
        np.testing.assert_array_equal(
            np.asarray(t_a.MappedBoundingBox.Dimensions), np.array([100.0, 80.0]))
        np.testing.assert_array_equal(
            np.asarray(t_b.MappedBoundingBox.Dimensions), np.array([50.0, 40.0]))


if __name__ == '__main__':
    unittest.main()
