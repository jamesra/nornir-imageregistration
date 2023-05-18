'''
Created on Apr 26, 2019

@author: u0490822
'''
import unittest

import numpy

import nornir_imageregistration


class Test(unittest.TestCase):
    tile_ID = 1

    @classmethod
    def create_tile(cls, target_center, shape):
        shape = numpy.array(shape)
        target_center = numpy.array(target_center)
        mapped_bbox = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), shape)
        target_center = target_center - mapped_bbox.Center
        transform = nornir_imageregistration.transforms.Rigid(target_center)
        temp_image = nornir_imageregistration.GenRandomData(height=shape[0], width=shape[1], mean=0.5, standardDev=0.25,
                                                            min_val=0.0, max_val=1.0)
        tile = nornir_imageregistration.tile.Tile(transform, temp_image, image_to_source_space_scale=1.0,
                                                  ID=cls.tile_ID)
        cls.tile_ID = cls.tile_ID + 1
        return tile

    def test_tile_overlaps_horizontal(self):
        A_fixed_center = (0, 40)
        B_fixed_center = (-0, -40)
        A_shape = (100, 100)
        B_shape = (100, 100)

        tile_A = self.create_tile(A_fixed_center, A_shape)
        tile_B = self.create_tile(B_fixed_center, B_shape)

        overlap = nornir_imageregistration.tile_overlap.TileOverlap(tile_A, tile_B, 1, 1)
        #        plot_tile_overlap(overlap)
        expected_target_space_overlap = nornir_imageregistration.Rectangle((-50, -10, 50, 10))
        numpy.testing.assert_array_equal(overlap.overlapping_target_rect.BoundingBox,
                                         expected_target_space_overlap.BoundingBox)
        expected_source_space_overlap_A = nornir_imageregistration.Rectangle((0, 0, 100, 20))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_A.BoundingBox,
                                         expected_source_space_overlap_A.BoundingBox)
        expected_source_space_overlap_B = nornir_imageregistration.Rectangle((0, 80, 100, 100))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_B.BoundingBox,
                                         expected_source_space_overlap_B.BoundingBox)
        return

    def test_tile_overlaps_vertical(self):
        A_fixed_center = (40, 0)
        B_fixed_center = (-40, -0)
        A_shape = (100, 100)
        B_shape = (100, 100)

        tile_A = self.create_tile(A_fixed_center, A_shape)
        tile_B = self.create_tile(B_fixed_center, B_shape)

        overlap = nornir_imageregistration.tile_overlap.TileOverlap(tile_A, tile_B, 1, 1)
        #        plot_tile_overlap(overlap)
        expected_target_space_overlap = nornir_imageregistration.Rectangle((-10, -50, 10, 50))
        numpy.testing.assert_array_equal(overlap.overlapping_target_rect.BoundingBox,
                                         expected_target_space_overlap.BoundingBox)
        expected_source_space_overlap_A = nornir_imageregistration.Rectangle((0, 0, 20, 100))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_A.BoundingBox,
                                         expected_source_space_overlap_A.BoundingBox)
        expected_source_space_overlap_B = nornir_imageregistration.Rectangle((80, 0, 100, 100))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_B.BoundingBox,
                                         expected_source_space_overlap_B.BoundingBox)
        return
        return

    def test_tile_overlaps_diagonal(self):
        A_fixed_center = (10, 40)
        B_fixed_center = (-10, -40)
        A_shape = (100, 100)
        B_shape = (100, 100)

        tile_A = self.create_tile(A_fixed_center, A_shape)
        tile_B = self.create_tile(B_fixed_center, B_shape)

        overlap = nornir_imageregistration.tile_overlap.TileOverlap(tile_A, tile_B, 1, 1)
        #       plot_tile_overlap(overlap) 8
        expected_target_space_overlap = nornir_imageregistration.Rectangle((-40, -10, 40, 10))
        numpy.testing.assert_array_equal(overlap.overlapping_target_rect.BoundingBox,
                                         expected_target_space_overlap.BoundingBox)
        expected_source_space_overlap_A = nornir_imageregistration.Rectangle((0, 0, 80, 20))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_A.BoundingBox,
                                         expected_source_space_overlap_A.BoundingBox)
        expected_source_space_overlap_B = nornir_imageregistration.Rectangle((20, 80, 100, 100))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_B.BoundingBox,
                                         expected_source_space_overlap_B.BoundingBox)
        return


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_tile_overlaps']
    unittest.main()
