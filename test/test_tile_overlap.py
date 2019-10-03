'''
Created on Apr 26, 2019

@author: u0490822
'''
import unittest
import numpy
import nornir_imageregistration

from nornir_imageregistration.views import plot_tile_overlap

class Test(unittest.TestCase):
    
    tile_ID = 1
    
    @classmethod
    def create_tile(cls, target_center, shape):
        mapped_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea((0,0), shape)
        transform = nornir_imageregistration.transforms.Rigid(target_center, MappedBoundingBox=mapped_bbox)
        tile = nornir_imageregistration.tile.Tile(transform, None, cls.tile_ID)
        cls.tile_ID = cls.tile_ID + 1
        return tile

    def test_tile_overlaps_horizontal(self):
        A_fixed_center = (0, 40)
        B_fixed_center = (-0,-40)
        A_shape = (100,100)
        B_shape = (100,100)
        
        tile_A = self.create_tile(A_fixed_center, A_shape)
        tile_B = self.create_tile(B_fixed_center, B_shape)
                
        overlap = nornir_imageregistration.tile_overlap.TileOverlap(tile_A, tile_B, 1, 1)
#        plot_tile_overlap(overlap) 
        expected_overlap_A = nornir_imageregistration.Rectangle((-50,-50,50,-30))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_A.BoundingBox, expected_overlap_A.BoundingBox)
        expected_overlap_B = nornir_imageregistration.Rectangle((-50,30,50,50))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_B.BoundingBox, expected_overlap_B.BoundingBox)
        return
    
    def test_tile_overlaps_vertical(self):
        A_fixed_center = (40, 0)
        B_fixed_center = (-40,-0)
        A_shape = (100,100)
        B_shape = (100,100)
        
        tile_A = self.create_tile(A_fixed_center, A_shape)
        tile_B = self.create_tile(B_fixed_center, B_shape)
                
        overlap = nornir_imageregistration.tile_overlap.TileOverlap(tile_A, tile_B, 1, 1)
#        plot_tile_overlap(overlap) 
        expected_overlap_A = nornir_imageregistration.Rectangle((-50,-50,-30,50))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_A.BoundingBox, expected_overlap_A.BoundingBox)
        expected_overlap_B = nornir_imageregistration.Rectangle((30,-50,50,50))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_B.BoundingBox, expected_overlap_B.BoundingBox)
        return
    
    def test_tile_overlaps_diagonal(self):
        A_fixed_center = (10, 40)
        B_fixed_center = (-10,-40)
        A_shape = (100,100)
        B_shape = (100,100)
        
        tile_A = self.create_tile(A_fixed_center, A_shape)
        tile_B = self.create_tile(B_fixed_center, B_shape)
                
        overlap = nornir_imageregistration.tile_overlap.TileOverlap(tile_A, tile_B, 1, 1)
 #       plot_tile_overlap(overlap) 
        expected_overlap_A = nornir_imageregistration.Rectangle((-50,-50, 30, -30))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_A.BoundingBox, expected_overlap_A.BoundingBox)
        expected_overlap_B = nornir_imageregistration.Rectangle((-30,30,50,50))
        numpy.testing.assert_array_equal(overlap.scaled_overlapping_source_rect_B.BoundingBox, expected_overlap_B.BoundingBox)
        return
    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_tile_overlaps']
    unittest.main()