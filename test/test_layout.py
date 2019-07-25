'''
Created on Mar 2, 2015

@author: u0490822
'''
import unittest

from nornir_imageregistration.layout import *

import nornir_pools
import nornir_shared.plot 
import nornir_imageregistration
import numpy as np

import os

from . import setup_imagetest
from . import test_arrange

def _MaxTension(layout):    
    net_tension_vectors = layout.NetTensionVectors()
    return np.max(setup_imagetest.array_distance(net_tension_vectors[:,1:]))
    

def _Relax_Layout(layout_obj, MovieImageDir, max_tension_cutoff=0.5, max_iter=100):
            
    max_tension = _MaxTension(layout_obj)
     
    i = 0
    
    #pool = nornir_pools.GetGlobalMultithreadingPool()
      
    os.makedirs(MovieImageDir, exist_ok=True)
        
    filename = os.path.join(MovieImageDir, "%d.png" % i) 
    #pool.add_task("Plot step #%d" % (i), nornir_shared.plot.VectorField, layout_obj.GetPositions(), layout_obj.NetTensionVectors(), OutputFilename=filename)
    nornir_imageregistration.views.plot_layout(layout_obj, OutputFilename=filename)
    
    output_interval = 5
        
    while max_tension > max_tension_cutoff and i < max_iter:
        print("%d %g" % (i, max_tension))
        node_movement = nornir_imageregistration.layout.Layout.RelaxNodes(layout_obj)
        max_tension = _MaxTension(layout_obj)
        # node_distance = setup_imagetest.array_distance(node_movement[:,1:3])             
        # max_distance = np.max(node_distance,0)
        i += 1
        
        filename = os.path.join(MovieImageDir, "%d.png" % i)
        
        #pool.add_task("Plot step #%d" % (i), nornir_imageregistration.views.plot_layout, layout_obj, OutputFilename=filename)
        if i % output_interval == 0 or i < output_interval:
            nornir_imageregistration.views.plot_layout(layout_obj, OutputFilename=filename)
        # nornir_shared.plot.VectorField(layout_obj.GetPositions(), layout_obj.NetTensionVectors(), OutputFilename=filename)
        
        
    return layout_obj

class TestLayoutPosition(setup_imagetest.TestBase):


    def _CheckLayoutPositionCreate(self, layout_position, ID, position):
        self.assertEqual(layout_position.ID, ID, "ID not set correctly")
        self.assertTrue(np.all(layout_position.Position == np.array(position)), "Position not set correctly")
        
    def SetOffset(self, A, B, offset, weight=1.0):
        A.SetOffset(B.ID, offset, weight)
        B.SetOffset(A.ID, -offset, weight)

    def test_LayoutPosition_Basics(self):
        '''
        Test the creation of Layout Positions and setting offsets without a Layout object'''
        
        p_position = (0, 0)
        p = nornir_imageregistration.layout.LayoutPosition(1, p_position)
        self._CheckLayoutPositionCreate(p, 1, p_position)
         
        p2_position = (10, 10)
        p2 = nornir_imageregistration.layout.LayoutPosition(2, p2_position)
        self._CheckLayoutPositionCreate(p2, 2, p2_position) 
        
        offset = np.array((5, 5))
        weight = 1.0
        self.SetOffset(p, p2, offset, weight)
          
        return 
    
    def test_cross(self):
        '''
        Create a cross of five positions that should perfectly cancel to produce an offset vector of zero for the center
        '''
        
        spring_layout = nornir_imageregistration.layout.Layout()
        
        positions = np.array([[0, 0],
                                [10, 0],
                                [-10, 0],
                                [0, 10],
                                [0, -10]])
        
        spring_layout.CreateNode(0, positions[0, :])
        spring_layout.CreateNode(1, positions[1, :])
        spring_layout.CreateNode(2, positions[2, :])
        spring_layout.CreateNode(3, positions[3, :])
        spring_layout.CreateNode(4, positions[4, :])
        
        spring_layout.SetOffset(0, 1, positions[1, :])
        spring_layout.SetOffset(0, 2, positions[2, :])
        spring_layout.SetOffset(0, 3, positions[3, :])
        spring_layout.SetOffset(0, 4, positions[4, :])
        
        max_tension = spring_layout.MaxTensionMagnitude
        self.assertTrue(max_tension[1] == 0)
        self.assertTrue(spring_layout.MinTensionMagnitude[1] == 0)
        
        np.testing.assert_equal(spring_layout.PairTensionVector(0,1), np.array((0,0)))
         
        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(3) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(4) == np.array([0, 0])))
    
    def test_singularity(self):
        '''
        Same as test_cross, but position all of the points at the same location
        '''
        spring_layout = nornir_imageregistration.layout.Layout()
        
        positions = np.array([[0, 0],
                                [10, 0],
                                [-10, 0],
                                [0, 10],
                                [0, -10]])
        
        spring_layout.CreateNode(0, positions[0, :])
        spring_layout.CreateNode(1, positions[0, :])
        spring_layout.CreateNode(2, positions[0, :])
        spring_layout.CreateNode(3, positions[0, :])
        spring_layout.CreateNode(4, positions[0, :])
        
        spring_layout.SetOffset(0, 1, positions[1, :])
        spring_layout.SetOffset(0, 2, positions[2, :])
        spring_layout.SetOffset(0, 3, positions[3, :])
        spring_layout.SetOffset(0, 4, positions[4, :])
         
        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array(positions[1, :])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array(positions[2, :])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(3) == np.array(positions[3, :])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(4) == np.array(positions[4, :])))
        
        # OK, try to relax the layout and see where the nodes land
        displacements = nornir_imageregistration.layout.Layout.RelaxNodes(spring_layout, vector_scalar=1.0)
        
        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(3) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(4) == np.array([0, 0])))
        
        self.assertTrue(np.all(spring_layout.GetPosition(0) == np.array(positions[0, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(1) == np.array(positions[1, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(2) == np.array(positions[2, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(3) == np.array(positions[3, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(4) == np.array(positions[4, :])))
        
        # Nothing should happen on the second pass
        displacements = nornir_imageregistration.layout.Layout.RelaxNodes(spring_layout)
        
        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(3) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(4) == np.array([0, 0])))
        
        self.assertTrue(np.all(spring_layout.GetPosition(0) == np.array(positions[0, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(1) == np.array(positions[1, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(2) == np.array(positions[2, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(3) == np.array(positions[3, :])))
        self.assertTrue(np.all(spring_layout.GetPosition(4) == np.array(positions[4, :])))
 
        print("Node Positions")
        print(spring_layout.GetPositions())
        
        #Todo: translate layout to (0,0) and ensure nodes are within a pixel of the expected position
    
    
    def test_weighted_line(self):
        '''
        Three points on a line
        '''
        spring_layout = nornir_imageregistration.layout.Layout()
        
        positions = np.array([[0, 0],
                                [10, 5],
                                [-10, 5]])
                               
        
        spring_layout.CreateNode(0, positions[0, :])
        spring_layout.CreateNode(1, positions[1, :])
        spring_layout.CreateNode(2, positions[2, :]) 
        
        spring_layout.SetOffset(0, 1, positions[1, :], weight=1)
        spring_layout.SetOffset(0, 2, positions[2, :], weight=1)
        spring_layout.SetOffset(1, 2, positions[2, :] - positions[1, :], weight=1)
         
        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array([0, 0])))
        
        # OK, try to relax the layout and see where the nodes land
        max_vector_magnitude = 0.05
        _Relax_Layout(spring_layout, MovieImageDir=self.TestOutputPath, max_tension_cutoff=max_vector_magnitude, max_iter=100) 
        
        for ID in spring_layout.nodes.keys():
            self.assertTrue(setup_imagetest.array_distance(spring_layout.NetTensionVector(ID)) < max_vector_magnitude, "Node %d should have net tension vector below relax cutoff")
        
        self.assertTrue(np.allclose(spring_layout.GetPosition(0), positions[0, :], atol=max_vector_magnitude))
        self.assertTrue(np.allclose(spring_layout.GetPosition(1), positions[1, :], atol=max_vector_magnitude)) 
        self.assertTrue(np.allclose(spring_layout.GetPosition(2), positions[2, :], atol=max_vector_magnitude))  
 
        print("Node Positions")
        print(spring_layout.GetPositions())
        
        
    def test_uneven_weighted_line(self):
        '''
        Three points on a line with inconsistent desired offsets and weights
        '''
        spring_layout = nornir_imageregistration.layout.Layout()

        positions = np.array([[0, 0],
                                [10, 5],
                                [-10, 5]])


        spring_layout.CreateNode(0, positions[0, :])
        spring_layout.CreateNode(1, positions[1, :])
        spring_layout.CreateNode(2, positions[2, :]) 

        #The node in the center is neutral, but the 1,2 nodes are trying to pull together.
        spring_layout.SetOffset(0, 1, positions[1, :], weight=1)
        spring_layout.SetOffset(0, 2, positions[2, :], weight=1)
        spring_layout.SetOffset(1, 2, np.array([-15, 0]), weight=1)

        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array([-5, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array([5, 0])))
        
        np.testing.assert_equal(spring_layout.PairTensionVector(0,2), np.array((0,0)))
        np.testing.assert_equal(spring_layout.PairTensionVector(1,2), np.array((-5,0)))

        # OK, try to relax the layout and see where the nodes land
        max_vector_magnitude = 0.001
        _Relax_Layout(spring_layout,
                      MovieImageDir=self.TestOutputPath,
                      max_tension_cutoff=max_vector_magnitude,
                      max_iter=100) 

        for ID in spring_layout.nodes.keys():
            self.assertTrue(setup_imagetest.array_distance(spring_layout.NetTensionVector(ID)) < max_vector_magnitude, "Node %d should have net tension vector below relax cutoff")

        self.assertTrue(np.allclose(spring_layout.GetPosition(0) - spring_layout.GetPosition(1), (-8.333,-5), atol=max_vector_magnitude*2)) #Since the nodes at the ends both move we expect equilibrium at 8.33 instead of 7.5
        self.assertTrue(np.allclose(spring_layout.GetPosition(0) - spring_layout.GetPosition(2), (8.333,-5), atol=max_vector_magnitude*2))

        print("Node Positions")
        print(spring_layout.GetPositions())
        
        nornir_imageregistration.views.plot_layout(spring_layout)
        
class TestLayout(setup_imagetest.TestBase):
    
    @staticmethod
    def enumerate_eight_adjacent(pos, grid_dims):
        '''
        yields coordinates of all 8-way adjacent cells on a grid
        :param tuple pos: (Y,X) position on a grid
        :param tuple grid_dims: (Y,X) size of grid
        '''
        (y,x) = pos
        
        min_x = x - 1
        max_x = x + 1
        min_y = y - 1
        max_y = y + 1
        
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x >= grid_dims[1]:
            max_x = grid_dims[1] - 1
        if max_y >= grid_dims[0]:
            max_y = grid_dims[0] - 1
            
        for iY in range(min_y,max_y+1):
            for iX in range(min_x,max_x+1):
                if iY == y and iX == x:
                    continue
                    
                yield np.asarray((iY, iX), dtype=np.int64)
                    
    @staticmethod
    def enumerate_four_adjacent(pos, grid_dims):
        '''
        yields coordinates of all 8-way adjacent cells on a grid
        :param tuple pos: (Y,X) position on a grid
        :param tuple grid_dims: (Y,X) size of grid
        :returns: (adjacentY, adjacentX) as array
        '''
        (y,x) = pos
        
        min_x = x - 1
        max_x = x + 1
        min_y = y - 1
        max_y = y + 1
        
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x >= grid_dims[1]:
            max_x = grid_dims[1] - 1
        if max_y >= grid_dims[0]:
            max_y = grid_dims[0] - 1
            
        for iY in range(min_y,max_y+1):
            for iX in range(min_x,max_x+1):
                if (iY != y) ^ (iX != x):
                    yield np.asarray((iY, iX), dtype=np.int64)
                    
    def test_layout_properties(self):
        '''Generate a 10x10 grid of tiles with fixed positions and correct tension vectors. Ensure relax can move the tiles to approximately correct positions'''
        num_cols = 10
        num_rows = 10
        grid_dims = (num_rows, num_cols)
        layout = nornir_imageregistration.layout.Layout()
        tile_dims = np.asarray((10,10))
        
        minX = 0
        maxX = num_cols * tile_dims[1]
        minY = 0
        maxY = num_rows * tile_dims[0]
                
        pos_to_tileid = {}
        tileid_to_pos = {}
        #Tiles are size 10x10, and are randomly placed somewhere in the bounds of the grid
        iTile = 0
        for iRow in range(0,num_rows):
            for iCol in range(0,num_cols): 
                position = np.array((iRow * tile_dims[0], iCol * tile_dims[1]), dtype=np.float64)
                position = position + (tile_dims / 2.0)
                
                layout.CreateNode(iTile, position, tile_dims)
#                layout.CreateNode(iTile, np.asarray((iRow, iCol)) * tile_dims, tile_dims)
                pos_to_tileid[(iRow, iCol)] = iTile
                tileid_to_pos[iTile] = (iRow, iCol)
                iTile = iTile + 1 
                
                
        expected_center = (grid_dims * tile_dims) / 2.0
        np.testing.assert_array_almost_equal_nulp(layout.average_center, expected_center)

        #50% of the tiles are candidates to be linked, create layouts for each independent region
        create_offset = np.random.random_integers(0,1, grid_dims).astype(np.bool)
        
        nornir_imageregistration.ShowGrayscale(create_offset, title="Offset bitmask")
        
        tile_offset_dict = {}
                       
        for iRow in range(0,num_rows):
            for iCol in range(0,num_cols):
                tileID = (iRow * num_cols) + iCol
                pos = np.asarray((iRow, iCol))
                for adj in TestLayout.enumerate_four_adjacent(pos, grid_dims):
                    if adj[0] < iRow or adj[1] < iCol:
                        continue
                    
                    adjID = (adj[0] * num_cols) + adj[1]
                    
                    tile_offset_dict[(tileID, adjID)] = (adj - np.array((iRow,iCol))) * tile_dims
                    
        added_to_layout = np.zeros(grid_dims, dtype=np.bool)
        checked_mask = np.zeros(grid_dims, dtype=np.bool)
        layout_lists = []
                    
        for iRow in range(0,num_rows):
            for iCol in range(0,num_cols):
                
                if added_to_layout[iRow, iCol]:
                    continue
                
                iTile = (iRow * num_cols) + iCol
                 
                layout = nornir_imageregistration.layout.Layout()        
                layout_lists.append(layout)
                       
                for adj in TestLayout.flood_fill(create_offset, np.array((iRow, iCol), dtype=np.int), added_to_layout):
                    iAdj = (adj[0] * num_cols) + adj[1] 
                    position = adj * tile_dims
                    layout.CreateNode(int(iAdj), position, tile_dims)
        
        #Add the offsets for the nodes connected in the layout
        for layout in layout_lists:
            for node in layout.nodes.values():
                iRow = node.ID / grid_dims[1]
                iCol = node.ID % grid_dims[1]
                for adj in TestLayout.enumerate_four_adjacent((int(iRow), int(iCol)), grid_dims):
                    adj_id = (adj[0] * num_cols) + adj[1]
                    
                    if (node.ID, adj_id) in tile_offset_dict and adj_id in layout.nodes:
                        layout.SetOffset(node.ID, adj_id, tile_offset_dict[(node.ID, adj_id)], 1)
                        
        #Add one extra layout that is not connected to anything via tile_offset_dict
        extra_layout = nornir_imageregistration.layout.Layout()
        layout_lists.append(extra_layout)
        
        extra_coord = grid_dims + np.asarray((1,1))
        extra_id = int((extra_coord[0] * num_cols) + extra_coord[1])
        extra_layout.CreateNode(extra_id, extra_coord * tile_dims, tile_dims )
                        
        merged_layout = MergeDisconnectedLayoutsWithOffsets(layout_lists, tile_offset_dict)
        
        self.assertTrue(extra_id in merged_layout.nodes, "Unconnected layout is not in the merged layout")
                    
        nornir_imageregistration.views.plot_layout(merged_layout)
        
        self.assertTrue(len(merged_layout.nodes) == 100)
        #for layout in layout_lists:
        #    nornir_imageregistration.views.plot_layout(layout, ylim=(-tile_dims[0], (grid_dims[0] + 1) * tile_dims[0]), xlim=(-tile_dims[1], (grid_dims[1] + 1) * tile_dims[1]))

    @classmethod
    def flood_fill(cls, bit_mask, origin, checked_mask=None, desired_value=None):
        '''
        An iterator that yields all indicies connected to true bits of the mask
        If the origin is over a false bit, then only the origin is returned
        '''
        
        if checked_mask is None:
            checked_mask = np.zeros(bit_mask.shape, dtype=np.bool)
        
        key = tuple(origin)
        checked_mask[key] = True
        
        if desired_value is None:
            desired_value = bit_mask[key]
        
        if bit_mask[key] == desired_value:
            yield origin
        else:
            return
        
        for adj in TestLayout.enumerate_four_adjacent(origin, bit_mask.shape):
            adj_key = tuple(adj)
            if checked_mask[adj_key]:
                continue
            
            if bit_mask[adj_key] == desired_value:
                yield from cls.flood_fill(bit_mask, adj_key, checked_mask)
                 
        return
        
        
    
    def test_layout_relax_into_grid(self):
        '''Generate a 10x10 grid of tiles with random positions but correct tension vectors. Ensure relax can move the tiles to approximately correct positions'''
        num_cols = 10
        num_rows = 10
        grid_dims = (num_rows, num_cols)
        layout = nornir_imageregistration.layout.Layout()
        tile_dims = np.asarray((10,10))
        
        minX = 0
        maxX = num_cols * tile_dims[1]
        minY = 0
        maxY = num_rows * tile_dims[0]
        
        positions = np.random.rand(num_cols * num_rows, 2) * np.asarray((maxY, maxY), dtype=np.float64) 
        
        pos_to_tileid = {}
        tileid_to_pos = {}
        #Tiles are size 10x10, and are randomly placed somewhere in the bounds of the grid
        iTile = 0
        for iRow in range(0,num_rows):
            for iCol in range(0,num_cols): 
                position = positions[iTile,:]
                layout.CreateNode(iTile, position, tile_dims)
#                layout.CreateNode(iTile, np.asarray((iRow, iCol)) * tile_dims, tile_dims)
                pos_to_tileid[(iRow, iCol)] = iTile
                tileid_to_pos[iTile] = (iRow, iCol)
                iTile = iTile + 1 
                
        for iRow in range(0,num_rows):
            for iCol in range(0,num_cols):
                pos = np.asarray((iRow, iCol))
                pos_tile_id = pos_to_tileid[(iRow, iCol)]
                for adj in TestLayout.enumerate_four_adjacent(pos, grid_dims):
                    offset = (adj - pos) * tile_dims
                    adj_tile_id = pos_to_tileid[tuple(adj)]
                    
                    layout.SetOffset(pos_tile_id, adj_tile_id, offset)
                    
        #nornir_imageregistration.views.plot_layout(layout)
        
        _Relax_Layout(layout, MovieImageDir=self.TestOutputPath, 
                      max_tension_cutoff=0.1, max_iter=1000)
                    
    def test_layout_relax_into_grid_random_weight(self):
        '''Generate a 10x10 grid of tiles with random positions but correct tension vectors. Ensure relax can move the tiles to approximately correct positions'''
        num_cols = 10
        num_rows = 10
        grid_dims = (num_rows, num_cols)
        layout = nornir_imageregistration.layout.Layout()
        tile_dims = np.asarray((10,10))
        
        minX = 0
        maxX = num_cols * tile_dims[1]
        minY = 0
        maxY = num_rows * tile_dims[0]
        
        positions = np.random.rand(num_cols * num_rows, 2) * np.asarray((maxY, maxY), dtype=np.float64) 
        
        pos_to_tileid = {}
        tileid_to_pos = {}
        #Tiles are size 10x10, and are randomly placed somewhere in the bounds of the grid
        iTile = 0
        for iRow in range(0,num_rows):
            for iCol in range(0,num_cols): 
                position = positions[iTile,:]
                layout.CreateNode(iTile, position, tile_dims)
#                layout.CreateNode(iTile, np.asarray((iRow, iCol)) * tile_dims, tile_dims)
                pos_to_tileid[(iRow, iCol)] = iTile
                tileid_to_pos[iTile] = (iRow, iCol)
                iTile = iTile + 1 
                
        for iRow in range(0,num_rows):
            for iCol in range(0,num_cols):
                pos = np.asarray((iRow, iCol))
                pos_tile_id = pos_to_tileid[(iRow, iCol)]
                for adj in TestLayout.enumerate_four_adjacent(pos, grid_dims):
                    offset = (adj - pos) * tile_dims
                    adj_tile_id = pos_to_tileid[tuple(adj)]
                    
                    layout.SetOffset(pos_tile_id, adj_tile_id, offset, weight=np.random.rand())
                    
        nornir_imageregistration.views.plot_layout(layout)
        
        _Relax_Layout(layout, MovieImageDir=self.TestOutputPath, 
                      max_tension_cutoff=0.1, max_iter=1000)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
