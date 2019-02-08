'''
Created on Mar 2, 2015

@author: u0490822
'''
import unittest

from nornir_imageregistration.layout import *

import nornir_pools
import nornir_shared.plot
import nornir_shared.plot as plot
import numpy as np
import os

from . import setup_imagetest
from . import test_arrange


class TestLayoutPosition(setup_imagetest.TestBase):


    def _CheckLayoutPositionCreate(self, layout_position, ID, position):
        self.assertEqual(layout_position.ID, ID, "ID not set correctly")
        self.assertTrue(np.all(layout_position.Position == np.array(position)), "Position not set correctly")
        
    def SetOffset(self, A, B, offset, weight=1.0):
        A.SetOffset(B.ID, offset, weight)
        B.SetOffset(A.ID, -offset, weight)

    def test_Basics(self):
        
        p_position = (0, 0)
        p = LayoutPosition(1, p_position)
        self._CheckLayoutPositionCreate(p, 1, p_position)
        
        
        p2_position = (10, 10)
        p2 = LayoutPosition(2, p2_position)
        self._CheckLayoutPositionCreate(p2, 2, p2_position) 
        
        offset = np.array((5, 5))
        weight = 1.0
        self.SetOffset(p, p2, offset, weight)
         
                
        return 
    
    def test_cross(self):
        '''
        Create a cross of five positions that should perfectly cancel to produce an offset vector of zero for the center
        '''
        
        spring_layout = Layout()
        
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
         
        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(3) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(4) == np.array([0, 0])))
    
    def test_singularity(self):
        '''
        Same as test_cross, but position all of the points at the same location
        '''
        spring_layout = Layout()
        
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
        displacements = Layout.RelaxNodes(spring_layout, vector_scalar=1.0)
        
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
        displacements = Layout.RelaxNodes(spring_layout)
        
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
        
    def _MaxTension(self, layout):
        
        net_tension_vectors = layout.WeightedNetTensionVectors()
        return np.max(setup_imagetest.array_distance(net_tension_vectors))
        
    
    def _Relax_Layout(self, layout_obj, max_tension_cutoff=0.5, max_iter=100):
                
        max_tension = self._MaxTension(layout_obj)
         
        i = 0
        
        pool = nornir_pools.GetGlobalMultithreadingPool()
        
        MovieImageDir = os.path.join(self.TestOutputPath, "relax_movie")
        if not os.path.exists(MovieImageDir):
            os.makedirs(MovieImageDir)
            
        filename = os.path.join(MovieImageDir, "%d.png" % i) 
        pool.add_task("Plot step #%d" % (i), nornir_shared.plot.VectorField, layout_obj.GetPositions(), layout_obj.NetTensionVectors(), filename)
            
        while max_tension > max_tension_cutoff and i < max_iter:
            print("%d %g" % (i, max_tension))
            node_movement = nornir_imageregistration.layout.Layout.RelaxNodes(layout_obj)
            max_tension = self._MaxTension(layout_obj)
            # node_distance = setup_imagetest.array_distance(node_movement[:,1:3])             
            # max_distance = np.max(node_distance,0)
            i += 1
            
            filename = os.path.join(MovieImageDir, "%d.png" % i)
            
            pool.add_task("Plot step #%d" % (i), nornir_shared.plot.VectorField, layout_obj.GetPositions(), layout_obj.NetTensionVectors(), filename)
            # nornir_shared.plot.VectorField(layout_obj.GetPositions(), layout_obj.NetTensionVectors(), filename)
            
        return layout_obj
    
    def test_weighted_line(self):
        '''
        Three points on a line
        '''
        spring_layout = Layout()
        
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
        self._Relax_Layout(spring_layout, max_tension_cutoff=max_vector_magnitude, max_iter=100) 
        
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
        spring_layout = Layout()

        positions = np.array([[0, 0],
                                [10, 5],
                                [-10, 5]])


        spring_layout.CreateNode(0, positions[0, :])
        spring_layout.CreateNode(1, positions[1, :])
        spring_layout.CreateNode(2, positions[2, :]) 

        spring_layout.SetOffset(0, 1, positions[1, :], weight=1)
        spring_layout.SetOffset(0, 2, positions[2, :], weight=1)
        spring_layout.SetOffset(1, 2, np.array([-15, 0]), weight=1)

        self.assertTrue(np.all(spring_layout.NetTensionVector(0) == np.array([0, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(1) == np.array([-5, 0])))
        self.assertTrue(np.all(spring_layout.NetTensionVector(2) == np.array([5, 0])))

        # OK, try to relax the layout and see where the nodes land
        max_vector_magnitude = 0.05
        self._Relax_Layout(spring_layout, max_tension_cutoff=max_vector_magnitude, max_iter=100) 

        for ID in spring_layout.nodes.keys():
            self.assertTrue(setup_imagetest.array_distance(spring_layout.NetTensionVector(ID)) < max_vector_magnitude, "Node %d should have net tension vector below relax cutoff")

        self.assertTrue(np.allclose(spring_layout.GetPosition(0) - spring_layout.GetPosition(1), positions[1, :], atol=max_vector_magnitude))
        self.assertTrue(np.allclose(spring_layout.GetPosition(0) - spring_layout.GetPosition(2), positions[2, :], atol=max_vector_magnitude))

        print("Node Positions")
        print(spring_layout.GetPositions())

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
