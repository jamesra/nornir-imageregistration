'''
Created on Mar 2, 2015

@author: u0490822
'''
import unittest
import numpy as np
import nornir_shared.plot as plot
from . import setup_imagetest
from nornir_imageregistration.layout import *


class TestLayoutPosition(setup_imagetest.TestBase):


    def _CheckLayoutPositionCreate(self, layout_position, ID, position):
        self.assertEqual(layout_position.ID, ID, "ID not set correctly")
        self.assertTrue(np.all(layout_position.Position == np.array(position)), "Position not set correctly")
        
    def SetOffset(self, A, B, offset, weight=1.0):
        A.SetOffset(B.ID, offset, weight)
        B.SetOffset(A.ID, -offset, weight)

    def test_Basics(self):
        
        p_position = (0,0)
        p = LayoutPosition(1, p_position)
        self._CheckLayoutPositionCreate(p, 1, p_position)
        
        
        p2_position = (10,10)
        p2 = LayoutPosition(2, p2_position)
        self._CheckLayoutPositionCreate(p2, 2, p2_position) 
        
        offset = np.array((5,5))
        weight = 1.0
        self.SetOffset(p, p2, offset, weight)
         
                
        return 
    
    def test_cross(self):
        '''
        Create a cross of five positions that should perfectly cancel to produce an offset vector of zero for the center
        '''
        
        sprint_layout = Layout()
        
        positions = np.array([[0,0],
                                [10,0],
                                [-10,0],
                                [0,10],
                                [0,-10]])
        
        sprint_layout.CreateNode(0, positions[0,:])
        sprint_layout.CreateNode(1, positions[1,:])
        sprint_layout.CreateNode(2, positions[2,:])
        sprint_layout.CreateNode(3, positions[3,:])
        sprint_layout.CreateNode(4, positions[4,:])
        
        sprint_layout.SetOffset(0, 1, positions[1,:])
        sprint_layout.SetOffset(0, 2, positions[2,:])
        sprint_layout.SetOffset(0, 3, positions[3,:])
        sprint_layout.SetOffset(0, 4, positions[4,:])
         
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(3) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(4) == np.array([0,0])))
    
    def test_singularity(self):
        '''
        Same as test_cross, but position all of the points at the same location
        '''
        sprint_layout = Layout()
        
        positions = np.array([[0,0],
                                [10,0],
                                [-10,0],
                                [0,10],
                                [0,-10]])
        
        sprint_layout.CreateNode(0, positions[0,:])
        sprint_layout.CreateNode(1, positions[0,:])
        sprint_layout.CreateNode(2, positions[0,:])
        sprint_layout.CreateNode(3, positions[0,:])
        sprint_layout.CreateNode(4, positions[0,:])
        
        sprint_layout.SetOffset(0, 1, positions[1,:])
        sprint_layout.SetOffset(0, 2, positions[2,:])
        sprint_layout.SetOffset(0, 3, positions[3,:])
        sprint_layout.SetOffset(0, 4, positions[4,:])
         
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array(positions[1,:])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array(positions[2,:])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(3) == np.array(positions[3,:])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(4) == np.array(positions[4,:])))
        
        #OK, try to relax the layout and see where the nodes land
        displacements = Layout.RelaxNodes(sprint_layout)
        
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(3) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(4) == np.array([0,0])))
        
        self.assertTrue(np.all(sprint_layout.GetPosition(0) == np.array(positions[0,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(1) == np.array(positions[1,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(2) == np.array(positions[2,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(3) == np.array(positions[3,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(4) == np.array(positions[4,:])))
        
        #Nothing should happen on the second pass
        displacements = Layout.RelaxNodes(sprint_layout)
        
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(3) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(4) == np.array([0,0])))
        
        self.assertTrue(np.all(sprint_layout.GetPosition(0) == np.array(positions[0,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(1) == np.array(positions[1,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(2) == np.array(positions[2,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(3) == np.array(positions[3,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(4) == np.array(positions[4,:])))
 
        print("Node Positions")
        print(sprint_layout.GetPositions())
        
    def test_weighted_line(self):
        '''
        Three points on a line with inconsistent desired offsets and weights
        '''
        sprint_layout = Layout()
        
        positions = np.array([[0,0],
                                [10,5],
                                [-10,5]])
                               
        
        sprint_layout.CreateNode(0, positions[0,:])
        sprint_layout.CreateNode(1, positions[1,:])
        sprint_layout.CreateNode(2, positions[2,:]) 
        
        sprint_layout.SetOffset(0, 1, positions[1,:], weight=1)
        sprint_layout.SetOffset(0, 2, positions[2,:], weight=1)
        sprint_layout.SetOffset(1, 2, np.array([-15,0]), weight=1)
         
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array([-5,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array([5,0])))
        
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        
        #OK, try to relax the layout and see where the nodes land
        displacements = Layout.RelaxNodes(sprint_layout)
        
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        
        displacements = Layout.RelaxNodes(sprint_layout)
         
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        #Nothing should happen on the second pass
        displacements = Layout.RelaxNodes(sprint_layout)
        
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(3) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(4) == np.array([0,0])))
        
        self.assertTrue(np.all(sprint_layout.GetPosition(0) == np.array(positions[0,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(1) == np.array(positions[1,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(2) == np.array(positions[2,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(3) == np.array(positions[3,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(4) == np.array(positions[4,:])))
 
        print("Node Positions")
        print(sprint_layout.GetPositions())
        
    def test_weighted_line(self):
        '''
        Three points on a line with inconsistent desired offsets and weights
        '''
        sprint_layout = Layout()
        
        positions = np.array([[0,0],
                                [10,5],
                                [-10,5]])
                               
        
        sprint_layout.CreateNode(0, positions[0,:])
        sprint_layout.CreateNode(1, positions[1,:])
        sprint_layout.CreateNode(2, positions[2,:]) 
        
        sprint_layout.SetOffset(0, 1, positions[1,:], weight=1)
        sprint_layout.SetOffset(0, 2, positions[2,:], weight=1)
        sprint_layout.SetOffset(1, 2, np.array([-15,0]), weight=1)
         
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array([-5,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array([5,0])))
        
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        
        #OK, try to relax the layout and see where the nodes land
        displacements = Layout.RelaxNodes(sprint_layout)
        
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        
        displacements = Layout.RelaxNodes(sprint_layout)
         
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        #Nothing should happen on the second pass
        displacements = Layout.RelaxNodes(sprint_layout)
        
        plot.VectorField(sprint_layout.GetPositions(), sprint_layout.NetTensionVectors())
        
        self.assertTrue(np.all(sprint_layout.NetTensionVector(0) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(1) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(2) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(3) == np.array([0,0])))
        self.assertTrue(np.all(sprint_layout.NetTensionVector(4) == np.array([0,0])))
        
        self.assertTrue(np.all(sprint_layout.GetPosition(0) == np.array(positions[0,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(1) == np.array(positions[1,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(2) == np.array(positions[2,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(3) == np.array(positions[3,:])))
        self.assertTrue(np.all(sprint_layout.GetPosition(4) == np.array(positions[4,:])))
 
        print("Node Positions")
        print(sprint_layout.GetPositions())
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()