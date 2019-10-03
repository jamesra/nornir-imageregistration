import logging
import os

import collections
import nornir_imageregistration.tile

import nornir_imageregistration.transforms.factory as tfactory
import nornir_pools
import numpy as np

from . import alignment_record
from . import core
from . import spatial 
from operator import itemgetter

ID_Value = collections.namedtuple('ID_Magnitude', ['ID', 'Value'])

def _sort_array_on_column(a, iCol, ascending=False):
    '''Sort the numpy array on the specfied column'''
    
    iSorted = np.argsort(a[:, iCol], 0)
    if not ascending:
        iSorted = np.flipud(iSorted)
    return a[iSorted, :]


def CreatePairID(A,B=None):
    '''
    :return: A tuple where the lowest ID number is in the first position and IDs are cast to integers
    '''
    
    if isinstance(A, collections.Iterable) and B is None:
        B = A[1]
        A = A[0]
        
    A_ID = None
    if isinstance(A, LayoutPosition):
        A_ID = A.ID
    else:
        A_ID = A
        
    if isinstance(B, LayoutPosition):
        B_ID = B.ID
    else:
        B_ID = B
        
    A_ID = int(A_ID)
    B_ID = int(B_ID)
        
    if A_ID < B_ID:
        return (A_ID, B_ID)
    else:
        return (B_ID,A_ID)


class LayoutPosition(object):
    '''This is an anchor with a number of springs of a certain length attached.  In our use the anchor is a tile and the spring size
       and strength is determined by the offset to overlap an adjacent tile
       
       Offsets is a numpy array of the form [[ID Y X Weight]]
    '''
    
    iOffsetID = 0
    iOffsetY = 1 
    iOffsetX = 2 
    iOffsetWeight = 3
    
    # offset_dtype = np.dtype([('ID', np.int32), ('Y', np.float32), ('X', np.float32), ('Weight', np.float32)])
         
    @property
    def ID(self):
        return self._ID
    
    @property
    def Position(self):
        '''Our position in the layout'''
        return self._position
    
    @property
    def OffsetArray(self):
        '''Read-only use please'''
        return self._OffsetArray
    
    @property
    def IsIsolated(self):
        '''Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no offsets'''
        return len(self._OffsetArray) == 0
    
    @Position.setter
    def Position(self, value):
        '''Our position in the layout'''
        if not isinstance(value, np.ndarray):
            self._position = np.array(value, dtype=np.float64)
        else:
            self._position = value.astype(np.float64)
        
        assert(self._position.ndim == 1)
        return 
        
    @property
    def ConnectedIDs(self):
        return self._OffsetArray[:, LayoutPosition.iOffsetID].astype(np.int)
    
    @property
    def dims(self):
        return self._dims
    
    def GetOffset(self, ID):
        iKnown = self.ConnectedIDs == ID
        return self.OffsetArray[iKnown, LayoutPosition.iOffsetY:LayoutPosition.iOffsetX + 1].flatten()
    
    def ContainsOffset(self, ID):
        iKnown = self.ConnectedIDs == ID
        return np.any(iKnown)
    
    def GetWeight(self, ID):
        iKnown = self.ConnectedIDs == ID
        return self.OffsetArray[iKnown, LayoutPosition.iOffsetWeight]
    
    def SetOffset(self, ID, offset, weight):
        '''Set the offset for the specified Layout position ID.  
           This means that when we subtract our position from the other ID's position we hope to obtain this offset value. 
        '''
         
        if np.isnan(weight):
            raise ValueError("weight is not a number")
        
        new_row = np.array((ID, offset[0], offset[1], weight))  # , dtype=LayoutPosition.offset_dtype, ndmin=2)
        iKnown = self.ConnectedIDs == ID
        if np.any(iKnown):
            # Update a row
            self._OffsetArray[iKnown] = new_row            
        else:
            # Insert a new row
            
            self._OffsetArray = np.vstack((self._OffsetArray, new_row))
            if self._OffsetArray.ndim == 1:
                self._OffsetArray = np.reshape(self._OffsetArray, (1, self._OffsetArray.shape[0]))
            else:
                self._OffsetArray = _sort_array_on_column(self._OffsetArray, 0, ascending=True)
        return
    
    def RemoveOffset(self, ID):
        '''
        Remove the offset to the other tile entirely
        '''
        
        iKnown = self.ConnectedIDs == ID
        if np.any(iKnown):
            self._OffsetArray = self._OffsetArray[iKnown == False,:]
        
        Warning('Removing non-existent offset: {0}->{1}'.format(self.ID, ID))
        return 
    
    def get_row_indicies(self, connected_nodes=None):
        '''
        Given a set of connected nodes, return the index into our _OffsetArray
        :return: A numpy array of row indicies
        '''
        if connected_nodes is None:
            iRows = np.array(range(0,len(self.ConnectedIDs)), dtype=np.int)
        else:
            connected_IDs = [n.ID for n in connected_nodes]
            iRows = nornir_imageregistration.IndexOfValues(self.ConnectedIDs, connected_IDs)
            return iRows
    
    def TensionVectors(self, connected_nodes=None):
        '''The difference between the current connected_positions and the expected positions based on our offsets
        :param ndarray connected_positions: [ID Y X] Position of the connected nodes'''
        if len(connected_nodes) == 0:
            return np.zeros((1,2), dtype=np.float64)
        
        connected_positions = np.vstack([n.Position for n in connected_nodes])
        relative_connected_positions = connected_positions - self.Position
        iRows = self.get_row_indicies(connected_nodes)
        
        return relative_connected_positions - self._OffsetArray[iRows, LayoutPosition.iOffsetY:LayoutPosition.iOffsetX + 1]
    
    def NetTensionVector(self, connected_nodes):
        position_difference = self.TensionVectors(connected_nodes)
        return np.sum(position_difference, 0)
      
    def WeightedNetTensionVector(self, connected_nodes):
        '''The direction of the vector this tile wants to move after summing all of the offsets
        :param ndarray connected_positions: Position of the connected nodes'''
        if len(connected_nodes) == 0:
            return np.zeros((1,2), dtype=np.float64)
        
        position_difference = self.TensionVectors(connected_nodes)
 
        # Cannot weight more than 1.0
        # normalized_weight = self._OffsetArray[:,LayoutPosition.iOffsetWeight] / np.max(self._OffsetArray[:,LayoutPosition.iOffsetWeight])
        iRows = self.get_row_indicies(connected_nodes)
        weights = self._OffsetArray[iRows, LayoutPosition.iOffsetWeight]
        total_weight = np.sum(weights)
        if total_weight != 0:
            normalized_weight = weights / total_weight
        else:
            normalized_weight = weights
            
        assert(np.all(weights >= 0))
        assert(np.all(weights <= 1.0))
        #assert(np.sum(normalized_weight) == 1.0)
        weighted_position_difference = position_difference * normalized_weight.reshape((normalized_weight.shape[0], 1))
        
        return np.sum(weighted_position_difference, 0)
    
    def MaxTensionVector(self, connected_nodes):
        '''
        The largest tension vector
        :return: tuple of (ID, magnitude) of the largest tension vector
        '''
        if len(connected_nodes) == 0:
            return ID_Value(None, np.array((0,0)))
        
        position_difference = self.TensionVectors(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference ** 2, 1))
        i_max_tension = magnitudes.argmax()
        return ID_Value(self.OffsetArray[i_max_tension,self.iOffsetID], position_difference[i_max_tension,:])
    
    def MinTensionVector(self, connected_nodes):
        '''
        The smallest tension vector
        :return: tuple of (ID, magnitude) of the smallest tension vector
        '''
        if len(connected_nodes) == 0:
            return ID_Value(None, np.array((0,0)))
        
        position_difference = self.TensionVectors(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference ** 2, 1))
        i_min_tension = magnitudes.argmin()
        return ID_Value(self.OffsetArray[i_min_tension,self.iOffsetID], position_difference[i_min_tension,:])
    
    def MaxTensionMagnitude(self, connected_nodes):
        '''
        The largest tension vector
        :return: tuple of (ID, magnitude) of the largest tension vector
        '''
        if len(connected_nodes) == 0:
            return ID_Value(None, 0)
        
        position_difference = self.MaxTensionVector(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference ** 2, 1))
        i_max_tension = magnitudes.argmax()
        return ID_Value(self.OffsetArray[i_max_tension,self.iOffsetID], magnitudes[i_max_tension])
    
    def MinTensionMagnitude(self, connected_nodes):
        '''
        The smallest tension vector
        :return: tuple of (ID, magnitude) of the smallest tension vector
        '''
        if len(connected_nodes) == 0:
            return ID_Value(None, 0)
        
        position_difference = self.TensionVectors(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference ** 2, 1))
        i_min_tension = magnitudes.argmin()
        return ID_Value(self.OffsetArray[i_min_tension,self.iOffsetID], magnitudes[i_min_tension])
    
    def ScaleOffsetWeightsByPosition(self, connected_nodes):
        '''
        Reweight our set of weights based on how far from this expectation our offsets are.  THis is useful if we believe our initial positions are largely accurate but
        our calculated desired offsets may have errors.
        :param ndarray connected_nodes: The locations we believe our connected positions should be.  
        '''
        position_difference = self.TensionVectors(connected_nodes)
        distance = np.sqrt(np.sum(position_difference ** 2, 1))
        medianDistance = np.median(distance)
        
        new_weight = distance / medianDistance
        
        raise NotImplementedError("Update this to take LayoutPosition List as argument")
        self._OffsetArray[:, LayoutPosition.iOffsetWeight] = new_weight
            
        return 
     
    def __init__(self, ID, position, dims=None, *args, **kwargs):
        '''
        :param int ID: ID number
        :param tuple position: Center position (Y,X)
        :param tuple dims: Dimensions of node (Y,X) 
        '''
        if not isinstance(ID, int):
            raise TypeError("Node ID must be an integer: {0}".format(ID))
        
        self._ID = ID 
        self.Position =position
        self._OffsetArray = np.empty((0, 4), dtype=np.float64)  # dtype=LayoutPosition.offset_dtype)
        self._dims = dims
        
    def __eq__(self, other):
        if isinstance(other, LayoutPosition):
            return self._ID == other.ID # change that to your needs 
        
        return False
    
    def __ne__(self, other):
        if isinstance(other, LayoutPosition):
            return self._ID != other.ID # change that to your needs 
        
        return True
    
    def __hash__(self):
        return self._ID
        
    def copy(self):
        ''':return: A copy of the object'''
        c = LayoutPosition(self._ID, 
                              position=self.Position.copy())
        c._OffsetArray = self._OffsetArray.copy()
        return c
        
    def _str_(self):
        return "%d y:%g x:%g" % (self._ID, self.Position[0], self.Position[1]) 
        

class Layout(object):
    ''' Records the optimal offset from each tile in a mosaic tile to overlapping tiles. 
        IDs of nodes should be incremental and match the row index of the array.'''
    
    # Offsets into node position array
    iNodeID = 0
    iNodeY = 1 
    iNodeX = 2 
    
    @classmethod 
    def _parameter_to_offset_IDs(cls, param):
        '''
        :param param: Either a TileOverlap object or a tuple of node ID's.
        :return: A tuple of node ID's
        ''' 
        if isinstance(param, nornir_imageregistration.tile_overlap.TileOverlap):
            return (param.A.ID, param.B.ID)
        else:
            return param
    
    @property
    def nodes(self):
        '''
        :return: A dictionary mapping ID to LayoutPosition objects
        '''
        return self._nodes
    
    @property
    def linked_nodes(self):
        '''
        :return: A set of tuples of linked IDs, lowest ID value in the first position
        '''
        #Return the set of linked nodes
        pairs = set()
        for node in self.nodes.values():
            #pairs
            #for connected_ID in node.ConnectedIDs:
            node_pairs = [tuple(sorted([node.ID, connected_ID])) for connected_ID in node.ConnectedIDs]
                #node_pairs.append(tuple(sorted([node.ID, connected_ID])))
            
            pairs = pairs.union(node_pairs)

        return pairs
    
    @property
    def average_center(self):
        '''
        :return: The average of the center positions of all tiles in the layout
        '''
        
        centers = [n.Position for n in self.nodes.values()]
        centers_stacked = np.vstack(centers)
        avg_center = np.average(centers_stacked, axis=0)
        return avg_center

    @property    
    def MaxWeightedNetTensionMagnitude(self):
        '''Returns the (ID, Magnitude) of the node with the largest weighted net tension vector.'''
        net_tension_vectors = self.WeightedNetTensionVectors()
        tension_magnitude = core.array_distance(net_tension_vectors[:,1:]) 
        i_max = np.argmax(tension_magnitude)
        return ID_Value(net_tension_vectors[i_max,0], tension_magnitude[i_max])
        #return np.max(core.array_distance(net_tension_vectors))
    
    
    @property    
    def MaxNetTensionMagnitude(self):
        '''Returns the (ID, Magnitude) of the node with the largest net tension vector.'''
        net_tension_vectors = self.NetTensionVectors()
        tension_magnitude = core.array_distance(net_tension_vectors[:,1:]) 
        i_max = np.argmax(tension_magnitude)
        return ID_Value(net_tension_vectors[i_max,0], tension_magnitude[i_max])
    
    @property    
    def MaxTensionMagnitude(self):
        '''
        The largest single tension between any two nodes in the layout
        :return: An array of (A,B,Magnitude) where A,B are IDs
        '''
        
        tension_vectors = self.MaxTensionVectors
        tension_magnitude = core.array_distance(tension_vectors[:,2:4]) 
        if len(tension_magnitude) == 0:
            return None
        
        i_max = tension_magnitude.argmax()
        return ID_Value(CreatePairID(tension_vectors[i_max,0:2]), tension_magnitude[i_max])
    
    @property    
    def MinTensionMagnitude(self):
        '''
        The smallest tension between any two nodes in the layout
        :return: A tuple of (A,B,Magnitude) where A,B are IDs
        '''
        
        tension_vectors = self.MinTensionVectors
        tension_magnitude = core.array_distance(tension_vectors[:,2:4])
        if len(tension_magnitude) == 0:
            return None
         
        i_min = tension_magnitude.argmin()
        return ID_Value(CreatePairID(tension_vectors[i_min,0:2]), tension_magnitude[i_min])
    
    
    def __str__(self):
        return "Layout {0} nodes {1} Connections {2}".format(self.ID, len(self.nodes.keys()), len(self.linked_nodes))
    
    def Contains(self, ID):
        '''
        :rtype: bool
        :return: True if layout contains the ID
        '''
        return ID in self._nodes.keys()
    
    def SetOffset(self, A_ID, B_ID, offset, weight=1.0):
        '''
        Specify the expected offset between two nodes in the spring model.
        '''
        A = self.nodes[A_ID]
        B = self.nodes[B_ID]
        A.SetOffset(B.ID, offset, weight)
        B.SetOffset(A.ID, -offset, weight)
        
    def ContainsOffset(self, overlap):
        ''':return: True if the layout has an offset between the two nodes'''
        (A_ID, B_ID) = Layout._parameter_to_offset_IDs(overlap)
        
        if not (self.Contains(A_ID) and self.Contains(B_ID)):
            return False
        
        A = self.nodes[A_ID]
        B = self.nodes[B_ID]
        
        return A.ContainsOffset(B_ID) and B.ContainsOffset(A_ID)
        
    def RemoveOverlap(self, overlap):
        (A_ID, B_ID) = Layout._parameter_to_offset_IDs(overlap)
        
        if self.Contains(A_ID) and self.Contains(B_ID):
            A = self.nodes[A_ID]
            B = self.nodes[B_ID]
            A.RemoveOffset(B.ID)
            B.RemoveOffset(A.ID)
        return
    
    def RemoveNode(self, node_ID):
        if node_ID in self.nodes:
            node = self.nodes[node_ID]
            
            for connected_ID in node.ConnectedIDs:
                self.RemoveOverlap(node_ID, connected_ID)
                
            del self.nodes[node_ID]
            
            return True
        
        return False
        
    def GetPosition(self, ID):
        '''Return the position array for a set of nodes, sorted by node ID'''
        return self.nodes[ID].Position
        
    def GetPositions(self, IDs=None):
        '''Return the position array for a set of nodes, sorted by node ID'''
        
        if IDs is None:
            IDs = list(self.nodes.keys())
            IDs.sort()
            
        if isinstance(IDs, int):
            IDs = [IDs]  
            
        positions = np.empty((len(IDs), 2))
        for i, tileID in enumerate(IDs):
            positions[i, :] = self.nodes[tileID].Position 
                                         
        return positions
    
    def GetNodes(self, IDs=None):
        '''Return the sorted subset of nodes by IDs as a list'''
        
        if IDs is None:
            IDs = list(self.nodes.keys())
            IDs.sort()
            
        if isinstance(IDs, int):
            IDs = [IDs]  
            
        nodes = [None] * len(IDs)
        for i, tileID in enumerate(IDs):
            nodes[i] = self.nodes[tileID] 
                                         
        return nodes
    
    def GetOffsetWeightExtrema(self):
        '''
        :return: A tuple with the (min,max) weight values of offsets in the layout
        '''
        
        maxWeight = np.NaN
        minWeight = np.NaN
        
        first = True
        for node in self._nodes.values():
            # Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no scores
            if node.IsIsolated:
                continue
            
            weights = node.OffsetArray[:, LayoutPosition.iOffsetWeight]
            
            if first:
                first = False
                minWeight = np.min(weights)
                maxWeight = np.max(weights)
            else:
                minWeight = min((minWeight, np.min(weights)))
                maxWeight = max((maxWeight, np.max(weights)))
                
        return (minWeight, maxWeight)
    
    def NetTensionVector(self, ID):
        '''Return the net tension vector of the specified ID'''
        
        node = self.nodes[ID]
        linked_nodes = self.GetNodes(node.ConnectedIDs)
        
        return node.NetTensionVector(linked_nodes)
    
    def NetTensionVectors(self):
        '''Return all net tension vectors for our nodes'''
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 3))
        for (i, ID) in enumerate(IDs):
            output[i, 0] = ID
            output[i, 1:] = self.NetTensionVector(ID)
            
        return output
    
    def PairTensionVector(self, A, B):
        '''Return the tension vector between A and B
        :return: The ideal offset between A and B
        '''
        
        pair = CreatePairID(A,B)
        node = self.nodes[pair[0]]
        linked_nodes = self.GetNodes(pair[1])
        
        return node.NetTensionVector(linked_nodes)
    
#     def PairTensionMagnitude(self, A, B):
#         '''Return the tension vector between A and B
#         :return: The ideal offset between A and B
#         '''
#         
#         pair = CreatePairID(A,B)
#         node = self.nodes[pair[0]]
#         linked_nodes = self.GetNodes(pair[1])
#         
#         net = node.NetTensionVector(linked_nodes)
        
    
    
    def WeightedNetTensionVector(self, ID):
        '''Return the net tension vector of the specified ID'''
        
        node = self.nodes[ID]
        linked_node_positions = self.GetNodes(node.ConnectedIDs)
        
        return node.WeightedNetTensionVector(linked_node_positions)
        
    def WeightedNetTensionVectors(self):
        '''Return all net tension vectors for our nodes'''
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 3))
        for i, TileID in enumerate(IDs):
            output[i, 0] = TileID
            output[i, 1:] = self.WeightedNetTensionVector(TileID)
            
        return output
    
    @property
    def MaxTensionVectors(self):
        '''
        Return the maximum tension vector for each node
        '''
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 4))
        i = 0
        for ID in IDs:
            node = self.nodes[ID]
            node_max = node.MaxTensionVector(self.GetNodes(node.ConnectedIDs))
            if node_max[0] is None:
                continue
            
            pair = CreatePairID(ID, node_max[0])
            output[i,:] = (pair[0], pair[1], node_max[1][0], node_max[1][1])
            i = i + 1
            
        return np.array(output[0:i,:])
    
    @property
    def MinTensionVectors(self):
        '''
        Return the minimum tension vector for each node
        '''
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 4))
        i = 0
        for ID in IDs:
            node = self.nodes[ID]
            node_min = node.MinTensionVector(self.GetNodes(node.ConnectedIDs))
            if node_min[0] is None:
                continue
            
            pair = CreatePairID(ID, node_min[0])
            output[i,:] = (pair[0], pair[1], node_min[1][0], node_min[1][1])
            i = i + 1
            
        return np.array(output[0:i,:])
    
    def _nextID(self):
        '''Generate the next ID number for a position'''
        return self._nodepositions.shape[0]
        
    def CreateNode(self, ID, position, dims=None):
          
        assert(not ID in self.nodes)
        node = LayoutPosition(ID, position, dims)
        self.nodes[ID] = node
        return
        
    def CreateOffsetNode(self, Existing_ID, New_ID, scaled_offset, Weight):
        '''Add a new position to the layout.  Place the new relative to the specified existing position plus an offset'''
        
        new_position = self.GetPosition(Existing_ID) + scaled_offset
        self.CreateNode(New_ID, new_position)
        self.SetOffset(Existing_ID, New_ID, scaled_offset, Weight)
        return 
    
    NextLayoutID = 0
    def __init__(self):
        
        self.ID = Layout.NextLayoutID
        Layout.NextLayoutID = Layout.NextLayoutID + 1
        self._nodes = {}
        return
        
    def copy(self):
        c = Layout()
        c._nodes = {n.ID: n.copy() for n in self._nodes.values()}
        return c
    
    def Translate(self, vector):
        '''Move all nodes by offset'''
        for node in self.nodes.values():
            node.Position = node.Position + vector
            
        return
    
    def Merge(self, layoutB):
        '''Merge layout directly into our layout'''
        self.nodes.update(layoutB.copy().nodes)
    
    @classmethod
    def RelaxNodes(cls, layout_obj, vector_scalar=None):
        '''Adjust the position of each node along its tension vector
        :param Layout layout_obj: The layout to relax
        :param float vector_scalar: Multiply the weighted tension vectors by this amount before adjusting the position.  A high value is faster but may not be constrained.  A low value is slower but safe.
        :return: nx2 array of tile movement distance
        '''
        
        # TODO: Get rid of vector scalar.  Instead calculate the net tension vector at the new position.  Then add them and apply the merged vector. 
        
        node_movement = np.zeros((len(layout_obj.nodes), 3))
        
        if vector_scalar is None:
            vector_scalar = 0.95
        
        # vectors = {}
        
        i = 0
        nodes = layout_obj.nodes.values()
        
        for (i, node) in enumerate(nodes):
            vector = layout_obj.WeightedNetTensionVector(node.ID) * vector_scalar
            # vectors[ID] = vector
            row = np.array([node.ID, vector[0], vector[1]])
            node_movement[i, :] = row
            i += 1
            
        #OK, move all of the nodes according to the net movement
        for (i,node) in enumerate(nodes): 
            # Skip the first node, the others can move around it
            node.Position = node.Position + (node_movement[i,1:])
        
        return node_movement
    
    @classmethod
    def MergeLayouts(cls, layoutA, layoutB, offset):
        '''
        Merge B with A by translating all B transforms by offset.
        Then update the dictionary of A
        '''

        layoutB.Translate(offset)
        layoutA.nodes.update(layoutB.nodes) 
        return layoutA
    
    def CreateTransform(self, ID, bounding_box):
        '''
        Create a transform for the position in the layout
        '''
        OriginalImageSize = (bounding_box[spatial.iRect.MaxY], bounding_box[spatial.iRect.MaxX])
        
        return tfactory.CreateRigidMeshTransform(target_image_shape=OriginalImageSize,
                                             source_image_shape=OriginalImageSize,
                                             rangle=0,
                                             warped_offset=self.GetPosition(ID)) 
#         return tfactory.CreateRigidTransform(target_image_shape=OriginalImageSize,
#                                              source_image_shape=OriginalImageSize,
#                                              rangle=0,
#                                              warped_offset=self.GetPosition(ID)) 
    
    def ToTransforms(self, tiles):
        '''
        Create a new set of transform for each tile in the tiles dictionary
        :param tiles: Dictionary of tile ID to tiles
        :return: sorted list of transforms for ID's found tiles
        '''
        
        transforms = []
        
        for ID in sorted(tiles.keys()):
            if not ID in self.nodes:
                continue
            
            tile = tiles[ID]
            
            transform = self.CreateTransform(self, ID, tile.MappedBoundingBox)
            transforms.append(transform)
            
        return transforms
    
    def UpdateTileTransforms(self, tiles):
        '''
        Create a new set of transform for each tile in the tiles dictionary
        :param tiles: Dictionary of tile ID to tiles
        :return: sorted list of transforms for ID's found tiles
        '''
        
        transforms = []
        
        for ID in sorted(tiles.keys()):
            if not ID in self.nodes:
                continue
            
            tile = tiles[ID]
            
            transform = self.CreateTransform(ID, tile.MappedBoundingBox)
            
            #Copy the bounding box of the image the transform maps if it is not already present
            if transform.MappedBoundingBox is None:
                transform.MappedBoundingBox = tile.Transform.MappedBoundingBox
                
            tile.Transform = transform
            
        return transforms
    
    def ToMosaic(self, tiles):
        '''
        Generate a Mosaic object from a dictionary mapping tile numbers to tile paths that can be used to create a .mosaic file 
        :param dict tiles: Maps tile ID used in layout to a Tile object
        '''
        
        mosaic = nornir_imageregistration.Mosaic()
    
        for ID in sorted(tiles.keys()):
            if not ID in self.nodes:
                continue
            
            tile = tiles[ID]
            
            transform = self.CreateTransform(ID, tile.MappedBoundingBox)
            
            #Copy the bounding box of the image the transform maps if it is not already present
            if transform.MappedBoundingBox is None:
                transform.MappedBoundingBox = tile.Transform.MappedBoundingBox
            mosaic.ImageToTransform[tile.ImagePath] = transform
    
        mosaic.TranslateToZeroOrigin()
    
        return mosaic
    
    
def OffsetsSortedByWeight(layout):
    '''
    Return all of a layouts offsets sorted by weight.  
    :return: An array [[TileA_ID, TileB_ID, OffsetY, OffsetX, Weight]] To prevent duplicates we only report offsets where TileA_ID < TileB_ID
    ''' 
    ret_array = np.empty((0, 5))
    for node in layout.nodes.values():
        if node.IsIsolated:
            continue
        
        # Prevent duplicates by skipping IDs less than the nodes
        iNewRows = node.OffsetArray[:, 0] > node.ID 
        if not np.any(iNewRows):
            continue 
        
        new_column = np.ones((np.sum(iNewRows), 1)) * node.ID
        new_rows = np.hstack((new_column, node.OffsetArray[iNewRows, :]))
        ret_array = np.vstack((ret_array, new_rows))
        
    return _sort_array_on_column(ret_array, 4)  


def ScaleOffsetWeightsByPosition(original_layout):
    
    for node in original_layout.nodes.values():
        linked_node_positions = original_layout.GetNodes(node.ConnectedIDs)
        node.ScaleOffsetWeightsByPosition(linked_node_positions)
        
    return


def NormalizeOffsetWeights(original_layout):
    '''
    Proportionally scale offset weights so the highest weight is 1.0
    '''
    
    (minWeight, maxWeight) = original_layout.GetOffsetWeightExtrema()
    
    # All the weights are equal... odd
    if maxWeight == minWeight:
        for node in original_layout.nodes.values():
            if node.IsIsolated:
                continue
            
            node.OffsetArray[:, LayoutPosition.iOffsetWeight] = 1.0
    
        return
    
    for node in original_layout.nodes.values():
        # Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no scores
        if node.IsIsolated:
            continue
        
        node.OffsetArray[:, LayoutPosition.iOffsetWeight] = node.OffsetArray[:, LayoutPosition.iOffsetWeight] / maxWeight
        assert(np.alltrue(node.OffsetArray[:, LayoutPosition.iOffsetWeight] >= 0))
        assert(np.alltrue(node.OffsetArray[:, LayoutPosition.iOffsetWeight] <= 1.0))
                
    return 


def ScaleOffsetWeightsByPopulationRank(original_layout, min_allowed_weight=0, max_allowed_weight=1.0):
    '''
    Remap offset weights so the highest weight is 1.0 and the lowest is 0
    '''
    
    if min_allowed_weight >= max_allowed_weight:
        raise ValueError("Min allowed weight must be below the max allowed weight")
    
    (minWeight, maxWeight) = original_layout.GetOffsetWeightExtrema()
    
    # All the weights are equal... odd
    if maxWeight == minWeight:
        for node in original_layout.nodes.values():
            if node.IsIsolated:
                continue
            
            node.OffsetArray[:, LayoutPosition.iOffsetWeight] = max_allowed_weight
        return
    
    maxWeight -= minWeight
    
    allowed_weight_range = max_allowed_weight - min_allowed_weight
    
    for node in original_layout.nodes.values():
        # Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no scores
        if node.IsIsolated:
            continue
        
        node.OffsetArray[:, LayoutPosition.iOffsetWeight] = (node.OffsetArray[:, LayoutPosition.iOffsetWeight] - minWeight) / maxWeight
        node.OffsetArray[:, LayoutPosition.iOffsetWeight] *= allowed_weight_range
        node.OffsetArray[:, LayoutPosition.iOffsetWeight] += min_allowed_weight
        assert(np.alltrue(node.OffsetArray[:, LayoutPosition.iOffsetWeight] >= min_allowed_weight))
        assert(np.alltrue(node.OffsetArray[:, LayoutPosition.iOffsetWeight] <= max_allowed_weight))
                
    return 

    
def RelaxLayout(layout_obj, max_tension_cutoff=None, max_iter=None, vector_scale=None, 
                plotting_output_path=None, plotting_interval=None):
    ''' 
    :param layout_obj: Layout to refine
    :param float max_tension_cutoff: Stop iteration after the maximum tension vector has a magnitude below this value
    :param int max_iter: Maximum number of iterations
    '''
     
    max_tension = layout_obj.MaxWeightedNetTensionMagnitude[1]
    
    if max_tension_cutoff is None:
        max_tension_cutoff = 0.1
        
    if max_iter is None:
        max_iter = 500
        
    if plotting_interval is None:
        plotting_interval = 10
             
    i = 0
    min_plotting_tension = max_tension_cutoff * 20
    plotting_max_tension = max(min_plotting_tension, max_tension)
    
#         MovieImageDir = os.path.join(self.TestOutputPath, "relax_movie")
#         if not os.path.exists(MovieImageDir):
#             os.makedirs(MovieImageDir)
    pool = None
    if plotting_output_path is not None:
        os.makedirs(plotting_output_path, exist_ok=True)
        pool = nornir_pools.GetGlobalMultithreadingPool()
        
    print("Relax Layout")
    
    while max_tension > max_tension_cutoff and i < max_iter:
        print("\t%d %g" % (i, max_tension))
        Layout.RelaxNodes(layout_obj, vector_scalar=vector_scale)
        max_tension = layout_obj.MaxWeightedNetTensionMagnitude[1]
        
        plotting_max_tension = max(min_plotting_tension, max_tension)
        
        if plotting_output_path is not None and (i % plotting_interval == 0 or i < 10):
            filename = os.path.join(plotting_output_path, "%d.svg" % i)
#             nornir_imageregistration.views.plot_layout( 
#                            layout_obj=layout_obj.copy(),
#                            OutputFilename=filename,
#                            max_tension=plotting_max_tension)
            pool.add_task("Plot step #%d" % (i),
                          nornir_imageregistration.views.plot_layout,
                          layout_obj=layout_obj.copy(),
                          OutputFilename=filename,
                          max_tension=plotting_max_tension)   
        
        # node_distance = setup_imagetest.array_distance(node_movement[:,1:3])             
        # max_distance = np.max(node_distance,0)
        i += 1
        
        # nornir_shared.plot.VectorField(layout_obj.GetPositions(), layout_obj.NetTensionVectors(), OutputFilename=filename)
        # pool.add_task("Plot step #%d" % (i), nornir_shared.plot.VectorField,layout_obj.GetPositions(), layout_obj.WeightedNetTensionVectors(), OutputFilename=filename)
        
    return layout_obj
        

def BuildLayoutWithHighestWeightsFirst(original_layout):
    '''
    Constructs a mosaic by sorting all of the match results according to strength. 
    
    :param dict tiles: Dictionary of tile objects containing alignment records to other tiles
    '''

    placedTiles = dict()
    
    sorted_offsets = OffsetsSortedByWeight(original_layout) 

    LayoutList = [] 
    for iRow in range(0, sorted_offsets.shape[0]):
        row = sorted_offsets[iRow, :]      
        A_ID = int(row[0])
        B_ID = int(row[1])
        YOffset = row[2]
        XOffset = row[3]
        Weight = row[4]
        offset = row[2:4]
        
        print("%d -> %d (%g,%g w: %g)" % (A_ID, B_ID, YOffset, XOffset, Weight))

        if np.isnan(Weight):
            print("Skip: Invalid weight, not a number")
            continue

        ALayout = GetLayoutForID(LayoutList, A_ID)
        BLayout = GetLayoutForID(LayoutList, B_ID)

        if ALayout is None and BLayout is None:
            new_layout = Layout()
            A_pos = original_layout.GetPosition(A_ID)
            new_layout.CreateNode(A_ID, A_pos)
            new_layout.CreateNode(B_ID, A_pos + offset)
            new_layout.SetOffset(A_ID, B_ID, offset, Weight) 
            LayoutList.append(new_layout)
            print("New layout")

        elif (not ALayout is None) and (not BLayout is None):
            # Need to merge the layouts? See if they are the same
            if ALayout == BLayout:
                # Already mapped
                if B_ID in ALayout.nodes[A_ID].ConnectedIDs: 
                    print("Skip: Already mapped")
                else:
                    ALayout.SetOffset(A_ID, B_ID, offset, Weight)
            else:
                MergeLayoutsWithNodeOffset(ALayout, BLayout, A_ID, B_ID, offset, Weight)
                print("Merged")
                LayoutList.remove(BLayout)
        else:
            
            if ALayout is None and not BLayout is  None:
                BLayout.CreateOffsetNode(B_ID, A_ID, -offset, Weight)
                # We'll pick it up on the next pass
                # print("Skip: Getting it next time")
                # continue
                
            else:
                ALayout.CreateOffsetNode(A_ID, B_ID, offset, Weight)

    # OK, we should have a single list of layouts
    #LargestLayout = LayoutList[0]

    return LayoutList

def MergeDisconnectedLayouts(layout_list):
    '''Given a list of layouts, generate a single layout with all nodes in the same positions'''
    if len(layout_list) == 1:
        return layout_list[0]
    
    merged_layout = layout_list[0].copy()
 
    for (i,layout) in enumerate(layout_list):
        if i == 0:
            continue
        
        merged_layout.Merge(layout)
        
    return merged_layout


def _generate_combinations(list_of_lists):
    '''Given a list of iterables containing integers, returns all of the pairs of numbers without pairs appearing in the same list'''
    for (iList, id_list) in enumerate(list_of_lists):
        for (iOther, other_list) in enumerate(list_of_lists):
            if iOther <= iList:
                continue 
            
            for A in id_list:
                for B in other_list:
                    if A < B:
                        yield (A,B)

def MergeDisconnectedLayoutsWithOffsets(layout_list, tile_offset_dict=None):
    '''
    Given a list of layouts, generate a single layout, if possible using the offsets in tile_offset_dict
    :param dict tile_offset_dict: Keys are (A,B) values are offsets [Y,X]
    '''
    if len(layout_list) == 1:
        return layout_list[0]
    
    if tile_offset_dict is None:
        tile_offset_dict = {}
    else:
        #Clone the dictionary so we don't change the passed parameter
        tile_offset_dict = dict(tile_offset_dict)
        
    #First, to minimize our search time, remove offsets the layouts already encode
    # and build a frozenset of IDs in the layouts
    layout_IDs = []
    tile_to_layout = {} #Track which tile belongs to which layout
    for (iLayout,layout) in enumerate(layout_list):
        layout_IDs.append( frozenset([node_id for node_id in layout.nodes.keys()]) )
        
        for link in layout.linked_nodes:
            assert(link[0] < link[1])
            if link in tile_offset_dict:
                del tile_offset_dict[link]
                
        for node_id in layout.nodes.keys():
            tile_to_layout[node_id] = iLayout
            
    #Remove the tile links who are in the same layout but not linked in the layout
    for key in list(tile_offset_dict.keys()):
        if key[0] in tile_to_layout and key[1] in tile_to_layout:
            if tile_to_layout[key[0]] == tile_to_layout[key[1]]:
                del tile_offset_dict[key]
        
    cross_layout_keys = {}

    #Identify all of the tile_offsets that can describe where the layouts are relative to each other
    for offset_key in tile_offset_dict.keys():
        try:
            iLayout_A = tile_to_layout[offset_key[0]] 
            iLayout_B = tile_to_layout[offset_key[1]]
        except KeyError:
            continue
        
        assert(iLayout_A != iLayout_B)
        
        layout_pair = (iLayout_A, iLayout_B)
        if iLayout_B < iLayout_A:
            layout_pair = (iLayout_B, iLayout_A)
        
        if iLayout_A != iLayout_B:
            if layout_pair in cross_layout_keys:
                cross_layout_keys[layout_pair].append(offset_key)
            else:
                cross_layout_keys[layout_pair] = [offset_key]
            
    #Generate a list of the layouts pairings with the greatest number of keys first, merge largest to smallest
    layout_pair_offset_count = [(key, len(cross_layout_keys[key])) for key in cross_layout_keys.keys()]
    sorted_layout_pair_offset_count = sorted(layout_pair_offset_count, key=itemgetter(1), reverse=True)
    
    #layout_centers = np.vstack([l.average_center for l in layout_list])
    
    #Merge the largest layouts to the smallest layouts until all are merged that can be merged
    while len(sorted_layout_pair_offset_count) > 0:
        layout_pair_to_merge = sorted_layout_pair_offset_count.pop(0)
        layout_pair = layout_pair_to_merge[0]
        (iLayout_A, iLayout_B) = layout_pair
        print("Layout {0} absorbing {1}".format(iLayout_A, iLayout_B))
        assert(iLayout_A != iLayout_B)
        ALayout = layout_list[iLayout_A]
        BLayout = layout_list[iLayout_B]
           
        tile_offsets = cross_layout_keys[layout_pair]
        
        A_To_B_offset_measures = np.zeros((len(tile_offsets), 2))
        
        for (iRow, offset_key) in enumerate(tile_offsets):
            
            #Using each tile offset key, average the offset between the disconnected layouts
            tile_offset = tile_offset_dict[offset_key]
            iLayout = (tile_to_layout[offset_key[0]], tile_to_layout[offset_key[1]])
            #assert(iLayout[0] != iLayout[1])
            if iLayout[0] == iLayout[1]:
                continue
            
            tile_layouts = (layout_list[iLayout[0]], layout_list[iLayout[1]])
            A_Pos = tile_layouts[0].GetPosition(offset_key[0])
            B_Pos = tile_layouts[1].GetPosition(offset_key[1])
            #Layout_A_Center = layout_centers[iLayout[0], :]
            #Layout_B_Center = layout_centers[iLayout[1], :]
            #A_To_Layout = A_Pos - Layout_A_Center
            #B_To_Layout = B_Pos - Layout_B_Center
            #A_To_B = A_To_Layout + tile_offset + B_To_Layout
            
            #A_To_B_offset_measures[iRow, :] = A_To_B
            A_To_B_offset_measures[iRow, :] = (B_Pos - A_Pos) - tile_offset
            
            #MergeLayoutsWithNodeOffset(ALayout, BLayout, offset_key[0], offset_key[1], tile_offset.offset, Weight=0)
            #print("Merged")
            
        MergeLayoutsWithAbsoluteOffset(ALayout, BLayout, np.mean(A_To_B_offset_measures, axis=0))
        
        for (iLayout, layout) in enumerate(layout_list):
            if layout.ID == BLayout.ID:
                layout_list[iLayout] = ALayout
                
        #layout_list[iLayout_B] = ALayout
        #layout_centers[iLayout_A] = ALayout.average_center
        #layout_centers[iLayout_B] = ALayout.average_center
    
    #Now we check for layouts that are completely disconnected
    distinct_IDs = set([ll.ID for ll in layout_list])
    unmerged_layouts = []
    for layout in layout_list:
        if layout.ID in distinct_IDs:
            unmerged_layouts.append(layout)
            distinct_IDs = distinct_IDs - set([layout.ID])
        
    return MergeDisconnectedLayouts(unmerged_layouts)
    
 
def GetLayoutForID(listLayouts, ID):
    '''Given a list of tile layouts, returns the layout containing the given ID'''

    if listLayouts is None:
        return None

    for layout in listLayouts:
        if layout.Contains(ID):
            return layout

    return None


def MergeLayoutsWithNodeOffset(layoutA, layoutB, NodeInA, NodeInB, offset, weight):
    '''
    Merge B with A by translating all B transforms by offset.
    Then update the dictionary of A
    '''

    PositionInA = layoutA.GetPosition(NodeInA)
    PositionInB = layoutB.GetPosition(NodeInB)
    
    ExpectedMovingTilePosition = offset + PositionInA
    MovingPositionDifference = ExpectedMovingTilePosition - PositionInB
    
    layoutB.Translate(MovingPositionDifference)
    layoutA.Merge(layoutB)
    
    layoutA.SetOffset(NodeInA, NodeInB, offset, weight)
    

def MergeLayoutsWithAbsoluteOffset(layoutA, layoutB, offset):
    '''
    Merge B with A by translating all B transforms by offset.
    Then update the dictionary of A
    '''
    
    layoutB.Translate(offset)
    layoutA.Merge(layoutB)