'''
Created on Oct 22, 2018

@author: u0490822

Contains some helper classes for organizing a grid of points placed over an image. 
Used with the slice-to-slice grid refinement code
'''

import numpy as np
import nornir_imageregistration


class GridDivisionBase(object):
    '''Abstract class for structures that divide images into grids of possibly overlapping cells'''
    
    def __init__(self):
        self.cell_size = None
        self.grid_dims = None
        self.grid_spacing = None
        self.coords = None
        self.FixedPoints = None
        self.WarpedPoints = None
        self.fixed_shape = None
        
    def PopulateWarpedPoints(self, transform):
        if transform is not None:
            self.WarpedPoints = transform.InverseTransform(self.FixedPoints).astype(np.int64)
            return self.WarpedPoints 
        
    def RemoveMaskedPoints(self, mask):
        '''
        :param array mask: a boolean mask that determines which points are kept. 
        '''
        
        self.coords = self.coords[mask,:]
        self.FixedPoints = self.FixedPoints[mask,:]
        
        if self.WarpedPoints is not None:
            self.WarpedPoints = self.WarpedPoints[mask,:]
        
    def ApplyFixedImageMask(self, fixed_mask):
        if fixed_mask is not None:
            self.FilterOutofBoundsFixedPoints(fixed_mask.shape)
            valid = nornir_imageregistration.index_with_array(fixed_mask, self.FixedPoints)
            
            self.RemoveMaskedPoints(valid)
            
    def RemoveCellsUsingFixedImageMask(self, fixed_mask, min_overlap=0.5):
        '''
        :param float min_overlap: Amount of cell area that must be valid according to mask
        '''
        if fixed_mask is not None:
            
            cell_true_count = np.asarray([False] * self.FixedPoints.shape[0], dtype=np.float64)
            half_cell = (self.cell_size / 2.0).astype(np.int32)
            cell_area = np.prod(self.cell_size)
            
            origins = self.FixedPoints - half_cell
            
            for iRow in range(0,self.FixedPoints.shape[0]):
                o = origins[iRow,:]
                
                cell = nornir_imageregistration.CropImage(fixed_mask,
                                                          int(o[1]), int(o[0]),
                                                          int(self.cell_size[1]), int(self.cell_size[0]),
                                                          cval=False)
                cell_true_count[iRow] = np.count_nonzero(cell)
                
            overlaps = cell_true_count / float(cell_area)
            valid = overlaps > min_overlap
            self.RemoveMaskedPoints(valid)
            
    def ApplyWarpedImageMask(self, warped_mask):
        if warped_mask is not None:
            self.FilterOutofBoundsWarpedPoints(warped_mask.shape)
            valid = nornir_imageregistration.index_with_array(warped_mask, self.WarpedPoints)
            self.RemoveMaskedPoints(valid)
            
    def RemoveCellsUsingWarpedImageMask(self, warped_mask, min_overlap=0.5):
        '''
        :param float min_overlap: Amount of cell area that must be valid according to mask
        '''
        if warped_mask is not None:
            
            cell_true_count = np.asarray([False] * self.WarpedPoints.shape[0], dtype=np.float64)
            half_cell = (self.cell_size / 2.0).astype(np.int32)
            cell_area = np.prod(self.cell_size)
            
            origins = self.WarpedPoints - half_cell
            
            for iRow in range(0,self.WarpedPoints.shape[0]):
                o = origins[iRow,:]
                
                cell = nornir_imageregistration.CropImage(warped_mask,
                                                          int(o[1]), int(o[0]),
                                                          int(self.cell_size[1]), int(self.cell_size[0]),
                                                          cval=False)
                cell_true_count[iRow] = np.count_nonzero(cell)
                
            overlaps = cell_true_count / float(cell_area)
            valid = overlaps > min_overlap
            self.RemoveMaskedPoints(valid)
        
    def FilterOutofBoundsFixedPoints(self, fixed_shape=None):
        
        if fixed_shape is None:
            fixed_shape = self.fixed_shape
        
        valid_inbounds = np.logical_and(np.all(self.FixedPoints >= np.asarray((0, 0)), 1),
                                        np.all(self.FixedPoints < fixed_shape, 1))
        self.RemoveMaskedPoints(valid_inbounds)
        
    def FilterOutofBoundsWarpedPoints(self, warped_shape):
        valid_inbounds = np.logical_and(np.all(self.WarpedPoints >= np.asarray((0, 0)), 1),
                                        np.all(self.WarpedPoints < warped_shape, 1))
        self.RemoveMaskedPoints(valid_inbounds)

class ITKGridDivision(GridDivisionBase): 
    '''
     Align the grid so the centers of the edge cells touch the edge of the image
    '''

    def __init__(self, fixed_shape, cell_size, grid_dims=None, grid_spacing=None, transform = None):
        '''
        Divides an image into a grid, of possibly overlapping cells.
        
        :param tuple fixed_shape: (Rows, Columns) of image we are dividing 
        :param tuple cell_size: The dimensions of each grid cell
        :param tuple grid_dims: The number of (Rows, Columns) in the grid
        :param tuple grid_spacing: The distance between the centers of grid cells, possibly allowing overlapping cells
        
        '''
        
        self.cell_size = np.asarray(cell_size, np.int32)
        fixed_shape = np.asarray(fixed_shape, np.int64)
        
        if cell_size is None:
            raise ValueError("cell_size must be specified")
        
        if grid_dims is not None and grid_spacing is not None:
            raise ValueError("Either grid_dims or grid_spacing must be specified but not both")
        
        #We want the coordinates of grid centers to go from edge to edge in the image because ITK expects this behavior
        #Due to this fact we do not guarantee the grid_spacing requested        
        if grid_dims is None and grid_spacing is None: 
            self.grid_dims = nornir_imageregistration.TileGridShape(fixed_shape, cell_size) + 1 #Add one because we center the boundary points on the edge and not the center
        elif grid_spacing is None:
            self.grid_dims =  np.asarray(grid_dims, np.int32)
        elif grid_dims is None:
            self.grid_dims = nornir_imageregistration.TileGridShape(fixed_shape, grid_spacing)
            
        self.grid_spacing = (fixed_shape-1) / (self.grid_dims-1)
            
        self.coords = [np.asarray((iRow, iCol), dtype=np.int64) for iRow in range(self.grid_dims[0]) for iCol in range(self.grid_dims[1])]
        self.coords = np.vstack(self.coords)
        
        self.FixedPoints = self.coords * self.grid_spacing
        self.FixedPoints = np.floor(self.FixedPoints).astype(np.int64)
        
        self.fixed_shape = fixed_shape
        
        if transform is not None:
            self.WarpedPoints = self.PopulateWarpedPoints(transform)
        else:
            self.WarpedPoints = None
            
class CenteredGridDivision(GridDivisionBase):
    '''
    Align the grid so the edges of the edge cells touch the edge of the image
    '''

    def __init__(self, fixed_shape, cell_size, grid_dims=None, grid_spacing=None, transform = None):
        '''
        Divides an image into a grid, of possibly overlapping cells.
        
        :param tuple fixed_shape: (Rows, Columns) of image we are dividing 
        :param tuple cell_size: The dimensions of each grid cell
        :param tuple grid_dims: The number of (Rows, Columns) in the grid
        :param tuple grid_spacing: The distance between the centers of grid cells, possibly allowing overlapping cells
        '''
        
        self.cell_size = np.asarray(cell_size, np.int32)
        fixed_shape = np.asarray(fixed_shape, np.int64)
        
        if cell_size is None:
            raise ValueError("cell_size must be specified")
        
        if grid_dims is not None and grid_spacing is not None:
            raise ValueError("Either grid_dims or grid_spacing must be specified but not both")
        
        #We want the coordinates of grid centers to go from edge to edge in the image because ITK expects this behavior
        #Due to this fact we do not guarantee the grid_spacing requested        
        if grid_dims is None and grid_spacing is None: 
            self.grid_spacing = cell_size
            self.grid_dims = nornir_imageregistration.TileGridShape(fixed_shape, self.grid_spacing) 
        elif grid_spacing is None:
            self.grid_dims =  np.asarray(grid_dims, np.int32)
            self.grid_spacing = np.asarray( (fixed_shape-1) / self.grid_dims, np.int64)
        elif grid_dims is None:
            self.grid_spacing = np.asarray(grid_dims, np.int64)
            self.grid_dims = nornir_imageregistration.TileGridShape(fixed_shape, grid_spacing)
             
        self.coords = [np.asarray((iRow, iCol), dtype=np.int64) for iRow in range(self.grid_dims[0]) for iCol in range(self.grid_dims[1])]
        self.coords = np.vstack(self.coords)
        
        self.FixedPoints = self.coords * self.grid_spacing
        #Grid dimensions round up, so if we are larger than image find out by how much and adjust the points so they are centered on the image
        overage = ((self.grid_dims * self.grid_spacing) - fixed_shape) / 2.0
        self.FixedPoints = np.round(self.FixedPoints - overage).astype(np.int64)
        
        self.fixed_shape = fixed_shape
        
        if transform is not None:
            self.WarpedPoints = self.PopulateWarpedPoints(transform)
        else:
            self.WarpedPoints = None
            
    