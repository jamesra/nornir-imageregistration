'''
Created on Apr 26, 2022

@author: u0490822
'''

import numpy as np
import nornir_imageregistration

class GridRefinement(object):
    '''
    Settings for grid refinement
    '''


    def __init__(self,   
                target_image,
                source_image,
                target_mask=None,
                source_mask=None,
                num_iterations:int=None,
                cell_size=None,
                grid_spacing=None,
                angles_to_search=None,
                final_pass_angles=None,
                max_travel_for_finalization: float=None,
                min_alignment_overlap: float = None,
                min_unmasked_area: float = None):
        '''
        :param target_image: ndarray or path to file, fixed space image
        :param source_image: ndarray or path to file, source space image
        :param target_mask: ndarray or path to file, fixed space image mask
        :param source_mask: ndarray or path to file, source space image mask
        :param int num_iterations: The maximum number of iterations to perform
        :param tuple cell_size: (width, height) area of image around control points to use for registration
        :param tuple grid_spacing: (width, height) of separation between control points on the grid
        :param array angles_to_search: An array of floats or None.  Images are rotated by the degrees indicated in the array.  The single best alignment across all angles is selected.  This set is used on all registrations.  Can have a performance impact.
        :param array final_pass_angles: An array of floats or None.  Images are rotated by the degrees indicated in the array.  The single best alignment across all angles is selected.  This set is usually a reduced range to fine tune a registration. Can have a performance impact.
        :param float max_travel_for_finalization: The maximum amount of travel a point can have from its predicted position for it to be considered "good enough" and considered for finalization
        :param float min_alignment_overlap: Limits how far control points can be translated.  The cells from fixed and target space must overlap by this minimum amount.
        :param float min_unmasked_area: Amount of cell area that must be unmasked to utilize the cell
        '''
        
        
        if target_image is None:
            raise ValueError("target_iamge must be specified")
        
        if source_image is None:
            raise ValueError("source_image must be specified")
        
        if cell_size is None:
            self.cell_size = np.array((128, 128))
        elif isinstance(cell_size, int):
            self.cell_size = np.array((cell_size, cell_size))
        
        if grid_spacing is None:
            self.grid_spacing = np.array((96, 96))
        elif isinstance(grid_spacing, int):
            self.grid_spacing = np.array((grid_spacing, grid_spacing))
        
        target_image = nornir_imageregistration.ImageParamToImageArray(target_image, dtype=np.float16)
        source_image = nornir_imageregistration.ImageParamToImageArray(source_image, dtype=np.float16)
        
        #Create extrema masks
        target_extrema_mask = nornir_imageregistration.CreateExtremaMask(target_image, int(np.prod(cell_size)))
        source_extrema_mask = nornir_imageregistration.CreateExtremaMask(source_image, int(np.prod(cell_size)))
        
        if target_mask is not None:
            target_mask = nornir_imageregistration.ImageParamToImageArray(target_mask, dtype=np.bool)
            self.target_mask = np.logical_and(target_mask, target_extrema_mask)
        else:
            self.target_mask = target_extrema_mask 
     
        if source_mask is not None:
            source_mask = nornir_imageregistration.ImageParamToImageArray(source_mask, dtype=np.bool)
            self.source_mask = np.logical_and(source_mask, source_extrema_mask)
        else:
            self.source_mask = source_extrema_mask
            
        self.target_image = nornir_imageregistration.RandomNoiseMask(target_image, target_mask, Copy=False)
        self.source_image = nornir_imageregistration.RandomNoiseMask(source_image, source_mask, Copy=False)
              
        if angles_to_search is None:
            self.angles_to_search = [0]
        else:
            self.angles_to_search = angles_to_search
            
        if final_pass_angles is None:
            self.final_pass_angles = [0]
        else:
            self.final_pass_angles = final_pass_angles
            
        if num_iterations is None:
            self.num_iterations = 10
        else:
            self.num_iterations = num_iterations
            
        if max_travel_for_finalization is None:
            self.max_travel_for_finalization = np.sqrt(np.max(cell_size))
        else:
            self.max_travel_for_finalization = max_travel_for_finalization
            
        if min_alignment_overlap is None:
            self.min_alignment_overlap = 0.5
        else:
            self.min_alignment_overlap = min_alignment_overlap
            
        if min_unmasked_area is None:
            self.min_unmasked_area = 0.49
        else:
            self.min_unmasked_area = min_unmasked_area
            
    def __str__(self):
        return f'{self.cell_size[0]}x{self.cell_size[1]} spaced {self.grid_spacing[0]}x{self.grid_spacing[1]} {self.num_iterations} iterations'
        
        