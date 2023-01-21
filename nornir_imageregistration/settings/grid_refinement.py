"""
Created on Apr 26, 2022

@author: u0490822
"""

import numpy as np
import nornir_imageregistration


class GridRefinement(object):
    """
    Settings for grid refinement
    """

    def __init__(self,
                 target_image,
                 source_image,
                 target_mask=None,
                 source_mask=None,
                 num_iterations: int = None,
                 cell_size=None,
                 grid_spacing=None,
                 angles_to_search=None,
                 final_pass_angles=None,
                 max_travel_for_finalization: float = None,
                 max_travel_for_finalization_improvement: float = None,
                 min_alignment_overlap: float = None,
                 min_unmasked_area: float = None):
        """
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
        :param max_travel_for_finalization_improvement: When finalized points are checked to see if they need to be nudged, they must move less than this distance to be considered.  If None, no limit is applied
        :param float min_alignment_overlap: Limits how far control points can be translated.  The cells from fixed and target space must still overlap by this minimum amount after being registered.
        :param float min_unmasked_area: Area of cell that must be unmasked in both images to utilize that cell
        """

        if target_image is None:
            raise ValueError("target_image must be specified")

        if source_image is None:
            raise ValueError("source_image must be specified")

        if cell_size is None:
            self.cell_size = np.array((128, 128))
        elif isinstance(cell_size, int):
            self.cell_size = np.array((cell_size, cell_size))
        elif isinstance(cell_size, np.ndarray):
            self.cell_size = cell_size
        else:
            self.cell_size = np.array(cell_size)

        if self.cell_size.shape[0] != 2:
            raise ValueError(f'cell_size is supposed to be an array with two entries, got: {self.cell_size}')

        if grid_spacing is None:
            self.grid_spacing = np.array((96, 96))
        elif isinstance(grid_spacing, int):
            self.grid_spacing = np.array((grid_spacing, grid_spacing))
        elif isinstance(grid_spacing, np.ndarray):
            self.grid_spacing = grid_spacing
        else:
            self.grid_spacing = np.array(grid_spacing)

        if self.grid_spacing.shape[0] != 2:
            raise ValueError(f'grid_spacing is supposed to be an array with two entries, got: {self.grid_spacing}')

        target_image = nornir_imageregistration.ImageParamToImageArray(target_image, dtype=np.float16)
        source_image = nornir_imageregistration.ImageParamToImageArray(source_image, dtype=np.float16)

        if target_mask is not None:
            target_mask = nornir_imageregistration.ImageParamToImageArray(target_mask, dtype=np.bool)

        if source_mask is not None:
            source_mask = nornir_imageregistration.ImageParamToImageArray(source_mask, dtype=np.bool)

        # Create extrema masks
        target_extrema_mask = nornir_imageregistration.CreateExtremaMask(target_image, target_mask,
                                                                         size_cutoff=int(np.prod(cell_size)))
        source_extrema_mask = nornir_imageregistration.CreateExtremaMask(source_image, source_mask,
                                                                         size_cutoff=int(np.prod(cell_size)))

        if target_mask is not None:
            self.target_mask = np.logical_and(target_mask, target_extrema_mask)
        else:
            self.target_mask = target_extrema_mask

        if source_mask is not None:
            self.source_mask = np.logical_and(source_mask, source_extrema_mask)
        else:
            self.source_mask = source_extrema_mask

        self.target_image_stats = nornir_imageregistration.ImageStats.Create(target_image[target_mask])
        self.source_image_stats = nornir_imageregistration.ImageStats.Create(source_image[source_mask])

        self.target_image = nornir_imageregistration.RandomNoiseMask(target_image, self.target_mask, Copy=False)
        self.source_image = nornir_imageregistration.RandomNoiseMask(source_image, self.source_mask, Copy=False)

        self.angles_to_search = [0] if angles_to_search is None else angles_to_search
        self.final_pass_angles = [0] if final_pass_angles is None else final_pass_angles
        self.num_iterations = 10 if num_iterations is None else num_iterations
        self.max_travel_for_finalization = np.sqrt(
            np.max(cell_size)) if max_travel_for_finalization is None else max_travel_for_finalization
        self.max_travel_for_finalization_improvement = float("inf") if max_travel_for_finalization_improvement is None else max_travel_for_finalization_improvement
        self.min_alignment_overlap = 0.5 if min_alignment_overlap is None else min_alignment_overlap
        self.min_unmasked_area = 0.49 if min_unmasked_area is None else min_unmasked_area

    def __str__(self):
        return f'{self.cell_size[0]}x{self.cell_size[1]} spaced {self.grid_spacing[0]}x{self.grid_spacing[1]} {self.num_iterations} iterations'
