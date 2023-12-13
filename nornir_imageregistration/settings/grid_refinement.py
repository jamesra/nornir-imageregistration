"""
Created on Apr 26, 2022

@author: u0490822
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration


class GridRefinement(object):
    """
    Settings for grid refinement
    """
    source_image: NDArray[float]
    target_image: NDArray[float]
    source_mask: NDArray[bool]
    target_mask: NDArray[bool]
    cell_size: NDArray[int]
    grid_spacing: NDArray[int]
    angles_to_search: list[int]
    final_pass_angles: list[int]
    num_iterations: int
    max_travel_for_finalization: float
    max_travel_for_finalization_improvement: float
    min_alignment_overlap: float
    min_unmasked_area: float
    _single_thread_processing: bool
    _cupy_processing: bool

    @property
    def cupy_processing(self) -> bool:
        '''True if the settings are for processing the images with cupy.
        Currently this indicates if the images are stored in cupy arrays instead
        of numpy arrays
        '''
        return self._cupy_processing

    @property
    def single_thread_processing(self) -> bool:
        '''True if the settings are for processing the images on a single thread.
        Currently this indicates if the images are stored in shared memory or not
        '''
        return self._single_thread_processing or self._cupy_processing

    def __getstate__(self):
        # Return a dict that contains only the name attribute
        output = {}.update(self.__dict__)
        del output['target_image']
        del output['target_image']
        del output['source_mask']
        del output['target_mask']

        return output

    def __setstate__(self, state):
        # Restore the name attribute from the state dict
        self.__dict__.update(state)
        self.target_image = nornir_imageregistration.ImageParamToImageArray(self.target_image_meta)
        self.source_image = nornir_imageregistration.ImageParamToImageArray(self.source_image_meta)
        self.source_mask = nornir_imageregistration.ImageParamToImageArray(self.source_mask_meta)
        self.target_mask = nornir_imageregistration.ImageParamToImageArray(self.target_mask_meta)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        nornir_imageregistration.unlink_shared_memory(self.target_image_meta)
        nornir_imageregistration.unlink_shared_memory(self.source_image_meta)
        nornir_imageregistration.unlink_shared_memory(self.source_mask_meta)
        nornir_imageregistration.unlink_shared_memory(self.target_mask_meta)

    def __init__(self,
                 target_image: NDArray[float],
                 source_image: NDArray[float],
                 target_image_stats: nornir_imageregistration.ImageStats,
                 source_image_stats: nornir_imageregistration.ImageStats,
                 target_mask: NDArray[bool] | None = None,
                 source_mask: NDArray[bool] | None = None,
                 num_iterations: int = None,
                 cell_size=None,
                 grid_spacing=None,
                 angles_to_search=None,
                 final_pass_angles=None,
                 max_travel_for_finalization: float = None,
                 max_travel_for_finalization_improvement: float = None,
                 min_alignment_overlap: float = None,
                 min_unmasked_area: float = None,
                 single_thread_processing: bool = False):
        """
        Contains the settings that will be passed to RefineGrid.  It is the responsibility of the caller
        to ensure input images have been properly masked with random noise.  image_permutations_helper.py
        contains a class to aid in masking input images.

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
        :param bool single_thread_processing: True if the refinement should not use threads.  When set, arrays are not placed in shared memory
        :param bool cupy_processing: True if the refinement will be done on the GPU.  When set, arrays are created as cupy arrays instead of NDArrays
        """

        self._single_thread_processing = single_thread_processing or nornir_imageregistration.UsingCupy()  

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

        self.target_image = target_image
        self.source_image = source_image
        self.target_mask = target_mask
        self.source_mask = source_mask

        self.target_image_stats = target_image_stats
        self.source_image_stats = source_image_stats

        if not self._single_thread_processing:
            self.source_image_meta, self.source_image = nornir_imageregistration.npArrayToSharedArray(self.source_image)
            self.target_image_meta, self.target_image = nornir_imageregistration.npArrayToSharedArray(self.target_image)
            self.source_mask_meta, self.source_mask = nornir_imageregistration.npArrayToSharedArray(self.source_mask)
            self.target_mask_meta, self.target_mask = nornir_imageregistration.npArrayToSharedArray(self.target_mask)
        else:
            self.source_image_meta = self.source_image
            self.target_image_meta = self.target_image
            self.source_mask_meta = self.source_mask
            self.target_mask_meta = self.target_mask

        self.angles_to_search = [0] if angles_to_search is None else angles_to_search
        self.final_pass_angles = [0] if final_pass_angles is None else final_pass_angles
        self.num_iterations = 10 if num_iterations is None else num_iterations
        self.max_travel_for_finalization = np.sqrt(
            np.max(cell_size)) if max_travel_for_finalization is None else max_travel_for_finalization
        self.max_travel_for_finalization_improvement = float(
            "inf") if max_travel_for_finalization_improvement is None else max_travel_for_finalization_improvement
        self.min_alignment_overlap = 0.5 if min_alignment_overlap is None else min_alignment_overlap
        self.min_unmasked_area = 0.49 if min_unmasked_area is None else min_unmasked_area

    @staticmethod
    def CreateWithPreprocessedImages(target_img_data: nornir_imageregistration.ImagePermutationHelper,
                                     source_img_data: nornir_imageregistration.ImagePermutationHelper,
                                     num_iterations: int = None,
                                     cell_size=None,
                                     grid_spacing=None,
                                     angles_to_search=None,
                                     final_pass_angles=None,
                                     max_travel_for_finalization: float = None,
                                     max_travel_for_finalization_improvement: float = None,
                                     min_alignment_overlap: float = None,
                                     min_unmasked_area: float = None,
                                     single_thread_processing: bool = False) -> GridRefinement:
        '''Creates a settings object for imags that require no further processing.  For example
        masked areas and extrema regions have been filled with random noise.'''

        return GridRefinement(target_image=target_img_data.ImageWithMaskAsNoise,
                              source_image=source_img_data.ImageWithMaskAsNoise,
                              target_image_stats=target_img_data.Stats,
                              source_image_stats=source_img_data.Stats,
                              target_mask=target_img_data.BlendedMask,
                              source_mask=source_img_data.BlendedMask,
                              num_iterations=num_iterations,
                              cell_size=cell_size,
                              grid_spacing=grid_spacing,
                              angles_to_search=angles_to_search,
                              final_pass_angles=final_pass_angles,
                              max_travel_for_finalization=max_travel_for_finalization,
                              max_travel_for_finalization_improvement=max_travel_for_finalization_improvement,
                              min_alignment_overlap=min_alignment_overlap,
                              min_unmasked_area=min_unmasked_area,
                              single_thread_processing=single_thread_processing)

    @staticmethod
    def CreateWithUnproccessedImages(
            target_image: nornir_imageregistration.ImageLike,
            source_image: nornir_imageregistration.ImageLike,
            target_mask: nornir_imageregistration.ImageLike | None = None,
            source_mask: nornir_imageregistration.ImageLike | None = None,
            extrema_mask_size_cuttoff: float | int | NDArray | None = None,
            num_iterations: int = None,
            cell_size=None,
            grid_spacing=None,
            angles_to_search=None,
            final_pass_angles=None,
            max_travel_for_finalization: float = None,
            max_travel_for_finalization_improvement: float = None,
            min_alignment_overlap: float = None,
            min_unmasked_area: float = None,
            single_thread_processing: bool = False) -> GridRefinement:
        '''Creates a settings objects and adds noise to images according to the provided masks'''
        target_img_data = nornir_imageregistration.ImagePermutationHelper(target_image, target_mask,
                                                                          extrema_mask_size_cuttoff=extrema_mask_size_cuttoff)
        source_img_data = nornir_imageregistration.ImagePermutationHelper(source_image, source_mask,
                                                                          extrema_mask_size_cuttoff=extrema_mask_size_cuttoff)

        return GridRefinement.CreateWithPreprocessedImages(target_img_data, source_img_data,
                                                           num_iterations=num_iterations,
                                                           cell_size=cell_size,
                                                           grid_spacing=grid_spacing,
                                                           angles_to_search=angles_to_search,
                                                           final_pass_angles=final_pass_angles,
                                                           max_travel_for_finalization=max_travel_for_finalization,
                                                           max_travel_for_finalization_improvement=max_travel_for_finalization_improvement,
                                                           min_alignment_overlap=min_alignment_overlap,
                                                           min_unmasked_area=min_unmasked_area,
                                                           single_thread_processing=single_thread_processing)

    def __str__(self):
        return f'{self.cell_size[0]}x{self.cell_size[1]} spaced {self.grid_spacing[0]}x{self.grid_spacing[1]} {self.num_iterations} iterations'
