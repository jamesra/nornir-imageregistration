'''
Created on Jan 27, 2014

@author: u0490822
'''

import numpy as np
import nornir_imageregistration.transforms.utils as tutils

def _SortArrayByColumn(arr, SortColumn=None):
    '''Sort an array using the column order, either an integer or a list, and return the array'''
    (numRows, numCols) = arr.shape()

    if SortColumn is None:
        SortColumn = numCols - 1

    columns = []
    if isinstance(SortColumn, list) or isinstance(SortColumn, tuple):

        for col in SortColumn:
            columns.append(arr[:, col])
    else:
        columns = arr[:, SortColumn]

    indicies = np.lexsort(columns)

    return (arr[:, indicies], indicies)


class Volume(object):
    '''
    A collection of slice-to-volume transforms that can map a point from any section into the volume
    '''
    @property
    def SectionToVolumeTransforms(self):
        return self._SectionToVolumeTransforms

    def __init__(self):
        '''
        '''
        self._SectionToVolumeTransforms = dict()

    def AddSection(self, SectionID, transform):
        '''Adds a transform for a section, replacing it if it already exists'''
        self._SectionToVolumeTransforms[SectionID] = transform

    ##############################
    # Transformations            #
    ##############################

    def SectionToVolume2D(self, SectionID, Points):
        return self._SectionToVolumeTransforms[SectionID].Transform(Points)

    def VolumeToSection2D(self, SectionID, Points):
        self._SectionToVolumeTransforms[SectionID].InverseTransform(Points)

    def SectionToVolume3D(self, points):
        '''Maps array of [X Y Z] points on sections to volume space'''
        self.__ApplyTransformFor3DPoints(points, transformFunc=self.SectionToVolume2D)

    def VolumeToSection3D(self, points):
        '''Maps array of [X Y Z] points on sections to volume space'''
        self.__ApplyTransformFor3DPoints(points, transformFunc=self.VolumeToSection2D)

    def __ApplyTransformFor3DPoints(self, points, transformFunc):
        '''Maps array of [X Y Z] points on sections to volume space'''
        outputPoints = points.copy()

        (numRows, numCols) = points.shape()
        assert(numCols == 3)
        sorted_points, unsorted_indicies = _SortArrayByColumn(points, ColumnNumber=2)

        # sectionNumbers, sectionIndicies = np.unique(points[:, 2], return_index=False, return_inverse=True)
        iRow = 0
        while iRow < numRows:

            sectionNumber = sorted_points[iRow, 2]
            iEnd = np.searchsorted(sorted_points[:, 2], sectionNumber, side='right')

            sectionPoints = sorted_points[iRow:iEnd, 0:2]

            transformedPoints = transformFunc(sectionNumber, sectionPoints)
            input_array_position = unsorted_indicies[iRow:iEnd]

            outputPoints[input_array_position, :] = transformedPoints

            iRow = iEnd


    ##############################
    # Boundary data              #
    ##############################

    @property
    def VolumeBounds(self):
        # Boundaries of the volume based on locations where sections will map points into the volume
        return tutils.FixedBoundingBox(self.SectionToVolumeTransforms.values())
    
    def IsOriginAtZero(self):
        return tutils.IsOriginAtZero(self._SectionToVolumeTransforms.values())

    def TranslateToZeroOrigin(self):
        '''Ensure that the transforms in the mosaic do not map to negative coordinates'''
        return tutils.TranslateToZeroOrigin(self._SectionToVolumeTransforms.values())
