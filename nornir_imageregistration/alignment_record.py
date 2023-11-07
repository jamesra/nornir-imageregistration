'''
'''

from math import pi
import os

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.transforms import ITransform


class AlignmentRecord(object):
    '''
    Records basic registration information as an angle and offset between a fixed and moving image
    If the offset is zero the center of both images occupy the same point.  
    The offset determines the translation of the moving image over the fixed image.
    There is no support for scale, and there should not be unless added as another variable to the alignment record
    
    :param array peak: Translation vector for moving image
    :param float weight: The strength of the alignment
    :param float angle: Angle to rotate moving image in degrees
    
    '''

    @property
    def angle(self) -> float:
        '''Rotation in degrees'''
        return self._angle

    @property
    def rangle(self) -> float:
        '''Rotation in radians'''
        return self._angle * (pi / 180.0)

    @property
    def weight(self) -> float:
        '''Quantifies the quality of the alignment'''
        return self._weight

    @weight.setter
    def weight(self, value: float):
        self._weight = float(value)

    @property
    def flippedud(self) -> bool:
        '''True if the warped image was flipped vertically for the alignment'''
        return self._flippedud

    @flippedud.setter
    def flippedud(self, value: bool):
        self._flippedud = value

    @property
    def peak(self) -> NDArray[float]:
        '''Translation vector for the alignment'''
        return self._peak

    def WeightKey(self) -> float:
        return self._weight

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        '''Scales the source space to target space, including peak'''
        self._scale = value

    def translate(self, value: NDArray[float]):
        '''Translates the peak position using tuple (Y,X)'''
        self._peak += value

    def Invert(self):
        '''
        Returns a new alignment record with the coordinates of the peak reversed
        Used to change the frame of reference of the alignment from one tile to another
        '''
        return AlignmentRecord((-self.peak[0], -self.peak[1]), self.weight, self.angle)

    def __repr__(self):
        s = '{x:.2f}x, {y:.2f}y Weight: {w:.2f}'.format(x=self._peak[1], y=self._peak[0], w=self._weight)

        if self._angle != 0:
            s += f' Angle: {self._angle:.2f}'

        # s = 'angle: ' + str(self._angle) + ' offset: ' + str(self._peak) + ' weight: ' + str(self._weight)
        if self.flippedud:
            s += ' Flipped up/down'

        return s

    def __str__(self):
        return f'Offset: {self.__repr__()}'

    def __init__(self, peak: NDArray[float] | tuple[float, float], weight: float, angle: float = 0.0, flipped_ud: bool = False,
                 scale: float = 1.0):
        '''
        :param float scale: Scales source space by this factor to map into target space
        '''
        if not isinstance(angle, float):
            angle = float(angle)

        self._angle = angle

        if not isinstance(peak, np.ndarray):
            peak = np.array(peak)

        self._scale = scale
        self._peak = peak
        self._weight = float(weight)
        self._flippedud = flipped_ud

    def CorrectPeakForOriginalImageSize(self, FixedImageShape: NDArray[int], MovingImageShape: NDArray[int]):

        offset = self.peak
        if self.peak is None:
            offset = (0, 0)

        return nornir_imageregistration.transforms.factory.__CorrectOffsetForMismatchedImageSizes(offset=offset,
                                                                                                  FixedImageShape=FixedImageShape,
                                                                                                  MovingImageShape=MovingImageShape)

    def GetTransformedCornerPoints(self, warpedImageSize: NDArray[int]) -> NDArray[float]:
        '''
        '''
        # Adjust image size by 1 since the images are indexed by 0
        return nornir_imageregistration.transforms.factory.GetTransformedRigidCornerPoints(warpedImageSize - 1,
                                                                                           self.rangle,
                                                                                           self.peak, self.flippedud)

    def ToTransform(self, fixedImageSize: NDArray | tuple[int, int],
                    warpedImageSize: NDArray | tuple[int, int] | None = None) -> ITransform:
        '''
        Generates a rigid transform for the alignment record.
        :param (Height, Width) fixedImageSize: Size of translated image in fixed space
        :param (Height, Width) warpedImageSize: Size of translated image in warped space.   If unspecified defaults to fixedImageSize
        :return: A rigid rotation+translation transform described by the alignment record
        '''

        if warpedImageSize is None:
            warpedImageSize = fixedImageSize

        warpedImageSize = nornir_imageregistration.EnsurePointsAre1DNumpyArray(warpedImageSize)
        fixedImageSize = nornir_imageregistration.EnsurePointsAre1DNumpyArray(fixedImageSize)

        source_center_of_rotation = (warpedImageSize - 1) / 2.0  # Subtract 1 because images are indexed starting at zero
        # Adjust the center of rotation by 0.5 if there is an even dimension
        # source_center_of_rotation[np.mod(warpedImageSize, 2) == 0] -= 0.5
        target_center = (fixedImageSize - 1) / 2.0

        target_translation = (source_center_of_rotation - target_center) + self.peak

        if GetActiveComputationalLib() == ComputationLib.cupy:
            return nornir_imageregistration.transforms.Rigid_GPU(target_offset=target_translation,
                                                             source_rotation_center=source_center_of_rotation,
                                                             angle=self.rangle,
                                                             flip_ud=self.flippedud)
        else:
            return nornir_imageregistration.transforms.Rigid(target_offset=target_translation,
                                                         source_rotation_center=source_center_of_rotation,
                                                         angle=self.rangle,
                                                         flip_ud=self.flippedud)

        # return nornir_imageregistration.transforms.factory.CreateRigidTransform(warped_offset=self.peak,
        #                                                                        rangle=self.rangle,
        #                                                                        target_image_shape=fixedImageSize,
        #                                                                        source_image_shape=warpedImageSize,
        #                                                                        flip_ud=self.flippedud
        #                                                                        )

        # return nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=fixedImageSize,
        #                                                                         source_image_shape=warpedImageSize,
        #                                                                         rangle=self.rangle,
        #                                                                         warped_offset=self.peak,
        #                                                                         flip_ud=self.flippedud,
        #                                                                         scale=self.scale)

    def ToStos(self, ImagePath, WarpedImagePath, FixedImageMaskPath=None, WarpedImageMaskPath=None, PixelSpacing=1):
        stos = nornir_imageregistration.StosFile()
        stos.ControlImageName = os.path.basename(ImagePath)
        stos.ControlImagePath = os.path.dirname(ImagePath)

        stos.MappedImageName = os.path.basename(WarpedImagePath)
        stos.MappedImagePath = os.path.dirname(WarpedImagePath)

        if FixedImageMaskPath is not None:
            stos.ControlMaskName = os.path.basename(FixedImageMaskPath)
            stos.ControlMaskPath = os.path.dirname(FixedImageMaskPath)

        if WarpedImageMaskPath is not None:
            stos.MappedMaskName = os.path.basename(WarpedImageMaskPath)
            stos.MappedMaskPath = os.path.dirname(WarpedImageMaskPath)

        (ControlHeight, ControlWidth) = nornir_imageregistration.core.GetImageSize(ImagePath)
        stos.ControlImageDim = (ControlWidth, ControlHeight)

        (MappedHeight, MappedWidth) = nornir_imageregistration.core.GetImageSize(WarpedImagePath)
        stos.MappedImageDim = (MappedWidth, MappedHeight)

        # transformTemplate = "FixedCenterOfRotationAffineTransform_double_2_2 vp 8 %(cos)g %(negsin)g %(sin)g %(cos)g %(x)g %(y)g 1 1 fp 2 %(mapwidth)d %(mapheight)d"

        # stos.Transform = transformTemplate % {'cos' : cos(Match.angle * numpy.pi / 180),
        #                                 'sin' : sin(Match.angle * numpy.pi / 180),
        #                                 'negsin' : -sin(Match.angle * numpy.pi / 180),
        #                                 'x' : Match.peak[0],
        #                                 'y' : -Match.peak[1],
        #                                 'mapwidth' : stos.MappedImageDim[0]/2,
        #                                 'mapheight' : stos.MappedImageDim[1]/2}

        # transformTemplate = "GridTransform_double_2_2 vp 8 %(coordString)s fp 7 0 1 1 0 0 %(width)d %(height)d"

        # I have checked the dimensions that should be written for Grid transform against the original SCI code.  The image dimensions should be the actual dimensions and not
        # have a -1 to account for the zero origin
        # stos.Transform = transformTemplate % {'coordString': coordString,
        # 'width': stos.MappedImageDim[0],
        # 'height': stos.MappedImageDim[1]}

        transform = self.ToTransform(fixedImageSize=(ControlHeight, ControlWidth),
                                     warpedImageSize=(MappedHeight, MappedWidth))

        stos.Transform = transform.ToITKString()

        stos.Downsample = PixelSpacing

        #        print "Done!"

        return stos


class EnhancedAlignmentRecord(AlignmentRecord):
    '''
    An extension of the AlignmentRecord class that also records the Fixed and Warped Points
    '''

    @property
    def ID(self):
        return self._ID

    @property
    def TargetPoint(self):
        return self._TargetPoint

    @property
    def SourcePoint(self):
        return self._SourcePoint

    @property
    def AdjustedTargetPoint(self):
        return self._TargetPoint + self.peak

    @property
    def AdjustedSourcePoint(self):
        '''Note if there is rotation involved this point is not reliable'''
        return self._SourcePoint - self.peak

    def __init__(self, ID, TargetPoint: NDArray[float], SourcePoint, peak, weight, angle=0.0, flipped_ud=False):
        super(EnhancedAlignmentRecord, self).__init__(peak=peak, weight=weight, angle=angle, flipped_ud=flipped_ud)
        self._ID = ID
        self._TargetPoint = TargetPoint
        self._SourcePoint = SourcePoint

    def __repr__(self):
        return f'ID: {self._ID} {super(EnhancedAlignmentRecord, self).__repr__()}'

    def __str__(self):
        return self.__repr__()
