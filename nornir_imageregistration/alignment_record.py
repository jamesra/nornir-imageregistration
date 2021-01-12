'''
'''

import os

import nornir_imageregistration
import nornir_imageregistration.files.stosfile
from scipy import pi

import numpy as np


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
    def angle(self):
        '''Rotation in degrees'''
        return self._angle

    @property
    def rangle(self):
        '''Rotation in radians'''
        return self._angle * (pi / 180.0)

    @property
    def weight(self):
        '''Quantifies the quality of the alignment'''
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = float(value)

    @property
    def flippedud(self):
        '''True if the warped image was flipped vertically for the alignment'''
        return self._flippedud

    @flippedud.setter
    def flippedud(self, value):
        self._flippedud = value

    @property
    def peak(self):
        '''Translation vector for the alignment'''
        return self._peak

    def WeightKey(self):
        return self._weight
    
    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        '''Scales the source space to target space, including peak'''
        self._scale = value

    def translate(self, value):
        '''Translates the peak position using tuple (Y,X)'''
        self._peak = self._peak + value

    def Invert(self):
        '''
        Returns a new alignment record with the coordinates of the peak reversed
        Used to change the frame of reference of the alignment from one tile to another
        '''
        return AlignmentRecord((-self.peak[0], -self.peak[1]), self.weight, self.angle)

    def __str__(self):
        s = 'Offset: {x:.2f}x, {y:.2f}y Weight: {w:.2f}'.format(x=self._peak[1], y=self._peak[0], w=self._weight)
        
        if self._angle != 0:
            s += ' Angle: {0:.2f}'.format(self._angle)
            
        # s = 'angle: ' + str(self._angle) + ' offset: ' + str(self._peak) + ' weight: ' + str(self._weight)
        if self.flippedud:
            s += ' Flipped up/down'
            
        return s

    def __init__(self, peak, weight, angle=0.0, flipped_ud=False, scale=1.0):
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

    def CorrectPeakForOriginalImageSize(self, FixedImageShape, MovingImageShape):

        if self.peak is None:
            self.peak = (0, 0)

        return nornir_imageregistration.transforms.factory.__CorrectOffsetForMismatchedImageSizes(FixedImageShape, MovingImageShape)

    def GetTransformedCornerPoints(self, warpedImageSize):
        '''
        '''
        return nornir_imageregistration.transforms.factory.GetTransformedRigidCornerPoints(warpedImageSize, self.rangle, self.peak, self.flippedud)

    def ToTransform(self, fixedImageSize, warpedImageSize=None):
        '''
        Generates a rigid transform for the alignment record.
        :param (Height, Width) fixedImageSize: Size of translated image in fixed space
        :param (Height, Width) warpedImageSize: Size of translated image in warped space.   If unspecified defaults to fixedImageSize
        :return: A rigid rotation+translation transform described by the alignment record
        '''

        if warpedImageSize is None:
            warpedImageSize = fixedImageSize

        return nornir_imageregistration.transforms.factory.CreateRigidMeshTransform(target_image_shape=fixedImageSize,
                                                                                source_image_shape=warpedImageSize,
                                                                                rangle=self.rangle,
                                                                                warped_offset=self.peak,
                                                                                flip_ud=self.flippedud,
                                                                                scale=self.scale)

    def __ToGridTransformString(self, fixedImageSize, warpedImageSize):

        transform = self.ToTransform(fixedImageSize, warpedImageSize)

        # Flipped is always false because the transform is already flipped if needed
        warpedSpaceCorners = nornir_imageregistration.transforms.factory.GetTransformedRigidCornerPoints(warpedImageSize, rangle=0, offset=(0, 0), flip_ud=False)

        fixedSpaceCorners = transform.Transform(warpedSpaceCorners)

#        list = [str(BotLeft.item(0)),
#                str(BotLeft.item(1)),
#                str(BotRight.item(0)),
#                str(BotRight.item(1)),
#                str(TopLeft.item(0)),
#                str(TopLeft.item(1)),
#                str(TopRight.item(0)),
#                str(TopRight.item(1))]

        string = ""

        fixedSpaceCorners = np.fliplr(fixedSpaceCorners)

        for s in fixedSpaceCorners.flat:
            string = string + ' %g' % s

        return string

    def ToStos(self, ImagePath, WarpedImagePath, FixedImageMaskPath=None, WarpedImageMaskPath=None, PixelSpacing=1):
        stos = nornir_imageregistration.files.stosfile.StosFile()
        stos.ControlImageName = os.path.basename(ImagePath)
        stos.ControlImagePath = os.path.dirname(ImagePath)

        stos.MappedImageName = os.path.basename(WarpedImagePath)
        stos.MappedImagePath = os.path.dirname(WarpedImagePath)

        if not FixedImageMaskPath is None:
            stos.ControlMaskName = os.path.basename(FixedImageMaskPath)
            stos.ControlMaskPath = os.path.dirname(FixedImageMaskPath)

        if not WarpedImageMaskPath is None:
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

        transformTemplate = "GridTransform_double_2_2 vp 8 %(coordString)s fp 7 0 1 1 0 0 %(width)f %(height)f"

        # We use Y,X ordering in memory due to Numpy.  Ir-Tools coordinates are written X,Y.
        coordString = self.__ToGridTransformString((stos.ControlImageDim[1], stos.ControlImageDim[0]), (stos.MappedImageDim[1], stos.MappedImageDim[0]))

        stos.Transform = transformTemplate % {'coordString' : coordString,
                                              'width' : stos.MappedImageDim[0] - 1,
                                              'height' : stos.MappedImageDim[1] - 1}

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
    
    def __init__(self, ID, TargetPoint, SourcePoint, peak, weight, angle=0.0, flipped_ud=False):
        
        super(EnhancedAlignmentRecord, self).__init__(peak=peak, weight=weight, angle=angle, flipped_ud=flipped_ud)
        self._ID = ID
        self._TargetPoint = TargetPoint
        self._SourcePoint = SourcePoint
        
    
