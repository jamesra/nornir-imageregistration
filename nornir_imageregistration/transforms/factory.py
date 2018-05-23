'''
Created on Nov 13, 2012

@author: u0490822

The factory is focused on the loading and saving of transforms
'''

from scipy import *

from nornir_imageregistration.spatial.indicies import *
import nornir_imageregistration.transforms.meshwithrbffallback as meshwithrbffallback
import numpy as np

from . import utils


def TransformToIRToolsString(transformObj, bounds=None):
    if hasattr(transformObj, 'gridWidth') and hasattr(transformObj, 'gridHeight'):
        return _TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight, bounds=bounds)
    else:
        return _TransformToIRToolsString(transformObj, bounds)  # , bounds=NewStosFile.MappedImageDim)


def _TransformToIRToolsGridString(Transform, XDim, YDim, bounds=None):

    numPoints = Transform.points.shape[0]

    # Find the extent of the mapped boundaries
    (bottom, left, top, right) = (None, None, None, None)
    if bounds is None:
        (bottom, left, top, right) = Transform.MappedBoundingBox.ToTuple()
    else:
        (bottom, left, top, right) = bounds

    output = []
    output.append("GridTransform_double_2_2 vp " + str(numPoints * 2))
    
    template = " %(cx)g %(cy)g"

    NumAdded = int(0)
    for CY, CX, MY, MX in Transform.points:
        pstr = template % {'cx' : CX, 'cy' : CY}
        output.append(pstr)
        NumAdded = NumAdded + 1

   # print str(NumAdded) + " points added"

    boundsStr = " ".join(map(str, [left, bottom, right - left, top - bottom]))
    
    output.append(" fp 7 0 " + str(int(YDim - 1)) + " " + str(int(XDim - 1)) + " " + boundsStr)
    # output = output + " fp 7 0 " + str(int(YDim - 1)) + " " + str(int(XDim - 1)) + " " + boundsStr
    transform_string = ''.join(output)
    
    return transform_string


def _TransformToIRToolsString(Transform, bounds=None):
    numPoints = Transform.points.shape[0]

    # Find the extent of the mapped boundaries
    (bottom, left, top, right) = (None, None, None, None)
    if bounds is None:
        (bottom, left, top, right) = Transform.MappedBoundingBox.ToTuple()
    else:
        (bottom, left, top, right) = bounds

    output = []
    output.append("MeshTransform_double_2_2 vp " + str(numPoints * 4))

    template = " %(mx)f %(my)f %(cx)g %(cy)g"

    width = right - left
    height = top - bottom

    for CY, CX, MY, MX in Transform.points:
        pstr = template % {'cx' : CX, 'cy' : CY, 'mx' : (MX - left) / width , 'my' : (MY - bottom) / height}
        output.append(pstr)

    boundsStr = " ".join(map(str, [left, bottom, width, height]))
    
    output.append(" fp 8 0 16 16 ")
    output.append(boundsStr)
    output.append(' ' + str(numPoints))
    
    # boundsStr = " ".join(map(str, [0, 0, width, height]))
    # transform_string = #output + " fp 8 0 16 16 " + boundsStr + ' ' + str(numPoints)
    transform_string = ''.join(output)

    return transform_string


def __ParseParameters(parts):
    '''Input is a transform split on white space, returns the variable and fixed parameters as a list'''

    iVP = 0
    iFP = 0

    VariableParameters = []
    FixedParameters = []

    for i, val in enumerate(parts):
        if val == 'vp':
            iVP = i
        elif val == 'fp':
            iFP = i
        elif iFP > 0 and i > iFP + 1:
            FixedParameters.append(float(val))
            if FixedParameters[-1] >= 1.79769e+308:
                raise ValueError("Unexpected value in transform, probably invalid output from ir-tools")

        elif iVP > 0 and i > iVP + 1:
            VariableParameters.append(float(val))
            if VariableParameters[-1] >= 1.79769e+308:
                raise ValueError("Unexpected value in transform, probably invalid output from ir-tools")

    return (VariableParameters, FixedParameters)


def SplitTrasform(transformstring):
    '''Returns transform name, variable points, fixed points'''
    parts = transformstring.split()
    transformName = parts[0]
    assert(parts[1] == 'vp')

    VariableParts = []
    iVp = 2
    while(parts[iVp] != 'fp'):
        VariableParts = float(parts[iVp])
        iVp = iVp + 1

    # skip vp # entries
    iVp = iVp + 2
    FixedParts = []
    for iVp in range(iVp, len(parts)):
        FixedParts = float(parts[iVp])

    return (transformName, FixedParts, VariableParts)


def LoadTransform(Transform, pixelSpacing=None):
    '''Transform is a string from either a stos or mosiac file'''

    parts = Transform.split()

    transformType = parts[0]

    if transformType == "GridTransform_double_2_2":
        return ParseGridTransform(parts, pixelSpacing)
    elif transformType == "MeshTransform_double_2_2":
        return ParseMeshTransform(parts, pixelSpacing)
    elif transformType == "LegendrePolynomialTransform_double_2_2_1":
        return ParseLegendrePolynomialTransform(parts, pixelSpacing)
    elif transformType == "Rigid2DTransform_double_2_2":
        return ParseRigid2DTransform(parts, pixelSpacing)


def ParseGridTransform(parts, pixelSpacing=None):

    if pixelSpacing is None:
        pixelSpacing = 1.0

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    gridWidth = int(FixedParameters[2]) + 1
    gridHeight = int(FixedParameters[1]) + 1

    ImageWidth = float(FixedParameters[5]) * pixelSpacing
    ImageHeight = float(FixedParameters[6]) * pixelSpacing

    PointPairs = []

    for i in range(0, len(VariableParameters) - 1, 2):
        iY = (i / 2) // gridWidth
        iX = (i / 2) % gridWidth

        # print str((iX,iY))

        mappedX = (float(iX) / float(gridWidth - 1)) * ImageWidth
        mappedY = (float(iY) / float(gridHeight - 1)) * ImageHeight
        ControlX = VariableParameters[i]
        ControlY = VariableParameters[i + 1]
        PointPairs.append((ControlY, ControlX, mappedY, mappedX))

    T = meshwithrbffallback.MeshWithRBFFallback(PointPairs)
    T.gridWidth = gridWidth
    T.gridHeight = gridHeight
    return T


def ParseMeshTransform(parts, pixelSpacing=None):

    if pixelSpacing is None:
        pixelSpacing = 1.0

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    Left = float(FixedParameters[3]) * pixelSpacing
    Bottom = float(FixedParameters[4]) * pixelSpacing
    ImageWidth = float(FixedParameters[5]) * pixelSpacing
    ImageHeight = float(FixedParameters[6]) * pixelSpacing

    PointPairs = []

    for i in range(0, len(VariableParameters) - 1, 4):

        mappedX = (VariableParameters[i + 0] * ImageWidth) + Left
        mappedY = (VariableParameters[i + 1] * ImageHeight) + Bottom
        ControlX = float(VariableParameters[i + 2]) * pixelSpacing
        ControlY = float(VariableParameters[i + 3]) * pixelSpacing

        PointPairs.append((ControlY, ControlX, mappedY, mappedX))

    T = meshwithrbffallback.MeshWithRBFFallback(PointPairs)
    return T


def ParseLegendrePolynomialTransform(parts, pixelSpacing=None):

    # Example: LegendrePolynomialTransform_double_2_2_1 vp 6 1 0 1 1 1 0 fp 4 10770 -10770 2040 2040

    if pixelSpacing is None:
        pixelSpacing = 1.0

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    # We don't support anything but translation from this transform at the moment:
    # assert(VariableParameters == [1, 0, 1, 1, 1, 0])  # Sequence for transform only transformation

    Left = 0 * pixelSpacing
    Bottom = 0 * pixelSpacing
    ImageWidth = float(FixedParameters[2]) * pixelSpacing * 2.0
    ImageHeight = float(FixedParameters[3]) * pixelSpacing * 2.0

    array = np.array([[Left, Bottom],
                  [Left, Bottom + ImageHeight],
                  [Left + ImageWidth, Bottom],
                  [Left + ImageWidth, Bottom + ImageHeight]])
    PointPairs = []

    for i in range(0, 4):
        mappedX, mappedY = array[i, :]
        ControlX = (FixedParameters[0] * pixelSpacing) + mappedX
        ControlY = (FixedParameters[1] * pixelSpacing) + mappedY
        PointPairs.append((ControlY, ControlX, mappedY, mappedX))

    T = meshwithrbffallback.MeshWithRBFFallback(PointPairs)
    return T


def ParseRigid2DTransform(parts, pixelSpacing=None):

    raise Exception("Not implemented")

    if pixelSpacing is None:
        pixelSpacing = 1.0

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    gridWidth = int(float(FixedParameters[2])) + 1
    gridHeight = int(float(FixedParameters[1])) + 1

    Left = float(FixedParameters[3]) * pixelSpacing
    Bottom = float(FixedParameters[4]) * pixelSpacing
    ImageWidth = float(FixedParameters[5]) * pixelSpacing
    ImageHeight = float(FixedParameters[6]) * pixelSpacing

    PointPairs = []

    for i in range(0, len(VariableParameters) - 1, 4):
        iY = i // gridWidth
        iX = i % gridWidth

        mappedX = (VariableParameters[i + 0] * ImageWidth) + Left
        mappedY = (VariableParameters[i + 1] * ImageHeight) + Bottom
        ControlX = float(VariableParameters[i + 2])
        ControlY = float(VariableParameters[i + 3])
        PointPairs.append((ControlX, ControlY, mappedX, mappedY))

    T = meshwithrbffallback.MeshWithRBFFallback(PointPairs)
    return T


def __CorrectOffsetForMismatchedImageSizes(offset, FixedImageShape, MovingImageShape):

    return (offset[0] + ((FixedImageShape[0] - MovingImageShape[0]) / 2.0), offset[1] + ((FixedImageShape[1] - MovingImageShape[1]) / 2.0))


def CreateRigidTransform(FixedImageSize, WarpedImageSize, rangle, warped_offset, flip_ud=False):
    '''Returns a transform, the fixed image defines the boundaries of the transform.
       The warped image '''

    assert(FixedImageSize[0] > 0)
    assert(FixedImageSize[1] > 0)
    assert(WarpedImageSize[0] > 0)
    assert(WarpedImageSize[1] > 0)

    # Adjust offset for any mismatch in dimensions
    AdjustedOffset = __CorrectOffsetForMismatchedImageSizes(warped_offset, FixedImageSize, WarpedImageSize)

    # The offset is the translation of the warped image over the fixed image.  If we translate 0,0 from the warped space into
    # fixed space we should obtain the warped_offset value
    FixedPoints = GetTransformedRigidCornerPoints(WarpedImageSize, rangle, AdjustedOffset)
    WarpedPoints = GetTransformedRigidCornerPoints(WarpedImageSize, rangle=0, offset=(0, 0), flip_ud=flip_ud)

    ControlPoints = np.append(FixedPoints, WarpedPoints, 1)

    transform = meshwithrbffallback.MeshWithRBFFallback(ControlPoints)

    return transform


def GetTransformedRigidCornerPoints(size, rangle, offset, flip_ud=False):
    '''Returns positions of the four corners of a warped image in a fixed space using the rotation and peak offset.  Rotation occurs at the center.
       Flip, if requested, is performed before the rotation and translation
    :param tuple size: (Height, Width)
    :param float rangle: Angle in radians
    :param tuple offset: (Y, X)
    :return: Nx2 array of points [[BotLeft], [BotRight], [TopLeft],  [TopRight]]
    :rtype: numpy.ndarray
    '''
    CenteredRotation = utils.RotationMatrix(rangle)

    HalfWidth = (size[iArea.Width] - 1) / 2.0
    HalfHeight = (size[iArea.Height] - 1) / 2.0

#     BotLeft = CenteredRotation * matrix([[-HalfWidth], [-HalfHeight], [1]])
#     TopLeft = CenteredRotation * matrix([[-HalfWidth], [HalfHeight], [1]])
#     BotRight = CenteredRotation * matrix([[HalfWidth], [-HalfHeight], [1]])
#     TopRight = CenteredRotation * matrix([[HalfWidth], [HalfHeight], [1]])

    if not flip_ud:
        BotLeft = CenteredRotation * matrix([[-HalfHeight], [-HalfWidth], [1]])
        TopLeft = CenteredRotation * matrix([[HalfHeight], [-HalfWidth], [1]])
        BotRight = CenteredRotation * matrix([[-HalfHeight], [HalfWidth], [1]])
        TopRight = CenteredRotation * matrix([[HalfHeight], [HalfWidth], [1]])
    else:
        BotLeft = CenteredRotation * matrix([[HalfHeight], [-HalfWidth], [1]])
        TopLeft = CenteredRotation * matrix([[-HalfHeight], [-HalfWidth], [1]])
        BotRight = CenteredRotation * matrix([[HalfHeight], [HalfWidth], [1]])
        TopRight = CenteredRotation * matrix([[-HalfHeight], [HalfWidth], [1]])

    Translation = matrix([[1, 0, HalfHeight], [0, 1, HalfWidth], [0, 0, 1]])

    BotLeft = Translation * BotLeft
    BotRight = Translation * BotRight
    TopLeft = Translation * TopLeft
    TopRight = Translation * TopRight

    # Adjust to the peak location
    PeakTranslation = matrix([[1, 0, offset[iPoint.Y]], [0, 1, offset[iPoint.X]], [0, 0, 1]])

    BotLeft = PeakTranslation * BotLeft
    TopLeft = PeakTranslation * TopLeft
    BotRight = PeakTranslation * BotRight
    TopRight = PeakTranslation * TopRight

    arr = np.vstack([BotLeft[:2].getA1(), BotRight[:2].getA1(), TopLeft[:2].getA1(), TopRight[:2].getA1()])
    # arr[:, [0, 1]] = arr[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points
    return arr
