'''
Created on Nov 13, 2012

@author: u0490822

The factory is focused on the loading and saving of transforms
'''

from scipy import *

import nornir_imageregistration.transforms.meshwithrbffallback as meshwithrbffallback
import numpy as np
import utils

def TransformToIRToolsString(transformObj, bounds=None):
    if hasattr(transformObj, 'gridWidth') and hasattr(transformObj, 'gridHeight'):
        return _TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight, bounds=bounds)
    else:
        return _TransformToIRToolsString(transformObj, bounds)  # , bounds=NewStosFile.MappedImageDim)


def _TransformToIRToolsGridString(Transform, XDim, YDim, bounds=None):

    numPoints = Transform.points.shape[0]

    # Find the extent of the mapped boundaries
    (left, bottom, right, top) = (None, None, None, None)
    if bounds is None:
        (left, bottom, right, top) = Transform.MappedBounds
    else:
        (left, bottom, right, top) = bounds


    output = "GridTransform_double_2_2 vp " + str(numPoints * 2)

    template = " %(cx)g %(cy)g"

    NumAdded = int(0)
    for CY, CX, MY, MX in Transform.points:
        pstr = template % {'cx' : CX, 'cy' : CY}
        output = output + pstr
        NumAdded = NumAdded + 1

   # print str(NumAdded) + " points added"

    boundsStr = " ".join(map(str, [left, bottom, right - left, top - bottom]))
    output = output + " fp 7 0 " + str(int(YDim - 1)) + " " + str(int(XDim - 1)) + " " + boundsStr

    return output


def _TransformToIRToolsString(Transform, bounds=None):
    numPoints = Transform.points.shape[0]

    # Find the extent of the mapped boundaries
    (left, bottom, right, top) = (None, None, None, None)
    if bounds is None:
        (left, bottom, right, top) = Transform.MappedBounds
    else:
        (left, bottom, right, top) = bounds

    output = "MeshTransform_double_2_2 vp " + str(numPoints * 4)

    template = " %(mx)g %(my)g %(cx)g %(cy)g"

    width = right - left
    height = top - bottom

    for CY, CX, MY, MX in Transform.points:
        pstr = template % {'cx' : CX, 'cy' : CY, 'mx' : (MX - left) / width , 'my' : (MY - bottom) / height}
        output = output + pstr

    boundsStr = " ".join(map(str, [left, bottom, width, height]))
    # boundsStr = " ".join(map(str, [0, 0, width, height]))
    output = output + " fp 8 0 16 16 " + boundsStr + ' ' + str(numPoints)

    return output


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
        elif iVP > 0 and i > iVP + 1:
            VariableParameters.append(float(val))

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
        pixelSpacing = 1

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    gridWidth = int(FixedParameters[2]) + 1
    gridHeight = int(FixedParameters[1]) + 1

    ImageWidth = int(FixedParameters[5]) * pixelSpacing
    ImageHeight = int(FixedParameters[6]) * pixelSpacing

    PointPairs = []

    for i in range(0, len(VariableParameters) - 1, 2):
        iY = (i / 2) / gridWidth
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
        pixelSpacing = 1

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    Left = float(FixedParameters[3]) * pixelSpacing
    Bottom = float(FixedParameters[4]) * pixelSpacing
    ImageWidth = int(FixedParameters[5]) * pixelSpacing
    ImageHeight = int(FixedParameters[6]) * pixelSpacing

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
        pixelSpacing = 1

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    # We don't support anything but translation from this transform at the moment:
    # assert(VariableParameters == [1, 0, 1, 1, 1, 0])  # Sequence for transform only transformation

    Left = 0 * pixelSpacing
    Bottom = 0 * pixelSpacing
    ImageWidth = int(FixedParameters[2]) * pixelSpacing * 2.0
    ImageHeight = int(FixedParameters[3]) * pixelSpacing * 2.0

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
        pixelSpacing = 1

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    gridWidth = int(float(FixedParameters[2])) + 1
    gridHeight = int(float(FixedParameters[1])) + 1

    Left = float(FixedParameters[3]) * pixelSpacing
    Bottom = float(FixedParameters[4]) * pixelSpacing
    ImageWidth = int(FixedParameters[5]) * pixelSpacing
    ImageHeight = int(FixedParameters[6]) * pixelSpacing

    PointPairs = []

    for i in range(0, len(VariableParameters) - 1, 4):
        iY = i / gridWidth
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


def CreateRigidTransform(FixedImageSize, WarpedImageSize, rangle, warped_offset):
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
    WarpedPoints = GetTransformedRigidCornerPoints(WarpedImageSize, rangle=0, offset=(0, 0))

    ControlPoints = np.append(FixedPoints, WarpedPoints, 1)

    transform = meshwithrbffallback.MeshWithRBFFallback(ControlPoints)

    return transform


def GetTransformedRigidCornerPoints(size, rangle, offset):
    '''Returns positions of the four corners of a warped image in a fixed space using the rotation and peak offset.
       return [BotLeft, TopLeft, BotRight, TopRight]'''

    CenteredRotation = utils.RotationMatrix(rangle)

    BotLeft = CenteredRotation * matrix([[-(size[0] - 1) / 2.0], [-(size[1] - 1) / 2.0], [1]])
    TopLeft = CenteredRotation * matrix([[-(size[0] - 1) / 2.0], [(size[1] - 1) / 2.0], [1]])
    BotRight = CenteredRotation * matrix([[(size[0] - 1) / 2.0], [-(size[1] - 1) / 2.0], [1]])
    TopRight = CenteredRotation * matrix([[(size[0] - 1) / 2.0], [(size[1] - 1) / 2.0], [1]])

    Translation = matrix([[1, 0, (size[0] - 1) / 2.0], [0, 1, (size[1] - 1) / 2.0], [0, 0, 1]])

    BotLeft = Translation * BotLeft
    BotRight = Translation * BotRight
    TopLeft = Translation * TopLeft
    TopRight = Translation * TopRight

    # Adjust to the peak location
    PeakTranslation = matrix([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])

    BotLeft = PeakTranslation * BotLeft
    TopLeft = PeakTranslation * TopLeft
    BotRight = PeakTranslation * BotRight
    TopRight = PeakTranslation * TopRight

    arr = np.vstack([BotLeft[:2].getA1(), TopLeft[:2].getA1(), BotRight[:2].getA1(), TopRight[:2].getA1()])
    return arr