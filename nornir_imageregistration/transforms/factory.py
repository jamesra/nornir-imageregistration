"""
Created on Nov 13, 2012

@author: u0490822

The factory is focused on the loading and saving of transforms
"""

from collections.abc import Iterable
from typing import Sequence

import numpy as np

import nornir_imageregistration.transforms
from nornir_imageregistration.spatial import *
from nornir_imageregistration.transforms.base import *
from . import float_to_shortest_string


def TransformToIRToolsString(transformObj, bounds=None):
    return transformObj.ToITKString()

    # if hasattr(transformObj, 'gridWidth') and hasattr(transformObj, 'gridHeight'):
    #     return _TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight,
    #                                          bounds=bounds)
    # if isinstance(transformObj, nornir_imageregistration.transforms.RigidNoRotation):
    #     return transformObj.ToITKString()
    # else:
    #     return _TransformToIRToolsString(transformObj, bounds)  # , bounds=NewStosFile.MappedImageDim)


def _GetMappedBoundsExtents(transform, bounds=None):
    '''
    Find the extent of the mapped boundaries
    '''
    (bottom, left, top, right) = (None, None, None, None)
    if bounds is None:
        (bottom, left, top, right) = transform.MappedBoundingBox.ToTuple()
    elif isinstance(bounds, Rectangle):
        (bottom, left, top, right) = bounds.BoundingBox
    else:
        (bottom, left, top, right) = bounds

    return bottom, left, top, right


def _TransformToIRToolsGridString(Transform: IControlPoints, XDim: int, YDim: int) -> str:
    """
    Write an ITK GridTransform_double_2_2 string.
    :param Transform:
    :param XDim: Grid dimensions, ITK expects the reported value to be one less than actual
    :param YDim: Grid dimensions, ITK expects the reported value to be one less than actual
    :param bounds:
    :return:
    """

    if not isinstance(Transform, nornir_imageregistration.IControlPoints):
        raise ValueError("Transform must implement IControlPoints to generate an ITK Grid transform")
    numPoints = Transform.SourcePoints.shape[0]

    # Find the extent of the mapped boundaries
    (bottom, left, top, right) = Transform.MappedBoundingBox.ToTuple()
    image_width = (
            right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
    image_height = (top - bottom)

    output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]

    # template = " %(cx).3f %(cy).3f"
    template = " %(cx)s %(cy)s"

    NumAdded = int(0)
    for CY, CX, MY, MX in Transform.points:
        pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
        output.append(pstr)
        NumAdded += 1

    # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
    # then 1024 should be written to the file.
    output.append(f" fp 7 0 {YDim - 1:d} {XDim - 1:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
    transform_string = ''.join(output)

    return transform_string


def _MeshTransformToIRToolsString(Transform: IControlPoints, bounds=None):
    if not isinstance(Transform, IControlPoints):
        raise ValueError("Transform must implement IControlPoints to generate an ITK Mesh transform")

    numPoints = Transform.points.shape[0]

    # Find the extent of the mapped boundaries
    (bottom, left, top, right) = _GetMappedBoundsExtents(Transform, bounds)
    image_width = (
                          right - left) + 1  # We add one because a 10x10 image is mappped from 0,0 to 9,9, which means the bounding box will be Left=0, Right=9, and width is 9 unless we correct for it.
    image_height = (top - bottom) + 1

    output = [f"MeshTransform_double_2_2 vp {numPoints * 4}"]

    # template = " %(mx).10f %(my).10f %(cx).3f %(cy).3f"
    template = " %(mx)s %(my)s %(cx)s %(cy)s"

    width = right - left
    height = top - bottom

    for CY, CX, MY, MX in Transform.points:
        pstr = template % {'cx': float_to_shortest_string(CX, 3),
                           'cy': float_to_shortest_string(CY, 3),
                           'mx': float_to_shortest_string((MX - left) / width, 10),
                           'my': float_to_shortest_string((MY - bottom) / height, 10)}
        output.append(pstr)

    output.append(f" fp 8 0 16 16 {left:g} {bottom:g} {image_width:g} {image_height:g} {numPoints:d}")

    transform_string = ''.join(output)

    return transform_string


def __ParseParameters(parts: Sequence[str]) -> (list[str], list[str]):
    '''Input is a transform split on white space, returns the variable and fixed parameters as a list'''

    iVP = None
    iFP = None

    VariableParameters = []
    FixedParameters = []

    for i, val in enumerate(parts):
        if val == 'vp':
            iVP = i
            iFP = None
        elif val == 'fp':
            iFP = i
            iVP = None
        elif iFP is not None and iFP > 0 and i > iFP + 1:
            FixedParameters.append(float(val))
            if FixedParameters[-1] >= 1.79769e+308:
                raise ValueError("Unexpected value in transform, probably invalid output from ir-tools")
        elif iVP is not None and iVP > 0 and i > iVP + 1:
            VariableParameters.append(float(val))
            if VariableParameters[-1] >= 1.79769e+308:
                raise ValueError("Unexpected value in transform, probably invalid output from ir-tools")

    return VariableParameters, FixedParameters


def SplitTransform(transformstring):
    '''Returns transform name, variable points, fixed points'''
    parts = transformstring.split()
    transformName = parts[0]
    assert (parts[1] == 'vp')

    VariableParts = []
    iVp = 2
    while parts[iVp] != 'fp':
        VariableParts = float(parts[iVp])
        iVp += 1

    # skip vp # entries
    iVp += 2
    FixedParts = []
    for iVp in range(iVp, len(parts)):
        FixedParts = float(parts[iVp])

    return transformName, FixedParts, VariableParts


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
    elif transformType == "CenteredSimilarity2DTransform_double_2_2":
        return ParseCenteredSimilarity2DTransform(parts, pixelSpacing)
    elif transformType == "FixedCenterOfRotationAffineTransform_double_2_2":
        return ParseFixedCenterOfRotationAffineTransform(parts, pixelSpacing)

    raise ValueError(f"LoadTransform was passed an unknown transform type: {transformType}")


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

        # We subtract one from ImageWidth because the pixels are indexed at zero->Width-1
        mappedX = (float(iX) / float(gridWidth - 1)) * ImageWidth
        mappedY = (float(iY) / float(gridHeight - 1)) * ImageHeight
        ControlX = VariableParameters[i]
        ControlY = VariableParameters[i + 1]
        PointPairs.append((ControlY, ControlX, mappedY, mappedX))

    PointPairs = np.array(PointPairs)
    grid = nornir_imageregistration.ITKGridDivision((ImageHeight, ImageWidth),
                                                    cell_size=(256, 256),
                                                    # cell_size doesn't matter for how this object is going to be used
                                                    grid_dims=(gridHeight, gridWidth))
    grid.TargetPoints = PointPairs[:, 0:2]

    # discrete_transform = nornir_imageregistration.transforms.GridTransform(grid)
    # continuous_transform = nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection(grid.SourcePoints, grid.TargetPoints)
    T = nornir_imageregistration.transforms.GridWithRBFFallback(grid)

    # T = nornir_imageregistration.transforms.MeshWithRBFFallback(PointPairs)
    # T.gridWidth = gridWidth
    # T.gridHeight = gridHeight
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

    T = nornir_imageregistration.transforms.MeshWithRBFFallback(PointPairs)
    return T


def ParseLegendrePolynomialTransform(parts, pixelSpacing=None):
    # Example: LegendrePolynomialTransform_double_2_2_1 vp 6 1 0 1 1 1 0 fp 4 10770 -10770 2040 2040

    if pixelSpacing is None:
        pixelSpacing = 1.0

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    X = FixedParameters[0]
    Y = FixedParameters[1]
    target_offset = (Y, X)

    # if VariableParameters[0]
    if not np.array_equal(np.array(VariableParameters), np.array([1, 0, 1, 1, 1, 0])):
        raise ValueError("We don't support anything but translation from polynomial transforms")

    return nornir_imageregistration.transforms.RigidNoRotation(target_offset)

    # # We don't support anything but translation from this transform at the moment:
    # # assert(VariableParameters == [1, 0, 1, 1, 1, 0])  # Sequence for transform only transformation
    #
    # Left = 0 * pixelSpacing
    # Bottom = 0 * pixelSpacing
    # ImageWidth = float(FixedParameters[2]) * pixelSpacing * 2.0
    # ImageHeight = float(FixedParameters[3]) * pixelSpacing * 2.0
    #
    # array = np.array([[Left, Bottom],
    #               [Left, Bottom + ImageHeight],
    #               [Left + ImageWidth, Bottom],
    #               [Left + ImageWidth, Bottom + ImageHeight]])
    # PointPairs = []
    #
    # for i in range(0, 4):
    #     mappedX, mappedY = array[i, :]
    #     ControlX = (FixedParameters[0] * pixelSpacing) + mappedX
    #     ControlY = (FixedParameters[1] * pixelSpacing) + mappedY
    #     PointPairs.append((ControlY, ControlX, mappedY, mappedX))
    #
    # T = nornir_imageregistration.transforms.MeshWithRBFFallback(PointPairs)
    return T


def ParseFixedCenterOfRotationAffineTransform(parts: list[str], pixelSpacing: float = None):
    if pixelSpacing is None:
        pixelSpacing = 1.0

    # FixedCenterOfRotationAffineTransform_double_2_2 vp 8 0.615661 -0.788011 0.788011 0.615661 -7.60573 -17.8375 5.26637e-67 3.06321e-322 fp 2 122 61

    # I do not have a lot of input data to test with, so I'm assuming the center of rotation is in source space
    (VariableParameters, FixedParameters) = __ParseParameters(parts)
    src_image_center_x = float(FixedParameters[0])
    src_image_center_y = float(FixedParameters[1])

    vp = [float(vp) for vp in VariableParameters]

    post_transform_translation_x = vp[4]
    post_transform_translation_y = vp[5]

    matrix = np.array(((vp[1], vp[0]), (vp[3], vp[2])))
    post_transform_translation = np.array((post_transform_translation_y, post_transform_translation_x))

    return nornir_imageregistration.transforms.AffineMatrixTransform(matrix=matrix,
                                                                     pre_transform_translation=np.array(
                                                                         (-src_image_center_y, -src_image_center_x)),
                                                                     post_transform_translation=post_transform_translation)


def ParseRigid2DTransform(parts: Sequence[str], pixelSpacing: float | None = None):
    # Example: Rigid2DTransform_double_2_2 vp 3 0 0 0 fp 2 0 0
    if pixelSpacing is None:
        pixelSpacing = 1.0

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    angle = -float(VariableParameters[0])  # We negate angle to be compatible with ITK
    xoffset = float(VariableParameters[1])
    yoffset = float(VariableParameters[2])

    x_center = float(FixedParameters[0])
    y_center = float(FixedParameters[1])

    target_offset = (yoffset, xoffset)

    if angle == 0:
        return nornir_imageregistration.transforms.RigidNoRotation(target_offset)
    else:
        return nornir_imageregistration.transforms.Rigid(target_offset=target_offset,
                                                         source_rotation_center=(y_center, x_center),
                                                         angle=angle)


def ParseCenteredSimilarity2DTransform(parts: Sequence[str], pixelSpacing=None):
    # Example: CenteredSimilarity2DTransform_double_2_2 vp 6 0 0 0 0 0 0
    if pixelSpacing is None:
        pixelSpacing = 1.0

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    scale = float(VariableParameters[0])
    angle = -float(VariableParameters[1])  # We negate angle to be compatible with ITK
    x_center = float(VariableParameters[2])
    y_center = float(VariableParameters[3])
    xoffset = float(VariableParameters[4])
    yoffset = float(VariableParameters[5])

    target_offset = (yoffset, xoffset)
    source_center = (y_center, x_center)

    if scale == 1.0 and angle == 0:
        return nornir_imageregistration.transforms.RigidNoRotation(target_offset)
    elif scale == 1.0:
        return nornir_imageregistration.transforms.Rigid(target_offset=target_offset,
                                                         source_rotation_center=source_center,
                                                         angle=angle)
    else:
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=target_offset,
                                                                                 source_center=source_center,
                                                                                 angle=angle,
                                                                                 scale=scale)


def __CorrectOffsetForMismatchedImageSizes(
        offset: NDArray[float] | NDArray[int] | tuple[float, float] | tuple[int, int],
        FixedImageShape: NDArray[int],
        MovingImageShape: NDArray[int],
        scale: float = 1.0) -> tuple[float, float]:
    '''
    :param float scale: Scale the movingImageShape by this amount before correcting to match scaling done to the moving image when passed to the registration algorithm
    '''

    if isinstance(scale, float):
        scale = (scale, scale)
    elif isinstance(scale, int):
        scale = float(scale)
        scale = (scale, scale)
    elif not isinstance(scale, Iterable):
        raise NotImplementedError("Unsupported type")

    return (offset[0] + ((FixedImageShape[0] - MovingImageShape[0] * scale[0]) / 2.0),
            offset[1] + ((FixedImageShape[1] - MovingImageShape[1] * scale[1]) / 2.0))


def CreateRigidTransform(warped_offset, rangle: float, target_image_shape, source_image_shape, flip_ud: bool = False):
    '''Returns a transform, the fixed image defines the boundaries of the transform.
       The warped image '''

    use_mesh_transform = flip_ud
    scalar = 1.0
    source_image_shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(source_image_shape)
    target_image_shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(target_image_shape)

    #     if not np.array_equal(source_image_shape, target_image_shape):
    #         shape_ratio = target_image_shape.astype(np.float64) / source_image_shape.astype(np.float64)
    #         if shape_ratio[0] != shape_ratio[1]:
    #             use_mesh_transform = True
    #         else:
    #             scalar = shape_ratio[0]

    if use_mesh_transform:
        return CreateRigidMeshTransform(target_image_shape=target_image_shape,
                                        source_image_shape=source_image_shape,
                                        rangle=rangle,
                                        warped_offset=warped_offset,
                                        flip_ud=flip_ud)

    assert (source_image_shape[0] > 0)
    assert (source_image_shape[1] > 0)
    assert (target_image_shape[0] > 0)
    assert (target_image_shape[1] > 0)

    source_bounding_rect = Rectangle.CreateFromPointAndArea((0, 0), source_image_shape)
    # target_bounding_rect = Rectangle.CreateFromPointAndArea((0, 0), target_image_shape)

    # Subtract 1 because we are defining this transform as a rotation of the center of an image.
    # the image will be indexed from 0 to N-1, so the center point as indexed for a 10x10 image is 4.5 since it is indexed from 0 to 9
    source_rotation_center = source_bounding_rect.Center  # - 0.5

    # Adjust offset for any mismatch in dimensions
    # Adjust the center of rotation to be consistent with the original ir-tools
    AdjustedOffset = __CorrectOffsetForMismatchedImageSizes(warped_offset, target_image_shape,  # - np.array((1, 1)),
                                                            source_image_shape)  # - np.array((1, 1)))

    # The offset is the translation of the warped image over the fixed image.  If we translate 0,0 from the warped space into
    # fixed space we should obtain the warped_offset value
    # TargetPoints = GetTransformedRigidCornerPoints(WarpedImageSize, rangle, AdjustedOffset)
    # SourcePoints = GetTransformedRigidCornerPoints(WarpedImageSize, rangle=0, offset=(0, 0), flip_ud=flip_ud)

    # ControlPoints = np.append(TargetPoints, SourcePoints, 1)

    if rangle != 0:
        transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=AdjustedOffset,
                                                                                      source_rotation_center=source_rotation_center,
                                                                                      angle=rangle,
                                                                                      scalar=scalar)
    else:
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=AdjustedOffset)

    return transform


def CreateRigidMeshTransform(target_image_shape: NDArray[int] | tuple[int, int],
                             source_image_shape: NDArray[int] | tuple[int, int],
                             rangle: float,
                             warped_offset: NDArray | tuple[float, float],
                             flip_ud: bool = False,
                             scale: float = 1.0) -> ITransform:
    '''
    Returns a MeshWithRBFFallback transform, the fixed image defines the boundaries of the transform.
    '''
    source_image_shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(source_image_shape)
    target_image_shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(target_image_shape)

    assert (source_image_shape[0] > 0)
    assert (source_image_shape[1] > 0)
    assert (target_image_shape[0] > 0)
    assert (target_image_shape[1] > 0)

    # Adjust offset for any mismatch in dimensions
    # Adjust the center of rotation to be consistent with the original ir-tools
    AdjustedOffset = __CorrectOffsetForMismatchedImageSizes(warped_offset, target_image_shape, source_image_shape,
                                                            scale)

    return CreateRigidMeshTransformWithOffset(source_image_shape=source_image_shape,
                                              rangle=rangle,
                                              target_space_offset=AdjustedOffset,
                                              flip_ud=flip_ud,
                                              scale=scale)


def CreateRigidMeshTransformWithOffset(source_image_shape: tuple[int, int] | NDArray[int],
                                       rangle: float,
                                       target_space_offset: tuple[float, float] | NDArray[float],
                                       scale: float = 1.0,
                                       flip_ud: bool = False) -> ITransform:
    # The offset is the translation of the warped image over the fixed image.  If we translate 0,0 from the warped space into
    # fixed space we should obtain the warped_offset value
    TargetPoints = GetTransformedRigidCornerPoints(source_image_shape, rangle, target_space_offset, scale=scale)
    SourcePoints = GetTransformedRigidCornerPoints(source_image_shape, rangle=0, offset=(0, 0), flip_ud=flip_ud)

    ControlPoints = np.append(TargetPoints, SourcePoints, 1)

    transform = nornir_imageregistration.transforms.MeshWithRBFFallback(ControlPoints)

    return transform


def GetTransformedRigidCornerPoints(size: tuple[float, float] | NDArray[float], rangle: float, offset: tuple[float, float] | NDArray[float],
                                    flip_ud: bool = False,
                                    scale: float = 1.0) -> NDArray[float]:
    '''Returns positions of the four corners of a warped image in a fixed space using the rotation and peak offset.  Rotation occurs at the center.
       Flip, if requested, is performed before the rotation and translation
    :param flip_ud:
    :param tuple size: (Height, Width)
    :param float rangle: Angle in radians
    :param tuple offset: (Y, X)
    :param float scale: Scale the corners by this factor. Ex: 0.5 produces points that shrink the source space by 50% in target space.
    :return: Nx2 array of points [[BotLeft], [BotRight], [TopLeft],  [TopRight]]
    :rtype: numpy.ndarray
    '''

    # The corners of an X,Y image that starts at 0,0 are located at X-1,Y-1.  So we subtract one from the size
    size = np.array(size, int)  # - np.array((1, 1))
    r = nornir_imageregistration.transforms.Rigid(angle=rangle, target_offset=offset, source_rotation_center=size / 2.0,
                                                  flip_ud=flip_ud)

    ymax, xmax = size
    corners = np.array([[0, 0],
                        [0, xmax],
                        [ymax, 0],
                        [ymax, xmax]])

    out_corners = r.Transform(corners)
    return out_corners
    #
    # c = np.cos(rangle)
    # s = np.sin(rangle)
    #
    # CenteredRotation = np.array([[ c, s],
    #                              [-s, c]])
    #
    # ymax, xmax = size
    #
    # corners = [[0, 0, ymax, ymax],
    #            [0, xmax, 0, xmax]]
    #
    # out_bounds_corners = CenteredRotation @ corners
    # out_plane_shape = (out_bounds_corners.ptp(axis=1) + 0.5).astype(int)
    # #out_bounds = out_bounds.T[:, 0:2]
    # #out_plane_shape = (out_bounds.ptp(axis=0) + 0.5).astype(int)
    #
    # in_center = (size - 1) / 2.0
    # out_center = CenteredRotation @ (out_plane_shape - 1) / 2
    # center_offset = in_center - out_center
    #
    # out_plane_corners = out_bounds_corners.T + center_offset
    # #out_plane_corners = out_plane_corners + offset
    # return out_plane_corners

    # ScaleMatrix = None
    # if scale is not None:
    #     ScaleMatrix = utils.ScaleMatrixXY(scale)
    # else:
    #     ScaleMatrix = np.identity(3)
    #
    # HalfWidth = (size[iArea.Width]) / 2.0
    # HalfHeight = (size[iArea.Height]) / 2.0
    #
    # RotHalfHeight = HalfHeight
    # RotHalfWidth = HalfWidth
    #
    # if not flip_ud:
    #     BotLeft = np.array([[-RotHalfHeight, -RotHalfWidth, 1]])
    #     TopLeft = np.array([[RotHalfHeight, -RotHalfWidth, 1]])
    #     BotRight = np.array([[-RotHalfHeight, RotHalfWidth, 1]])
    #     TopRight = np.array([[RotHalfHeight, RotHalfWidth, 1]])
    # else:
    #     BotLeft = np.array([RotHalfHeight, -RotHalfWidth, 1])
    #     TopLeft = np.array([-RotHalfHeight, -RotHalfWidth, 1])
    #     BotRight = np.array([RotHalfHeight, RotHalfWidth, 1])
    #     TopRight = np.array([-RotHalfHeight, RotHalfWidth, 1])
    #
    # corners = np.vstack((BotLeft, BotRight, TopLeft, TopRight)).T
    #
    # # Adjust to the peak location
    # PeakTranslation = np.array([[1, 0, 0], [0, 1, 0], [offset[iPoint.Y], offset[iPoint.X], 1]])
    #
    # # Center the image
    # # Subtract 0.5 because we are defining this transform as a rotation of the center of an image.
    # # the image will be indexed from 0 to N-1, so the center point as indexed for a 10x10 image is 4.5 since it is indexed from 0 to 9
    # #Translation = np.array([[1, 0, HalfHeight - 0.5], [0, 1, HalfWidth - 0.5], [0, 0, 1]])
    # Translation = np.array([[1, 0, 0], [0, 1, 0], [HalfHeight, HalfWidth, 1]])
    #
    # transform = (ScaleMatrix @ Translation @ PeakTranslation).T @ CenteredRotation
    # #c = np.copy(corners)
    #
    # corners = transform @ corners
    #
    # #corners = CenteredRotation @ corners
    # #corners = corners.T @ Translation
    # #corners = corners @ PeakTranslation
    # #corners = corners @ ScaleMatrix
    #
    # #corners = corners[:, :2]
    # corners = corners.T[:, :2]
    # #if not np.allclose(ct, corners):
    # #    raise ValueError()
    # # arr[:, [0, 1]] = arr[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points
    # return corners
