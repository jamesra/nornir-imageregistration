"""
Created on Nov 13, 2012

@author: u0490822

The factory is focused on the loading and saving of transforms
"""

from collections.abc import Iterable
from typing import Sequence
import math
import numpy as np

try:
    import cupy as cp
    # import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

import nornir_imageregistration.transforms
from nornir_imageregistration.spatial import *
from nornir_imageregistration.transforms.base import *
from . import float_to_shortest_string


def TransformToIRToolsString(transformObj: ITransform, bounds=None) -> str:
    """Return the transform as an ITK-style string. Returns str."""
    return transformObj.ToITKString()


def _GetMappedBoundsExtents(transform, bounds=None):
    """Return the extent of the mapped (source) boundaries as (bottom, left, top, right).

    :param transform: Transform with MappedBoundingBox (or bounds used when bounds is not None).
    :param bounds: Optional Rectangle or (bottom, left, top, right); if None, uses transform.MappedBoundingBox.
    :return: Tuple (bottom, left, top, right) in mapped space.
    """
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
    (bottom, left, top, right) = Transform.MappedBoundingBox.ToTuple()  # type: ignore[attr-defined]
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


def _MeshTransformToIRToolsString(Transform: IControlPoints, bounds=None) -> str:
    """Serialize a control-point (mesh) transform to an ITK MeshTransform_double_2_2 string.

    :param Transform: Transform implementing IControlPoints (e.g. MeshWithRBFFallback).
    :param bounds: Optional Rectangle or (bottom, left, top, right) for mapped extent.
    :return: ITK-format transform string.
    """
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


def __ParseParameters(parts: Sequence[str]) -> tuple[list[float], list[float]]:
    """Parse variable (vp) and fixed (fp) parameter lists from a whitespace-split transform string.

    :param parts: List of tokens from splitting an ITK-style transform string on whitespace.
    :return: Tuple (VariableParameters, FixedParameters) as lists of floats.
    """

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


def SplitTransform(transformstring: str) -> tuple[str, list[float], list[float]]:
    """Split an ITK-style transform string into transform name, variable parameters, and fixed parameters. Returns tuple."""
    parts = transformstring.split()
    transformName = parts[0]
    assert (parts[1] == 'vp')

    VariableParts = []
    iVp = 2
    nVar = int(parts[iVp])
    iVp += 1
    for _ in range(nVar):
        VariableParts.append(float(parts[iVp]))
        iVp += 1

    assert parts[iVp] == 'fp'
    iVp += 1
    nFixed = int(parts[iVp])
    iVp += 1
    FixedParts = []
    for _ in range(nFixed):
        FixedParts.append(float(parts[iVp]))
        iVp += 1

    return transformName, VariableParts, FixedParts


def LoadTransform(Transform: str, pixelSpacing: float | None = None) -> ITransform:
    """Parse a transform string (from a stos or mosaic file) and return an ITransform (Grid, Mesh, Rigid, etc.)."""

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


def _verify_grid_dimension_and_variable_parameters_match(grid_dim: tuple[int, int], variable_parameters: list[float]) -> None:
    """Ensure grid point count is consistent with the number of variable parameters (2 per point).

    ITK grid transforms sometimes store fewer displacement vectors than the declared grid size
    (e.g. boundary nodes that fall outside the image are omitted). We therefore accept any
    num_vp_points <= num_grid_points.  More points than the declared grid is always an error.

    :param grid_dim: (width, height) or (rows, cols) of the grid.
    :param variable_parameters: List of variable parameters (2 per grid point).
    :raises ValueError: If len(variable_parameters) / 2 exceeds the declared grid point count.
    """
    num_grid_points = math.prod(grid_dim)
    num_vp_points = len(variable_parameters) / 2
    if num_vp_points > num_grid_points:
        grid_height, grid_width = grid_dim[0], grid_dim[1]
        raise ValueError(
            f"The grid transform has {num_vp_points} points but declares a grid of {grid_width}x{grid_height} = {num_grid_points} points")


def ParseGridTransform(parts, pixelSpacing: float | None = None):
    """Parse ITK GridTransform_double_2_2 parts into a GridWithRBFFallback transform. Returns ITransform."""
    if pixelSpacing is None:
        pixelSpacing = 1.0

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    gridWidth = int(FixedParameters[2]) + 1
    gridHeight = int(FixedParameters[1]) + 1

    _verify_grid_dimension_and_variable_parameters_match((gridWidth, gridHeight), VariableParameters)

    # Sparse ITK grids omit trailing boundary nodes; pad with zeros (zero displacement = node at its
    # regular grid position) so ITKGridDivision receives the full expected point count.
    num_grid_points = gridWidth * gridHeight
    expected_vp_len = num_grid_points * 2
    if len(VariableParameters) < expected_vp_len:
        VariableParameters = list(VariableParameters) + [0.0] * (expected_vp_len - len(VariableParameters))

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
    grid = nornir_imageregistration.ITKGridDivision((ImageHeight, ImageWidth),  # type: ignore[arg-type]
                                                    cell_size=(256, 256),
                                                    # cell_size doesn't matter for how this object is going to be used
                                                    grid_dims=(gridHeight, gridWidth))
    grid.TargetPoints = PointPairs[:, 0:2]

    # discrete_transform = nornir_imageregistration.transforms.GridTransform(grid)
    # continuous_transform = nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection(grid.SourcePoints, grid.TargetPoints)
    if use_cp:
        # Grid with RBF Fallback, using GPU component
        T = nornir_imageregistration.transforms.GridWithRBFFallback_GPUComponent(grid)
        # Option 1- Direct RBF interpolation on grid (full GPU usage)
        # T = nornir_imageregistration.transforms.GridWithRBFInterpolator_Direct_GPU(grid)
        # Option 2 - RegularGridInterpolator with cupy RBFInterpolator (full GPU usage)
        # T = nornir_imageregistration.transforms.GridWithRBFInterpolator_GPU(grid)
    else:
        # Grid with RBF fallback
        T = nornir_imageregistration.transforms.GridWithRBFFallback(grid)
        # Option 1 - Direct RBF interpolation on grid (experiments on CPU)
        # T = nornir_imageregistration.transforms.GridWithRBFInterpolator_Direct_CPU(grid)
        # Option 2 - RegularGridInterpolator with scipy RBFInterpolator (CPU only)
        # T = nornir_imageregistration.transforms.GridWithRBFInterpolator_CPU(grid)

    # T = nornir_imageregistration.transforms.MeshWithRBFFallback(PointPairs)
    # T.gridWidth = gridWidth
    # T.gridHeight = gridHeight
    return T


def ParseMeshTransform(parts, pixelSpacing: float | None = None):
    """Parse ITK MeshTransform_double_2_2 parts into a MeshWithRBFFallback transform. Returns ITransform."""
    if pixelSpacing is None:
        pixelSpacing = 1.0

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

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

    if use_cp:
        T = nornir_imageregistration.transforms.MeshWithRBFFallback_GPUComponent(PointPairs)  # type: ignore[arg-type]
        # Option - direct RBF interpolation on mesh (for GPU)
        # T = nornir_imageregistration.transforms.MeshWithRBFInterpolator_GPU(PointPairs)
    else:
        T = nornir_imageregistration.transforms.MeshWithRBFFallback(PointPairs)  # type: ignore[arg-type]
        # Option - direct RBF interpolation on mesh (via CPU)
        # T = nornir_imageregistration.transforms.MeshWithRBFInterpolator_CPU(PointPairs)
    return T


def ParseLegendrePolynomialTransform(parts, pixelSpacing: float | None = None):
    """Parse ITK LegendrePolynomialTransform parts; returns RigidTranslation (translation-only support)."""
    if pixelSpacing is None:
        pixelSpacing = 1.0

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    X = FixedParameters[0]
    Y = FixedParameters[1]
    target_offset = (Y, X)

    # if VariableParameters[0]
    if not np.array_equal(np.array(VariableParameters), np.array([1, 0, 1, 1, 1, 0])):
        raise ValueError("We don't support anything but translation from polynomial transforms")

    return nornir_imageregistration.transforms.RigidTranslation(target_offset)

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
    # return T


def ParseFixedCenterOfRotationAffineTransform(parts: list[str], pixelSpacing: float | None = None):
    """Parse ITK FixedCenterOfRotationAffineTransform parts into AffineMatrixTransform. Returns ITransform."""
    if pixelSpacing is None:
        pixelSpacing = 1.0

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    # FixedCenterOfRotationAffineTransform_double_2_2 vp 8 0.615661 -0.788011 0.788011 0.615661 -7.60573 -17.8375 5.26637e-67 3.06321e-322 fp 2 122 61

    # I do not have a lot of input data to test with, so I'm assuming the center of rotation is in source space
    (VariableParameters, FixedParameters) = __ParseParameters(parts)
    src_image_center_x = float(FixedParameters[0])
    src_image_center_y = float(FixedParameters[1])

    vp = [float(vp) for vp in VariableParameters]

    post_transform_translation_x = vp[4]
    post_transform_translation_y = vp[5]

    xp = cp if use_cp else np
    matrix = xp.array(((vp[1], vp[0]), (vp[3], vp[2])))
    post_transform_translation = xp.array((post_transform_translation_y, post_transform_translation_x))
    pre_transform_translation = xp.array((-src_image_center_y, -src_image_center_x))

    if use_cp:
        affine = nornir_imageregistration.transforms.AffineMatrixTransform_GPU(
            matrix=matrix,  # type: ignore[abstract]
            pre_transform_translation=pre_transform_translation,
            post_transform_translation=post_transform_translation)
    else:
        affine = nornir_imageregistration.transforms.AffineMatrixTransform(
            matrix=matrix,  # type: ignore[abstract]
            pre_transform_translation=pre_transform_translation,
            post_transform_translation=post_transform_translation)

    return _try_decompose_affine_to_rigid_with_flip(affine)


def _try_decompose_affine_to_rigid_with_flip(affine) -> ITransform:
    """If *affine* is a Y-reflected similarity, return Rigid/CS2D with flip_ud=True.

    Non-flipped rigids never serialize as Affine; only flipped saves use this ITK
    type. Unreflected affines are left as AffineMatrixTransform.
    """
    try:
        pre = nornir_imageregistration.EnsureNumpyArray(affine.pre_transform_translation)
        center = -np.asarray(pre, dtype=np.float64).ravel()[:2]
        # Sample a small cloud about the center for rigid-component estimation.
        offsets = np.array(
            [[0.0, 0.0], [25.0, 0.0], [0.0, 25.0], [25.0, 25.0],
             [-20.0, 10.0], [10.0, -20.0], [15.0, -15.0], [-15.0, -10.0]],
            dtype=np.float64)
        source = center + offsets
        target = nornir_imageregistration.EnsureNumpyArray(affine.Transform(source))
        target = np.asarray(target, dtype=np.float64)
        components = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(
            target, source)
        if not components.reflected:
            return affine
        rebuilt = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=components.translation,
            source_rotation_center=components.source_rotation_center,
            angle=components.angle,
            scalar=components.scale,
            flip_ud=True)
        rebuilt_target = nornir_imageregistration.EnsureNumpyArray(rebuilt.Transform(source))
        residual = float(np.max(np.abs(np.asarray(rebuilt_target, dtype=np.float64) - target)))
        if residual > 0.05:
            return affine
        if np.isclose(components.scale, 1.0):
            return nornir_imageregistration.transforms.Rigid(
                target_offset=components.translation,
                source_rotation_center=components.source_rotation_center,
                angle=components.angle,
                flip_ud=True)
        return rebuilt
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return affine


def ParseRigid2DTransform(parts: Sequence[str], pixelSpacing: float | None = None, negate_angle: bool = False):
    """Parse ITK Rigid2DTransform parts into Rigid2DTransform. negate_angle supports pre-11/28/2023 format. Returns ITransform."""
    # Example: Rigid2DTransform_double_2_2 vp 3 0 0 0 fp 2 0 0
    if pixelSpacing is None:
        pixelSpacing = 1.0

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    angle = float(VariableParameters[0])  # We negate angle to be compatible with ITK
    if negate_angle:
        angle = -angle

    xoffset = float(VariableParameters[1])
    yoffset = float(VariableParameters[2])

    x_center = float(FixedParameters[0])
    y_center = float(FixedParameters[1])

    target_offset = (yoffset, xoffset)

    if angle == 0:
        return nornir_imageregistration.transforms.RigidTranslation(target_offset)
    else:
        return nornir_imageregistration.transforms.Rigid(target_offset=target_offset,
                                                         source_rotation_center=(y_center, x_center),
                                                         angle=angle)


def ParseCenteredSimilarity2DTransform(parts: Sequence[str], pixelSpacing: float | None = None,
                                       negate_angle: bool = False):
    """Parse ITK CenteredSimilarity2DTransform parts into Rigid/RigidTranslation/Similarity2D. Returns ITransform."""
    if pixelSpacing is None:
        pixelSpacing = 1.0

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    (VariableParameters, FixedParameters) = __ParseParameters(parts)

    scale = float(VariableParameters[0])
    angle = -float(VariableParameters[1])  # We negate angle to be compatible with ITK
    if negate_angle:
        angle = -angle
    x_center = float(VariableParameters[2])
    y_center = float(VariableParameters[3])
    xoffset = float(VariableParameters[4])
    yoffset = float(VariableParameters[5])

    target_offset = (yoffset, xoffset)
    source_center = (y_center, x_center)

    if scale == 1.0 and angle == 0:
        return nornir_imageregistration.transforms.RigidTranslation(target_offset)
    elif scale == 1.0:
        return nornir_imageregistration.transforms.Rigid(target_offset=target_offset,
                                                         source_rotation_center=source_center,
                                                         angle=angle)
    else:
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=target_offset,
                                                                                 source_rotation_center=source_center,
                                                                                 angle=angle,
                                                                                 scalar=scale)


def __CorrectOffsetForMismatchedImageSizes(
        offset: NDArray[np.floating] | NDArray[np.integer] | tuple[float, float] | tuple[int, int],
        target_image_shape: NDArray[np.integer],
        source_image_shape: NDArray[np.integer],
        scale: float = 1.0) -> tuple[float, float]:
    '''
    :param float scale: Scale the movingImageShape by this amount before correcting to match scaling done to the moving image when passed to the registration algorithm
    '''

    scale_tuple: tuple[float, float]
    if isinstance(scale, float):
        scale_tuple = (scale, scale)
    elif isinstance(scale, int):
        scale_tuple = (float(scale), float(scale))
    elif isinstance(scale, Iterable):
        scale_tuple = (scale[0], scale[1])  # type: ignore[index]
    else:
        raise NotImplementedError("Unsupported type")

    return (offset[0] + ((target_image_shape[0] - source_image_shape[0] * scale_tuple[0]) / 2.0),  # type: ignore[index]
            offset[1] + ((target_image_shape[1] - source_image_shape[1] * scale_tuple[1]) / 2.0))


def CreateRigidTransform(warped_offset, rangle: float, target_image_shape: NDArray, source_image_shape: NDArray,
                         flip_ud: bool = False) -> ITransform:
    """Create a rigid (or rigid+flip) transform mapping source image into target space.

    Fixed (target) image defines the output boundaries. If flip_ud is True, returns a
    mesh-based rigid transform; otherwise returns CenteredSimilarity2D or RigidTranslation.

    :param warped_offset: (Y, X) translation of warped image in fixed space.
    :param rangle: Rotation angle in radians.
    :param target_image_shape: (Height, Width) of the fixed/target image.
    :param source_image_shape: (Height, Width) of the source image.
    :param flip_ud: If True, use mesh transform to support vertical flip.
    :return: ITransform (RigidTranslation, CenteredSimilarity2D, or MeshWithRBFFallback).
    """

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
        transform = nornir_imageregistration.transforms.RigidTranslation(target_offset=AdjustedOffset)

    return transform


def CreateRigidMeshTransform(target_image_shape: NDArray[np.integer] | tuple[int, int],
                             source_image_shape: NDArray[np.integer] | tuple[int, int],
                             rangle: float,
                             warped_offset: NDArray | tuple[float, float],
                             flip_ud: bool = False,
                             scale: float = 1.0) -> ITransform:
    """Create a rigid mesh transform (MeshWithRBFFallback) for the given image shapes and rotation.

    Fixed (target) image defines the transform boundaries. Offset is adjusted for image size mismatch.

    :param target_image_shape: (Height, Width) of the fixed/target image.
    :param source_image_shape: (Height, Width) of the source image.
    :param rangle: Rotation angle in radians.
    :param warped_offset: (Y, X) translation of warped image in fixed space.
    :param flip_ud: If True, apply vertical flip before rotation/translation.
    :param scale: Scale factor applied to source when computing offset.
    :return: MeshWithRBFFallback transform.
    """
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


def CreateRigidMeshTransformWithOffset(source_image_shape: tuple[int, int] | NDArray[np.integer],
                                       rangle: float,
                                       target_space_offset: tuple[float, float] | NDArray[np.floating],
                                       scale: float = 1.0,
                                       flip_ud: bool = False) -> ITransform:
    """Build a MeshWithRBFFallback from source shape, rotation, and target-space offset.

    Offset is the translation of the warped image in fixed space (0,0 in warped → offset in fixed).

    :param source_image_shape: (Height, Width) of the source image.
    :param rangle: Rotation angle in radians.
    :param target_space_offset: (Y, X) translation in fixed/target space.
    :param scale: Scale factor for corner positions.
    :param flip_ud: If True, apply vertical flip.
    :return: MeshWithRBFFallback transform.
    """
    TargetPoints = GetTransformedRigidCornerPoints(source_image_shape, rangle, target_space_offset, scale=scale)  # type: ignore[arg-type]
    SourcePoints = GetTransformedRigidCornerPoints(source_image_shape, rangle=0, offset=(0, 0), flip_ud=flip_ud)  # type: ignore[arg-type]

    ControlPoints = np.append(TargetPoints, SourcePoints, 1)

    transform = nornir_imageregistration.transforms.MeshWithRBFFallback(ControlPoints)

    return transform


def GetTransformedRigidCornerPointsForImage(size: tuple[int, int] | NDArray[np.integer],
                                            rangle: float,
                                            offset: tuple[float, float] | NDArray[np.floating],
                                            flip_ud: bool = False,
                                            scale: float = 1.0) -> NDArray[np.floating]:
    """Return rigid-transformed corner points using image-style center (dimension - 1) / 2.

    Delegates to GetTransformedRigidCornerPoints with size adjusted so rotation center is at
    (H-1)/2, (W-1)/2 (pixel center convention).

    :param size: (Height, Width) of the image.
    :param rangle: Rotation angle in radians.
    :param offset: (Y, X) translation in target space.
    :param flip_ud: If True, apply vertical flip.
    :param scale: Scale factor for corners.
    :return: Nx2 array of corner positions in fixed space (same order as GetTransformedRigidCornerPoints).
    """
    return GetTransformedRigidCornerPoints(size - 1, rangle, offset, flip_ud, scale)  # type: ignore[arg-type]


def GetTransformedRigidCornerPoints(size: tuple[float, float] | NDArray[np.floating],
                                    rangle: float,
                                    offset: tuple[float, float] | NDArray[np.floating],
                                    flip_ud: bool = False,
                                    scale: float = 1.0) -> NDArray[np.floating]:
    """Return positions of the four corners of a rectangle in fixed space after rigid transform.

    Rotation is about the center (size / 2). If flip_ud is True, flip is applied before
    rotation and translation.

    :param size: (Height, Width) of the source rectangle.
    :param rangle: Rotation angle in radians.
    :param offset: (Y, X) translation in target space.
    :param flip_ud: If True, flip vertically before rotation and translation.
    :param scale: Scale factor for corners (only 1.0 is implemented).
    :return: Nx2 array of corner points: bottom-left, bottom-right, top-left, top-right.
    :raises NotImplementedError: If scale is not 1.0.
    """

    if scale is not None and scale != 1.0:
        raise NotImplementedError('scale')

    size = np.array(size, int)
    r = nornir_imageregistration.transforms.Rigid(angle=rangle, target_offset=offset, source_rotation_center=size / 2.0,
                                                  flip_ud=flip_ud)

    ymax, xmax = size
    corners = np.array([[0, 0],
                        [0, xmax],
                        [ymax, 0],
                        [ymax, xmax]])

    out_corners = r.Transform(corners)
    return out_corners
