from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray
import scipy

import nornir_imageregistration
from nornir_imageregistration.transforms import IControlPoints, ITransform, TransformType
from nornir_imageregistration.transforms.pointrelations import ControlPointRelation, \
    calculate_control_points_relationship

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

tau = np.pi * 2


class RigidComponents(NamedTuple):
    source_rotation_center: NDArray[np.floating]
    angle: float
    scale: float
    translation: NDArray[np.floating]
    reflected: bool


def _kabsch_umeyama(target_points: NDArray[np.floating], source_points: NDArray[np.floating]) -> tuple[
    NDArray[np.floating], float, NDArray[np.floating]]:
    """
    This function is used to get the translation, rotation and scaling factors when aligning
    points in B on reference points in A.

    The R,c,t componenets once return can be used to obtain B'

    To be compatible with Rigid transforms used by nornir the order of operations must be
    1. Scaling
    2. Rotation
    3. Translation
    4. Flip
    """
    A = target_points.astype(np.float64, copy=False)
    B = source_points.astype(np.float64, copy=False)
    assert A.shape == B.shape
    num_pts, num_dims = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    source_rotation_center = EB
    centered_A = A - EA
    centered_B = B - EB
    VarA = np.mean(np.linalg.norm(centered_A, axis=1) ** 2)
    # VarB = np.mean(np.linalg.norm(centered_B, axis=1) ** 2)

    H = (centered_A.T @ centered_B) / num_pts
    U, D, VT = np.linalg.svd(H)
    # VT = VT.T # The determinate approach to flip detection was not working according to hypothesis testing
    # d = np.sign(np.linalg.det(VT.T @ U))
    # reflected = d < 0
    relation = calculate_control_points_relationship(source_points, target_points)
    reflected = relation == ControlPointRelation.FLIPPED
    d = -1 if reflected else 1
    if reflected:
        sp = np.array(source_points)
        sp -= source_rotation_center  # Flip points across center
        sp[:, 0] = -sp[:, 0]
        sp += source_rotation_center
        ignore_source_rotation_center, rotation_matrix, scale, translation, ignore_reflected = _kabsch_umeyama(
            target_points, sp)
        if not np.allclose(ignore_source_rotation_center, source_rotation_center):
            raise ArithmeticError("Flipping control points should not change the center of rotation")

        return source_rotation_center, rotation_matrix, scale, translation, True

    S = np.diag([1] * (num_dims - 1) + [d])
    # if reflected:
    #    U[:, -1] = -U[:, -1]

    rotation_matrix = U @ S @ VT

    sp = source_points

    scale = VarA / np.trace(np.diag(D) @ S)

    # Total translation, does not factor in translation of B to EB for rotation
    translation = EA - (scale * rotation_matrix @ EB)

    # if reflected:
    #    S = np.diag([1] * (num_dims - 1) + [d])
    #    rotation_matrix = U @ S @ VT 
    #    rotation_matrix[1, 1] = -rotation_matrix[1, 1]
    #    rotation_matrix[0, 0] = -rotation_matrix[0, 0]

    return source_rotation_center, rotation_matrix, scale, translation, reflected


def _kabsch_umeyama_translation_scaling(target_points: NDArray[np.floating], source_points: NDArray[np.floating]) -> \
        tuple[
            NDArray[np.floating], float, NDArray[np.floating]]:
    '''
    This function is used to get the translation and scaling factors when aligning
    points in B on reference points in A.

    The R,c,t componenets once return can be used to obtain B'
    '''
    A = target_points
    B = source_points
    assert A.shape == B.shape
    num_pts, num_dims = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    centered_A = A - EA
    centered_B = B - EB
    VarA = np.mean(np.linalg.norm(centered_A, axis=1) ** 2)
    # VarB = np.mean(np.linalg.norm(centered_B, axis=1) ** 2)

    H = (centered_A.T @ centered_B) / num_pts
    D = np.linalg.svd(H, compute_uv=False)

    scale = VarA / np.trace(np.diag(D))
    # Total translation, does not factor in translation of B to EB for rotation
    translation = EA - (scale * EB)

    return scale, translation


def EstimateScale(source_points: NDArray[np.floating],
                  target_points: NDArray[np.floating]) -> float:
    """
    Given a set of two points, estimate the scale factor to achieve the same root mean square distance to the origin.
    Assumes the points in the transform have been centered around the origin and not translated.
    :param source_points: 
    :param target_points: 
    :return: 
    """

    mean_source_points = np.mean(source_points, axis=0)
    mean_target_points = np.mean(target_points, axis=0)

    centered_source_points = source_points - mean_source_points
    centered_target_points = target_points - mean_target_points

    target_rms = np.sum(np.sqrt(np.sum(centered_target_points ** 2, axis=1)))
    source_rms = np.sum(np.sqrt(np.sum(centered_source_points ** 2, axis=1)))

    scale = target_rms / source_rms
    return scale


def EstimateRigidComponentsFromControlPoints(target_points: NDArray[np.floating],
                                             source_points: NDArray[np.floating]) -> RigidComponents:
    xp = cp.get_array_module(source_points)

    num_pts, m = source_points.shape

    source_center = xp.mean(source_points, axis=0)
    target_center = xp.mean(target_points, axis=0)
    centered_source_points = source_points - source_center
    centered_target_points = target_points - target_center

    scale_estimate = nornir_imageregistration.transforms.converters.EstimateScale(centered_source_points,
                                                                                  centered_target_points)

    ###################################################################################
    # We know the scale now, remove the scalar from the target_points, and
    # determine the rotation
    ###################################################################################

    unscaled_target_points = target_points / scale_estimate
    unscaled_target_center = xp.mean(unscaled_target_points, axis=0)
    unscaled_centered_target_points = unscaled_target_points - unscaled_target_center

    zeros_z_column = xp.zeros((num_pts, 1))
    rotation = scipy.spatial.transform.Rotation.align_vectors(
        xp.hstack((zeros_z_column, centered_source_points)),
        xp.hstack(
            (zeros_z_column, unscaled_centered_target_points))
    )
    euler_angles = rotation[0].as_euler('zyx')
    estimated_angle = euler_angles[2]

    # Ensure the angle is in the range of -pi to pi
    if estimated_angle <= -xp.pi:
        estimated_angle += xp.pi * 2

    ###################################################################################
    # Determine if the transform is reflected
    relation = nornir_imageregistration.transforms.converters.calculate_control_points_relationship(source_points,
                                                                                                    target_points)
    reflected = relation == nornir_imageregistration.transforms.ControlPointRelation.FLIPPED

    if relation == nornir_imageregistration.transforms.ControlPointRelation.COLINEAR:
        raise ValueError("Colinear points detected")

    ###################################################################################
    # The angle and reflection is estimated.  We remove the angle and reflection from the target
    # points and determine translation
    ###################################################################################

    rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(estimated_angle)

    estimated_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
        target_offset=xp.zeros((2,)),
        source_rotation_center=source_center,
        angle=estimated_angle,
        scalar=scale_estimate,
        flip_ud=reflected)

    test_target_points = estimated_transform.Transform(source_points)
    test_target_center = test_target_points.mean(axis=0)
    tranlsation_estimate = target_center - test_target_center

    return RigidComponents(source_rotation_center=source_center, angle=estimated_angle,
                           translation=tranlsation_estimate, scale=scale_estimate, reflected=reflected)


def ConvertTransform(input: ITransform, transform_type: TransformType,
                     **kwargs) -> ITransform:
    """
    Creates a new transform that is as close as possible to the input transform
    :param input:
    :param transform_type:
    :return:
    """
    if input.type == transform_type:
        return input

    if transform_type == nornir_imageregistration.transforms.TransformType.RIGID:
        return ConvertTransformToRigidTransform(input, **kwargs)

    if transform_type == nornir_imageregistration.transforms.TransformType.MESH:
        return ConvertTransformToMeshTransform(input, **kwargs)

    if transform_type == nornir_imageregistration.transforms.TransformType.GRID:
        return ConvertTransformToGridTransform(input, **kwargs)

    if transform_type == nornir_imageregistration.transforms.TransformType.RBF:
        return ConvertTransformToRBFTransform(input, **kwargs)

    raise NotImplemented()


def ConvertRigidTransformToCenteredSimilarityTransform(input_transform: ITransform):
    if isinstance(input_transform, nornir_imageregistration.transforms.CenteredSimilarity2DTransform):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle,
            scalar=input_transform.scalar)
    elif isinstance(input_transform, nornir_imageregistration.transforms.Rigid):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle,
            scalar=input_transform.scalar)
    elif isinstance(input_transform, nornir_imageregistration.transforms.RigidNoRotation):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle,
            scalar=input_transform.scalar)

    raise NotImplemented()


def ConvertTransformToRigidTransform(input_transform: ITransform, ignore_rotation: bool = False, **kwargs):
    if isinstance(input_transform, IControlPoints):
        if ignore_rotation:
            raise ValueError("Ignore rotation is no longer supported for control points")
        components = EstimateRigidComponentsFromControlPoints(input_transform.TargetPoints,
                                                              input_transform.SourcePoints)

        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=components.translation,
                                                                                 source_rotation_center=components.source_rotation_center,
                                                                                 angle=components.angle,
                                                                                 scalar=components.scale,
                                                                                 flip_ud=components.reflected)

    if isinstance(input_transform, nornir_imageregistration.transforms.CenteredSimilarity2DTransform):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle,
            scalar=input_transform.scalar)
    elif isinstance(input_transform, nornir_imageregistration.transforms.Rigid):
        return nornir_imageregistration.transforms.Rigid(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle)
    elif isinstance(input_transform, nornir_imageregistration.transforms.RigidNoRotation):
        return nornir_imageregistration.transforms.RigidNoRotation(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle)

    raise NotImplemented()


def ConvertTransformToMeshTransform(input_transform: ITransform,
                                    source_image_shape: NDArray | None = None) -> ITransform:
    if isinstance(input_transform, IControlPoints):
        return nornir_imageregistration.transforms.MeshWithRBFFallback(input_transform.points)

    if isinstance(input_transform, nornir_imageregistration.transforms.Rigid) or \
            isinstance(input_transform, nornir_imageregistration.transforms.RigidNoRotation):
        control_points = GetControlPointsForRigidTransform(input_transform, source_image_shape)
        transform = nornir_imageregistration.transforms.MeshWithRBFFallback(control_points)
        return transform

    raise NotImplemented()


def GetTargetSpaceCornerPoints(input_transform: ITransform,
                               source_image_shape: NDArray) -> NDArray[np.floating]:
    ymax, xmax = source_image_shape
    corners = np.array([[0, 0],
                        [0, xmax],
                        [ymax, 0],
                        [ymax, xmax]])
    return input_transform.Transform(corners)


def GetControlPointsForRigidTransform(input_transform: ITransform,
                                      source_image_shape: NDArray) -> NDArray[np.floating]:
    ymax, xmax = source_image_shape
    corners = np.array([[0, 0],
                        [0, xmax],
                        [ymax, 0],
                        [ymax, xmax]])
    out_corners = input_transform.Transform(corners)
    return np.append(out_corners, corners, 1)


def ConvertTransformToGridTransform(input_transform: ITransform, source_image_shape: NDArray,
                                    cell_size: NDArray | None = None, grid_dims: NDArray | None = None,
                                    grid_spacing: NDArray | None = None) -> ITransform:
    """
    Converts a set of EnhancedAlignmentRecord peaks from the _RefineGridPointsForTwoImages function into a transform

    """

    grid_data = nornir_imageregistration.ITKGridDivision(source_image_shape, cell_size=cell_size,
                                                         grid_spacing=grid_spacing, grid_dims=grid_dims)
    grid_data.PopulateTargetPoints(input_transform)

    point_pairs = np.hstack((grid_data.TargetPoints, grid_data.SourcePoints))

    # TODO, create a specific grid transform object that uses numpy's RegularGridInterpolator

    return nornir_imageregistration.transforms.GridWithRBFFallback(grid_data)


def ConvertTransformToRBFTransform(input_transform: ITransform,
                                   source_image_shape: NDArray | None = None) -> ITransform:
    """
    Converts a set of EnhancedAlignmentRecord peaks from the _RefineGridPointsForTwoImages function into a transform
    """

    if isinstance(input_transform, IControlPoints):
        return nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection(input_transform.SourcePoints,
                                                                                 input_transform.TargetPoints)
    # elif isinstance(input_transform, nornir_imageregistration.transforms.RigidNoRotation):
    # TargetPoints = GetTransformedRigidCornerPoints(source_image_shape, input_transform.angle, target_space_offset, scale=scale)
    # SourcePoints = GetTransformedRigidCornerPoints(source_image_shape, rangle=0, offset=(0, 0), flip_ud=flip_ud)

    # TODO, create a specific grid transform object that uses numpy's RegularGridInterpolator

    raise NotImplementedError()
