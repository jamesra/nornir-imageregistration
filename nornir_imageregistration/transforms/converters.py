from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

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
    '''
    This function is used to get the translation, rotation and scaling factors when aligning
    points in B on reference points in A.

    The R,c,t componenets once return can be used to obtain B'
    '''
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


def EstimateRigidComponentsFromControlPoints(target_points: NDArray[np.floating],
                                             source_points: NDArray[np.floating],
                                             ignore_rotation: bool = False) -> RigidComponents:
    source_rotation_center = np.zeros((1, 2))

    if not ignore_rotation:
        source_rotation_center, rotation_matrix, scale, translation, reflected = _kabsch_umeyama(target_points,
                                                                                                 source_points)

        # My rigid transform is probably written in a weird way.  It translates source points to the center of rotation, translates them back, and then
        # performs the final translation into target space.
        adjusted_translation = np.mean(target_points - source_points, axis=0) + translation
        rotate_angle = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
        if rotate_angle <= -tau:
            rotate_angle += tau
        elif rotate_angle >= tau:
            rotate_angle -= tau

        return RigidComponents(source_rotation_center=np.zeros((1, 2), float), angle=rotate_angle,
                               translation=translation, scale=scale, reflected=reflected)

    else:
        adjusted_translation = np.mean(target_points - source_points, axis=0)
        scale, translation = _kabsch_umeyama_translation_scaling(target_points, source_points)
        return RigidComponents(source_rotation_center=np.zeros((1, 2), float), angle=0,
                               translation=translation, scale=scale, reflected=False)


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
        components = EstimateRigidComponentsFromControlPoints(input_transform.TargetPoints,
                                                              input_transform.SourcePoints,
                                                              ignore_rotation)

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
