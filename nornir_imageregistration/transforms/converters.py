from typing import NamedTuple
import enum
import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.transforms import IControlPoints, ITransform, TransformType

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

tau = np.pi * 2
# _cross_product_abs_tolerance = 0.0001
_cross_product_abs_tolerance = None


class RigidComponents(NamedTuple):
    source_rotation_center: NDArray[np.floating]
    angle: float
    scale: float
    translation: NDArray[np.floating]
    reflected: bool


class ControlPointRelation(enum.IntEnum):
    LINEAR = 0  # Points are not flipped or colinear
    FLIPPED = 1
    COLINEAR = 2


def _get_pointset_crossproducts(points: NDArray[np.floating]) -> float:
    """
    Returns the cross products of the vectors between the first point to other points in the array
    """
    global _cross_product_abs_tolerance
    xp = cp.get_array_module(points)

    if points.shape[0] < 3:
        raise ValueError("Need at least 3 control points to determine if flipped")

    # Calculate vectors
    vectors = xp.diff(points, axis=0)  # Only need 2 vectors for speed

    # Grid transforms in particular have may colinear points.  So we start our search for a non-zero cross product
    # at the end of the list, and continue until we have two non-zero cross products
    cross_products = xp.cross(vectors[0], vectors[1:])

    return cross_products


def _get_pointset_crossproduct(points: NDArray[np.floating]) -> float:
    """
    Returns the product of the vectors between the first and last point in the array
    that creates a non-zero cross product if it exists.
    """
    cross_products = _get_pointset_crossproducts(points)

    # Grid transforms in particular have may colinear points.  So we start our search for a non-zero cross product
    # at the end of the list, and continue until we have two non-zero cross products
    i_max_cross = np.argmax(abs(cross_products))

    return cross_products[i_max_cross]


def are_points_colinear(points: NDArray[np.floating]) -> ControlPointRelation:
    """
    Returns true if the points are colinear. False otherwise.
    Assume source_points and target_points are numpy arrays of shape (n, 2)
    where n is the number of points and 2 corresponds to the x and y coordinates of each point
    """
    global _cross_product_abs_tolerance
    cross = _get_pointset_crossproduct(points)
    xp = cp.get_array_module(points)

    # Check if flippednp.isclose(cross, 0)
    # If either cross product is close to zero the points are colinear.
    cross = 0 if np.isclose(cross, 0) else cross

    # Both point sets are colinear
    if cross == 0:
        return ControlPointRelation.COLINEAR

    return ControlPointRelation.LINEAR if xp.sign(cross) >= 0 else ControlPointRelation.FLIPPED


def get_control_point_relationship(source_points, target_points) -> ControlPointRelation:
    """
    Returns true if the control points are flipped on an axis. False otherwise.
    Assume source_points and target_points are numpy arrays of shape (n, 2)
    where n is the number of points and 2 corresponds to the x and y coordinates of each point
    """

    xp = cp.get_array_module(source_points)

    # TODO, make sure cross product is from the same pair of points in both sets, and they are the maximum cross product possible
    source_crosses = _get_pointset_crossproducts(source_points)
    target_crosses = _get_pointset_crossproducts(target_points)

    source_crosses[np.isclose(source_crosses, 0)] = 0
    target_crosses[np.isclose(target_crosses, 0)] = 0

    non_zero_crosses = np.logical_and(source_crosses != 0, target_crosses != 0)

    source_cross_signs = np.sign(source_crosses[non_zero_crosses])
    target_cross_signs = np.sign(target_crosses[non_zero_crosses])

    # Check how many cross products have matching signs
    sign_comparisons = source_cross_signs == target_cross_signs
    num_matching_signs = np.sum(sign_comparisons)
    if num_matching_signs > len(sign_comparisons) / 2:
        return ControlPointRelation.LINEAR

    return ControlPointRelation.FLIPPED

    # Check the signs of the cross products, and if more matches are found than not, return that result
    sign_comparisons = np.sign(source_crosses) == np.sign(target_crosses)

    if source_cross == ControlPointRelation.COLINEAR and target_cross == ControlPointRelation.COLINEAR:
        return ControlPointRelation.COLINEAR

    # If the cross product is zero the points are colinear, use the cross product of the non-colinear points
    if source_cross == ControlPointRelation.COLINEAR:
        return ControlPointRelation.LINEAR if xp.sign(target_cross) >= 0 else ControlPointRelation.FLIPPED

    if target_cross == ControlPointRelation.COLINEAR:
        return ControlPointRelation.LINEAR if xp.sign(source_cross) >= 0 else ControlPointRelation.FLIPPED

    return ControlPointRelation.FLIPPED if xp.sign(source_cross) != xp.sign(
        target_cross) else ControlPointRelation.LINEAR


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
    relation = get_control_point_relationship(source_points, target_points)
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
