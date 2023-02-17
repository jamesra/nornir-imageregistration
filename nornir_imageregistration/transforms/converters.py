import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple


import nornir_imageregistration
from nornir_imageregistration.transforms import TransformType, ITransform, IControlPoints

class RigidComponents(NamedTuple):
    source_rotation_center: NDArray[float]
    angle: float
    scale: float
    translation: NDArray[float]
    reflected: bool


def _kabsch_umeyama(target_points: NDArray[float], source_points: NDArray[float]) -> tuple[NDArray[float], float, NDArray[float]]:
    '''
    This function is used to get the translation, rotation and scaling factors when aligning
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
    #VarB = np.mean(np.linalg.norm(centered_B, axis=1) ** 2)

    H = (centered_A.T @ centered_B) / num_pts
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (num_dims - 1) + [d])

    source_rotation_center = EB
    rotation_matrix = U @ S @ VT
    scale = VarA / np.trace(np.diag(D) @ S)
    #Total translation, does not factor in translation of B to EB for rotation
    translation = EA - (scale * rotation_matrix @ EB) + EB
    reflected = d < 0

    return source_rotation_center, rotation_matrix, scale, translation, reflected

def EstimateRigidComponentsFromControlPoints(target_points: NDArray[float], source_points: NDArray[float]) -> RigidComponents:
    source_rotation_center, rotation_matrix, scale, translation, reflected = _kabsch_umeyama(target_points, source_points)
    
    #My rigid transform is probably written in a weird way.  It translates source points to the center of rotation, translates them back, and then 
    #performs the final translation into target space.
    adjusted_translation = np.mean(target_points, axis=0) - np.mean(source_points, axis=0)
    rotate_angle = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
    if rotate_angle <= -np.pi * 2:
        rotate_angle += np.pi * 2

    return RigidComponents(source_rotation_center=source_rotation_center, angle=rotate_angle,
                           translation=adjusted_translation, scale=scale, reflected=reflected)

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

    if input.type == nornir_imageregistration.transforms.TransformType.RIGID:
        return ConvertTransformToRigidTransform(input, **kwargs)

    if input.type == nornir_imageregistration.transforms.TransformType.MESH:
        return ConvertTransformToMeshTransform(input, **kwargs)

    if input.type == nornir_imageregistration.transforms.TransformType.GRID:
        return ConvertTransformToGridTransform(input, **kwargs)

    raise NotImplemented()

def ConvertTransformToRigidTransform(input_transform: ITransform):
    if isinstance(input_transform, IControlPoints):
        components = EstimateRigidComponentsFromControlPoints(input_transform.TargetPoints,
                                                                             input_transform.SourcePoints) 

        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=components.translation,
                                                                                 source_rotation_center=components.source_rotation_center,
                                                                                 angle=components.angle,
                                                                                 scalar=components.scale, 
                                                                                 flip_ud=components.reflected)

    if isinstance(input_transform, nornir_imageregistration.transforms.CenteredSimilarity2DTransform):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(target_offset=input_transform.target_offset,
                                                                                 source_rotation_center=input_transform.source_rotation_center,
                                                                                 angle=input_transform.angle,
                                                                                 scalar=input_transform.scalar)
    elif isinstance(input_transform, nornir_imageregistration.transforms.Rigid):
        return nornir_imageregistration.transforms.Rigid(
            target_offset=input_transform.target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle)
    elif isinstance(input_transform, nornir_imageregistration.transforms.RigidNoRotation):
        return nornir_imageregistration.transforms.RigidNoRotation(
            target_offset=input_transform.target_offset,
            source_rotation_center=input_transform.source_rotation_center,
            angle=input_transform.angle)

    raise NotImplemented()


def ConvertTransformToMeshTransform(input_transform: ITransform,
                                    source_image_shape: NDArray | None = None) -> ITransform:
    if isinstance(input_transform, IControlPoints):
        return nornir_imageregistration.transforms.MeshWithRBFFallback(input_transform.points)

    if isinstance(input_transform, nornir_imageregistration.transforms.Rigid):
        return nornir_imageregistration.transforms.factory.CreateRigidMeshTransformWithOffset(
            source_image_shape=source_image_shape,
            rangle=input_transform.angle,
            target_space_offset=input_transform.target_offset)

    raise NotImplemented()


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

    t = nornir_imageregistration.transforms.triangulation.Triangulation(point_pairs)
    t.gridWidth = grid_data.grid_dims[1]
    t.gridHeight = grid_data.grid_dims[0]

    return t
