import copy

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

import nornir_imageregistration
import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import distance, ITransform, IControlPoints, IGridTransform, IRigidTransform
from nornir_imageregistration.transforms.rigid import (
    _NEGLIGIBLE_ANGLE_RADIANS,
    _NEGLIGIBLE_SCALE_DEVIATION,
)


def CentroidToVertexDistance(Centroids, TriangleVerts):
    """Minimum distance from each centroid to the vertices of its corresponding triangle.

    :param Centroids: Nx2 array of centroid points.
    :param TriangleVerts: Nx3x2 array of triangle vertices (3 points per triangle).
    :return: 1D array of length N (minimum centroid-to-vertex distance per row), same
        array module as *Centroids*.
    """
    xp = cp.get_array_module(Centroids)
    # Per-row 1×3 pairwise is a nearest-vertex check, not a cdist matrix.
    # CuVS launch cost dominates at this size; a vectorized norm stays on *xp*.
    diffs = Centroids[:, None, :] - TriangleVerts
    return xp.min(xp.linalg.norm(diffs, axis=-1), axis=1)


def AddTransforms(BToC_Unaltered_Transform: ITransform, AToB_mapped_Transform: IControlPoints,
                  EnrichTolerance: float | None = None,
                  create_copy: bool = True) -> ITransform:
    """Compose A->B and B->C to produce control points for a transform from A to C.

    :param BToC_Unaltered_Transform: Transform from space B to space C (unchanged).
    :param AToB_mapped_Transform: Control-point transform from space A to space B.
    :param EnrichTolerance: If set and point count < 250, enrich control points within this tolerance.
    :param create_copy: If True, return a new transform; if False, may modify AToB_mapped_Transform in place.
    :return: IControlPoints transform mapping A to C (type matches AToB when possible).
    """

    if isinstance(AToB_mapped_Transform, nornir_imageregistration.transforms.RigidTranslation):
        return _AddRigidTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform)
    elif isinstance(AToB_mapped_Transform, nornir_imageregistration.IGridTransform):
        return _AddGridTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform)

    elif isinstance(AToB_mapped_Transform, IControlPoints):
        if AToB_mapped_Transform.points.shape[0] < 250 and EnrichTolerance:
            return _AddAndEnrichTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, epsilon=EnrichTolerance,  # type: ignore[return-value]
                                           create_copy=create_copy)
        else:
            return _AddMeshTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, create_copy)  # type: ignore[return-value]
    else:
        raise ValueError(
            f'Unexpected transform types:\n A to B is {AToB_mapped_Transform.__class__}\n B to C is {BToC_Unaltered_Transform.__class__}')


def _rigid_is_pure_translation(transform: IRigidTransform) -> bool:
    """True when composition may add target_offset vectors without rotation/scale."""
    angle = float(getattr(transform, 'angle', 0.0) or 0.0)
    scalar = float(getattr(transform, 'scalar', 1.0))
    flip = bool(getattr(transform, 'flip_ud', False))
    return (abs(angle) < _NEGLIGIBLE_ANGLE_RADIANS
            and abs(scalar - 1.0) < _NEGLIGIBLE_SCALE_DEVIATION
            and not flip)


def _compose_rigid_via_point_fit(BToC_Unaltered_Transform: ITransform,
                                 AToB_mapped_Transform: IRigidTransform) -> ITransform:
    """Build A→C by fitting a rigid/similarity to B→C ∘ A→B on a local point cloud."""
    from nornir_imageregistration.transforms.converters import EstimateRigidComponentsFromControlPoints

    center = np.asarray(
        AToB_mapped_Transform.source_space_center_of_rotation, dtype=np.float64).ravel()[:2]
    offsets = np.array(
        [[0.0, 0.0], [25.0, 0.0], [0.0, 25.0], [25.0, 25.0],
         [-20.0, 10.0], [10.0, -20.0], [15.0, -15.0], [-15.0, -10.0]],
        dtype=np.float64)
    source = center + offsets
    target = BToC_Unaltered_Transform.Transform(AToB_mapped_Transform.Transform(source))
    source_np = nornir_imageregistration.EnsureNumpyArray(source)
    target_np = nornir_imageregistration.EnsureNumpyArray(target)
    components = EstimateRigidComponentsFromControlPoints(target_np, source_np)

    if components.reflected or not np.isclose(components.scale, 1.0):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=components.translation,
            source_rotation_center=components.source_rotation_center,
            angle=components.angle,
            scalar=components.scale,
            flip_ud=components.reflected)
    if abs(components.angle) < _NEGLIGIBLE_ANGLE_RADIANS:
        return nornir_imageregistration.transforms.RigidTranslation(
            target_offset=components.translation,
            source_rotation_center=components.source_rotation_center)
    return nornir_imageregistration.transforms.Rigid(
        target_offset=components.translation,
        source_rotation_center=components.source_rotation_center,
        angle=components.angle)


def _AddRigidTransforms(BToC_Unaltered_Transform: ITransform,
                        AToB_mapped_Transform: IRigidTransform):
    if isinstance(BToC_Unaltered_Transform, nornir_imageregistration.transforms.RigidTranslation):
        if (_rigid_is_pure_translation(AToB_mapped_Transform)
                and _rigid_is_pure_translation(BToC_Unaltered_Transform)):
            target_offset = (np.asarray(AToB_mapped_Transform.target_offset, dtype=np.float64).ravel()[:2]
                             + np.asarray(BToC_Unaltered_Transform.target_offset, dtype=np.float64).ravel()[:2])
            return nornir_imageregistration.transforms.RigidTranslation(
                target_offset=target_offset,
                source_rotation_center=AToB_mapped_Transform.source_space_center_of_rotation)
        return _compose_rigid_via_point_fit(BToC_Unaltered_Transform, AToB_mapped_Transform)
    elif isinstance(BToC_Unaltered_Transform, nornir_imageregistration.transforms.IGridTransform):
        old_grid = BToC_Unaltered_Transform.grid
        new_grid = nornir_imageregistration.ITKGridDivision(source_shape=old_grid.source_shape,
                                                            # Ideally this shape is the shape of the image the rigid transform is transforming
                                                            cell_size=old_grid.cell_size,
                                                            grid_dims=old_grid.grid_dims)
        AToB_target_points = AToB_mapped_Transform.Transform(new_grid.SourcePoints)
        AToC_target_points = BToC_Unaltered_Transform.Transform(AToB_target_points)
        new_grid.TargetPoints = AToC_target_points
        return nornir_imageregistration.transforms.GridWithRBFFallback(new_grid)
    elif isinstance(BToC_Unaltered_Transform, nornir_imageregistration.transforms.IControlPoints):
        AToC_source_points = AToB_mapped_Transform.InverseTransform(BToC_Unaltered_Transform.SourcePoints)  # type: ignore[attr-defined]
        xp = cp.get_array_module(BToC_Unaltered_Transform.TargetPoints, AToC_source_points)  # type: ignore[attr-defined]
        AToC_pointPairs = xp.hstack((BToC_Unaltered_Transform.TargetPoints, AToC_source_points))  # type: ignore[attr-defined]
        return nornir_imageregistration.transforms.MeshWithRBFFallback(AToC_pointPairs)

    raise NotImplementedError()


def _AddGridTransforms(BToC_Unaltered_Transform: ITransform,
                       AToB_mapped_Transform: IGridTransform):
    # Source stays A-space grid geometry; target is B→C(A→B targets).
    mappedControlPoints = AToB_mapped_Transform.TargetPoints  # type: ignore[attr-defined]
    txMappedControlPoints = BToC_Unaltered_Transform.Transform(mappedControlPoints)

    old_grid = AToB_mapped_Transform.grid
    new_grid = nornir_imageregistration.ITKGridDivision(source_shape=old_grid.source_shape,
                                                        cell_size=old_grid.cell_size,
                                                        grid_dims=old_grid.grid_dims)
    new_grid.TargetPoints = txMappedControlPoints
    new_transform = nornir_imageregistration.transforms.GridWithRBFFallback(new_grid)
    return new_transform


def _AddMeshTransforms(BToC_Unaltered_Transform: ITransform,
                       AToB_mapped_Transform: IControlPoints,
                       create_copy: bool = True):
    mappedControlPoints = AToB_mapped_Transform.TargetPoints
    txMappedControlPoints = BToC_Unaltered_Transform.Transform(mappedControlPoints)

    xp = cp.get_array_module(txMappedControlPoints, AToB_mapped_Transform.SourcePoints)
    AToC_pointPairs = xp.hstack((txMappedControlPoints, AToB_mapped_Transform.SourcePoints))

    newTransform = None
    if create_copy:
        newTransform = copy.deepcopy(AToB_mapped_Transform)
        newTransform.points = AToC_pointPairs  # type: ignore[misc]
        return newTransform
    else:
        AToB_mapped_Transform.points = AToC_pointPairs  # type: ignore[misc]
        return AToB_mapped_Transform


def _AddAndEnrichTransforms(BToC_Unaltered_Transform: ITransform, AToB_mapped_Transform: IControlPoints, epsilon=None,
                            create_copy=True):
    A_To_B_Transform = AToB_mapped_Transform
    B_To_C_Transform = BToC_Unaltered_Transform

    # print("Begin enrichment with %d verticies" % np.shape(A_To_B_Transform.points)[0])

    PointsAdded = True
    while PointsAdded:

        A_To_C_Transform = _AddMeshTransforms(BToC_Unaltered_Transform, A_To_B_Transform, create_copy=True)

        A_Centroids = A_To_B_Transform.GetWarpedCentroids()  # type: ignore[attr-defined]

        #   B_Centroids = A_To_B_Transform.transform(A_Centroids)
        # Get the centroids from B using A-B transform that correspond to A_Centroids
        B_Centroids = A_To_B_Transform.GetFixedCentroids(A_To_B_Transform.WarpedTriangles)  # type: ignore[attr-defined]

        # Warp the same centroids using both A->C and A->B transforms
        OC_Centroids = B_To_C_Transform.Transform(B_Centroids)
        AC_Centroids = A_To_C_Transform.Transform(A_Centroids)  # type: ignore[attr-defined]

        # Follow the arrays from Transform() (may be CuPy even when control points are NumPy).
        xp = cp.get_array_module(OC_Centroids, AC_Centroids, A_Centroids, B_Centroids)
        A_Centroids = xp.asarray(A_Centroids)
        B_Centroids = xp.asarray(B_Centroids)
        OC_Centroids = xp.asarray(OC_Centroids)
        AC_Centroids = xp.asarray(AC_Centroids)

        # Measure the discrepancy in the the results and create a bool array indicating which centroids failed
        Distances = distance(OC_Centroids, AC_Centroids)
        CentroidMisplaced = Distances > epsilon  # type: ignore[operator]

        # SciPy Delaunay.simplices are always NumPy; convert the mask at that host boundary
        # before indexing triangles, then index SourcePoints with the integer result.
        warped_triangles = A_To_B_Transform.WarpedTriangles  # type: ignore[attr-defined]
        if cp.get_array_module(CentroidMisplaced) is cp:
            misplaced_for_tri = np.asarray(CentroidMisplaced.get())
        else:
            misplaced_for_tri = np.asarray(CentroidMisplaced)

        # In extreme distortion we don't want to add new control points forever or converge on existing control points.
        # So ignore centroids falling too close to an existing vertex
        CentroidVertexDistances = xp.zeros(CentroidMisplaced.shape, dtype=Distances.dtype)
        if xp.any(CentroidMisplaced):
            source_points = xp.asarray(A_To_B_Transform.SourcePoints)  # type: ignore[attr-defined]
            A_CentroidTriangles = source_points[warped_triangles[misplaced_for_tri]]
            CentroidVertexDistances[CentroidMisplaced] = CentroidToVertexDistance(
                A_Centroids[CentroidMisplaced],
                A_CentroidTriangles)
        CentroidFarEnough = CentroidVertexDistances > epsilon

        # Add new verticies for the qualifying centroids
        AddCentroid = xp.logical_and(CentroidMisplaced, CentroidFarEnough)
        PointsAdded = bool(xp.any(AddCentroid))

        if PointsAdded:
            New_ControlPoints = xp.hstack((B_Centroids[AddCentroid], A_Centroids[AddCentroid]))
            starting_num_points = A_To_B_Transform.points.shape[0]
            A_To_B_Transform.AddPoints(New_ControlPoints)  # type: ignore[attr-defined]
            ending_num_points = A_To_B_Transform.points.shape[0]

            # If we have the same number of points after adding we must have had some duplicates in either fixed or warped space.  Continue onward
            if starting_num_points == ending_num_points:
                break

            # print("Mean Centroid Error: %g" % np.mean(Distances[AddCentroid]))
            # print("Added %d centroids, %d centroids OK" % (np.sum(AddCentroid), np.shape(AddCentroid)[0] - np.sum(AddCentroid)))
            # print("Total Verticies %d" % np.shape(A_To_B_Transform.points)[0])

            # TODO: Preserve the array indicating passing centroids to the next loop and do not repeat the test to save time.

    # print("End enrichment")

    if create_copy:
        output_transform = copy.deepcopy(AToB_mapped_Transform)
        output_transform.points = A_To_C_Transform.points  # type: ignore[misc, attr-defined]
        return output_transform
    else:
        AToB_mapped_Transform.points = A_To_C_Transform.points  # type: ignore[misc, attr-defined]
        return AToB_mapped_Transform


def AddTransformsWithLinearCorrection(BToC_Unaltered_Transform: ITransform, AToB_mapped_Transform: IControlPoints,
                                      EnrichTolerance: float | None = None,
                                      create_copy: bool = True,
                                      min_blend: float | None = None,
                                      travel_limit: float | None = None,
                                      ignore_rotation: bool = False,
                                      reblend_iterations: int = 1,
                                      reblend_tolerance: float | None = None,
                                      reblend_weight_tolerance: float | None = None,
                                      max_blend: float | None = None,
                                      B_To_C_Linear: ITransform | None = None,
                                      *,
                                      linear_factor: float | None = None):
    '''Takes the control points of a mapping from A to B and returns control points mapping from A to C
    :param BToC_Unaltered_Transform:
    :param AToB_mapped_Transform:
    :param EnrichTolerance:
    :param bool create_copy: True if a new transform should be returned.  If false replace the passed A to B transform points.  Default is True.
    :param B_To_C_Linear: Optional pre-composed rigid B→C (chain-consistent linear target). When None, rigid-fit B→C mesh.
    :return: ndarray of points that can be assigned as control points for a transform'''

    nonlinear_transform = AddTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, EnrichTolerance, True)
    if B_To_C_Linear is not None:
        linear_BToC_Ttransform = B_To_C_Linear
    else:
        linear_BToC_Ttransform = nornir_imageregistration.transforms.converters.ConvertTransformToRigidTransform(
            BToC_Unaltered_Transform,
            ignore_rotation=ignore_rotation)
    linear_transform = AddTransforms(linear_BToC_Ttransform, AToB_mapped_Transform, EnrichTolerance, True)

    blend_kwargs: dict = {
        'min_blend': min_blend,
        'travel_limit': travel_limit,
        'reblend_iterations': reblend_iterations,
        'linear_factor': linear_factor,
    }
    if reblend_tolerance is not None:
        blend_kwargs['reblend_tolerance'] = reblend_tolerance
    if reblend_weight_tolerance is not None:
        blend_kwargs['reblend_weight_tolerance'] = reblend_weight_tolerance
    if max_blend is not None:
        blend_kwargs['max_blend'] = max_blend

    blended_transform = nornir_imageregistration.transforms.utils.BlendTransformsIteratively(
        nonlinear_transform,  # type: ignore[arg-type]
        linear_transform,
        **blend_kwargs)

    return blended_transform
