import copy

import numpy as np
import scipy

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import distance, ITransform, IControlPoints, IGridTransform
from Tools.scripts.objgraph import ignore


def CentroidToVertexDistance(Centroids, TriangleVerts):
    '''
    :param ndarray Centroids: An Nx2 array of centroid points
    :param ndarray TriangleVerts: An Nx3x2 array of verticies of triangles
    '''
    numCentroids = Centroids.shape[0]
    distance = np.zeros(numCentroids)
    for i in range(0, Centroids.shape[0]):
        distances = scipy.spatial.distance.cdist([Centroids[i]], TriangleVerts[i])
        distance[i] = np.min(distances)

    return distance


def AddTransforms(BToC_Unaltered_Transform: ITransform, AToB_mapped_Transform: IControlPoints,
                  EnrichTolerance: float | None = None,
                  create_copy: bool = True):
    '''Takes the control points of a mapping from A to B and returns control points mapping from A to C
    :param BToC_Unaltered_Transform:
    :param AToB_mapped_Transform:
    :param EnrichTolerance:
    :param bool create_copy: True if a new transform should be returned.  If false replace the passed A to B transform points.  Default is True.
    :return: ndarray of points that can be assigned as control points for a transform'''

    if isinstance(AToB_mapped_Transform, nornir_imageregistration.IGridTransform):
        return _AddGridTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform)

    if AToB_mapped_Transform.points.shape[0] < 250 and EnrichTolerance:
        return _AddAndEnrichTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, epsilon=EnrichTolerance,
                                       create_copy=create_copy)
    else:
        return _AddMeshTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, create_copy)


def _AddGridTransforms(BToC_Unaltered_Transform: ITransform,
                       AToB_mapped_Transform: IGridTransform):
    mappedControlPoints = AToB_mapped_Transform.TargetPoints
    txMappedControlPoints = BToC_Unaltered_Transform.Transform(mappedControlPoints)

    AToC_pointPairs = np.hstack((txMappedControlPoints, AToB_mapped_Transform.SourcePoints))

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

    AToC_pointPairs = np.hstack((txMappedControlPoints, AToB_mapped_Transform.SourcePoints))

    newTransform = None
    if create_copy:
        newTransform = copy.deepcopy(AToB_mapped_Transform)
        newTransform.points = AToC_pointPairs
        return newTransform
    else:
        AToB_mapped_Transform.points = AToC_pointPairs
        return AToB_mapped_Transform


def _AddAndEnrichTransforms(BToC_Unaltered_Transform: ITransform, AToB_mapped_Transform: IControlPoints, epsilon=None,
                            create_copy=True):
    A_To_B_Transform = AToB_mapped_Transform
    B_To_C_Transform = BToC_Unaltered_Transform

    # print("Begin enrichment with %d verticies" % np.shape(A_To_B_Transform.points)[0])

    PointsAdded = True
    while PointsAdded:

        A_To_C_Transform = _AddMeshTransforms(BToC_Unaltered_Transform, A_To_B_Transform, create_copy=True)

        A_Centroids = A_To_B_Transform.GetWarpedCentroids()

        #   B_Centroids = A_To_B_Transform.Transform(A_Centroids)
        # Get the centroids from B using A-B transform that correspond to A_Centroids
        B_Centroids = A_To_B_Transform.GetFixedCentroids(A_To_B_Transform.WarpedTriangles)

        # Warp the same centroids using both A->C and A->B transforms
        OC_Centroids = B_To_C_Transform.Transform(B_Centroids)
        AC_Centroids = A_To_C_Transform.Transform(A_Centroids)

        # Measure the discrepancy in the the results and create a bool array indicating which centroids failed
        Distances = distance(OC_Centroids, AC_Centroids)
        CentroidMisplaced = Distances > epsilon

        # In extreme distortion we don't want to add new control points forever or converge on existing control points.
        # So ignore centroids falling too close to an existing vertex
        CentroidVertexDistances = np.zeros(CentroidMisplaced.shape, np.bool)
        A_CentroidTriangles = A_To_B_Transform.SourcePoints[A_To_B_Transform.WarpedTriangles[CentroidMisplaced]]
        CentroidVertexDistances[CentroidMisplaced] = CentroidToVertexDistance(A_Centroids[CentroidMisplaced],
                                                                              A_CentroidTriangles)
        CentroidFarEnough = CentroidVertexDistances > epsilon

        # Add new verticies for the qualifying centroids
        AddCentroid = np.logical_and(CentroidMisplaced, CentroidFarEnough)
        PointsAdded = np.any(AddCentroid)

        if PointsAdded:
            New_ControlPoints = np.hstack((B_Centroids[AddCentroid], A_Centroids[AddCentroid]))
            starting_num_points = A_To_B_Transform.points.shape[0]
            A_To_B_Transform.AddPoints(New_ControlPoints)
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
        output_transform.points = A_To_C_Transform.points
        return output_transform
    else:
        AToB_mapped_Transform.points = A_To_C_Transform.points
        return AToB_mapped_Transform
    
def AddTransformsWithLinearCorrection(BToC_Unaltered_Transform: ITransform, AToB_mapped_Transform: IControlPoints,
                  EnrichTolerance: float | None = None,
                  create_copy: bool = True,
                  linear_factor: float | None = None,
                  travel_limit: float | None = None,
                  ignore_rotation: bool = False):
    '''Takes the control points of a mapping from A to B and returns control points mapping from A to C
    :param BToC_Unaltered_Transform:
    :param AToB_mapped_Transform:
    :param EnrichTolerance:
    :param bool create_copy: True if a new transform should be returned.  If false replace the passed A to B transform points.  Default is True.
    :return: ndarray of points that can be assigned as control points for a transform'''
    
    nonlinear_transform = AddTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, EnrichTolerance, True)
    linear_BToC_Ttransform = nornir_imageregistration.transforms.converters.ConvertTransformToRigidTransform(BToC_Unaltered_Transform,
                                                                                                       ignore_rotation=ignore_rotation)
    linear_transform = AddTransforms(linear_BToC_Ttransform, AToB_mapped_Transform, EnrichTolerance, True)
    
    
    blended_transform = nornir_imageregistration.transforms.utils.BlendTransforms(nonlinear_transform, linear_transform, linear_factor=linear_factor,
                                                              travel_limit=travel_limit)
    
    return blended_transform
     
