from __future__ import annotations

from typing import Any, NamedTuple
import math
import numpy as np
from numpy.typing import NDArray
import scipy

import nornir_imageregistration
from nornir_imageregistration import cp
from nornir_imageregistration.spatial_distance import array_to_numpy_host
from nornir_imageregistration.transforms import IControlPoints, ITransform, TransformType
from nornir_imageregistration.transforms.pointrelations import ControlPointRelation, \
    calculate_control_points_relationship

tau = np.pi * 2


class RigidComponents(NamedTuple):
    source_rotation_center: NDArray[np.floating]
    angle: float
    scale: float
    translation: NDArray[np.floating]
    reflected: bool


def _kabsch_umeyama(target_points: NDArray[np.floating], source_points: NDArray[np.floating]) -> tuple[
    NDArray[np.floating], NDArray[np.floating], float, NDArray[np.floating], bool]:
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
        tuple[float, NDArray[np.floating]]:
    '''
    This function is used to get the translation and scaling factors when aligning
    points in B on reference points in A.

    The R,c,t componenets once return can be used to obtain B'
    '''
    # Host-only: uses NumPy linear algebra; CuPy inputs are transferred once at this boundary.
    A = array_to_numpy_host(target_points)
    B = array_to_numpy_host(source_points)
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

    return float(scale), translation


def EstimateScale(source_points: NDArray[np.floating],
                  target_points: NDArray[np.floating]) -> float:
    """
    Given a set of two points, estimate the scale factor to achieve the same root mean square distance to the origin.
    Assumes the points in the transform have been centered around the origin and not translated.
    :param source_points: 
    :param target_points: 
    :return: 
    """
    xp_src = cp.get_array_module(source_points)
    xp_tgt = cp.get_array_module(target_points)
    xp = cp if (xp_src is cp or xp_tgt is cp) else np
    source_points = xp.asarray(source_points)
    target_points = xp.asarray(target_points)

    mean_source_points = xp.mean(source_points, axis=0)
    mean_target_points = xp.mean(target_points, axis=0)

    centered_source_points = source_points - mean_source_points
    centered_target_points = target_points - mean_target_points

    target_rms = xp.sum(xp.sqrt(xp.sum(centered_target_points ** 2, axis=1)))
    source_rms = xp.sum(xp.sqrt(xp.sum(centered_source_points ** 2, axis=1)))

    scale = target_rms / source_rms
    return float(scale)


def _coerce_to_array_module(arr: NDArray[np.floating], xp: Any) -> NDArray[np.floating]:
    """Place *arr* on *xp* without a host round-trip when already resident there."""
    if cp.get_array_module(arr) is xp:
        return arr
    if xp is np:
        return nornir_imageregistration.EnsureNumpyArray(arr)
    return xp.asarray(arr)


def _translation_only_rigid_components(
        source_points: NDArray[np.floating],
        target_points: NDArray[np.floating],
        reflected: bool,
        xp: Any) -> RigidComponents:
    """Centroid translation when rotation or scale cannot be estimated."""
    source_center = xp.mean(source_points, axis=0)
    target_center = xp.mean(target_points, axis=0)
    estimated_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
        target_offset=xp.zeros((2,)),
        source_rotation_center=source_center,
        angle=0.0,
        scalar=1.0,
        flip_ud=reflected)
    test_target_points = _coerce_to_array_module(
        estimated_transform.Transform(source_points), xp)
    test_target_center = xp.mean(test_target_points, axis=0)
    return RigidComponents(
        source_rotation_center=source_center,
        angle=0.0,
        scale=1.0,
        translation=target_center - test_target_center,
        reflected=reflected)


def _rotation_vectors_are_usable(vecs_a: NDArray[np.floating], vecs_b: NDArray[np.floating]) -> bool:
    """Return False when scipy ``align_vectors`` would see a degenerate embedding."""
    if not (np.isfinite(vecs_a).all() and np.isfinite(vecs_b).all()):
        return False
    return float(np.linalg.norm(vecs_a)) >= 1e-12 and float(np.linalg.norm(vecs_b)) >= 1e-12


def EstimateRigidComponentsFromControlPoints(target_points: NDArray[np.floating],
                                             source_points: NDArray[np.floating],
                                             reflected_override: bool | None = None) -> RigidComponents:
    xp_src = cp.get_array_module(source_points)
    xp_tgt = cp.get_array_module(target_points)
    xp = cp if (xp_src is cp or xp_tgt is cp) else np
    source_points = xp.asarray(source_points)
    target_points = xp.asarray(target_points)

    num_pts, m = source_points.shape

    source_center = xp.mean(source_points, axis=0)
    target_center = xp.mean(target_points, axis=0)
    centered_source_points = source_points - source_center
    centered_target_points = target_points - target_center

    if reflected_override is not None:
        reflected = reflected_override
    else:
        relation = nornir_imageregistration.transforms.converters.calculate_control_points_relationship(
            source_points, target_points)
        reflected = relation == nornir_imageregistration.transforms.ControlPointRelation.FLIPPED
        if relation == nornir_imageregistration.transforms.ControlPointRelation.COLINEAR:
            raise ValueError("Colinear points detected")

    centered_source_np = array_to_numpy_host(centered_source_points)
    if (not np.isfinite(centered_source_np).all()
            or float(np.linalg.norm(centered_source_np)) < 1e-12):
        return _translation_only_rigid_components(source_points, target_points, reflected, xp)

    with np.errstate(divide='ignore', invalid='ignore'):
        scale_estimate = nornir_imageregistration.transforms.converters.EstimateScale(
            centered_source_points, centered_target_points)
    if not math.isfinite(float(scale_estimate)) or abs(float(scale_estimate)) < 1e-15:
        return _translation_only_rigid_components(source_points, target_points, reflected, xp)

    ###################################################################################
    # We know the scale now, remove the scalar from the target_points, and
    # determine the rotation
    ###################################################################################

    unscaled_target_points = target_points / scale_estimate
    unscaled_target_center = xp.mean(unscaled_target_points, axis=0)
    unscaled_centered_target_points = unscaled_target_points - unscaled_target_center

    zeros_z_column = xp.zeros((num_pts, 1))
    vecs_a = xp.hstack((zeros_z_column, centered_source_points))
    vecs_b = xp.hstack((zeros_z_column, unscaled_centered_target_points))
    # scipy.spatial.transform has no CuPy implementation; host arrays only.
    vecs_a_np = array_to_numpy_host(vecs_a)
    vecs_b_np = array_to_numpy_host(vecs_b)
    if not _rotation_vectors_are_usable(vecs_a_np, vecs_b_np):
        return _translation_only_rigid_components(source_points, target_points, reflected, xp)
    try:
        rotation = scipy.spatial.transform.Rotation.align_vectors(vecs_a_np, vecs_b_np)
    except np.linalg.LinAlgError:
        # Collapsed or ill-conditioned meshes (common after a sparse refine) make SVD fail.
        return _translation_only_rigid_components(source_points, target_points, reflected, xp)
    euler_angles = rotation[0].as_euler('zyx')
    estimated_angle = float(euler_angles[2])

    # Ensure the angle is in the range of -pi to pi
    if estimated_angle <= -math.pi or math.isclose(estimated_angle, -math.pi, abs_tol=1e-10):
        estimated_angle += math.pi * 2

    ###################################################################################
    # The angle and reflection is estimated.  We remove the angle and reflection from the target
    # points and determine translation
    ###################################################################################

    # rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(estimated_angle)

    estimated_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
        target_offset=xp.zeros((2,)),
        source_rotation_center=source_center,
        angle=estimated_angle,
        scalar=scale_estimate,
        flip_ud=reflected)

    test_target_points = _coerce_to_array_module(
        estimated_transform.Transform(source_points), xp)
    test_target_center = xp.mean(test_target_points, axis=0)
    tranlsation_estimate = target_center - test_target_center

    return RigidComponents(source_rotation_center=source_center, angle=estimated_angle,
                           translation=tranlsation_estimate, scale=float(scale_estimate), reflected=reflected)


def _batched_ring_reflected(source_rings: NDArray[np.floating],
                            target_rings: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Vectorized reflection flags for ``(N, R, 2)`` control-point rings."""
    if source_rings.shape[1] < 3:
        raise ValueError("Need at least 3 control points to determine if flipped")

    src_vectors = np.diff(source_rings, axis=1)
    tgt_vectors = np.diff(target_rings, axis=1)
    # signed_cross_product_2d(v0, v1:) per ring: v0_y * v_x - v0_x * v_y
    source_crosses = (src_vectors[:, 0:1, 1] * src_vectors[:, 1:, 0]
                      - src_vectors[:, 0:1, 0] * src_vectors[:, 1:, 1])
    target_crosses = (tgt_vectors[:, 0:1, 1] * tgt_vectors[:, 1:, 0]
                      - tgt_vectors[:, 0:1, 0] * tgt_vectors[:, 1:, 1])
    source_crosses = source_crosses.copy()
    target_crosses = target_crosses.copy()
    source_crosses[np.isclose(source_crosses, 0)] = 0
    target_crosses[np.isclose(target_crosses, 0)] = 0

    src_all_zero = np.all(np.isclose(source_crosses, 0, atol=1e-10), axis=1)
    tgt_all_zero = np.all(np.isclose(target_crosses, 0, atol=1e-10), axis=1)
    if bool(np.any(src_all_zero | tgt_all_zero)):
        raise ValueError("Colinear points detected")

    non_zero = (source_crosses != 0) & (target_crosses != 0)
    # Match scalar: among paired non-zero crosses, flipped if fewer than half share sign.
    sign_match = np.sign(source_crosses) == np.sign(target_crosses)
    match_count = np.sum(sign_match & non_zero, axis=1)
    pair_count = np.sum(non_zero, axis=1)
    # Avoid divide-by-zero; empty pair_count should not occur after colinear check.
    return match_count < (pair_count / 2.0)


def _batched_similarity_angles(centered_source: NDArray[np.floating],
                               unscaled_centered_target: NDArray[np.floating]) -> NDArray[np.floating]:
    """Batched SO(3) Kabsch on planar ``(0, y, x)`` embeddings; return euler-Z angles.

    Matches ``scipy.spatial.transform.Rotation.align_vectors`` used by the scalar
    ``EstimateRigidComponentsFromControlPoints`` path (including reflected rings).
    """
    num_sets, num_pts, _ = centered_source.shape
    vecs_a = np.zeros((num_sets, num_pts, 3), dtype=np.float64)
    vecs_b = np.zeros((num_sets, num_pts, 3), dtype=np.float64)
    vecs_a[:, :, 1:] = centered_source
    vecs_b[:, :, 1:] = unscaled_centered_target
    # Covariance for R @ b ≈ a (same convention as scipy align_vectors).
    covariance = np.einsum('nri,nrj->nij', vecs_b, vecs_a)
    u_mat, _singular, vt_mat = np.linalg.svd(covariance)
    det_signs = np.sign(np.linalg.det(u_mat) * np.linalg.det(vt_mat))
    det_signs = np.where(det_signs == 0.0, 1.0, det_signs)
    correction = np.zeros((num_sets, 3, 3), dtype=np.float64)
    correction[:, 0, 0] = 1.0
    correction[:, 1, 1] = 1.0
    correction[:, 2, 2] = det_signs
    rotation_mats = np.einsum(
        'nij,njk,nkl->nil',
        vt_mat.transpose(0, 2, 1),
        correction,
        u_mat.transpose(0, 2, 1))
    angles = scipy.spatial.transform.Rotation.from_matrix(rotation_mats).as_euler('zyx')[:, 2]
    wrap_mask = (angles <= -math.pi) | np.isclose(angles, -math.pi, atol=1e-10)
    angles = np.where(wrap_mask, angles + (math.pi * 2), angles)
    return angles.astype(np.float64, copy=False)


def _batched_zero_translation_targets(source_rings: NDArray[np.floating],
                                      source_centers: NDArray[np.floating],
                                      angles: NDArray[np.floating],
                                      scales: NDArray[np.floating],
                                      reflected: NDArray[np.bool_]) -> NDArray[np.floating]:
    """Apply CenteredSimilarity2D (t=0) to rings: ``Flip @ R @ s @ (p - c) + c``, with Rigid rounding."""
    # Match Rigid/CenteredSimilarity2DTransform float32 center storage.
    centers = np.asarray(source_centers, dtype=np.float32)
    points = np.asarray(source_rings, dtype=np.float64)
    delta = points - centers[:, None, :]
    scaled = delta * scales.astype(np.float64, copy=False)[:, None, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # RotationMatrix convention: [[c, s], [-s, c]] on (Y, X).
    rot_y = cos_a[:, None] * scaled[:, :, 0] + sin_a[:, None] * scaled[:, :, 1]
    rot_x = -sin_a[:, None] * scaled[:, :, 0] + cos_a[:, None] * scaled[:, :, 1]
    rotated = np.stack((rot_y, rot_x), axis=-1)
    flipped = rotated.copy()
    flipped[:, :, 0] = np.where(reflected[:, None], -rotated[:, :, 0], rotated[:, :, 0])
    targets = flipped + centers[:, None, :]
    precision = nornir_imageregistration.RoundingPrecision(targets.dtype)
    return np.around(targets, decimals=precision)


def EstimateRigidComponentsFromControlPointsBatched(
        source_rings: NDArray[np.floating],
        target_rings: NDArray[np.floating],
        reflected_override: bool | NDArray[np.bool_] | None = None) -> list[RigidComponents]:
    """Estimate rigid/similarity components for many ``(R, 2)`` rings at once.

    Host NumPy after a single transfer. Uses batched SO(3) Kabsch (planar embedding)
    to match scalar ``EstimateRigidComponentsFromControlPoints`` / scipy
    ``Rotation.align_vectors``, avoiding a Python loop of align_vectors calls.

    :param source_rings: Shape ``(N, R, 2)`` source control points.
    :param target_rings: Shape ``(N, R, 2)`` corresponding target points.
    :param reflected_override: Optional per-ring or scalar reflection flag.
    :returns: Length-``N`` list of ``RigidComponents``.
    """
    source = array_to_numpy_host(source_rings).astype(np.float64, copy=False)
    target = array_to_numpy_host(target_rings).astype(np.float64, copy=False)
    if source.ndim != 3 or target.ndim != 3 or source.shape != target.shape or source.shape[-1] != 2:
        raise ValueError(
            f"source_rings/target_rings must share shape (N, R, 2); got {source.shape} and {target.shape}")

    num_sets = int(source.shape[0])
    if num_sets == 0:
        return []

    source_centers = source.mean(axis=1)
    target_centers = target.mean(axis=1)
    centered_source = source - source_centers[:, None, :]
    centered_target = target - target_centers[:, None, :]

    source_rms = np.sqrt(np.sum(centered_source ** 2, axis=2)).sum(axis=1)
    target_rms = np.sqrt(np.sum(centered_target ** 2, axis=2)).sum(axis=1)
    scales = np.divide(
        target_rms, source_rms, out=np.full(num_sets, np.nan, dtype=np.float64), where=source_rms > 1e-12)
    rotation_ok = (
        np.isfinite(source).all(axis=(1, 2))
        & np.isfinite(target).all(axis=(1, 2))
        & np.isfinite(scales)
        & (np.abs(scales) >= 1e-15)
        & (target_rms > 1e-12)
    )
    angles = np.zeros(num_sets, dtype=np.float64)
    scales_out = np.ones(num_sets, dtype=np.float64)
    scales_out[rotation_ok] = scales[rotation_ok]
    if np.any(rotation_ok):
        unscaled_target = target[rotation_ok] / scales[rotation_ok, None, None]
        unscaled_centered_target = unscaled_target - unscaled_target.mean(axis=1)[:, None, :]
        try:
            angles[rotation_ok] = _batched_similarity_angles(
                centered_source[rotation_ok], unscaled_centered_target)
        except np.linalg.LinAlgError:
            override_arr = None if reflected_override is None else np.broadcast_to(
                np.asarray(reflected_override, dtype=bool), (num_sets,))
            return [
                EstimateRigidComponentsFromControlPoints(
                    target[i], source[i],
                    reflected_override=None if override_arr is None else bool(override_arr[i]))
                for i in range(num_sets)
            ]
    scales = scales_out

    if reflected_override is None:
        reflected = np.zeros(num_sets, dtype=bool)
        if np.any(rotation_ok):
            try:
                reflected[rotation_ok] = _batched_ring_reflected(source[rotation_ok], target[rotation_ok])
            except ValueError:
                return [
                    EstimateRigidComponentsFromControlPoints(target[i], source[i])
                    for i in range(num_sets)
                ]
    else:
        reflected = np.broadcast_to(np.asarray(reflected_override, dtype=bool), (num_sets,)).copy()

    test_targets = _batched_zero_translation_targets(
        source, source_centers, angles, scales, reflected)
    translations = target_centers - test_targets.mean(axis=1)

    return [
        RigidComponents(
            source_rotation_center=source_centers[i],
            angle=float(angles[i]),
            scale=float(scales[i]),
            translation=translations[i],
            reflected=bool(reflected[i]),
        )
        for i in range(num_sets)
    ]


def RigidComponentsToCenteredSimilarityTransform(components: RigidComponents
                                                 ) -> nornir_imageregistration.transforms.CenteredSimilarity2DTransform:
    """Build a centered similarity transform from estimated rigid components."""
    return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
        target_offset=components.translation,
        source_rotation_center=components.source_rotation_center,
        angle=components.angle,
        scalar=components.scale,
        flip_ud=components.reflected)


def _rigid_fit_residual(target_points: NDArray[np.floating],
                        source_points: NDArray[np.floating],
                        rigid: ITransform) -> float:
    """Mean Euclidean residual between rigid prediction and target control points."""
    xp = cp.get_array_module(target_points, source_points)
    predicted = rigid.Transform(source_points)
    predicted = xp.asarray(predicted)
    target_points = xp.asarray(target_points)
    return float(xp.mean(xp.linalg.norm(predicted - target_points, axis=1)))


def ConvertControlPointsToRigidTransformForBlend(input_transform: IControlPoints,
                                                 ignore_rotation: bool = False) -> ITransform:
    """Estimate a rigid linear target for blend that preserves mesh Y-orientation.

    When the mesh has a strong inverse-map Y correlation, pick the reflected or
    non-reflected rigid fit that matches the mesh orientation sign (lowest residual
    among orientation-consistent candidates).
    """
    if ignore_rotation:
        raise ValueError("Ignore rotation is no longer supported for control points")

    from nornir_imageregistration.transforms.utils import estimate_inverse_map_y_correlation

    target_points = input_transform.TargetPoints
    source_points = input_transform.SourcePoints
    mesh_corr = estimate_inverse_map_y_correlation(input_transform)

    candidates: list[nornir_imageregistration.transforms.CenteredSimilarity2DTransform] = []
    for reflected in (False, True):
        try:
            components = EstimateRigidComponentsFromControlPoints(target_points,
                                                                  source_points,
                                                                  reflected_override=reflected)
        except (ValueError, np.linalg.LinAlgError):
            xp = cp.get_array_module(source_points, target_points)
            components = _translation_only_rigid_components(
                xp.asarray(source_points), xp.asarray(target_points), reflected, xp)
        candidates.append(RigidComponentsToCenteredSimilarityTransform(components))

    if abs(mesh_corr) < 0.9:
        return min(candidates, key=lambda rigid: _rigid_fit_residual(target_points, source_points, rigid))

    mesh_sign = 1.0 if mesh_corr >= 0.0 else -1.0
    orientation_matched: list[nornir_imageregistration.transforms.CenteredSimilarity2DTransform] = []
    for rigid in candidates:
        rigid_corr = estimate_inverse_map_y_correlation(rigid, source_points=source_points)
        if rigid_corr == 0.0 or np.sign(rigid_corr) == mesh_sign:
            orientation_matched.append(rigid)

    pool = orientation_matched if orientation_matched else candidates
    return min(pool, key=lambda rigid: _rigid_fit_residual(target_points, source_points, rigid))


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
    flip_ud = bool(getattr(input_transform, 'flip_ud', False))
    if isinstance(input_transform, nornir_imageregistration.transforms.CenteredSimilarity2DTransform):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_space_center_of_rotation,
            angle=input_transform.angle,
            scalar=input_transform.scalar,
            flip_ud=flip_ud)
    elif isinstance(input_transform, nornir_imageregistration.transforms.Rigid):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_space_center_of_rotation,
            angle=input_transform.angle,
            scalar=input_transform.scalar,
            flip_ud=flip_ud)
    elif isinstance(input_transform, nornir_imageregistration.transforms.RigidTranslation):
        return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=input_transform._target_offset,
            source_rotation_center=input_transform.source_space_center_of_rotation,
            angle=input_transform.angle,
            scalar=input_transform.scalar,
            flip_ud=flip_ud)

    raise NotImplementedError()


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
            source_rotation_center=input_transform.source_space_center_of_rotation,
            angle=input_transform.angle,
            scalar=input_transform.scalar,
            flip_ud=input_transform.flip_ud)
    elif isinstance(input_transform, nornir_imageregistration.transforms.Rigid):
        return ConvertRigidTransformToCenteredSimilarityTransform(input_transform)
    elif isinstance(input_transform, nornir_imageregistration.transforms.RigidTranslation):
        return ConvertRigidTransformToCenteredSimilarityTransform(input_transform)

    raise NotImplementedError()


def ConvertTransformToMeshTransform(input_transform: ITransform,
                                    source_image_shape: NDArray | None = None) -> ITransform:
    if isinstance(input_transform, IControlPoints):
        return nornir_imageregistration.transforms.MeshWithRBFFallback(input_transform.points)

    if isinstance(input_transform, nornir_imageregistration.transforms.Rigid) or \
            isinstance(input_transform, nornir_imageregistration.transforms.RigidTranslation):
        control_points = GetControlPointsForRigidTransform(input_transform, source_image_shape)  # type: ignore[arg-type]
        transform = nornir_imageregistration.transforms.MeshWithRBFFallback(control_points)
        return transform

    raise NotImplementedError()


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
    xp = cp.get_array_module(out_corners)
    return xp.append(out_corners, xp.asarray(corners), 1)


def ConvertTransformToGridTransform(input_transform: ITransform, source_image_shape: NDArray,
                                    cell_size: NDArray | None = None, grid_dims: NDArray | None = None,
                                    grid_spacing: NDArray | None = None,
                                    prefer_gpu: bool = False) -> ITransform:
    """
    Converts a set of EnhancedAlignmentRecord peaks from the _RefineGridPointsForTwoImages function into a transform

    :param prefer_gpu: When True and CuPy is the active backend with cupyx
        LinearNDInterpolator available, build the GPU-component grid transform so
        the discrete inverse stays on-device (mirrors ``factory.ParseGridTransform``).
        Defaults to False so every existing caller keeps the CPU transform.
        Pass ``False`` explicitly at mosaic/STOS save even if other call sites opt in.
    """

    grid_data = nornir_imageregistration.ITKGridDivision(source_image_shape, cell_size=cell_size,
                                                         grid_spacing=grid_spacing, grid_dims=grid_dims)
    grid_data.PopulateTargetPoints(input_transform)

    # TODO, create a specific grid transform object that uses numpy's RegularGridInterpolator

    if prefer_gpu and (nornir_imageregistration.GetActiveComputationLib()
                       == nornir_imageregistration.ComputationLib.cupy):
        # Build the GPU component only when cupyx LinearND is importable; otherwise
        # fall through to the CPU transform so a missing cupyx build is not fatal.
        from nornir_imageregistration.transforms.gridtransform import cuLinearNDInterpolator
        if cuLinearNDInterpolator is not None:
            return nornir_imageregistration.transforms.GridWithRBFFallback_GPUComponent(grid_data)

    return nornir_imageregistration.transforms.GridWithRBFFallback(grid_data)


def ConvertTransformToRBFTransform(input_transform: ITransform,
                                   source_image_shape: NDArray | None = None) -> ITransform:
    """
    Converts a set of EnhancedAlignmentRecord peaks from the _RefineGridPointsForTwoImages function into a transform
    """

    if isinstance(input_transform, IControlPoints):
        return nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection(input_transform.SourcePoints,  # type: ignore[abstract]
                                                                                 input_transform.TargetPoints)
    if source_image_shape is None:
        raise ValueError("source_image_shape is required to convert this transform to RBF")

    # Convert via mesh so we can derive control points from non-control-point transforms (e.g. rigid).
    mesh_transform = ConvertTransformToMeshTransform(
        input_transform,
        source_image_shape=source_image_shape
    )
    if isinstance(mesh_transform, IControlPoints):
        return nornir_imageregistration.transforms.TwoWayRBFWithLinearCorrection(
            mesh_transform.SourcePoints,  # type: ignore[abstract]
            mesh_transform.TargetPoints   # type: ignore[abstract]
        )

    raise NotImplementedError(f"Unable to convert {input_transform.__class__.__name__} to RBF")
