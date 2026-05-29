"""
Created on Oct 4, 2012

@author: u0490822
"""
import multiprocessing
import multiprocessing.sharedctypes
from time import sleep
import numpy as np
from numpy.typing import NDArray
from typing import Sequence, AbstractSet
import logging
import skimage
import skimage.registration
import skimage.transform
import skimage.filters
from dataclasses import dataclass

from nornir_imageregistration import AlignmentRecord, IgnoreUnderflow
import nornir_imageregistration.phasecorrelation
from nornir_imageregistration.settings import StosBruteSettings, AngleSearchRange, SliceToSliceMethod
from nornir_imageregistration.nornir_image_types import ImageLike

# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp
    import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

import nornir_imageregistration
import nornir_shared.mathhelper
import nornir_pools
from nornir_imageregistration.hann_window_cache import HannWindowCache


@dataclass
class AngleScaleResult:
    angle: float
    scale: float
    weight: float
    translation: tuple[float, float]
    flippedud: bool = False


def _coerce_to_source_module(x: NDArray, xp) -> NDArray:
    """Place *x* on array module *xp* (numpy or cupy), matching ``cupy.get_array_module``."""
    if cp.get_array_module(x) is xp:
        return xp.asarray(x)
    if xp is np:
        if hasattr(x, "get"):
            return x.get()  # type: ignore[attr-defined, union-attr]
        return np.asarray(x)
    return xp.asarray(np.asarray(x))


def _coerce_registration_image_pair(source_image: NDArray, target_image: NDArray) -> tuple[NDArray, NDArray]:
    """Align *target_image* to the same array module as *source_image* (no global CuPy upgrade)."""
    xp = cp.get_array_module(source_image)
    return _coerce_to_source_module(source_image, xp), _coerce_to_source_module(target_image, xp)


def rotate_image(image: NDArray,
                 angle: float,
                 image_stats: nornir_imageregistration.ImageStats) -> NDArray:
    """Rotates an image, filling empty space with noise that matches the image stats
    :return: The rotated image and the image stats, the original objects if rotation is 0 / image_stats was passed"""

    if angle == 0:
        return image

    xp = cp.get_array_module(image)
    xp_scipy = cupyx.scipy.get_array_module(image)
    rotate = xp_scipy.ndimage.rotate

    # im_target = cp.asarray(im_target) if use_cp and not isinstance(im_target, cp.ndarray) else im_target
    # im_source = cp.asarray(im_source) if use_cp  and not isinstance(im_source, cp.ndarray)  else im_source

    # gc.set_debug(gc.DEBUG_LEAK)
    if image_stats is None:
        image_stats = nornir_imageregistration.ImageStats.CalcStats(image_stats)

    # This confused me for years, but the implementation of rotate calls affine_transform with
    # the rotation matrix.  However the docs for affine_transform state it needs to be called
    # with the inverse transform.  Hence negating the angle here.
    with IgnoreUnderflow():
        if xp is not np:
            im_rotated = rotate(image, axes=(1, 0), angle=-angle, cval=np.nan)
        else:
            im_rotated = rotate(image.astype(np.float32, copy=False), axes=(1, 0), angle=-angle,
                                cval=np.nan).astype(image.dtype, copy=False)  # Numpy cannot rotate float16 images

    xp_out = cp.get_array_module(im_rotated)
    im_result_empty_entries = xp_out.isnan(im_rotated)
    n_bad = int(xp_out.sum(im_result_empty_entries))
    if n_bad:
        noise = image_stats.GenerateNoise(n_bad, dtype=image.dtype, xp=xp_out)  # type: ignore[arg-type]
        if cp.get_array_module(noise) is not xp_out:
            if xp_out is np:
                noise = noise.get() if hasattr(noise, "get") else np.asarray(noise).ravel()
            else:
                noise = xp_out.asarray(np.asarray(noise))
        im_rotated[im_result_empty_entries] = noise

    return im_rotated


def pad_and_rotate_image(image: NDArray,
                         angle: float,
                         image_stats: nornir_imageregistration.ImageStats,
                         desired_shape: tuple[int, int] | None = None,
                         min_overlap: float = 0.75,
                         original_shape: NDArray | tuple[int, int] | None = None,
                         power_of_two: bool = False,
                         ) -> NDArray:
    """
    Rotates and image and pads it to ensure it has the requested dimensions, filling empty space with noise that matches the image stats.
    :param image:
    :param desired_shape: The desired shape of the image after rotation
    :param image_stats:
    :param min_overlap:
    :param original_shape: If the input image has been previously padded, this is the original shape of the image
    :param power_of_two: If True, the image will be padded to the nearest power of two.  This may be largest than the desired_shape
    :return: The rotated image and the image stats, the original objects if rotation is 0 / image_stats was passed
    """

    if original_shape is None:
        orginal_shape = image.shape

    if desired_shape is None:
        desired_shape = (None, None)  # type: ignore[assignment]

    rotated_image = rotate_image(image, angle=angle, image_stats=image_stats) if angle != 0 else image

    # if desired_shape is not None and rotated_image.shape[0] > desired_shape[0] or rotated_image.shape[1] > desired_shape[1]:
    #    raise ValueError("Need to add support to pad_and_rotate_image for expanding the desired image size")

    if power_of_two:
        desired_shape = nornir_imageregistration.NearestPowerOfTwo(rotated_image.shape)  # type: ignore[assignment]

    padded_rotated_image = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(rotated_image,
                                                                                                     image_median=image_stats.median,
                                                                                                     image_stddev=image_stats.std,
                                                                                                     min_overlap=min_overlap,
                                                                                                     original_shape=original_shape,
                                                                                                     new_height=
                                                                                                     desired_shape[0],  # type: ignore[index]
                                                                                                     new_width=
                                                                                                     desired_shape[  # type: ignore[index]
                                                                                                         1])
    return padded_rotated_image


# from memory_profiler import profile
def SliceToSliceRigidRegistration(target_image: ImageLike,
                                  source_image: ImageLike,
                                  target_mask: ImageLike | None = None,
                                  source_mask: ImageLike | None = None,
                                  LargestDimension: int | None = None,
                                  AngleSearchRange: Sequence[float] | AbstractSet[float] | None = None,
                                  MinOverlap: float = 0.5,
                                  WarpedImageScaleFactors=None,
                                  SingleThread: bool = False,
                                  Cluster: bool = False,
                                  TestFlip: bool = True,
                                  estimate_angle: bool = True,
                                  method: SliceToSliceMethod = SliceToSliceMethod.LogPolar) -> nornir_imageregistration.AlignmentRecord:
    """Given two images this function returns the rotation angle which best aligns them
       Largest dimension determines how large the images used for alignment should be.

       :param target_image: Source
       :param source_image: Target
       :param target_mask:
       :param source_mask:
       :param SingleThread:
       :param Cluster:
       :param TestFlip:
       :param estimate_angle: If true, run a log_polar registration and append the result to the angles to search
       :param int LargestDimension: The input images should be scaled so the largest image dimension is equal to this value, default is None
       :param float MinOverlap: The minimum amount of overlap we require in the images.  Higher values reduce false positives but may not register offset images
       :param float AngleSearchRange: A list of rotation angles to test.  Pass None for the default which is every two degrees
       :param float WarpedImageScaleFactors: Scale the warped image input by this amount before attempting registration
       """
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    if AngleSearchRange is not None:
        if not isinstance(AngleSearchRange, set):
            AngleSearchRange = set(AngleSearchRange)  # type: ignore[assignment]
        # if isinstance(AngleSearchRange, np.ndarray):

        if 0 not in set(AngleSearchRange):  # type: ignore[arg-type]
            logger = logging.getLogger(__name__ + '.SliceToSliceRigidRegistration')
            logger.warning("AngleSearchRange should contain 0 degrees to ensure the best match is found")
    else: 
        AngleSearchRange = set(map(float, range(0, 358, 2)))

    SingleThread = True if use_cp else SingleThread

    source_image_data = nornir_imageregistration.ImagePermutationHelper(source_image, source_mask)
    target_image_data = nornir_imageregistration.ImagePermutationHelper(target_image, target_mask)

    if estimate_angle and method == SliceToSliceMethod.LogPolar:
        estimate_angle = False
        # raise ValueError("LogPolar method is redundant with setting estimate_angle to true")

    if estimate_angle:
        # Estimate the angle and scale
        estimated_angle_best_match = _find_angle_and_scale_with_logpolar(
            source_image=source_image_data.ImageWithMaskAsNoise,
            target_image=target_image_data.ImageWithMaskAsNoise,
            source_stats=source_image_data.Stats,
            target_stats=target_image_data.Stats,
            min_overlap=MinOverlap)
        if abs(estimated_angle_best_match.angle) > 0.25:
            AngleSearchRange.add(estimated_angle_best_match.angle)  # type: ignore[union-attr]

    settings = StosBruteSettings(method=method,
                                 angles=AngleSearchRange,
                                 min_overlap=MinOverlap,
                                 source_image_scale_factors=WarpedImageScaleFactors,
                                 larget_dimension=LargestDimension,
                                 try_flipped=TestFlip)

    return SliceToSliceRigidRegistrationWithPreprocessedImages(source_image_data=source_image_data,
                                                               target_image_data=target_image_data,
                                                               settings=settings,
                                                               SingleThread=SingleThread,
                                                               Cluster=Cluster)


def NarrowAngleSearchRangeWithResult(angle_range: NDArray[np.floating],
                                     min_step_size: float,
                                     target_angle: float) -> set[float]:
    """
    Given a range of angles, returns a smaller search range around an estimated correct angle
    :param angle_range: The original search range of angles we want to narrow down
    :param min_step_size: Minimum difference between angles in the results
    :param target_angle: The angle previously estimated to be the best match
    :return: A narrower search range to refine the angle search in a future iteration
    """
    if len(angle_range) < 2:
        raise ValueError("Angle search range must contain at least two angles to be refined")

    sorted_angles = sorted(angle_range)
    iMatch = sorted_angles.index(target_angle)
    iBelow = iMatch - 1 if iMatch - 1 >= 0 else len(sorted_angles) - 1
    iAbove = iMatch + 1 if iMatch + 1 < len(sorted_angles) else 0
    below = sorted_angles[iMatch - 1] if iMatch - 1 >= 0 else sorted_angles[0] - np.abs(
        sorted_angles[1] - sorted_angles[0])
    above = sorted_angles[iMatch + 1] if iMatch + 1 < len(sorted_angles) else sorted_angles[
                                                                                  iMatch] + np.abs(
        sorted_angles[iMatch] - sorted_angles[iMatch - 1])
    refine_search_range = above - below
    nSteps = 20
    stepsize = refine_search_range / nSteps

    if stepsize < min_step_size:
        nSteps = int(refine_search_range / min_step_size)
        stepsize = refine_search_range / nSteps

    refined_angle_search_range = {(x * stepsize) + below for x in range(1, nSteps)}

    # Ensure we include the best match angle
    refined_angle_search_range.add(target_angle)
    return refined_angle_search_range


def SliceToSliceRigidRegistrationWithPreprocessedImages(
        source_image_data: nornir_imageregistration.ImagePermutationHelper,
        target_image_data: nornir_imageregistration.ImagePermutationHelper,
        settings: StosBruteSettings,
        SingleThread: bool = False,
        Cluster: bool = False) -> nornir_imageregistration.AlignmentRecord:
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    target_image = target_image_data.ImageWithMaskAsNoise
    source_image = source_image_data.ImageWithMaskAsNoise

    target_stats = target_image_data.Stats
    source_stats = source_image_data.Stats

    del target_image_data
    del source_image_data

    target_image = cp.asarray(target_image) if use_cp and not isinstance(target_image, cp.ndarray) else target_image
    source_image = cp.asarray(source_image) if use_cp and not isinstance(source_image, cp.ndarray) else source_image

    scalar = 1.0
    if settings.larget_dimension is not None:
        scalar = nornir_imageregistration.ScalarForMaxDimension(settings.larget_dimension,
                                                                [target_image.shape, source_image.shape])
        if scalar > 1.0:
            scalar = 1.0

    if scalar != 1.0:
        target_image = nornir_imageregistration.ScaleImage(target_image, scalar)
        source_image = nornir_imageregistration.ScaleImage(source_image, scalar)

    # Replace extrema with noise
    if settings.method == nornir_imageregistration.settings.SliceToSliceMethod.LogPolar:
        best_match = _find_angle_and_scale_with_logpolar(source_image=source_image, target_image=target_image,
                                                         source_stats=source_stats, target_stats=target_stats,
                                                         min_overlap=settings.min_overlap)
    else:
        best_match = _find_best_angle(source_image=source_image, target_image=target_image,
                                      source_stats=source_stats, target_stats=target_stats,
                                      angle_range=settings.angle_range,
                                      min_overlap=settings.min_overlap,
                                      SingleThread=SingleThread,
                                      use_cluster=Cluster)

    is_flipped = False
    if settings.try_flipped:
        # source_flipped = np.copy(source_image)
        _xp_img = cp.get_array_module(source_image)
        source_flipped = _xp_img.flipud(source_image)

        if settings.method == nornir_imageregistration.settings.SliceToSliceMethod.LogPolar:
            best_match_flipped = _find_angle_and_scale_with_logpolar(source_image=source_flipped,
                                                                     target_image=target_image,
                                                                     source_stats=source_stats,
                                                                     target_stats=target_stats,
                                                                     min_overlap=settings.min_overlap)
        else:
            best_match_flipped = _find_best_angle(source_image=source_flipped, target_image=target_image,
                                                  source_stats=source_stats, target_stats=target_stats,
                                                  angle_range=settings.angle_range,
                                                  min_overlap=settings.min_overlap,
                                                  SingleThread=SingleThread, use_cluster=Cluster)
        best_match_flipped.flippedud = True

        # Determine if the best match is flipped or not
        is_flipped = best_match_flipped.weight > best_match.weight
        source_image = source_flipped if is_flipped else source_image
        best_match = best_match_flipped if is_flipped else best_match

    if not settings.method == nornir_imageregistration.settings.SliceToSliceMethod.LogPolar:

        # Todo: We do not utilize the scale information from the log-polar method

        # Note Clement - the RefinedAngleSearch list below is not centered around the current best angle
        # Default angle search range every 2 degrees
        # Old RefinedAngleSearch list: [(x * 0.1) + best_match.angle - 1.9 for x in range(0, 18)]
        # New RefinedAngleSearch list (length 39): [(x * 0.1 + best_match.angle) for x in range(-19, 20)]
        # New optional RefinedAngleSearch list (length 18): [(x * 0.2 + best_match.angle) for x in range(-9, 10)]
        if not settings.angle_range_defined():
            best_refined_match = _find_best_angle(source_image=source_image, target_image=target_image,
                                                  source_stats=source_stats, target_stats=target_stats,
                                                  # [(x * 0.1) + best_match.angle - 1.9 for x in range(0, 18)],
                                                  angle_range=[(x * 0.2 + best_match.angle) for x in range(-9, 10)],
                                                  min_overlap=settings.min_overlap, SingleThread=SingleThread)
            best_refined_match.flippedud = is_flipped
        else:
            min_step_size = 0.25
            if len(settings.angle_range) > 2:
                refined_angle_search_range = NarrowAngleSearchRangeWithResult(settings.angle_range, min_step_size,
                                                                              best_match.angle)
                best_refined_match = _find_best_angle(source_image=source_image, target_image=target_image,
                                                      source_stats=source_stats, target_stats=target_stats,
                                                      angle_range=np.array(list(refined_angle_search_range), float),
                                                      min_overlap=settings.min_overlap, SingleThread=SingleThread)
                best_refined_match.flippedud = is_flipped
            else:
                best_refined_match = best_match
                best_refined_match.flippedud = is_flipped
    else:
        # TODO: Preserve scale information
        translation_results = ScoreOneAngle(source_original=source_image,
                                            target_original=target_image,
                                            target_image_shape=target_image.shape,
                                            source_image_shape=source_image.shape,
                                            angle=best_match.angle,
                                            target_stats=target_stats,
                                            source_stats=source_stats,
                                            target_image_prepadded=False,
                                            min_overlap=settings.min_overlap)

        best_refined_match = nornir_imageregistration.AlignmentRecord(peak=translation_results.peak,
                                                                      weight=translation_results.weight,
                                                                      angle=best_match.angle,
                                                                      flipped_ud=best_match.flippedud,
                                                                      scale=best_match.scale)

    if scalar != 1.0:
        AdjustedPeak = (best_refined_match.peak[0] * (1 / scalar), best_refined_match.peak[1] * (1 / scalar))  # type: ignore[union-attr]
        best_refined_match = nornir_imageregistration.AlignmentRecord(AdjustedPeak, best_refined_match.weight,
                                                                      best_refined_match.angle, is_flipped)

    if settings.source_image_scaling_required and not settings.method == nornir_imageregistration.settings.SliceToSliceMethod.LogPolar:
        # AdjustedPeak = best_refined_match.peak * (1.0 / WarpedImageScaleFactors)
        best_refined_match = nornir_imageregistration.AlignmentRecord(best_refined_match.peak,  # type: ignore[union-attr]
                                                                      best_refined_match.weight,
                                                                      best_refined_match.angle, is_flipped,
                                                                      settings.source_image_scale_factors)  # type: ignore[arg-type]

    # best_refined_match.CorrectPeakForOriginalImageSize(imFixed.shape, source_image.shape)

    return best_refined_match  # type: ignore[return-value]


def ScoreOneAngle(target_original: NDArray, source_original: NDArray,
                  target_image_shape: tuple[int, int], source_image_shape: tuple[int, int],
                  angle: float,
                  target_stats: nornir_imageregistration.ImageStats | None = None,
                  source_stats: nornir_imageregistration.ImageStats | None = None,
                  target_image_prepadded: bool = True,
                  min_overlap: float = 0.75) -> nornir_imageregistration.AlignmentRecord:
    """Returns an alignment score for a fixed image and an image rotated at a specified angle"""

    # print(f'Scoring {angle} degrees')
    try:
        im_target = nornir_imageregistration.ImageParamToImageArray(target_original,
                                                                    dtype=nornir_imageregistration.default_image_dtype())
        im_source = nornir_imageregistration.ImageParamToImageArray(source_original,
                                                                    dtype=nornir_imageregistration.default_image_dtype())

        if source_stats is None:
            source_stats = nornir_imageregistration.ImageStats.CalcStats(im_source)

        if target_stats is None:
            target_stats = nornir_imageregistration.ImageStats.CalcStats(im_target)

        use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
        # Use of cupy or numpy
        xp = cp.get_array_module(target_original)
        # Use of cupyx.scipy.fft or scipy.fft
        xp_scipy = cupyx.scipy.get_array_module(target_original)

        rotated_source = pad_and_rotate_image(image=im_source,
                                              angle=angle,
                                              image_stats=source_stats,
                                              min_overlap=min_overlap)

        assert (rotated_source.shape[0] > 0)
        assert (rotated_source.shape[1] > 0)

        if not target_image_prepadded:
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(im_target,
                                                                                                      image_median=target_stats.median,
                                                                                                      image_stddev=target_stats.std,
                                                                                                      min_overlap=min_overlap,
                                                                                                      original_shape=target_image_shape)
        else:
            padded_target = im_target

        # print str(padded_target.shape) + ' ' +  str(rotated_padded_source.shape)

        TargetHeight = max([padded_target.shape[0], rotated_source.shape[0]])
        TargetWidth = max([padded_target.shape[1], rotated_source.shape[1]])

        # Why is MinOverlap hard-coded to 1.0?  To prevent padded_target from growing larger than the largest of the input dimensions
        # pad_image_for_phase_correlation will always return a copy, so don't call it unless we need to
        if not np.array_equal(im_target.shape, np.array((TargetHeight, TargetWidth))):
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(im_target,
                                                                                                      new_width=TargetWidth,
                                                                                                      new_height=TargetHeight,
                                                                                                      image_median=target_stats.median,
                                                                                                      image_stddev=target_stats.std,
                                                                                                      min_overlap=1.0)
            # print(f"{angle}: Padding target image to {padded_target.shape}")
        # else:
        #     print(f"{angle}: No additional padding   {padded_target.shape}")

        if np.array_equal(rotated_source.shape, np.array((TargetHeight, TargetWidth))):
            rotated_padded_source = rotated_source
        else:
            rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                rotated_source,
                new_width=TargetWidth,
                new_height=TargetHeight,
                image_median=source_stats.median,
                image_stddev=source_stats.std,
                min_overlap=1.0)

        assert (np.array_equal(padded_target.shape, rotated_padded_source.shape))

        # if use_cp and not isinstance(padded_target, cp.ndarray):
        #     padded_target = cp.asarray(padded_target)
        #
        # if use_cp and not isinstance(rotated_padded_source, cp.ndarray):
        #     rotated_padded_source = cp.asarray(rotated_padded_source)

        correlation_image = nornir_imageregistration.phasecorrelation.image_phase_correlation(
            target_image=padded_target,
            source_image=rotated_padded_source,
            target_mean=target_stats.mean,
            source_mean=source_stats.mean,
            correlation_coefficient=.66)

        # if OKToDelimWarped:
        del im_source
        del im_target

        del rotated_source

        del padded_target
        del rotated_padded_source

        correlation_image = xp_scipy.fft.fftshift(correlation_image)
        try:
            correlation_image -= correlation_image.min()
            # correlation_image /= correlation_image.max()
        except FloatingPointError as e:
            print(f"Floating point error: {e} for {correlation_image.min()} or {correlation_image.max()}")
            record = nornir_imageregistration.AlignmentRecord((0, 0), 0, 0)
            return record

        # Timer.Start('Find Peak')

        # Note - Clement: overlap_mask still uses numpy (and not cupy)
        overlap_mask = nornir_imageregistration.GetOverlapMask(target_image_shape, source_image_shape,
                                                                              correlation_image.shape, min_overlap,
                                                                              MaxOverlap=1.0)
        if use_cp and not isinstance(overlap_mask, cp.ndarray):
            overlap_mask = cp.asarray(overlap_mask)

        (peak, weight, cutoff_value, cutoff_percent) = nornir_imageregistration.phasecorrelation.find_peak(
            correlation_image, overlap_mask)
        del overlap_mask
        del correlation_image

        record = nornir_imageregistration.AlignmentRecord(peak, weight, angle)
        return record
    finally:
        nornir_imageregistration.close_shared_memory(target_original)  # type: ignore[arg-type]
        nornir_imageregistration.close_shared_memory(source_original)  # type: ignore[arg-type]


def GetFixedAndWarpedImageStats(imFixed: NDArray[np.floating], imWarped: NDArray[np.floating]) -> tuple[
    nornir_imageregistration.ImageStats, nornir_imageregistration.ImageStats]:
    tpool = nornir_pools.GetGlobalThreadPool()

    fixedStatsTask = tpool.add_task('FixedStats', nornir_imageregistration.ImageStats.CalcStats, imFixed)
    warpedStats = nornir_imageregistration.ImageStats.CalcStats(imWarped)

    fixedStats = fixedStatsTask.wait_return()

    return fixedStats, warpedStats


def _find_angle_and_scale_with_logpolar(source_image: NDArray[np.floating],
                                        target_image: NDArray[np.floating],
                                        source_stats: nornir_imageregistration.ImageStats,
                                        target_stats: nornir_imageregistration.ImageStats,
                                        min_overlap: float = 0.5) -> AngleScaleResult:
    """This function uses the log polar technique to determine the scale and angle of the best alignment between two images"""
    # skimage / numpy.fft: explicit host boundary when inputs are on device; keep NumPy path allocation-free.
    _xp_lp = cp.get_array_module(source_image)
    if _xp_lp is not np:
        source_image = source_image.get()  # type: ignore[attr-defined]
        target_image = target_image.get()  # type: ignore[attr-defined]

    desired_height = int(nornir_imageregistration.NearestPowerOfTwo(max([source_image.shape[0], target_image.shape[0]])))
    desired_width = int(nornir_imageregistration.NearestPowerOfTwo(max([source_image.shape[1], target_image.shape[1]])))
    desired_shape = np.array([desired_height, desired_width], dtype=int)

    max_dimension = max([desired_height, desired_width])
    radius = max_dimension // 4  # only take lower frequencies

    """Use the log-polar space to determine the best angle and then use the normal phase correlation to determine the best translation"""
    padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(target_image,
                                                                                              min_overlap=min_overlap,
                                                                                              image_median=target_stats.median,
                                                                                              image_stddev=target_stats.std,
                                                                                              new_height=desired_height,
                                                                                              new_width=desired_width)

    padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(source_image,
                                                                                              min_overlap=min_overlap,
                                                                                              image_median=source_stats.median,
                                                                                              image_stddev=source_stats.std,
                                                                                              new_height=desired_height,
                                                                                              new_width=desired_width)

    dg_target_image = skimage.filters.difference_of_gaussians(padded_target, low_sigma=4, high_sigma=20)
    dg_source_image = skimage.filters.difference_of_gaussians(padded_source, low_sigma=4, high_sigma=20)

    # target_window = skimage.filters.window('hann', padded_target.shape)
    # source_window = skimage.filters.window('hann', padded_source.shape)
    target_window = HannWindowCache.GetOrCreate(padded_target.shape)
    source_window = HannWindowCache.GetOrCreate(padded_source.shape)

    window_target_image = dg_target_image * target_window
    window_source_image = dg_source_image * source_window

    target_freq = np.fft.fft2(window_target_image)
    source_freq = np.fft.fft2(window_source_image)

    target_freq_shift = np.abs(np.fft.fftshift(target_freq))
    source_freq_shift = np.abs(np.fft.fftshift(source_freq))

    # Create a log-polar space image of both images

    target_image_log_polar = skimage.transform.warp_polar(target_freq_shift,
                                                          radius=radius,
                                                          output_shape=desired_shape,
                                                          scaling='log',
                                                          order=0)
    source_image_log_polar = skimage.transform.warp_polar(source_freq_shift,
                                                          radius=radius,
                                                          output_shape=desired_shape,
                                                          scaling='log',
                                                          order=0)

    target_image_log_polar_left_half = target_image_log_polar[:target_image_log_polar.shape[0] // 2, :]
    source_image_log_polar_left_half = source_image_log_polar[:source_image_log_polar.shape[0] // 2, :]

    phase_correlation = nornir_imageregistration.phasecorrelation.image_phase_correlation(
        target_image_log_polar_left_half, source_image_log_polar_left_half)

    # shifts, error, phasediff = skimage.registration.phase_cross_correlation(
    #     source_image_log_polar_left_half, target_image_log_polar_left_half, upsample_factor=2, normalization=None
    # )

    # phase_correlation_shifted = phase_correlation
    phase_correlation_shifted = np.fft.fftshift(phase_correlation)
    try:
        phase_correlation_shifted -= phase_correlation_shifted.min()
        phase_correlation_shifted /= phase_correlation_shifted.max()  # Remove before release, this is for visualization
    except FloatingPointError as e:
        print(f"Floating point error: {e} for {phase_correlation.min()} or {phase_correlation.max()}")
        record = AngleScaleResult(angle=0, scale=1.0, weight=0, translation=(0, 0))
        return record

    angle_scale_peak = nornir_imageregistration.phasecorrelation.find_peak(phase_correlation_shifted)

    # Because of the fftshift we need to add 180 to the angle to get the correct angle
    # recovered_angle = (360 / (desired_shape[0] / 2)) * (angle_scale_peak.scaled_offset[0] + (desired_shape[0] / 2))
    # recovered_angle = (360 / (desired_shape[0])) * (angle_scale_peak.scaled_offset[0] + (desired_shape[0] / 2))

    degrees_per_pixel = 360 / desired_shape[0]
    recovered_angle = (degrees_per_pixel * angle_scale_peak.scaled_offset[0])  # + 180
    klog = desired_shape[1] / np.log(radius)
    shift_scale = np.exp(angle_scale_peak.scaled_offset[1] / klog)

    # rotated_source = sp.ndimage.rotate(source_image.astype(np.float32), -recovered_angle, reshape=True)
    # rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(rotated_source,
    #                                                                                               MinOverlap=min_overlap,
    #                                                                                               ImageMedian=source_stats.median,
    #                                                                                               ImageStdDev=source_stats.std,
    #                                                                                               NewHeight=desired_height,
    #                                                                                               NewWidth=desired_width)

    # Check if we need to grow the boundaries to accomodate the rotation
    # rotated_bounds = nornir_imageregistration.transforms.utils.GetRotatedBoundaries(source_image.shape,angle=recovered_angle)
    # rotated_desired_height = max(desired_height, rotated_bounds.Height)
    # rotated_desired_width = max(desired_width, rotated_bounds.Width)
    # rotated_desired_shape = np.array((rotated_desired_height, rotated_desired_width), dtype=int)
    # rotated_desired_shape = nornir_imageregistration.NearestPowerOfTwo(rotated_desired_shape)
    # rotated_desired_height, rotated_desired_width = rotated_desired_shape

    # Check whether the angle is correct or needs to be adjusted by 180 degrees, also collect the translation vector
    rotated_padded_source = pad_and_rotate_image(image=source_image.astype(np.float32),
                                                 angle=recovered_angle,
                                                 image_stats=source_stats,
                                                 min_overlap=min_overlap,
                                                 desired_shape=[desired_height, desired_width],  # type: ignore[arg-type]
                                                 power_of_two=True)

    if not np.array_equal(rotated_padded_source.shape, padded_target.shape):
        # If the target image does not match the dimensions of the rotated source image, make the size equal
        rotated_desired_shape = nornir_shared.mathhelper.max_shape([rotated_padded_source.shape, padded_target.shape])  # type: ignore[arg-type]
        rotated_desired_height, rotated_desired_width = rotated_desired_shape
        padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(target_image,
                                                                                                  min_overlap=min_overlap,
                                                                                                  image_median=target_stats.median,
                                                                                                  image_stddev=target_stats.std,
                                                                                                  new_height=rotated_desired_height,
                                                                                                  new_width=rotated_desired_width)

        if not np.array_equal(rotated_padded_source.shape, rotated_desired_shape):
            # If the rotated source image does not match the dimensions of the target image, make the size equal
            # rotated_desired_shape = nornir_shared.mathhelper.max_shape([rotated_padded_source.shape, padded_target.shape])
            # rotated_desired_height, rotated_desired_width = rotated_desired_shape
            rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                target_image,
                min_overlap=min_overlap,
                image_median=target_stats.median,
                image_stddev=target_stats.std,
                new_height=rotated_desired_height,
                new_width=rotated_desired_width)
        target_window = HannWindowCache.GetOrCreate(rotated_desired_shape)
        source_window = HannWindowCache.GetOrCreate(rotated_desired_shape)
    else:
        rotated_desired_height, rotated_desired_width = desired_shape

    fft_target_ref = np.fft.fft2(padded_target * target_window)
    fft_source_ref = np.fft.fft2(rotated_padded_source * source_window)

    original_correlation = nornir_imageregistration.fft_phase_correlation(fft_target_ref, fft_source_ref)  # type: ignore[arg-type]

    # rotated_source = sp.ndimage.rotate(source_image.astype(np.float32), -recovered_angle + 180, reshape=True)
    # rotated_padded_source = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(rotated_source,
    #                                                                                               MinOverlap=min_overlap,
    #                                                                                               ImageMedian=source_stats.median,
    #                                                                                               ImageStdDev=source_stats.std,
    #                                                                                               NewHeight=desired_height,
    #                                                                                               NewWidth=desired_width)

    rotated_padded_source = pad_and_rotate_image(image=source_image.astype(np.float32),
                                                 angle=recovered_angle + 180,
                                                 image_stats=source_stats,
                                                 min_overlap=min_overlap,
                                                 desired_shape=[rotated_desired_height, rotated_desired_width])  # type: ignore[arg-type]

    rotated_source_freq = np.fft.fft2(rotated_padded_source * source_window)
    rotated_correlation = nornir_imageregistration.fft_phase_correlation(fft_target_ref, rotated_source_freq)  # type: ignore[arg-type]

    original_peak = nornir_imageregistration.phasecorrelation.find_peak(original_correlation)
    rotated_peak = nornir_imageregistration.phasecorrelation.find_peak(rotated_correlation)

    rotated_180 = False
    if original_peak.peak_strength >= rotated_peak.peak_strength:
        selected_peak = original_peak
    else:
        selected_peak = rotated_peak
        recovered_angle -= 180
        if recovered_angle < -180:
            recovered_angle += 360
        rotated_180 = True

    if nornir_imageregistration.in_debug_mode():
        print(
            f'{original_peak.peak_strength} vs {rotated_peak.peak_strength} @ recovered angle {recovered_angle} {'rotated_180' if rotated_180 else ""}')

    # shiftr, shiftc = shifts[:2]
    # degrees_per_pixel = 360 / desired_shape[0]
    # recovered_angle = degrees_per_pixel * shiftr
    # klog = desired_shape[1] / np.log(radius)
    # shift_scale = np.exp(shiftc / klog)

    #    return AngleScaleResult(angle=recovered_angle, scale=shift_scale, weight=angle_scale_peak.peak_strength)
    return AngleScaleResult(angle=recovered_angle, scale=shift_scale, weight=angle_scale_peak.peak_strength,
                            translation=selected_peak.scaled_offset)


def _find_best_angle(source_image: NDArray[np.floating],
                     target_image: NDArray[np.floating],
                     source_stats: nornir_imageregistration.ImageStats,
                     target_stats: nornir_imageregistration.ImageStats,
                     angle_range: NDArray[np.floating] | Sequence[float],
                     min_overlap: float = 0.5,
                     SingleThread: bool = False,
                     use_cluster: bool = False) -> nornir_imageregistration.AlignmentRecord:
    """Find the best angle to align two images.  This function can be very memory intensive.
       Setting SingleThread=True makes debugging easier"""

    try:
        Debug = False
        pool = None
        use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

        # Temporarily disable until we have  cluster pool working again.  Leaving this on eliminates shared memory which is a big optimization
        use_cluster = False

        if len(angle_range) <= 1:
            SingleThread = True

        if not SingleThread:
            if nornir_imageregistration.in_debug_mode():
                pool = nornir_pools.GetGlobalSerialPool()
            elif use_cluster:
                pool = nornir_pools.GetGlobalClusterPool()
            else:
                pool = nornir_pools.GetGlobalMultithreadingPool()

        # Preallocate lists to store results of each angle
        AngleMatchValues = list()  # type:  list[AlignmentRecord | None]
        taskList = list()  # type:  list[nornir_pools.Task | None]

        #    MaxRotatedDimension = max([max(imFixed), max(imWarped)]) * 1.4143
        #    MinRotatedDimension = max(min(imFixed), min(imWarped))
        #
        #    SmallPaddedFixed = pad_image_for_phase_correlation(imFixed, MaxOffset=0.1)
        #    LargePaddedFixed = pad_image_for_phase_correlation(imFixed, MaxOffset=0.1)

        padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(target_image,
                                                                                                  min_overlap=min_overlap,
                                                                                                  image_median=target_stats.median,
                                                                                                  image_stddev=target_stats.std)

        # Create a shared read-only memory map for the Padded fixed image

        if not (use_cluster or SingleThread):
            # temp_padded_fixed_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(padded_target)
            # temp_shared_warp_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(imWarped)

            # temp_padded_fixed_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data
            # temp_shared_warp_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data

            shared_target_metadata, shared_padded_target = nornir_imageregistration.npArrayToSharedArray(padded_target)
            shared_source_metadata, shared_source = nornir_imageregistration.npArrayToSharedArray(source_image
                                                                                                  )
            # shared_padded_target = np.save(padded_target, )
        else:
            shared_target_metadata = None
            shared_source_metadata = None
            shared_padded_target = padded_target.astype(nornir_imageregistration.default_image_dtype(),
                                                        copy=False) if not use_cp else cp.array(padded_target,
                                                                                                nornir_imageregistration.default_image_dtype())
            shared_source = source_image.astype(nornir_imageregistration.default_image_dtype(),
                                                copy=False) if not use_cp else cp.array(source_image,
                                                                                        nornir_imageregistration.default_image_dtype())

        CheckTaskInterval = 16

        source_shape = source_image.shape
        target_shape = target_image.shape
        max_task_count = multiprocessing.cpu_count() * 1.5

        for i, theta in enumerate(angle_range):
            if SingleThread:
                record = ScoreOneAngle(target_original=shared_padded_target, source_original=shared_source,
                                       target_image_shape=target_shape, source_image_shape=source_shape,
                                       angle=theta,
                                       target_stats=target_stats, source_stats=source_stats,
                                       min_overlap=min_overlap)
                AngleMatchValues.append(record)
            elif use_cluster:
                task = pool.add_task(str(theta), ScoreOneAngle,  # type: ignore[union-attr]
                                     target_original=shared_padded_target, source_original=shared_source,
                                     target_image_shape=target_shape, source_image_shape=source_shape,
                                     angle=theta,
                                     target_stats=target_stats, source_stats=source_stats,
                                     min_overlap=min_overlap)
                taskList.append(task)
            else:
                task = pool.add_task(str(theta), ScoreOneAngle,  # type: ignore[union-attr]
                                     target_original=shared_target_metadata, source_original=shared_source_metadata,
                                     target_image_shape=target_shape, source_image_shape=source_shape,
                                     angle=theta,
                                     target_stats=target_stats, source_stats=source_stats,
                                     min_overlap=min_overlap)
                taskList.append(task)

            if not i % CheckTaskInterval == 0:
                continue

            # I don't like this, but it lets me delete tasks before filling the queue which may save some memory.
            # No sense checking unless we've already filled the queue though
            if len(taskList) > max_task_count:
                for iTask in range(len(taskList) - 1, -1, -1):
                    if taskList[iTask].iscompleted:  # type: ignore[union-attr]
                        record = taskList[iTask].wait_return()  # type: ignore[union-attr]
                        AngleMatchValues.append(record)
                        del taskList[iTask]

            # TestOneAngle(shared_padded_target, shared_source, angle, None, MinOverlap)

        # taskList.sort(key=tpool.Task.name)

        while len(taskList) > 0:
            for iTask in range(len(taskList) - 1, -1, -1):
                if taskList[iTask].iscompleted:  # type: ignore[union-attr]
                    record = taskList[iTask].wait_return()  # type: ignore[union-attr]
                    AngleMatchValues.append(record)
                    del taskList[iTask]

            if len(taskList) > 0:
                # Wait a bit before checking the task list
                sleep(0.5)

            # print(str(record.angle) + ' = ' + str(record.peak) + ' weight: ' + str(record.weight) + '\n')

            # ShowGrayscale(NormCorrelationImage)

        # print(str(AngleMatchValues))

        # Delete the pool to ensure extra python threads do not stick around
        # if pool is not None:
        #    pool.shutdown()

        del padded_target

        BestMatch = max(AngleMatchValues, key=nornir_imageregistration.AlignmentRecord.WeightKey)  # type: ignore[arg-type]
        return BestMatch
    finally:

        if shared_target_metadata is not None:
            nornir_imageregistration.unlink_shared_memory(shared_target_metadata)
        if shared_source_metadata is not None:
            nornir_imageregistration.unlink_shared_memory(shared_source_metadata)

            # os.remove(temp_shared_warp_memmap.path)
            # os.remove(temp_padded_fixed_memmap.path)


def __ExecuteProfiler():
    SliceToSliceRigidRegistration('C:/Src/Git/nornir-testdata/Images/0162_ds32.png',
                                  'C:/Src/Git/nornir-testdata/Images/0164_ds32.png',
                                  AngleSearchRange=list(range(-175, -174, 1)),
                                  SingleThread=True)


if __name__ == '__main__':
    from nornir_shared import NearestPowerOfTwo, misc

    misc.RunWithProfiler("__ExecuteProfiler()", r"C:\Temp\StosBrute")
    # __ExecuteProfiler()
    pass

