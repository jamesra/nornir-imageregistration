"""
Phase correlation module for image registration.

This module provides functions for aligning images using phase correlation techniques.

The phase correlation method is based on the Fourier shift theorem, which states that
a shift in the spatial domain corresponds to a linear phase change in the frequency domain.
By computing the cross-power spectrum of two images and finding the location of the peak
in the inverse Fourier transform, we can determine the relative shift between the images.

Key functions:
- pad_image_for_phase_correlation: Prepares an image for phase correlation by padding it
- image_phase_correlation: Calculates the phase correlation between two images
- fft_phase_correlation: Calculates the phase correlation between two FFT-transformed images
- find_peak: Finds the peak in a phase correlation image
- find_offset: Finds the alignment between two images using phase correlation

This module supports both CPU (numpy) and GPU (cupy) computation, automatically selecting
the appropriate backend based on availability.
"""
import logging
from typing import Any, NamedTuple, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.peak_uniqueness import (
    DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
    masked_peak_ratio,
)
from nornir_imageregistration.core import (
    DimensionWithOverlap,
    GenRandomData,
    NearestPowerOfTwoWithOverlap,
    promote_dtype_for_value_range,
)
from nornir_imageregistration.mathfuncs import (
    CutoffMethod,
    estimate_cutoff,
    linear_percentile_curve,
)

try:
    import cupy as cp
    import cupyx
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx


def _xp_1d_to_float_pair(values, xp) -> tuple[float, float]:
    """Export a length-2 vector to Python floats without allocating a NumPy array."""
    flat = xp.ravel(values)
    if xp is np:
        return (float(flat[0]), float(flat[1]))
    return (float(flat[0].item()), float(flat[1].item()))


def _scaled_offset_from_center_of_mass(image_shape, peak_center_of_mass, xp) -> tuple[float, float]:
    """Peak offset from image center; float64 so large frames keep sub-pixel CoM digits."""
    scaled_offset_arr = (
        xp.asarray(image_shape, dtype=xp.float64) / xp.float64(2.0)
    ) - xp.asarray(peak_center_of_mass, dtype=xp.float64)
    return _xp_1d_to_float_pair(scaled_offset_arr, xp)


def _coerce_array_to_module(array: NDArray[Any], xp) -> NDArray[Any]:
    """Return *array* on *xp* (NumPy or CuPy) without extra copies when already there."""
    if cp.get_array_module(array) is xp:
        return array
    if xp is np:
        return array.get() if hasattr(array, "get") else np.asarray(array)
    return xp.asarray(array)


_logger = logging.getLogger(__name__)


def _no_peak_offset() -> tuple[float, float]:
    """Zero translation for a missing correlation peak.

    A successful peak reports ``(shape / 2) - peak_com``. Returning the image
    center coordinates instead (as if the peak were at array origin) applies a
    translation of half the padded FFT size and can place source and target
    outside each other's bounding boxes.
    """
    return (0.0, 0.0)


def _no_peak_result(reason: str, **context: Any) -> "FindPeakResult":
    """The degenerate all-zero result, logged with which of the paths produced it.

    ``find_peak`` reaches this from three unrelated conditions -- an overlap mask that
    admits nothing, a correlation surface with nothing above the cutoff, and labelled
    components that all sum to zero -- and every one of them returned a byte-identical
    ``FindPeakResult((0, 0), 0, 0.0, 0.0, 0.0)``. A caller sees ``weight == 0`` and
    cannot tell a misconfigured overlap window from a blank tile from a flat surface.
    The result stays identical, since callers gate on the weight; only the reason is
    now recoverable.

    Logged at debug because a blank or featureless tile is ordinary in a large mosaic
    and these would otherwise fire per tile.
    """
    if _logger.isEnabledFor(logging.DEBUG):
        detail = ', '.join(f'{k}={v}' for k, v in context.items())
        _logger.debug('find_peak found no usable peak (%s)%s',
                      reason, f': {detail}' if detail else '')
    return FindPeakResult(_no_peak_offset(), 0, 0.0, 0.0, 0.0)


def pad_image_for_phase_correlation(image: NDArray[np.floating],
                                    min_overlap: float = .05,
                                    image_median: Optional[float] = None,
                                    image_stddev: Optional[float] = None,
                                    original_shape: Optional[Union[Tuple[int, int], NDArray[np.integer]]] = None,
                                    new_width: Optional[int] = None,
                                    new_height: Optional[int] = None,
                                    power_of_two: bool = True,
                                    always_copy: bool = True) -> NDArray[np.floating]:
    """
    Prepare an image for use with the phase correlation operation.

    Padded areas are filled with noise matching the histogram of the original image.
    This ensures that the phase correlation algorithm works correctly by avoiding
    edge artifacts.

    The result uses the same array module (NumPy vs CuPy) as *image* — see
    ``cupy.get_array_module``.

    :param image: Input image to be padded
    :param min_overlap: Minimum overlap allowed between the input image and images it will be registered to, defaults to 0.05
    :param image_median: Median value of noise, calculated or pulled from cache if None, defaults to None
    :param image_stddev: Standard deviation of noise, calculated or pulled from cache if None, defaults to None
    :param original_shape: The original size of the image. If None, the shape of the input image is used. Set this if the image has been previously padded to prevent over-padding, defaults to None
    :param new_width: Pad input image to this width if not None, defaults to None
    :param new_height: Pad input image to this height if not None, defaults to None
    :param power_of_two: Pad the image to a power of two if True, defaults to True
    :param always_copy: If True, always copy the image even if no padding is needed, defaults to True
    :return: An image with the input image centered surrounded by noise
    :rtype: NDArray[np.floating]
    """

    xp = cp.get_array_module(image)
    image = xp.asarray(image)
    on_gpu = xp is not np

    if image_median is not None and image_stddev is not None:
        min_v = float(image_median) - 4.0 * float(image_stddev)
        max_v = float(image_median) + 4.0 * float(image_stddev)
        min_val = min_v
        max_val = max_v
    else:
        min_val = image.min()
        max_val = image.max()
        min_v = float(xp.asarray(min_val, dtype=xp.float64).ravel()[0])
        max_v = float(xp.asarray(max_val, dtype=xp.float64).ravel()[0])

    height = image.shape[0]
    width = image.shape[1]

    original_height = height
    original_width = width

    if original_shape is not None:
        original_width = original_shape[1]
        original_height = original_shape[0]

    if new_height is None:
        if power_of_two:
            new_height = int(NearestPowerOfTwoWithOverlap(original_height, min_overlap))
        else:
            new_height = int(DimensionWithOverlap(original_height, min_overlap))

    if new_width is None:
        if power_of_two:
            new_width = int(NearestPowerOfTwoWithOverlap(original_width, min_overlap))
        else:
            new_width = int(DimensionWithOverlap(original_width, min_overlap))

    # If we need a smaller size than we already are (from padding an image a 2nd time) then keep current size
    if new_width < image.shape[1]:
        new_width = image.shape[1]

    if new_height < image.shape[0]:
        new_height = image.shape[0]

    if width >= new_width and height >= new_height:
        if always_copy:
            return xp.copy(image)
        else:
            return image

    if image_median is None or image_stddev is None:
        image_1d = image.astype(xp.float64, copy=False)
        image_1d = image_1d.ravel() if on_gpu else image_1d.flat

        if image_median is None:
            image_median = float(xp.median(image_1d))
        if image_stddev is None:
            image_stddev = float(xp.std(image_1d))

        del image_1d

    desired_type = promote_dtype_for_value_range(image.dtype, min_v, max_v)

    assert new_height is not None and new_width is not None
    nh, nw = cast(int, new_height), cast(int, new_width)
    padded_image = xp.zeros((nh, nw), dtype=np.dtype(desired_type))

    padded_image_x_offset = int(np.floor((nw - width) / 2.0))
    padded_image_y_offset = int(np.floor((nh - height) / 2.0))

    # Copy image into padded image
    padded_image[padded_image_y_offset:padded_image_y_offset + height,
    padded_image_x_offset:padded_image_x_offset + width] = image[:, :]

    if not width == nw:
        left_border = GenRandomData(
            nh,
            padded_image_x_offset,
            image_median,
            image_stddev,
            min_val,
            max_val,
            dtype=desired_type,
            xp=xp,
        )
        right_border = GenRandomData(
            nh,
            nw - (width + padded_image_x_offset),
            image_median,
            image_stddev,
            min_val,
            max_val,
            dtype=desired_type,
            xp=xp,
        )

        padded_image[:, 0:padded_image_x_offset] = left_border
        padded_image[:, width + padded_image_x_offset:] = right_border

        del left_border
        del right_border

    if not height == nh:
        top_border = GenRandomData(
            padded_image_y_offset,
            width,
            image_median,
            image_stddev,
            min_val,
            max_val,
            dtype=desired_type,
            xp=xp,
        )
        bottom_border = GenRandomData(
            nh - (height + padded_image_y_offset),
            width,
            image_median,
            image_stddev,
            min_val,
            max_val,
            dtype=desired_type,
            xp=xp,
        )

        padded_image[0:padded_image_y_offset,
        padded_image_x_offset:padded_image_x_offset + width] = top_border
        padded_image[padded_image_y_offset + height:,
        padded_image_x_offset:padded_image_x_offset + width] = bottom_border

        del top_border
        del bottom_border

    return padded_image


def image_phase_correlation(target_image: NDArray[np.floating],
                            source_image: NDArray[np.floating],
                            target_mean: Optional[float] = None,
                            source_mean: Optional[float] = None,
                            correlation_coefficient: Optional[float] = None,
                            fft_target: Optional[NDArray[Any]] = None) -> NDArray[np.floating]:
    """
    Calculate the phase shift correlation of the FFT's of two images.

    This function computes the phase correlation between two images, which is useful
    for determining the relative shift between them.

    :param target_image: Target (fixed) grayscale image
    :param source_image: Source (moving) grayscale image. Dimensions must match target_image.
    :param target_mean: Mean value of the target image. If None, it will be calculated, defaults to None
    :param source_mean: Mean value of the source image. If None, it will be calculated, defaults to None
    :param correlation_coefficient: Controls the type of correlation. Setting this value to 1 is equivalent to using phase correlation. Setting it to 0 is equivalent to using Pearson Correlation. The default is 0.65. If you have a difficult to register section, changing this value to 1 may help, defaults to None
    :param fft_target: Optional precomputed FFT of ``(target_image - target_mean)``. When
        provided, the target FFT is reused and not freed (for multi-angle sweeps).
    :return: Correlation image of the FFT's. Light pixels indicate the phase is well aligned at that offset.

    Accepts NumPy or CuPy; ops follow ``cp.get_array_module``.
    :raises ValueError: If the dimensions of target_image and source_image do not match.
    """
    xp = cp.get_array_module(target_image if fft_target is None else fft_target)
    source_image = _coerce_array_to_module(source_image, xp)
    if fft_target is None:
        target_image = _coerce_array_to_module(target_image, xp)

    if fft_target is None and not (target_image.shape == source_image.shape):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")
    if fft_target is not None and fft_target.shape != source_image.shape:
        raise ValueError("ImagePhaseCorrelation: fft_target shape must match source_image shape")

    # --------------------------------
    # This is here in case this function ever needs to be revisited.  Scipy is a lot faster working with in-place operations so this
    # code has been obfuscated more than I like
    # target_fft = fftpack.rfft2(FixedImage)
    # source_fft = fftpack.rfft2(MovingImage)
    # conjFFTFixed = conj(target_fft)
    # Numerator = conjFFTFixed * source_fft
    # Divisor = abs(conjFFTFixed * source_fft)
    # T = Numerator / Divisor
    # CorrelationImage = real(fftpack.irfft2(T))
    # --------------------------------
    if source_mean is None:
        source_mean = float(xp.mean(source_image))

    source_fft = xp.fft.fft2(source_image - source_mean)

    if fft_target is not None:
        # Shared target FFT across angles: do not delete it; free only the source FFT after use.
        correlation = fft_phase_correlation(
            fft_target, source_fft, False, correlation_coefficient=correlation_coefficient)
        del source_fft
        return correlation

    if target_mean is None:
        target_mean = float(xp.mean(target_image))
    target_fft = xp.fft.fft2(target_image - target_mean)

    return fft_phase_correlation(target_fft, source_fft, True, correlation_coefficient=correlation_coefficient)


def fft_phase_correlation(fft_target: NDArray[Any],
                          fft_source: NDArray[Any],
                          delete_input: bool = False,
                          correlation_coefficient: Optional[float] = None) -> NDArray[np.floating]:
    """
    Calculate the phase shift correlation of the FFT's of two images.

    This function computes the phase correlation between two images that have already
    been transformed to the frequency domain using FFT.

    :param fft_target: FFT of the target (fixed) image
    :param fft_source: FFT of the source (moving) image. Dimensions must match fft_target.
    :param delete_input: If True, the input arrays will be deleted to save memory, defaults to False
    :param correlation_coefficient: Controls the type of correlation. Setting this value to 1 is equivalent to using phase correlation. Setting it to 0 is equivalent to using Pearson Correlation. The default is 0.65. If you have a difficult to register section, changing this value to 1 may help, defaults to None
    :return: Correlation image of the FFT's. Light pixels indicate the phase is well aligned at that offset.
    :rtype: NDArray[np.floating]
    :raises ValueError: If the dimensions of fft_target and fft_source do not match.
    :note:
    # --------------------------------
    # This is a working implementation of the phase correlation algorithm.  This code has been optimized and somewhat obfuscated as a result
    # code has been obfuscated more than I like
    # FFTFixed = fftpack.rfft2(FixedImage)
    # FFTMoving = fftpack.rfft2(MovingImage)
    # conj_fft_target = conj(FFTFixed)
    # Numerator = conj_fft_target * FFTMoving
    # Divisor = abs(conj_fft_target * FFTMoving)
    # T = Numerator / Divisor
    # CorrelationImage = real(fftpack.irfft2(T))
    # --------------------------------
    """

    if correlation_coefficient is None:
        correlation_coefficient = 0.65

    if not (fft_target.shape == fft_source.shape):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")

    # Ensure that correlation_coefficient is between 0 and 1
    if correlation_coefficient < 0 or correlation_coefficient > 1:
        raise ValueError("correlation_coefficient must be between 0 and 1")

    # Get the array module (numpy or cupy) based on the input arrays
    xp = cp.get_array_module(fft_target)
    # Step 1: Calculate the complex conjugate of the target FFT
    conj_fft_target = xp.conjugate(fft_target)
    if delete_input:
        del fft_target  # Free memory if requested

    # Step 2: Multiply the conjugate of the target FFT with the source FFT
    # This is the cross-power spectrum
    conj_fft_target *= fft_source

    if delete_input:
        del fft_source  # Free memory if requested

    # Step 3: Normalize the cross-power spectrum
    # This step is what makes it "phase correlation" rather than just cross-correlation
    abs_conj_target_fft = xp.absolute(conj_fft_target)

    # Only normalize values above a small threshold to avoid division by zero
    mask = abs_conj_target_fft > 1e-5

    # The correlation_coefficient controls the type of correlation:
    # - 1.0: Pure phase correlation (normalizes by absolute value)
    # - 0.0: Pearson correlation (no normalization)
    # - 0.65: Default, a blend that often works well in practice
    conj_fft_target[mask] /= xp.power(abs_conj_target_fft[mask], correlation_coefficient)
    del mask
    del abs_conj_target_fft

    # Step 4: Inverse FFT to get the correlation image
    correlation_image = xp.real(xp.fft.ifft2(conj_fft_target))
    del conj_fft_target  # Free memory

    return correlation_image


class FindPeakResult(NamedTuple):
    """
    Result of finding a peak in a phase correlation image.

    :attr scaled_offset: The offset of the peak from the center of the image (y, x)
    :attr peak_strength: The strength of the peak (signal-to-noise ratio)
    :attr cutoff_value: The cutoff value used to threshold the image
    :attr cutoff_percent: The percentile used to determine the cutoff value
    :attr peak_ratio: Primary / masked-2nd-peak uniqueness ratio
    """
    scaled_offset: tuple[float, float]
    peak_strength: float
    cutoff_value: float
    cutoff_percent: float
    peak_ratio: float = 0.0


# Masked-sample crossover from host percentile vs one CuPy sort (2D masked
# sweep: 384² / ~113k still host; 512² / ~201k device 1.5×). Dispatch on
# sample count, not image side — overlap masks shrink n.
_DEVICE_SORT_MIN_SAMPLES = 160_000


def _percentile_curve_for_cutoff(
        masked_values: NDArray[np.floating],
        percentiles: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the 101-point Raw-cutoff curve on host.

    Small CuPy surfaces download and use NumPy percentile. Larger surfaces sort
    once on-device and transfer only the curve.
    """
    xp = cp.get_array_module(masked_values)
    n = int(masked_values.size)
    if xp is not np and n >= _DEVICE_SORT_MIN_SAMPLES:
        curve = linear_percentile_curve(masked_values, percentiles)
        return nornir_imageregistration.EnsureNumpyArray(curve)
    host = nornir_imageregistration.EnsureNumpyArray(masked_values)
    try:
        return np.percentile(host, percentiles, method="linear")
    except TypeError:
        return np.percentile(host, percentiles)


def find_peak(image: NDArray[np.floating],
              overlap_mask: Optional[NDArray[np.bool_]] = None,
              cutoff: Optional[float] = None,
              allow_in_place: bool = False,
              peak_ratio_exclusion_radius: int = DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
              ) -> FindPeakResult:
    """
    Find the offset of the strongest response in a phase correlation image.

    This function identifies the peak in a phase correlation image, which corresponds
    to the most likely offset between the original images.

    :param image: Phase correlation image to find the peak in
    :param overlap_mask: Mask describing which pixels are eligible for consideration, defaults to None
    :param cutoff: Percentile used to threshold image. Values below the percentile are ignored. If None, an automatic cutoff is determined, defaults to None
    :param allow_in_place: If True, *image* may be overwritten by the overlap-mask
        multiply. Callers that discard the correlation image immediately after may
        pass True to avoid a full-size copy. The cutoff is *not* applied to *image*;
        it is evaluated into a separate boolean so the uniqueness ratio can still
        see competing peaks. Defaults to False.
    :param peak_ratio_exclusion_radius: Half-width cleared around the primary peak
        before measuring uniqueness (primary / 2nd peak).
    :return: A named tuple containing the offset of the peak, the strength of the peak,
        the cutoff value, the cutoff percentile, and the peak uniqueness ratio
    :rtype: FindPeakResult
    """
    # Get the appropriate array module (numpy or cupy) based on the input image
    xp = cp.get_array_module(image)
    sp = cupyx.scipy.get_array_module(image)

    n_valid: int | None = None
    if overlap_mask is not None:
        xp_mask = cp.get_array_module(overlap_mask)
        if xp_mask is not xp:
            overlap_mask = xp.asarray(overlap_mask)
        n_valid = int(xp.count_nonzero(overlap_mask))
        if n_valid == 0:
            return _no_peak_result('the overlap mask admits no pixels',
                                   mask_shape=tuple(overlap_mask.shape),
                                   image_shape=tuple(image.shape))

    # Fuse copy + mask: one allocation (or in-place) instead of copy + logical_not temp.
    if overlap_mask is not None:
        if allow_in_place:
            masked_image = image
            masked_image *= overlap_mask
        else:
            masked_image = image * overlap_mask
    elif allow_in_place:
        masked_image = image
    else:
        masked_image = xp.copy(image)

    # Mean over valid pixels BEFORE cutoff thresholding. Uses mask-entry count so
    # in-mask zeros are retained (matches xp.mean(image[overlap_mask])).
    if overlap_mask is not None:
        # n_valid already counted above; early return guarantees it is > 0.
        mean_pixel = float(masked_image.sum(dtype=xp.float64) / n_valid)  # type: ignore[operator]
    else:
        mean_pixel = float(masked_image.mean())

    # Determine the cutoff value for thresholding
    if cutoff is None:
        percentiles = np.linspace(0.95, 1, 101) * 100
        try:
            if overlap_mask is not None:
                masked_values = masked_image[overlap_mask].ravel()
            else:
                masked_values = masked_image.ravel()

            curve_host = _percentile_curve_for_cutoff(masked_values, percentiles)
            del masked_values
            result = estimate_cutoff(
                curve_host,
                percentiles,
                polyfit_degree=2,
                method=CutoffMethod.Raw,
                precomputed_percentile_values=curve_host,
            )
            del curve_host
            cutoff_percent = float(percentiles[result.cutoff_percentile_index])
            cutoff_value = result.cutoff_value
        except ValueError:
            cutoff_percent = 99.6
            if overlap_mask is not None:
                masked = masked_image[overlap_mask]
            else:
                masked = masked_image.ravel()
            cutoff_value = float(xp.percentile(masked, q=cutoff_percent))
            del masked
    else:
        # Use the provided cutoff value (fraction 0-1 -> percentile 0-100)
        cutoff_percent = cutoff * 100
        if overlap_mask is not None:
            masked = masked_image[overlap_mask]
        else:
            masked = masked_image.ravel()
        cutoff_value = float(xp.percentile(masked, q=cutoff_percent))
        del masked

    # Identify the above-cutoff region as a boolean rather than zeroing sub-cutoff
    # values in the surface itself. The old in-place multiply destroyed the
    # caller's correlation image when allow_in_place=True, and the uniqueness
    # measurement below needs that surface intact to see competing peaks. A bool
    # mask costs the same temporary the multiply already allocated for its
    # comparison, minus the multiply.
    above_cutoff = masked_image >= cutoff_value
    if cutoff_value <= 0.0:
        # The multiply left exact zeros as background even when the cutoff was
        # <= 0. Preserve that: otherwise zero regions join neighbouring
        # components and drag the center of mass off the peak.
        above_cutoff &= masked_image != 0

    # Label connected components of the above-cutoff region
    [label_image, num_labels] = sp.ndimage.label(above_cutoff)
    del above_cutoff

    # If no labels were found, there are no peaks
    if num_labels == 0:
        return _no_peak_result('nothing survived the cutoff',
                               cutoff=cutoff_value, percentile=cutoff_percent,
                               image_shape=tuple(image.shape))

    # Calculate the sum of pixel values for each label
    # The first interesting label starts at 1, 0 is the background
    # Weighted statistics read the un-thresholded surface: every pixel inside a
    # label is above the cutoff by construction, so these sums are identical to
    # the values the old thresholded buffer produced.
    label_sums = sp.ndimage.sum_labels(masked_image, label_image, xp.array(range(1, num_labels + 1)))

    # Find the label with the highest sum (strongest peak)
    peak_value_index = label_sums.argmax()
    peak_label_sum = float(label_sums[peak_value_index])
    del label_sums

    # center_of_mass normalises by the sum of the *selected* label, not by the total
    # across every label. Testing the total is the same test only while the surface is
    # non-negative, which is what the registration callers hand in. On a signed surface
    # the strongest component can sum to zero while the total does not, and the division
    # below then raised FloatingPointError instead of returning the degenerate result
    # this function produces for every other unusable surface. numpy runs with
    # divide='raise' here, so it is a hard error rather than a nan.
    if peak_label_sum <= 0:
        del label_image
        del masked_image

        return _no_peak_result('the strongest labelled component does not sum above zero',
                               num_labels=num_labels, cutoff=cutoff_value,
                               image_shape=tuple(image.shape))

    # Calculate the center of mass for the strongest peak
    # Because we offset the sum_labels call by 1, we must do the same for the peak_value_index
    peak_center_of_mass = sp.ndimage.center_of_mass(masked_image, label_image, int(peak_value_index + 1))

    # Signal-to-noise: peak max / pre-threshold mean of valid pixels
    peak_pixel = sp.ndimage.maximum(masked_image, label_image, int(peak_value_index + 1))
    del label_image
    del masked_image

    signal_to_noise = float(peak_pixel) / mean_pixel if mean_pixel != 0.0 else 0.0
    # Same array module as the input avoids implicit CuPy->NumPy conversions for
    # 0-d cupy.ndarray center-of-mass components; float64 keeps sub-pixel digits
    # on large frames (float32 loses ~4e-4 px near 16k).
    scaled_offset = _scaled_offset_from_center_of_mass(image.shape, peak_center_of_mass, xp)

    # Uniqueness on the original (pre-threshold) correlation surface.
    com = xp.asarray(peak_center_of_mass, dtype=xp.float64)
    if hasattr(com, 'get'):
        com = com.get()
    com_np = np.asarray(com, dtype=np.float64).reshape(-1)
    peak_row = int(np.clip(np.rint(com_np[0]), 0, image.shape[0] - 1))
    peak_col = int(np.clip(np.rint(com_np[1]), 0, image.shape[1] - 1))
    # Prefer the raw correlation sample at the COM; fall back to labeled max.
    raw_primary = float(image[peak_row, peak_col])
    if not np.isfinite(raw_primary) or raw_primary <= 0.0:
        raw_primary = float(peak_pixel)
    ratio = masked_peak_ratio(
        image,
        peak_row,
        peak_col,
        exclusion_radius=peak_ratio_exclusion_radius,
        overlap_mask=overlap_mask,
        primary_value=raw_primary,
    )
    if signal_to_noise <= 0.0:
        ratio = 0.0

    return FindPeakResult(
        scaled_offset,
        float(signal_to_noise),
        float(cutoff_value),
        float(cutoff_percent),
        float(ratio),
    )


def find_offset(target_image: NDArray[np.floating],
                source_image: NDArray[np.floating],
                min_overlap: float = 0.0,
                max_overlap: float = 1.0,
                fft_required: bool = True,
                target_shape: Optional[Union[Tuple[int, int], NDArray[np.integer]]] = None,
                source_shape: Optional[Union[Tuple[int, int], NDArray[np.integer]]] = None,
                correlation_coefficient: Optional[float] = None) -> nornir_imageregistration.AlignmentRecord:
    """
    Find the alignment between two images using phase correlation.

    This function returns an alignment record describing how the images overlap. 
    The alignment record indicates how much the source image must be rotated and 
    translated to align perfectly with the target image.

    :param target_image: Target (fixed) image we are registering into
    :param source_image: Source (moving) image we are registering from
    :param min_overlap: The minimum amount of overlap by area the registration must have, defaults to 0.0
    :param max_overlap: The maximum amount of overlap by area the registration must have, defaults to 1.0
    :param fft_required: If True, the input images will be transformed to FFT space. If False, the input images are assumed to already be in FFT space, defaults to True
    :param target_shape: If specified, contains the size of the target image before padding. Used to calculate mask for valid overlap values, defaults to None
    :param source_shape: If specified, contains the size of the source image before padding. Used to calculate mask for valid overlap values, defaults to None
    :param correlation_coefficient: Controls the type of correlation. See image_phase_correlation for details, defaults to None
    :return: An alignment record containing the peak offset and weight
    :rtype: nornir_imageregistration.AlignmentRecord
    :note: If adjusting control points, the peak can be added to the target image's control point, or subtracted from the source image's control point (accounting for any transform used to create the source image) to align the images.
    """

    if target_shape is None:
        target_shape = target_image.shape

    if source_shape is None:
        source_shape = source_image.shape

    xp = cp.get_array_module(target_image)

    # Find peak requires both the fixed and moving images have equal size
    if not ((target_image.shape[0] == source_image.shape[0]) and (target_image.shape[1] == source_image.shape[1])):
        # Pad the smaller image to the appropriate size
        (desired_height, desired_width) = (
            max((target_image.shape[0], source_image.shape[0])), max((target_image.shape[1], source_image.shape[1])))
        target_image = pad_image_for_phase_correlation(target_image, min_overlap=1, new_width=desired_width,
                                                       new_height=desired_height, always_copy=False)
        source_image = pad_image_for_phase_correlation(source_image, min_overlap=1, new_width=desired_width,
                                                       new_height=desired_height, always_copy=False)

    correlation_image = None
    if fft_required:
        correlation_image = image_phase_correlation(target_image,
                                                    source_image,
                                                    correlation_coefficient=correlation_coefficient)
    else:
        correlation_image = fft_phase_correlation(target_image,
                                                  source_image,
                                                  delete_input=False,
                                                  correlation_coefficient=correlation_coefficient)

    correlation_image = xp.fft.fftshift(correlation_image)

    # Normalize while guarding against flat/invalid responses. Some low-information
    # tiles can produce a near-constant correlation image; avoid divide-by-zero/NaN.
    correlation_image -= correlation_image.min()
    corr_max = correlation_image.max()
    corr_max_value = float(xp.asarray(corr_max, dtype=xp.float64).ravel()[0])
    if np.isfinite(corr_max_value) and corr_max_value > 0.0:
        correlation_image /= corr_max
    else:
        correlation_image[...] = 0

    # Get mask of valid overlap regions (upload once per geometry when on GPU).
    overlap_mask = nornir_imageregistration.overlapmasking.GetOverlapMaskOnDevice(
        target_shape,
        source_shape,
        correlation_image.shape,
        min_overlap,
        max_overlap,
        xp=xp)
    peak_result = find_peak(correlation_image, overlap_mask, allow_in_place=True)
    peak = peak_result.scaled_offset
    weight = peak_result.peak_strength

    del correlation_image

    record = nornir_imageregistration.AlignmentRecord(
        peak=peak, weight=weight, peak_ratio=float(peak_result.peak_ratio))

    return record


if __name__ == '__main__':
    """
    Legacy Windows fixture + cProfile harness. Opt-in only:

        NORNIR_PROFILE=1 python -m nornir_imageregistration.phasecorrelation

    Prefer ``scripts/audit_cupy_item_bench.py`` for CuPy audit timings.
    """
    import os
    import matplotlib.pyplot as plt

    _profile = os.environ.get('NORNIR_PROFILE', '').strip().lower()
    if _profile not in ('1', 'true', 'yes', 'on'):
        raise SystemExit(
            'phasecorrelation.py is a library module. '
            'Set NORNIR_PROFILE=1 to run the legacy fixture profiler.'
        )

    # Set up test files and output directory
    filename_a = 'C:\\BuildScript\\Test\\Images\\400.png'
    filename_b = 'C:\\BuildScript\\Test\\Images\\401.png'
    output_dir = 'C:\\Buildscript\\Test\\Results\\'

    os.makedirs(output_dir, exist_ok=True)


    def test_phase_correlation(im_a, im_b):
        """
        Test phase correlation between two images.

        :param im_a: First image to compare
        :param im_b: Second image to compare
        :return: None
        """
        # Pad images for phase correlation
        fixed_a = pad_image_for_phase_correlation(im_a)
        moving_b = pad_image_for_phase_correlation(im_b)

        # Find the offset between the images
        record = find_offset(fixed_a, moving_b, target_shape=im_a.shape, source_shape=im_b.shape)
        print(str(record))

        # Create and save a STOS file with the alignment
        stos = record.ToStos(filename_a, filename_b)
        stos.Save(os.path.join(output_dir, "TestPhaseCorrelation.stos"))
        return


    def main():
        """
        Main function to run the test multiple times.

        :return: None
        """
        im_a = plt.imread(filename_a)
        im_b = plt.imread(filename_b)

        for i in range(1, 5):
            print(f"Test run {i}")
            test_phase_correlation(im_a, im_b)


    import cProfile
    import pstats

    cProfile.run('main()', 'CoreProfile.pr')
    pr = pstats.Stats('CoreProfile.pr')
    pr.sort_stats('time')
    print(str(pr.print_stats(.5)))

