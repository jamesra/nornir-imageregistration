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
from typing import Any, NamedTuple, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.core import (
    DimensionWithOverlap,
    GenRandomData,
    NearestPowerOfTwoWithOverlap,
    promote_dtype_for_value_range,
)
from nornir_imageregistration.mathfuncs import CutoffMethod, estimate_cutoff

try:
    import cupy as cp
    import cupyx
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx


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
                            correlation_coefficient: Optional[float] = None) -> NDArray[np.floating]:
    """
    Calculate the phase shift correlation of the FFT's of two images.

    This function computes the phase correlation between two images, which is useful
    for determining the relative shift between them.

    :param target_image: Target (fixed) grayscale image
    :param source_image: Source (moving) grayscale image. Dimensions must match target_image.
    :param target_mean: Mean value of the target image. If None, it will be calculated, defaults to None
    :param source_mean: Mean value of the source image. If None, it will be calculated, defaults to None
    :param correlation_coefficient: Controls the type of correlation. Setting this value to 1 is equivalent to using phase correlation. Setting it to 0 is equivalent to using Pearson Correlation. The default is 0.65. If you have a difficult to register section, changing this value to 1 may help, defaults to None
    :return: Correlation image of the FFT's. Light pixels indicate the phase is well aligned at that offset.
    :raises ValueError: If the dimensions of target_image and source_image do not match.
    """
    xp = cp.get_array_module(target_image)

    if not (target_image.shape == source_image.shape):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")

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
    if target_mean is None:
        target_mean = float(xp.mean(target_image))
    if source_mean is None:
        source_mean = float(xp.mean(source_image))

    target_fft = xp.fft.fft2(target_image - target_mean)
    source_fft = xp.fft.fft2(source_image - source_mean)

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
    """
    scaled_offset: tuple[float, float]
    peak_strength: float
    cutoff_value: float
    cutoff_percent: float


def find_peak(image: NDArray[np.floating],
              overlap_mask: Optional[NDArray[np.bool_]] = None,
              cutoff: Optional[float] = None) -> FindPeakResult:
    """
    Find the offset of the strongest response in a phase correlation image.

    This function identifies the peak in a phase correlation image, which corresponds
    to the most likely offset between the original images.

    :param image: Phase correlation image to find the peak in
    :param overlap_mask: Mask describing which pixels are eligible for consideration, defaults to None
    :param cutoff: Percentile used to threshold image. Values below the percentile are ignored. If None, an automatic cutoff is determined, defaults to None
    :return: A named tuple containing the offset of the peak, the strength of the peak, the cutoff value, and the cutoff percentile
    :rtype: FindPeakResult
    """
    # Get the appropriate array module (numpy or cupy) based on the input image
    xp = cp.get_array_module(image)
    sp = cupyx.scipy.get_array_module(image)

    # Create a copy of the image for thresholding
    threshold_image = xp.copy(image)

    # Apply the overlap mask if provided
    if overlap_mask is not None:
        threshold_image[xp.logical_not(overlap_mask)] = 0

    # Determine the cutoff value for thresholding
    if cutoff is None:
        # Use percentiles between 95% and 100% to find an optimal cutoff
        percentiles = np.linspace(0.95, 1, 101) * 100
        try:
            # Use the estimate_cutoff function to automatically determine the best cutoff
            if overlap_mask is not None:
                result = estimate_cutoff(
                    image[overlap_mask].ravel(),
                    percentiles,
                    polyfit_degree=2,
                    method=CutoffMethod.Raw
                )
            else:
                result = estimate_cutoff(
                    image.ravel(),
                    percentiles,
                    polyfit_degree=2,
                    method=CutoffMethod.Raw
                )
            cutoff_percent = percentiles[result.cutoff_percentile_index] * 100
            cutoff_value = result.cutoff_value
        except ValueError:
            # Fallback to a fixed percentile if automatic estimation fails
            cutoff_percent = 99.6
            cutoff_value = xp.percentile(threshold_image[overlap_mask], q=cutoff_percent)
    else:
        # Use the provided cutoff value
        cutoff_percent = cutoff * 100
        cutoff_value = xp.percentile(threshold_image[overlap_mask], q=cutoff_percent)

    # Apply thresholding - set all values below the cutoff to zero
    threshold_image[threshold_image < cutoff_value] = 0

    # Label connected components in the thresholded image
    [label_image, num_labels] = sp.ndimage.label(threshold_image)

    # If no labels were found, there are no peaks
    if num_labels == 0:
        scaled_offset = tuple((np.asarray(image.shape, dtype=np.float32) / 2.0).tolist())
        peak_strength = 0
        return FindPeakResult(scaled_offset, peak_strength, 0.0, 0.0)

    # Calculate the sum of pixel values for each label
    # The first interesting label starts at 1, 0 is the background
    label_sums = sp.ndimage.sum_labels(threshold_image, label_image, xp.array(range(1, num_labels + 1)))

    if label_sums.sum() == 0:  # There are no peaks identified
        scaled_offset = tuple((np.asarray(image.shape, dtype=np.float32) / 2.0).tolist())
        peak_strength = 0
        return FindPeakResult(scaled_offset, peak_strength, 0.0, 0.0)
    else:
        # Find the label with the highest sum (strongest peak)
        peak_value_index = label_sums.argmax()
        peak_strength = label_sums[peak_value_index]

        # Calculate the center of mass for the strongest peak
        # Because we offset the sum_labels call by 1, we must do the same for the peak_value_index
        peak_center_of_mass = sp.ndimage.center_of_mass(threshold_image, label_image, int(peak_value_index + 1))

        # Calculate signal-to-noise ratio
        mean_pixel = xp.mean(image[overlap_mask])
        peak_pixel = sp.ndimage.maximum(threshold_image, label_image, int(peak_value_index + 1))
        signal_to_noise = peak_pixel / mean_pixel
        # Calculate the offset from the center of the image using the same array module as the input.
        # This avoids implicit CuPy->NumPy conversions for 0-d cupy.ndarray center-of-mass components.
        scaled_offset_arr = (
            xp.asarray(image.shape, dtype=xp.float32) / xp.float32(2.0)
        ) - xp.asarray(peak_center_of_mass, dtype=xp.float32)

        # Clean up memory
        del label_image
        del threshold_image
        del label_sums

        scaled_offset = tuple(nornir_imageregistration.EnsureNumpyArray(scaled_offset_arr).tolist())
        return FindPeakResult(
            scaled_offset,
            float(signal_to_noise),
            float(cutoff_value),
            float(cutoff_percent),
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

    # Crop the areas that cannot overlap
    correlation_image -= correlation_image.min()
    correlation_image /= correlation_image.max()

    # Get mask of valid overlap regions
    overlap_mask = nornir_imageregistration.GetOverlapMask(target_shape,
                                                           source_shape,
                                                           correlation_image.shape,
                                                           min_overlap,
                                                           max_overlap)
    peak_result = find_peak(correlation_image, overlap_mask)
    peak = peak_result.scaled_offset
    weight = peak_result.peak_strength

    del correlation_image

    record = nornir_imageregistration.AlignmentRecord(peak=peak, weight=weight)

    return record


if __name__ == '__main__':
    """
    Test code for phase correlation functionality.

    This section is executed when the module is run directly.
    It demonstrates how to use the phase correlation functions
    to align two images.

    :note: This is for testing and demonstration purposes only.
    """
    import os
    import matplotlib.pyplot as plt

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


    # Profile the execution to identify performance bottlenecks
    import cProfile
    import pstats

    cProfile.run('main()', 'CoreProfile.pr')
    pr = pstats.Stats('CoreProfile.pr')
    pr.sort_stats('time')
    print(str(pr.print_stats(.5)))

