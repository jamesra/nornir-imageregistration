"""
Phase correlation module for image registration.

This module provides functions for aligning images using phase correlation techniques.
"""
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.core import (DimensionWithOverlap, GenRandomData, NearestPowerOfTwoWithOverlap)
from nornir_imageregistration.mathfuncs import CutoffMethod

try:
    import cupy as cp
    import cupyx
    import cupy.fft as fftpack
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
    import numpy.fft as fftpack


def pad_image_for_phase_correlation(image: NDArray[np.floating],
                                    min_overlap: float = .05,
                                    image_median: Optional[float] = None,
                                    image_stddev: Optional[float] = None,
                                    original_shape: Optional[Union[Tuple[int, int], NDArray[int]]] = None,
                                    new_width: Optional[int] = None,
                                    new_height: Optional[int] = None,
                                    power_of_two: bool = True,
                                    always_copy: bool = True,
                                    return_numpy: bool = True) -> NDArray[np.floating]:
    """
    Prepare an image for use with the phase correlation operation.

    Padded areas are filled with noise matching the histogram of the original image.
    This ensures that the phase correlation algorithm works correctly by avoiding
    edge artifacts.

    :param image: Input image to be padded
    :param min_overlap: Minimum overlap allowed between the input image and images it will be registered to, defaults to 0.05
    :param image_median: Median value of noise, calculated or pulled from cache if None, defaults to None
    :param image_stddev: Standard deviation of noise, calculated or pulled from cache if None, defaults to None
    :param original_shape: The original size of the image. If None, the shape of the input image is used. Set this if the image has been previously padded to prevent over-padding, defaults to None
    :param new_width: Pad input image to this width if not None, defaults to None
    :param new_height: Pad input image to this height if not None, defaults to None
    :param power_of_two: Pad the image to a power of two if True, defaults to True
    :param always_copy: If True, always copy the image even if no padding is needed, defaults to True
    :param return_numpy: If True, ensure the returned array is a numpy array (not used currently), defaults to True
    :return: An image with the input image centered surrounded by noise
    :rtype: NDArray[np.floating]
    """

    min_val = image.min()
    max_val = image.max()

    height = image.shape[0]
    width = image.shape[1]

    original_height = height
    original_width = width

    use_cp = nornir_imageregistration.UsingCupy()
    image = nornir_imageregistration.EnsureArray(image)
    xp = cp.get_array_module(image)

    if original_shape is not None:
        original_width = original_shape[1]
        original_height = original_shape[0]

    if new_height is None:
        if power_of_two:
            new_height = NearestPowerOfTwoWithOverlap(original_height, min_overlap)
        else:
            new_height = DimensionWithOverlap(original_height, min_overlap)

    if new_width is None:
        if power_of_two:
            new_width = NearestPowerOfTwoWithOverlap(original_width, min_overlap)
        else:
            new_width = DimensionWithOverlap(original_width, min_overlap)

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
        image_1d = image_1d.ravel() if use_cp else image_1d.flat

        if image_median is None:
            image_median = xp.median(image_1d)
        if image_stddev is None:
            image_stddev = xp.std(image_1d)

        del image_1d

    desired_type = image.dtype
    if np.finfo(desired_type).max < max_val:
        desired_type = np.float32

    padded_image = xp.zeros((int(new_height), int(new_width)), dtype=desired_type)

    padded_image_x_offset = int(np.floor((new_width - width) / 2.0))
    padded_image_y_offset = int(np.floor((new_height - height) / 2.0))

    # Copy image into padded image
    padded_image[padded_image_y_offset:padded_image_y_offset + height,
    padded_image_x_offset:padded_image_x_offset + width] = image[:, :]

    if not width == new_width:
        left_border = GenRandomData(new_height, padded_image_x_offset, image_median, image_stddev, min_val, max_val)
        right_border = GenRandomData(new_height, new_width - (width + padded_image_x_offset),
                                     image_median, image_stddev, min_val, max_val)

        padded_image[:, 0:padded_image_x_offset] = left_border
        padded_image[:, width + padded_image_x_offset:] = right_border

        del left_border
        del right_border

    if not height == new_height:
        top_border = GenRandomData(padded_image_y_offset, width, image_median, image_stddev, min_val, max_val)
        bottom_border = GenRandomData(new_height - (height + padded_image_y_offset), width,
                                      image_median, image_stddev, min_val, max_val)

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
        target_mean = xp.mean(target_image)
    if source_mean is None:
        source_mean = xp.mean(source_image)

    target_fft = fftpack.fft2(target_image - target_mean)
    source_fft = fftpack.fft2(source_image - source_mean)

    return fft_phase_correlation(target_fft, source_fft, True, correlation_coefficient=correlation_coefficient)


def fft_phase_correlation(fft_target: NDArray[np.floating],
                          fft_source: NDArray[np.floating],
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
    """

    if correlation_coefficient is None:
        correlation_coefficient = 0.65

    if not (fft_target.shape == fft_source.shape):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")

    # --------------------------------
    # This is here in case this function ever needs to be revisited.  Scipy is a lot faster working with in-place operations so this
    # code has been obfuscated more than I like
    # FFTFixed = fftpack.rfft2(FixedImage)
    # FFTMoving = fftpack.rfft2(MovingImage)
    # conj_fft_target = conj(FFTFixed)
    # Numerator = conj_fft_target * FFTMoving
    # Divisor = abs(conj_fft_target * FFTMoving)
    # T = Numerator / Divisor
    # CorrelationImage = real(fftpack.irfft2(T))
    # --------------------------------

    xp = cp.get_array_module(fft_target)
    # sp = cupyx.scipy.get_array_module(FFTFixed)

    conj_fft_target = xp.conjugate(fft_target)
    if delete_input:
        del fft_target

    conj_fft_target *= fft_source

    if delete_input:
        del fft_source

    abs_conj_target_fft = xp.absolute(conj_fft_target)

    # Based on talk with Art Wetzel, apparently wht_expon = 1 is Phase Correlation.  0 is Pierson Correlation
    mask = abs_conj_target_fft > 1e-5
    # conj_fft_target[wht_mask] /= wht_scales  # Numerator / Divisor
    # conj_fft_target[mask] /= abs_conj_target_fft[mask]
    conj_fft_target[mask] /= xp.power(abs_conj_target_fft[mask], correlation_coefficient)
    # assert (np.array_equiv(WconjFFTFixed, conj_fft_target[mask]))
    del mask

    # wht_expon_adjustment = np.power(np.absolute(conj_fft_target[mask]), wht_expon)
    # conj_fft_target[mask] *= wht_expon_adjustment
    # wht_mask = conj_fft_target > 1e-5
    # conj_fft_target[wht_mask] *= np.power(conj_fft_target[wht_mask], -0.65)
    # del wht_expon_adjustment
    del abs_conj_target_fft

    CorrelationImage = xp.real(fftpack.ifft2(conj_fft_target))
    del conj_fft_target

    return CorrelationImage


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
              overlap_mask: Optional[NDArray[bool]] = None,
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
    xp = cp.get_array_module(image)
    sp = cupyx.scipy.get_array_module(image)

    threshold_image = xp.copy(image)
    if overlap_mask is not None:
        threshold_image[xp.logical_not(overlap_mask)] = 0

    if cutoff is None:
        percentiles = np.linspace(0.95, 1, 101) * 100
        try:
            result = nornir_imageregistration.mathfuncs.estimate_cutoff(
                image[overlap_mask].flat,
                percentiles,
                polyfit_degree=2,
                method=CutoffMethod.Raw
            ) if overlap_mask is not None else nornir_imageregistration.mathfuncs.estimate_cutoff(
                image.flat,
                percentiles,
                polyfit_degree=2,
                method=CutoffMethod.Raw
            )
            cutoff_percent = percentiles[result.cutoff_percentile_index] * 100
            cutoff_value = result.cutoff_value
        except ValueError:
            cutoff_percent = 99.6
            cutoff_value = xp.percentile(threshold_image[overlap_mask], q=cutoff_percent)
    else:
        cutoff_percent = cutoff * 100
        cutoff_value = xp.percentile(threshold_image[overlap_mask], q=cutoff_percent)

    threshold_image[threshold_image < cutoff_value] = 0

    [label_image, num_labels] = sp.ndimage.label(threshold_image)
    # The first interesting label starts at 1, 0 is the background
    label_sums = sp.ndimage.sum_labels(threshold_image, label_image, xp.array(range(1, num_labels + 1)))

    if label_sums.sum() == 0:  # There are no peaks identified
        scaled_offset = (np.asarray(image.shape, dtype=np.float32) / 2.0)
        peak_strength = 0
        return FindPeakResult(scaled_offset, peak_strength, 0, 0)
    else:
        peak_value_index = label_sums.argmax()
        peak_strength = label_sums[peak_value_index]
        # Because we offset the sum_labels call by 1, we must do the same for the peak_value_index
        peak_center_of_mass = sp.ndimage.center_of_mass(threshold_image, label_image, int(peak_value_index + 1))

        mean_pixel = xp.mean(image[overlap_mask])
        peak_pixel = sp.ndimage.maximum(threshold_image, label_image, int(peak_value_index + 1))
        signal_to_noise = peak_pixel / mean_pixel

        ########################################################################
        # This was my original implementation to understand signal strength.
        # Art Wetzel convinced me to use signal to noise by dividing the
        # peak pixel intensity by the median pixel intensity

        # center_of_mass returns results as (y,x)
        if nornir_imageregistration.UsingCupy():
            peak_center_of_mass = np.array((cp.asnumpy(peak_center_of_mass[0]), cp.asnumpy(peak_center_of_mass[1])))

        scaled_offset = (np.asarray(image.shape) / 2.0) - peak_center_of_mass

        del label_image
        del threshold_image
        del label_sums

        return FindPeakResult(scaled_offset, signal_to_noise, cutoff_value, cutoff_percent)


def find_offset(target_image: NDArray[np.floating],
                source_image: NDArray[np.floating],
                min_overlap: float = 0.0,
                max_overlap: float = 1.0,
                fft_required: bool = True,
                target_shape: Optional[Union[Tuple[int, int], NDArray[int]]] = None,
                source_shape: Optional[Union[Tuple[int, int], NDArray[int]]] = None,
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
