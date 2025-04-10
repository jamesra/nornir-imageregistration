from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from typing import NamedTuple

import nornir_imageregistration
from nornir_imageregistration.core import (DimensionWithOverlap, GenRandomData, NearestPowerOfTwoWithOverlap)
from nornir_imageregistration.mathfuncs import CutoffMethod

try:
    import cupy as cp
    import cupyx
    import cp.fft as fftpack
except (ModuleNotFoundError, ImportError) as e:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

    # prettyoutput.Log('cupy not installed on system')
    # try:
    #     import mkl_fft as fftpack
    # except (ModuleNotFoundError, ImportError) as e:
    # prettyoutput.Log('mkl_fft not installed on system')
    import numpy.fft as fftpack
    # pass


def PadImageForPhaseCorrelation(image: NDArray[np.floating],
                                MinOverlap: float = .05,
                                ImageMedian: float | None = None,
                                ImageStdDev: float | None = None,
                                OriginalShape: tuple[int, int] | NDArray[int] | None = None,
                                NewWidth: int | None = None,
                                NewHeight: int | None = None,
                                PowerOfTwo: bool = True,
                                AlwaysCopy: bool = True,
                                return_numpy: bool = True) -> NDArray[np.floating]:
    """
    Prepares an image for use with the phase correlation operation.  Padded areas are filled with noise matching the histogram of the
    original image.

    :param ndarray image: Input image
    :param float MinOverlap: Minimum overlap allowed between the input image and images it will be registered to
    :param float ImageMedian: Median value of noise, calculated or pulled from cache if none
    :param float ImageStdDev: Standard deviation of noise, calculated or pulled from cache if none
    :param tuple OriginalShape: The original size of the image, if None the shape of the input image is used.  Set this if the image has been previously been padded to prevent over-padding
    :param int NewWidth: Pad input image to this dimension if not none
    :param int NewHeight: Pad input image to this dimension if not none
    :param bool PowerOfTwo: Pad the image to a power of two if true
    :param bool AlwaysCopy: If true, always copy the image even if no padding is needed
    :return: An image with the input image centered surrounded by noise
    :rtype: ndimage
    """

    MinVal = image.min()
    MaxVal = image.max()

    Height = image.shape[0]
    Width = image.shape[1]

    OriginalHeight = Height
    OriginalWidth = Width

    use_cp = nornir_imageregistration.UsingCupy()
    image = nornir_imageregistration.EnsureArray(image)
    xp = cp.get_array_module(image)

    if OriginalShape is not None:
        OriginalWidth = OriginalShape[1]
        OriginalHeight = OriginalShape[0]

    if NewHeight is None:
        if PowerOfTwo:
            NewHeight = NearestPowerOfTwoWithOverlap(OriginalHeight, MinOverlap)
        else:
            NewHeight = DimensionWithOverlap(OriginalHeight,
                                             MinOverlap)  # # Height + (Height * (1 - MinOverlap))  # + 1

    if NewWidth is None:
        if PowerOfTwo:
            NewWidth = NearestPowerOfTwoWithOverlap(OriginalWidth, MinOverlap)
        else:
            NewWidth = DimensionWithOverlap(OriginalWidth, MinOverlap)  # # Height + (Height * (1 - MinOverlap))  # + 1

    if NewWidth < image.shape[
        1]:  # If we need a smaller size than we already are (from padding an image a 2nd time) then keep current size
        NewWidth = image.shape[1]

    if NewHeight < image.shape[0]:
        NewHeight = image.shape[0]

    if Width >= NewWidth and Height >= NewHeight:
        if AlwaysCopy:
            return xp.copy(image)
        else:
            return image

    if ImageMedian is None or ImageStdDev is None:
        Image1D = image.astype(xp.float64, copy=False)
        Image1D = Image1D.ravel() if use_cp else Image1D.flat

        if ImageMedian is None:
            ImageMedian = xp.median(Image1D)
        if ImageStdDev is None:
            ImageStdDev = xp.std(Image1D)

        del Image1D

    desired_type = image.dtype
    if np.finfo(desired_type).max < MaxVal:
        desired_type = np.float32

    # use_cp = use_cp or NewHeight * NewWidth > 4092
    # xp = cp if use_cp else np

    PaddedImage = xp.zeros((int(NewHeight), int(NewWidth)), dtype=desired_type)
    # if use_cp:
    #     image = xp.asarray(image)

    PaddedImageXOffset = int(np.floor((NewWidth - Width) / 2.0))
    PaddedImageYOffset = int(np.floor((NewHeight - Height) / 2.0))

    # Copy image into padded image
    PaddedImage[PaddedImageYOffset:PaddedImageYOffset + Height, PaddedImageXOffset:PaddedImageXOffset + Width] = image[
                                                                                                                 :, :]

    if not Width == NewWidth:
        LeftBorder = GenRandomData(NewHeight, PaddedImageXOffset, ImageMedian, ImageStdDev, MinVal, MaxVal)
        RightBorder = GenRandomData(NewHeight, NewWidth - (Width + PaddedImageXOffset), ImageMedian, ImageStdDev,
                                    MinVal, MaxVal)

        PaddedImage[:, 0:PaddedImageXOffset] = LeftBorder
        PaddedImage[:, Width + PaddedImageXOffset:] = RightBorder

        del LeftBorder
        del RightBorder

    if not Height == NewHeight:
        TopBorder = GenRandomData(PaddedImageYOffset, Width, ImageMedian, ImageStdDev, MinVal, MaxVal)
        BottomBorder = GenRandomData(NewHeight - (Height + PaddedImageYOffset), Width, ImageMedian, ImageStdDev, MinVal,
                                     MaxVal)

        PaddedImage[0:PaddedImageYOffset, PaddedImageXOffset:PaddedImageXOffset + Width] = TopBorder
        PaddedImage[PaddedImageYOffset + Height:, PaddedImageXOffset:PaddedImageXOffset + Width] = BottomBorder

        del TopBorder
        del BottomBorder

    return PaddedImage


def ImagePhaseCorrelation(target_image: NDArray[np.floating],
                          source_image: NDArray[np.floating],
                          target_mean: float | None = None,
                          source_mean: float | None = None,
                          correlation_coefficient: float | None = None) -> NDArray[np.floating]:
    """
    Returns the phase shift correlation of the FFT's of two images.

    Dimensions of Fixed and Moving images must match

    :param ndarray target_image: grayscale image
    :param ndarray source_image: grayscale image
    :param target_mean: Mean value of the fixed image
    :param source_mean: Mean value of the moving image
    :param CorrelationCoefficient: Setting this value to 1 is equivalent to using phase correlation.  Setting it to 0 is equivalent to using Pierson Correlation.  The default is .65.  If you have a difficult to register section changing this value to 1 may help.
    :returns: Correlation image of the FFT's.  Light pixels indicate the phase is well aligned at that offset.
    :rtype: ndimage
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

    return FFTPhaseCorrelation(target_fft, source_fft, True, correlation_coefficient=correlation_coefficient)


def FFTPhaseCorrelation(fft_target: NDArray[np.floating],
                        fft_source: NDArray[np.floating],
                        delete_input: bool = False,
                        correlation_coefficient: float | None = None) -> NDArray[np.floating]:
    """
    Returns the phase shift correlation of the FFT's of two images.

    Dimensions of Fixed and Moving images must match

    :param delete_input:
    :param ndarray fft_target: grayscale image
    :param ndarray fft_source: grayscale image
    :param CorrelationCoefficient: Setting this value to 1 is equivalent to using phase correlation.  Setting it to 0 is equivalent to using Pierson Correlation.  The default is .65.  If you have a difficult to register section changing this value to 1 may help.
    :returns: Correlation image of the FFT's.  Light pixels indicate the phase is well aligned at that offset.
    :rtype: ndimage
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
    scaled_offset: tuple[float, float]
    peak_strength: float
    cutoff_value: float
    cutoff_percent: float


def FindPeak(image: NDArray[np.floating],
             OverlapMask: NDArray[bool] | None = None,
             Cutoff: float = None) -> FindPeakResult:
    """
    Find the offset of the strongest response in a phase correlation image

    :param ndimage image: grayscale image
    :param float Cutoff: Percentile used to threshold image.  Values below the percentile are ignored
    :param ndimage OverlapMask: Mask describing which pixels are eligible
    :return: scaled_offset of peak from image center and sum of pixels values at peak
    :rtype: (tuple, float)
    """
    xp = cp.get_array_module(image)
    sp = cupyx.scipy.get_array_module(image)

    ThresholdImage = xp.copy(image)
    if OverlapMask is not None:
        ThresholdImage[xp.logical_not(OverlapMask)] = 0

    if Cutoff is None:
        percentiles = np.linspace(0.95, 1, 101) * 100
        try:
            result = nornir_imageregistration.mathfuncs.estimate_cutoff(image[OverlapMask].flat,
                                                                        percentiles,
                                                                        polyfit_degree=2,
                                                                        method=CutoffMethod.Raw) if OverlapMask is not None else \
                nornir_imageregistration.mathfuncs.estimate_cutoff(image.flat, percentiles, polyfit_degree=2,
                                                                   method=CutoffMethod.Raw)
            cutoff_percent = percentiles[result.cutoff_percentile_index] * 100
            CutoffValue = result.cutoff_value
        except ValueError:
            cutoff_percent = 99.6
            CutoffValue = xp.percentile(ThresholdImage[OverlapMask], q=cutoff_percent)
    else:
        cutoff_percent = Cutoff * 100
        CutoffValue = xp.percentile(ThresholdImage[OverlapMask], q=cutoff_percent)

        # Cutoff = 0.996
    #        num_pixels = np.prod(image.shape)

    #        if (1.0 - Cutoff) * num_pixels > 1000:
    #            Cutoff = 1.0 - (1000.0 / num_pixels)

    # CutoffValue = ImageIntensityAtPercent(image, Cutoff)

    # CutoffValue = scipy.stats.scoreatpercentile(image, per=Cutoff * 100.0)
    # ThresholdImage = xp.copy(image)  # np.copy(image)
    # OverlapMask = cp.array(OverlapMask)

    ThresholdImage[ThresholdImage < CutoffValue] = 0

    # ThresholdImage = scipy.stats.threshold(image, threshmin=CutoffValue, threshmax=None, newval=0)
    # nornir_imageregistration.ShowGrayscale([image, OverlapMask, ThresholdImage])

    [LabelImage, NumLabels] = sp.ndimage.label(ThresholdImage)
    # The first interesting label starts at 1, 0 is the background
    LabelSums = sp.ndimage.sum_labels(ThresholdImage, LabelImage, xp.array(range(1, NumLabels + 1)))
    if LabelSums.sum() == 0:  # There are no peaks identified
        scaled_offset = (np.asarray(image.shape, dtype=np.float32) / 2.0)
        PeakStrength = 0
        return scaled_offset, PeakStrength
    else:
        PeakValueIndex = LabelSums.argmax()
        PeakStrength = LabelSums[PeakValueIndex]
        # Because we offset the sum_labels call by 1, we must do the same for the PeakValueIndex
        PeakCenterOfMass = sp.ndimage.center_of_mass(ThresholdImage, LabelImage, int(PeakValueIndex + 1))
        # PeakArea = np.sum(LabelImage == PeakValueIndex + 1)
        # PeakMaximumPosition = scipy.ndimage.maximum_position(ThresholdImage, LabelImage, PeakValueIndex+1)
        # nPixelsInLabel = np.sum(LabelImage == PeakValueIndex+1)
        # if (nPixelsInLabel / np.prod(image.shape)) > 0.001: #Tighten up the cutoff until the peak contains only about 1 in 1000 pixels in the threshold image
        #     new_cutoff = Cutoff + ((1.0 - Cutoff) / 2.0)
        #     scaled_offset, Weight = FindPeak(image, OverlapMask, Cutoff=new_cutoff)
        #     return scaled_offset, Weight

        mean_pixel = xp.mean(image[OverlapMask])
        peak_pixel = sp.ndimage.maximum(ThresholdImage, LabelImage, int(PeakValueIndex + 1))
        signal_to_noise = peak_pixel / mean_pixel

        ########################################################################
        # This was my original implementation to understand signal strength.
        # Art Wetzel convinced me to use signal to noise by dividing the
        # peak pixel intensity by the median pixel intensity

        # if use_cupy:
        #    OtherPeaks = LabelSums[LabelSums != PeakStrength]
        # else:
        #     OtherPeaks = np.delete(LabelSums, PeakValueIndex)

        # FalsePeakStrength = xp.mean(OtherPeaks) if OtherPeaks.shape[0] > 0 else 1
        # FalsePeakStrength = OtherPeaks.max()

        # if FalsePeakStrength == 0:
        #    Weight = PeakStrength
        # else:
        #    Weight = PeakStrength / FalsePeakStrength

        # if PeakArea > 0:
        #    Weight /= PeakArea

        # print(f'{LabelSums.shape} Labels -> {PeakStrength} Peak')

        # center_of_mass returns results as (y,x)
        # scaled_offset = (image.shape[0] / 2.0 - PeakCenterOfMass[0], image.shape[1] / 2.0 - PeakCenterOfMass[1])
        # print(image.shape)
        # PeakCenterOfMass = np.array((cp.asnumpy(PeakCenterOfMass[0]), cp.asnumpy(PeakCenterOfMass[1])))
        if nornir_imageregistration.UsingCupy():
            PeakCenterOfMass = np.array((cp.asnumpy(PeakCenterOfMass[0]), cp.asnumpy(PeakCenterOfMass[1])))
            # print(PeakCenterOfMass)
        scaled_offset = (np.asarray(image.shape) / 2.0) - PeakCenterOfMass
        # scaled_offset = (scaled_offset[0], scaled_offset[1])

        del LabelImage
        del ThresholdImage
        del LabelSums

        return FindPeakResult(scaled_offset, signal_to_noise, CutoffValue, cutoff_percent)


def FindOffset(target_image: NDArray[np.floating],
               source_image: NDArray[np.floating],
               MinOverlap: float = 0.0,
               MaxOverlap: float = 1.0,
               FFT_Required: bool = True,
               target_shape: tuple[int, int] | NDArray[int] | None = None,
               source_shape: tuple[int, int] | NDArray[int] | None = None,
               correlation_coefficient: float | None = None):
    """return an alignment record describing how the images overlap. The alignment record indicates how much the
       moving image must be rotated and translated to align perfectly with the FixedImage.

       If adjusting control points the peak can be added to the fixed image's control point, or subtracted from the
       warped image's control point (accounting for any transform used to create the warped image) to align the images.

       :param ndarray target_image:  Target space we are registering into
       :param ndarray source_image: Source space we are coming from
       :param float MinOverlap: The minimum amount of overlap by area the registration must have
       :param float MaxOverlap: The maximum amount of overlap by area the registration must have
       :param bool FFT_Required: True by default, if False the input images are in FFT space already
       :param tuple target_shape: Defaults to None, if specified it contains the size of the fixed image before padding.  Used to calculate mask for valid overlap values.
       :param tuple source_shape: Defaults to None, if specified it contains the size of the moving image before padding.  Used to calculate mask for valid overlap values.
       """

    # nornir_imageregistration.ShowGrayscale([FixedImage, MovingImage])

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
        target_image = PadImageForPhaseCorrelation(target_image, MinOverlap=1, NewWidth=desired_width,
                                                   NewHeight=desired_height, AlwaysCopy=False)
        source_image = PadImageForPhaseCorrelation(source_image, MinOverlap=1, NewWidth=desired_width,
                                                   NewHeight=desired_height, AlwaysCopy=False)

    CorrelationImage = None
    if FFT_Required:
        CorrelationImage = ImagePhaseCorrelation(target_image,
                                                 source_image,
                                                 correlation_coefficient=correlation_coefficient)
    else:
        CorrelationImage = FFTPhaseCorrelation(target_image,
                                               source_image,
                                               delete_input=False,
                                               correlation_coefficient=correlation_coefficient)

    CorrelationImage = xp.fft.fftshift(CorrelationImage)

    # Crop the areas that cannot overlap
    CorrelationImage -= CorrelationImage.min()
    CorrelationImage /= CorrelationImage.max()

    # Timer.Start('Find Peak')
    OverlapMask = nornir_imageregistration.GetOverlapMask(target_shape,
                                                          source_shape,
                                                          CorrelationImage.shape,
                                                          MinOverlap,
                                                          MaxOverlap)
    (peak, weight, cutoff, cutoff_percent) = FindPeak(CorrelationImage, OverlapMask)

    del CorrelationImage

    record = nornir_imageregistration.AlignmentRecord(peak=peak, weight=weight)

    return record


if __name__ == '__main__':

    import os
    import matplotlib.pyplot as plt

    # arr = LoadImage('L:\\Neitz\\cped\\SEM\\1489\\SEM\\Leveled\\Images\\004\\1489_SEM_Leveled.png',
    #               dtype=np.float16)
    #
    FilenameA = 'C:\\BuildScript\\Test\\Images\\400.png'
    FilenameB = 'C:\\BuildScript\\Test\\Images\\401.png'
    OutputDir = 'C:\\Buildscript\\Test\\Results\\'

    os.makedirs(OutputDir, exist_ok=True)


    def TestPhaseCorrelation(imA, imB):

        # import TaskTimer
        # Timer = TaskTimer.TaskTimer()
        # Timer.Start('Correlate One Pair')

        # Timer.Start('Pad image One Pair')
        FixedA = PadImageForPhaseCorrelation(imA)
        MovingB = PadImageForPhaseCorrelation(imB)

        record = FindOffset(FixedA, MovingB, target_shape=imA.shape, source_shape=imB.shape)
        print(str(record))

        stos = record.ToStos(FilenameA, FilenameB)

        stos.Save(os.path.join(OutputDir, "TestPhaseCorrelation.stos"))

        # Timer.End('Find Peak', False)

        # Timer.End('Correlate One Pair', False)

        # print(str(Timer))

        # ShowGrayscale(NormCorrelationImage)
        return


    def SecondMain():
        imA = plt.imread(FilenameA)
        imB = plt.imread(FilenameB)

        for i in range(1, 5):
            print((str(i)))
            TestPhaseCorrelation(imA, imB)


    import cProfile
    import pstats

    cProfile.run('SecondMain()', 'CoreProfile.pr')
    pr = pstats.Stats('CoreProfile.pr')
    pr.sort_stats('time')
    print(str(pr.print_stats(.5)))
