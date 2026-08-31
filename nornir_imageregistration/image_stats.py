"""
Created on Jun 26, 2012

@author: James Anderson
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Sequence

import numpy
import numpy as np

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy as sp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
    import scipy as sp

try:
    import cupy.fft as fftpack
except (ModuleNotFoundError, ImportError):
    # try:
    #        import mkl_fft as fftpack
    #    except (ModuleNotFoundError, ImportError):
    import scipy.fft as fftpack

try:
    import cupy.random as random
except (ModuleNotFoundError, ImportError):
    # try:
    #     import mkl_random.mklrand as random
    # except (ModuleNotFoundError, ImportError):
    import numpy.random as random

from PIL import Image
from numpy.typing import NDArray, DTypeLike
from pylab import ceil, mod

import nornir_shared.histogram
import nornir_shared.prettyoutput as prettyoutput
import nornir_pools
import nornir_imageregistration


class ImageStats:
    """A container for image statistics"""
    _median: float | None = None
    _mean: float | None = None
    _std: float | None = None
    _min: float | None = None
    _max: float | None = None

    @property
    def median(self) -> float:
        return self._median  # type: ignore[return-value]

    @median.setter
    def median(self, val: float):
        self._median = val

    @property
    def mean(self) -> float:
        return self._mean  # type: ignore[return-value]

    @mean.setter
    def mean(self, val: float):
        self._mean = val

    @property
    def std(self) -> float:
        return self._std  # type: ignore[return-value]

    @std.setter
    def std(self, val: float):
        self._std = val

    @property
    def min(self) -> float:
        return self._min  # type: ignore[return-value]

    @min.setter
    def min(self, val: float):
        self._min = val

    @property
    def max(self) -> float:
        return self._max  # type: ignore[return-value]

    @max.setter
    def max(self, val: float):
        self._max = val

    def __init__(self):
        self._median = None
        self._mean = None
        self._std = None
        self._min = None
        self._max = None

    def __str__(self):
        return f'mean: {self._mean} std: {self._std} min: {self._min} max: {self._max}'

    def __getstate__(self):
        d = {'_median': self._median, '_mean': self._mean, '_std': self._std, '_min': self._min, '_max': self._max}
        return d

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    @classmethod
    def CalcStats(cls, image: nornir_imageregistration.ImageLike) -> ImageStats:  # type: ignore[invalid-type]
        return ImageStats.Create(image)

    @classmethod
    def Create(cls, image: NDArray) -> ImageStats:
        """Returns an object with the mean,median,std.dev of an image,
           this object is attached to the image object and only calculated once"""

        #        I removed this cache in the image object of the statistics.  I believe
        #        Python 3 had issues with it.  If there are performance problems we
        #        should add it back
        #         try:
        #             cachedVal = image.__IrToolsImageStats__
        #             if cachedVal is not None:
        #                 return cachedVal
        #         except AttributeError:
        #             pass

        xp = cp.get_array_module(image)

        obj = ImageStats()
        image = nornir_imageregistration.ImageParamToImageArray(image, dtype=numpy.float64)

        if image.size == 0 or image.shape[0] == 0:
            raise ValueError("Image has no data")

        # if image.dtype is not numpy.float64:  # Use float 64 to ensure accurate statistical results
        #    image = image.astype(dtype=numpy.float64)

        # This test for a masked array smells bad but began to be required
        # after upgrading to numpy 1.23.5
        if isinstance(image, np.ma.MaskedArray):
            obj._median = numpy.ma.median(image)
            obj._mean = numpy.ma.mean(image)
            obj._std = numpy.ma.std(image)
            obj._max = numpy.ma.max(image)
            obj._min = numpy.ma.min(image)
        else:
            # flatImage = image.ravel() if use_cp else image.flat
            flatImage = xp.ravel(image)
            obj._median = float(xp.median(flatImage))
            obj._mean = float(xp.mean(flatImage))
            obj._std = float(xp.std(flatImage))
            obj._max = float(xp.max(flatImage))
            obj._min = float(xp.min(flatImage))
            del flatImage

        #        image.__IrtoolsImageStats__ = obj
        return obj

    def GenerateNoise(self, shape: int | np.integer | np.ndarray | Any, dtype: DTypeLike, *, xp: Any | None = None):
        """
        Generate random data of shape with the specified mean and standard deviation.  Returned values will not be less than min or greater than max
        :param array shape: Shape of the returned array
        :param xp: Array module for output (``numpy`` or ``cupy``).  If ``None``, uses ``GetComputationModule()``.
        """

        size = None
        height = 1
        width = 1
        one_d_result = False
        if isinstance(shape, int) or isinstance(shape, np.integer):
            size = shape
            height = shape
            width = 1
            one_d_result = True
        elif isinstance(shape, np.ndarray) or isinstance(shape, cp.ndarray):
            shape_shape = shape.shape
            one_d_result = len(shape_shape) == 0
            height = int(shape) if one_d_result else shape[0]
            width = shape[1] if not one_d_result else 1
            size = int(shape) if one_d_result else shape
        else:
            one_d_result = len(shape) == 1
            height = shape[0] if not one_d_result else int(shape)
            width = shape[1] if not one_d_result else 1
            size = int(shape) if one_d_result else shape.shape

        if xp is None:
            xp = nornir_imageregistration.GetComputationModule()
        with nornir_imageregistration.IgnoreUnderAndOverflow():  # type: ignore[attr-defined]
            # Shares the generator behind GenRandomData, so seeding one seeds both. This
            # is the second noise source in a brute alignment -- padding fills the frame,
            # this fills the corners a rotation leaves empty -- and seeding only the
            # other one leaves the alignment as irreproducible as before.
            rng = nornir_imageregistration.random_generator(xp)
            data = ((rng.standard_normal(size) * self.std) + self.median).astype(dtype, copy=False)

        xp.clip(data, self.min, self.max, out=data)  # Ensure random data doesn't change range of the image

        return data


def Prune(filenames: str | Sequence[str], MaxOverlap: float | None = None):
    if isinstance(filenames, str):
        listfilenames = [filenames]
    else:
        listfilenames = filenames

    # logger = logging.getLogger('irtools.prune')

    if MaxOverlap is None:
        MaxOverlap = 0

    assert isinstance(listfilenames, list)

    FilenameToResult = __InvokeFunctionOnImageList__(listfilenames, Function=__PruneFileSciPy__, MaxOverlap=MaxOverlap)  # type: ignore[arg-type]

    # Convert results to a float
    for k in FilenameToResult.keys():
        FilenameToResult[k] = float(FilenameToResult[k])

    if isinstance(filenames, str):
        return list(FilenameToResult.items())[0]
    else:
        return FilenameToResult


def __InvokeFunctionOnImageList__(listfilenames: Sequence[str],
                                  Function: Callable[[str], None] | None = None,
                                  Pool: nornir_pools.IPool | None = None,
                                  **kwargs):
    """Return a number indicating how interesting the image is using SciPy
       """

    if Pool is None:
        TPool = nornir_pools.GetGlobalMultithreadingPool()
    else:
        TPool = Pool

    TileToScore = dict()
    tasklist = []
    for filename in listfilenames:
        task = TPool.add_task('Calc Feature Score: ' + os.path.basename(filename), Function, filename, **kwargs)  # type: ignore[arg-type]
        task.filename = filename  # type: ignore[attr-defined]
        tasklist.append(task)

    TPool.wait_completion()

    numTasks = len(tasklist)
    iTask = 0
    for task in tasklist:
        Result = task.wait_return()
        iTask += 1
        if Result is None:
            prettyoutput.LogErr('No return value for ' + task.filename)
            continue

        #         if Result[0] is None:
        #             PrettyOutput.LogErr('No filename for ' + task.name)
        #             continue

        prettyoutput.CurseProgress("ImageStats", iTask, numTasks)

        filename = task.filename
        TileToScore[filename] = Result

    return TileToScore


def ScoreImageWithPowerSpectralDensity(image: nornir_imageregistration.ImageLike) -> float:  # type: ignore[invalid-type]
    # Find all NaN values and replace with median value
    # finite_mask = numpy.isfinite(image)
    # infinite_mask = numpy.logical_not(finite_mask)
    # adjustment_value = numpy.mean(image[finite_mask].flat)
    # image[infinite_mask] = adjustment_value

    # Adjust image to have median value of zero, makes the PSD numbers more human-readable and possibly avoids floating point precision issues
    # Im_centered = image - adjustment_value

    Im_centered = image

    # Dispatch on the array for the same reason as the gaussian_filter call below: the
    # module-level `fftpack` is bound to cupy.fft at import whenever CuPy is installed, so a
    # host array reaching it raised "The input array a must be a cupy.ndarray" and took the
    # whole feature-score path down with it. (#249)
    fft = cp.get_array_module(Im_centered).fft.fft2(Im_centered)
    rfft = np.real(fft)  # type: ignore[call-overload]
    # fft = numpy.fft.fftshift(fft) 
    total_amp = numpy.sum(numpy.abs(rfft))
    score = total_amp / numpy.prod(Im_centered.shape)
    return score


def __CalculateFeatureScoreSciPy__(image: nornir_imageregistration.ImageLike,  # type: ignore[invalid-type]
                                   cell_size: tuple[int, int] | None = None,
                                   feature_coverage_percent: float | None = None, **kwargs) -> float:
    """
    Calculates a score indicating the amount of texture available for our phase correlation algorithm to use for alignment
    :param image: The image to score, either an ndarray or filename
    :param tuple cell_size: The dimensions of the subregions that will be evaluated across the image.
    :param float feature_coverage_percent: A value from 0 - 100 indicating what percentage of the image should contain textures scoring at or above the returned value.
    """

    if feature_coverage_percent is None:
        feature_coverage_percent = 75
    else:
        feature_coverage_percent = 100 - feature_coverage_percent
        assert (100 >= feature_coverage_percent >= 0)

    Im = nornir_imageregistration.ImageParamToImageArray(image, dtype=nornir_imageregistration.default_image_dtype())
    # #     Im_filtered = scipy.ndimage.filters.median_filter(Im, size=3)
    # #     sx = scipy.ndimage.sobel(Im_filtered, axis=0, mode='nearest')
    # #     sy = scipy.ndimage.sobel(Im_filtered, axis=1, mode='nearest')
    # #     sob = numpy.hypot(sx,sy)
    # #

    #
    # #
    #     logamp = numpy.log(amp) ** 2
    #     logampflat = numpy.asarray(logamp.flat)
    #     aboveMedian = numpy.median(logampflat)
    #     score = numpy.mean(logampflat[logampflat > aboveMedian])

    # score = numpy.max(Im_filtered.flat) - numpy.min(Im_filtered.flat)

    # score = numpy.var(Im_filtered.flat)
    # score = numpy.percentile(sob.flat, 90)
    # score = numpy.max(sob.flat)
    # #    score = numpy.mean(sob.flat)
    # score = numpy.median(sob.flat) - numpy.percentile(sob.flat, 10)
    # mode = numpy.stats.mode(sob.flat)

    # p10 = numpy.percentile(Im_filtered.flat, 10)
    # p90 = numpy.percentile(Im_filtered.flat, 90)
    # med = numpy.median(sob.flat)

    # score = (p90 - p10)

    # score = numpy.median(sob.flat) - numpy.percentile(sob.flat, 10))

    #     if score < .025:
    #         nornir_imageregistration.ShowGrayscale([Im, Im_filtered, sob], title=str(score))
    #         plt.figure()
    #         plt.hist(sob.flat, bins=100)
    #         a = 4
    # #     return score

    #     finite_subset = numpy.asarray(Im[numpy.isfinite(Im)].flat, dtype=numpy.float32)
    #     if len(finite_subset) < 3:
    #         return 0
    #
    #     return numpy.std(finite_subset)

    # Apply a basic gaussian smoothing to remove high frequency noise.
    #
    # Dispatch on the array instead of the module-level `sp`. That name is bound once at import
    # -- cupyx.scipy when CuPy is installed, scipy otherwise -- so it ignores where this
    # particular image actually lives. Two separate failures came out of that on a CuPy machine:
    # `cupyx.scipy.ndimage` has no `filters` submodule (AttributeError), and once that is
    # dropped, handing it the host array that ImageParamToImageArray returns raises TypeError.
    # `.filters` is deprecated on scipy as well, slated for removal in SciPy 2.0. Filtered
    # output agrees between the two backends to 1.2e-07, so feature_score_threshold keeps the
    # same meaning either way. (#249)
    sp_image = cupyx.scipy.get_array_module(Im)
    Im = sp_image.ndimage.gaussian_filter(Im.astype(np.float32), sigma=2.5, radius=5)

    if cell_size is None:
        # cell_size = numpy.max(numpy.vstack((numpy.asarray(numpy.asarray(Im.shape) / 64, dtype=numpy.int32), numpy.asarray((64,64),dtype=numpy.int32))),0) 
        cell_size = numpy.asarray((64, 64), dtype=numpy.int32)  # type: ignore[assignment]

    grid = nornir_imageregistration.CenteredGridDivision(Im.shape, cell_size=cell_size)  # type: ignore[arg-type]

    cell_area = numpy.prod(cell_size)  # type: ignore[arg-type]

    score_list = []

    for iPoint in range(0, grid.num_points):
        rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(grid.SourcePoints[iPoint, :],
                                                                               grid.cell_size)  # type: ignore[arg-type]
        subset = nornir_imageregistration.CropImageRect(Im, rect, cval=numpy.nan)
        finite_mask = numpy.isfinite(subset)  # type: ignore[arg-type]
        finite_subset = subset[finite_mask]  # type: ignore[index]
        if len(finite_subset) < (cell_area / 2.0):
            continue

        not_finite_subset = np.logical_not(finite_mask)
        subset[not_finite_subset] = subset[finite_mask].mean()  # type: ignore[index]

        std_val = ScoreImageWithPowerSpectralDensity(subset)

        # std_val = numpy.var(numpy.asarray(finite_subset, dtype=numpy.float32))
        # std_val = numpy.percentile(finite_subset, q=90) - numpy.percentile(finite_subset, q=10) 
        score_list.append(std_val)

        del subset

    del Im

    if len(score_list) == 0:
        return 0
    elif len(score_list) == 1:
        return float(score_list[0])
    else:
        # score_list holds one scalar per grid cell, so the reduction stays on whichever backend
        # produced the scores and only the returned scalar crosses to the host. numpy.percentile
        # cannot be handed a list of CuPy scalars -- it calls asanyarray, which refuses the
        # implicit transfer -- and this was the last step keeping a device image from scoring.
        # (#249)
        xp = cp.get_array_module(score_list[0])
        val = xp.percentile(xp.asarray(score_list), q=feature_coverage_percent)

        # val = numpy.max(score_list)
        # val = numpy.mean(score_list) #Median was less reliable when using the range of intensity values as a measure
        return float(val)


def __PruneFileSciPy__(filename: str, MaxOverlap: float = 0.15, **kwargs):
    """Returns a prune score for a single file
        Args:
           MaxOverlap = 0 to 1"""

    # TODO: This function should be updated to use the grid_subdivision module to create cells.  It should be used in the mosaic tile translation code to eliminate featureless 
    # overlap regions of adjacent tiles

    # logger = logging.getLogger('irtools.prune')
    # logger = multiprocessing.log_to_stderr()

    if MaxOverlap > 0.5:
        MaxOverlap = 0.5
    #
    #     if not os.path.exists(filename):
    #         # logger.error(filename + ' not found when attempting prune')
    #         # PrettyOutput.LogErr(filename + ' not found when attempting prune')
    #         return None

    Im = nornir_imageregistration.ImageParamToImageArray(filename)
    (Height, Width) = Im.shape

    StdDevList = []
    # MeanList = []

    MaxDim = Height
    if Width > Height:
        MaxDim = Width

    SampleSize = int(ceil(MaxDim / 32))

    VertOverlapPixelRange = int(MaxOverlap * float(Height))
    HorzOverlapPixelRange = int(MaxOverlap * float(Width))

    MaskTopBorder = VertOverlapPixelRange + (SampleSize - (mod(VertOverlapPixelRange, 32)))
    MaskBottomBorder = Height - VertOverlapPixelRange - (SampleSize - (mod(VertOverlapPixelRange, 32)))

    MaskLeftBorder = HorzOverlapPixelRange + (SampleSize - (mod(HorzOverlapPixelRange, 32)))
    MaskRightBorder = Width - HorzOverlapPixelRange - (SampleSize - (mod(HorzOverlapPixelRange, 32)))

    # Calculate the top
    for iHeight in range(0, MaskTopBorder - (SampleSize - 1), SampleSize):
        for iWidth in range(0, Width - 1, SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0

    # Calculate the sides
    for iHeight in range(MaskTopBorder, MaskBottomBorder, SampleSize):
        for iWidth in range(0, MaskLeftBorder - (SampleSize - 1), SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0.25

        for iWidth in range(MaskRightBorder, Width - SampleSize, SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0.5

    # Calculate the bottom
    for iHeight in range(MaskBottomBorder, Height - SampleSize, SampleSize):
        for iWidth in range(0, Width - 1, SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0.75

    del Im
    # nornir_imageregistration.ShowGrayscale(Im)
    return sum(StdDevList)


def Histogram(filenames: str | Sequence[str], Bpp: int | None = None, Scale: float | None = None,
              progress_task_key: str | None = None, progress_name: str | None = None,
              **kwargs) -> nornir_shared.histogram.Histogram:
    """Return a combined histogram of all images.

    If *Scale* is not None the images are scaled before the histogram is
    collected. Optional *progress_task_key* / *progress_name* publish a nested
    dashboard bar while tiles are processed.
    """

    if isinstance(filenames, str):
        listfilenames = [filenames]
    elif isinstance(filenames, Sequence):
        listfilenames = filenames
    else:
        raise ValueError("filenames has unexpected type ")

    numTiles = len(listfilenames)
    if numTiles == 0:
        raise ValueError("Cannot histogram with no input files")

    if Bpp is None:
        Bpp = nornir_shared.images.GetImageBpp(listfilenames[0])

    assert isinstance(listfilenames, list)

    reporter: prettyoutput.TaskProgressReporter | None = None
    if progress_task_key:
        reporter = prettyoutput.TaskProgressReporter(
            progress_task_key, numTiles, name=progress_name)
        reporter.start()

    FilenameToTask = {}
    # Pillow decode releases the GIL — prefer threads over process spawn/join.
    if len(listfilenames) > 2:
        pool = nornir_pools.GetGlobalThreadPool()
    else:
        pool = nornir_pools.GetGlobalSerialPool()

    for f in listfilenames:
        try:
            task = pool.add_task(f, __HistogramFileSciPy__, f, Bpp=Bpp, Scale=Scale, **kwargs)
            FilenameToTask[f] = task
        except Exception as e:
            # SerialPool runs inline and may raise at submit time.
            prettyoutput.Log(f"Skipping histogram for {f}: {type(e).__name__}: {e}")
            continue

    minVal = None
    maxVal = None
    histlist = []
    numBins = None
    completed = 0
    for f in list(FilenameToTask.keys()):
        task = FilenameToTask[f]
        try:
            h = task.wait_return()
        except (OSError, IOError, ValueError) as e:
            prettyoutput.Log(f"Skipping histogram for {f}: {e}")
            completed += 1
            if reporter is not None:
                reporter.update(completed)
            continue
        except Exception as e:
            # Pillow and codec errors vary by version; do not abort the mosaic hist.
            prettyoutput.Log(f"Skipping histogram for {f}: {type(e).__name__}: {e}")
            completed += 1
            if reporter is not None:
                reporter.update(completed)
            continue

        histlist.append(h)
        if minVal is None:
            minVal = h.MinValue
        else:
            minVal = min(minVal, h.MinValue)

        if maxVal is None:
            maxVal = h.MaxValue
        else:
            maxVal = max(maxVal, h.MaxValue)

        numBins = len(h.Bins)
        completed += 1
        if reporter is not None:
            reporter.update(completed)

    if reporter is not None:
        reporter.complete()

    if len(histlist) == 0:
        raise ValueError(
            f"Cannot build histogram: no readable tiles among {numTiles} input file(s)")

    HistogramComposite = nornir_shared.histogram.Histogram.Init(minVal=minVal, maxVal=maxVal, numBins=numBins)
    for h in histlist:
        HistogramComposite.AddHistogram(h)

    if Bpp > 8:  # type: ignore[operator]
        HistogramComposite = nornir_shared.histogram.Histogram.Trim(HistogramComposite)

    return HistogramComposite


def __Get_Histogram_For_Image_From_ImageMagick(filename: str, Bpp: int | None = None, Scale: float | None = None):
    Cmd = __CreateImageMagickCommandLineForHistogram(filename, Scale)  # type: ignore[arg-type]
    raw_output = __HistogramFileImageMagick__(filename, ProcPool, Bpp, Scale)  # type: ignore[name-defined]


def __HistogramFileSciPy__(filename: str,
                           Bpp: int | None = None,
                           NumSamples: int | None = None,
                           numBins: int | None = None,
                           Scale: float | None = None,
                           MinVal: float | None = None,
                           MaxVal: float | None = None,
                           stride: int | None = None) -> nornir_shared.histogram.Histogram:
    """Return the histogram of an image"""

    with Image.open(filename, mode='r') as img:
        img_I = img.convert("I")
        Im = np.asarray(img_I)

    return HistogramOfArray(Im, bpp=Bpp, num_samples=NumSamples, num_bins=numBins, scale=Scale,
                            min_val=MinVal, max_val=MaxVal, stride=stride)


def even_histogram_stride(sample_fraction: float = 0.02) -> int:
    """Return a 2D stride that retains roughly *sample_fraction* of pixels (1–5%)."""
    fraction = float(np.clip(sample_fraction, 0.01, 0.05))
    return max(1, int(round(1.0 / (fraction ** 0.5))))


def ApproximateHistogramOfArray(
        input: NDArray,
        *,
        sample_fraction: float = 0.02,
        bpp: int | None = 8,
        num_bins: int = 256,
        min_val: float = 0.0,
        max_val: float = 255.0,
) -> nornir_shared.histogram.Histogram | None:
    """Build a histogram from an even spatial subsample (~1–5% of pixels).

    CuPy inputs are transferred once at a host boundary for
    :func:`HistogramOfArray`. Returns None when *input* is empty.
    """
    if input is None:
        return None

    xp = cp.get_array_module(input)
    arr = input
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"ApproximateHistogramOfArray expects a 2D image, got shape {arr.shape}")
    if int(arr.size) == 0:
        return None

    # Host boundary: histogram construction and UI consume NumPy arrays.
    if xp is not np:
        arr = cp.asnumpy(arr)

    stride = even_histogram_stride(sample_fraction)
    return HistogramOfArray(
        arr,
        bpp=bpp,
        num_bins=num_bins,
        min_val=min_val,
        max_val=max_val,
        stride=stride,
    )


def HistogramOfArray(input: NDArray,
                     bpp: int | None = None,
                     num_samples: int | None = None,
                     num_bins: int | None = None,
                     scale: float | None = None,
                     min_val: float | None = None,
                     max_val: float | None = None,
                     stride: int | None = None) -> nornir_shared.histogram.Histogram:
    """Generate a histogram of the passed image
    :param bpp: The number of bits per pixel in the image
    :param num_samples: Approximate pixel count via even 1D subsample of the
        (optionally spatially strided) array. If None, all retained pixels are used.
    :param num_bins: The number of bins to use in the histogram
    :param scale: The percentage to scale the image before generating the histogram (Not currerntly supported, by the time the histogram is in memory it is faster to read it all, it has to be read to downsample it anyway.)
    :param min_val: The minimum value to use in the histogram
    :param max_val: The maximum value to use in the histogram
    :param stride: If > 1, histogram ``input[::stride, ::stride]`` (deterministic spatial subsample).

    """
    xp = cp.get_array_module(input)

    if stride is not None and stride > 1:
        input = input[::stride, ::stride]

    if input.ndim != 2:
        raise ValueError(f"HistogramOfArray expects a 2D image, got shape {input.shape}")

    (Height, Width) = input.shape
    num_pixels = Width * Height
    min_val = 0 if min_val is None else min_val

    if max_val is None:
        if bpp is None:
            bpp = nornir_imageregistration.ImageBpp(input)

        assert (isinstance(bpp, int))
        max_val = (1 << bpp) - 1

    if num_bins is None:
        num_bins = (max_val - min_val) + 1  # type: ignore[assignment]
    else:
        assert (isinstance(num_bins, int))
        if num_bins > (max_val - min_val) + 1:
            num_bins = (max_val - min_val) + 1  # type: ignore[assignment]

    # ravel works for NumPy and CuPy; flatiter + device indices does not.
    samples = input.ravel()

    if num_samples is None:
        num_samples = num_pixels
    elif num_samples > num_pixels:
        num_samples = num_pixels

    step_size = int(float(num_pixels) / float(num_samples))  # type: ignore[arg-type]
    if step_size > 1:
        # Even subsample (deterministic, backend-safe). Avoids cupy.random indices
        # into NumPy buffers that previously emptied Pyre contrast histograms.
        samples = samples[::step_size]

    # Host boundary: nornir_shared.Histogram is NumPy/list based.
    if xp is not np:
        samples = cp.asnumpy(samples)

    # In numpy's histogram, the max value must be at the end of the last bin, so for a 256 grayscale image MinVal=0 MaxVal=256
    [histogram_array, bin_edges] = numpy.histogram(samples, bins=num_bins, range=(min_val, max_val + 1))  # type: ignore[arg-type]
    binWidth = bin_edges[1] - bin_edges[0]
    assert (binWidth > 0)
    histogram_obj = nornir_shared.histogram.Histogram.FromArray(histogram_array, bin_edges[0], binWidth)  # type: ignore[arg-type]

    return histogram_obj


def __CreateImageMagickCommandLineForHistogram(filename: str, Scale: float):
    CmdTemplate = "magick convert %(filename)s -filter point -scale %(scale)g%% -define histogram:unique-colors=true -format %%c histogram:info:- && exit"
    return CmdTemplate % {'filename': filename, 'scale': Scale * 100}


def __HistogramFilePillow__(filename: str, Bpp: int | None = None, Scale: float | None = None):
    if Scale is None:
        Scale = 1

    # We only scale down, so if it is over 1 assume it is a percentage
    if Scale > 1:
        Scale /= 100.0

    if Scale > 1:
        Scale = 1

    with Image.open(filename) as im:
        histogram_array = im.convert('I').histogram()
    binWidth = (1 << Bpp) // len(histogram_array)  # type: ignore[operator]

    histogram_obj = nornir_shared.histogram.Histogram.FromArray(histogram_array, 0, binWidth)

    return histogram_obj


def __HistogramFileImageMagick__(filename: str,
                                 ProcPool: nornir_pools.IPool | None = None,
                                 Bpp: int | None = None,
                                 Scale: float | None = None):
    if Scale is None:
        Scale = 1

    # We only scale down, so if it is over 1 assume it is a percentage
    if Scale > 1:
        Scale /= 100.0

    if Scale > 1:
        Scale = 1

    Cmd = __CreateImageMagickCommandLineForHistogram(filename, Scale)
    task = ProcPool.add_process(os.path.basename(filename), Cmd, shell=True)  # type: ignore[union-attr]

    return task

# if __name__ == '__main__':
# 
#     Histogram = Histogram('C:\\Buildscript\\IrTools\\RawTile.png')
# 
#     import cProfile
#     import pstats
# 
#     score = Prune('C:\\Buildscript\\IrTools\\RawTile.png', 0.1)
#     PrettyOutput.Log("Score: " + str(score))
# 
#     ProfilePath = 'C:\\Buildscript\\IrTools\\BuildProfile.pr'
# 
#     ProfileDir = os.path.dirname(ProfilePath)
#     if not os.path.exists(ProfileDir):
# 
#         os.makedirs(ProfileDir)
# 
#     try:
#         cProfile.run("__PruneFileSciPy__('C:\\Buildscript\\IrTools\\RawTile.png', 0.1)", ProfilePath)
#     finally:
#         if not os.path.exists(ProfilePath):
#             PrettyOutput.LogErr("No profile file found" + ProfilePath)
#             sys.exit()
# 
#         pr = pstats.Stats(ProfilePath)
#         if not pr is None:
#             pr.sort_stats('time')
#             print(str(pr.print_stats(.05)))
