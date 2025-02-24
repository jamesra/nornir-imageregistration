'''
Created on Nov 2, 2018

@author: u0490822
'''

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.mathfuncs import find_inflection_points
import nornir_shared
import nornir_shared.histogram
import nornir_shared.plot


def PlotWeightHistogram(alignment_records: list[nornir_imageregistration.EnhancedAlignmentRecord],
                        filename: str, transform_cutoff: float, finalize_cutoff: float,
                        line_pos_list: list[float] | None, title: str | None = None):
    '''
    Plots the weights on the alignment records as a histogram
    :param list alignment_records: A list of EnhancedAlignmentRecords to render, uses a square to represent alignments in progress
    :param str filename: Filename to save plot as
    :param float cutoff: Cutoff value, a vertical line is drawn on the plot at this value
    '''
    title = "Histogram of Weights" if title is None else title

    weights = np.asarray(list(map(lambda a: a.weight, alignment_records)))
    h = nornir_shared.histogram.Histogram.Init(np.min(weights), np.max(weights),
                                               numBins=min(50, len(alignment_records)))
    h.Add(weights)
    nornir_shared.plot.Histogram(h, Title="Histogram of Weights", xlabel="Weight Value", ImageFilename=filename,
                                 MinCutoffPercent=transform_cutoff, MaxCutoffPercent=finalize_cutoff,
                                 LinePosList=line_pos_list)


def PlotPeakList(new_alignment_records: list[nornir_imageregistration.EnhancedAlignmentRecord],
                 finalized_alignment_records: list[nornir_imageregistration.EnhancedAlignmentRecord],
                 filename: str, ylim: tuple[float, float] | None = None, xlim: tuple[float, float] | None = None,
                 attrib=None):
    '''
    Converts a set of EnhancedAlignmentRecord peaks from the RefineTwoImages function into a transform
    :param attrib:
    :param new_alignment_records: A list of EnhancedAlignmentRecords to render, uses a square to represent alignments in progress
    :param finalized_alignment_records: A list of EnhancedAlignmentRecords to render, uses a circle to represent alignments are fixed
    :param filename: Filename to save plot as
    :param ylim: y-limit of plot
    :param xlim: x-limit of plot
    '''

    if attrib is None:
        attrib = 'weight'

    all_records = new_alignment_records + finalized_alignment_records
    shapes = ['s' for a in new_alignment_records]
    shapes.extend(['.' for a in finalized_alignment_records])
    colors = ['b' for a in new_alignment_records]
    colors.extend(['g' for a in finalized_alignment_records])

    TargetPoints = np.asarray(list(map(lambda a: a.TargetPoint, all_records)))
    # OriginalWarpedPoints = np.asarray(list(map(lambda a: a.SourcePoint, all_records)))
    AdjustedTargetPoints = np.asarray(list(map(lambda a: a.AdjustedTargetPoint, all_records)))
    # AdjustedWarpedPoints = np.asarray(list(map(lambda a: a.AdjustedWarpedPoint, all_records)))
    weights = np.asarray(list(map(lambda a: getattr(a, attrib), all_records)))
    percentile_prep = weights - weights.min()
    percentiles = (percentile_prep / np.max(percentile_prep)) * 100
    TargetPeaks = AdjustedTargetPoints - TargetPoints

    nornir_shared.plot.VectorField(TargetPoints.astype(np.float32, copy=False), TargetPeaks, shapes, percentiles,
                                   OutputFilename=filename, ylim=ylim, xlim=xlim, colors=colors)

    return


def _gray_to_rgba(img: NDArray, alpha: float = 1.0):
    rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=img.dtype)
    rgba[:, :, 0] = img
    rgba[:, :, 1] = img
    rgba[:, :, 2] = img
    rgba[:, :, 3] = alpha
    return rgba


def plot_aligned_images(alignment_record, image_A: NDArray, image_B: NDArray):
    '''
    Plot the two images in the same axis using the provided alignment record    
    '''

    image_A = nornir_imageregistration.ImageParamToImageArray(image_A)
    image_B = nornir_imageregistration.ImageParamToImageArray(image_B)

    plt.clf()

    a = _gray_to_rgba(image_A, alpha=128)
    b = _gray_to_rgba(image_B, alpha=128)

    a[:, :, 3] = a[:, :, 0]  # Scale alpha by luminosity
    b[:, :, 3] = b[:, :, 0]  # Scale alpha by luminosity
    a[:, :, 1] = 0

    b[:, :, 0] = 0
    b[:, :, 2] = 0

    a_extent = (0, image_A.shape[1], 0, image_A.shape[0])
    b_extent = (alignment_record.peak[1], alignment_record.peak[1] + image_B.shape[1], alignment_record.peak[0],
                alignment_record.peak[0] + image_B.shape[0])
    plt.imshow(a, origin='lower', extent=a_extent)
    plt.imshow(b, origin='lower', extent=b_extent)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return


def plot_percentiles(records: NDArray[np.floating],
                     filename: str | None = None,
                     title: str | None = None,
                     horz_line_pos_list: list[tuple[float, dict]] | None = None):
    '''
    Plot the percentiles of the records
    '''
    if title is None:
        title = "Percentiles"

    a = records
    p = np.linspace(0, 100, 101)
    plt.clf()
    ax = plt.gca()
    lines = [
        ('linear', '-', 'C0'),
        # ('inverted_cdf', ':', 'C1'),
        # # Almost the same as `inverted_cdf`:
        # ('averaged_inverted_cdf', '-.', 'C1'),
        # ('closest_observation', ':', 'C2'),
        # ('interpolated_inverted_cdf', '--', 'C1'),
        # ('hazen', '--', 'C3'),
        # ('weibull', '-.', 'C4'),
        # ('median_unbiased', '--', 'C5'),
        # ('normal_unbiased', '-.', 'C6'),
    ]

    percentile_values = np.percentile(a, p, method='linear')

    for method, style, color in lines:
        ax.plot(
            p, np.percentile(a, p, method=method),
            label=method, linestyle=style, color=color)
    ax.set(
        title='Percentiles for different methods',
        xlabel='Percentile',
        ylabel='Estimated percentile value')

    # Add a polyfit to the linear line
    degree = 5
    coefficients = np.polyfit(p, percentile_values, degree)
    # Generate the polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(p)
    ax.plot(p, y_fit, 'k--', label='Polyfit')

    min_a = min(a)
    max_a = max(a)
    inflection_indicies, inflections = find_inflection_points(x=p, y=y_fit)
    label = 'Inflection Point'
    for inflection in inflections:
        ax.plot([inflection, inflection], [min_a, max_a], 'b--', label=label)
        label = None

    # max_rate_of_change = np.gradient(y_fit, p)
    # max_accel_of_rate = np.gradient(max_rate_of_change, p)
    # max_rate_of_change_index = np.argmax(abs(max_accel_of_rate))
    # max_rate_of_change_percentile = p[max_rate_of_change_index]
    #
    # ax.plot([max_rate_of_change_percentile, max_rate_of_change_percentile], [min_a, max_a], 'r--', label='Max rate of change')

    label = 'Cutoff Threshold'

    if horz_line_pos_list is not None:
        c = 1
        for line in horz_line_pos_list:
            if isinstance(line, float):
                min_a = min(min_a, line)
                color = 'C' + str(c)
                ax.plot([0, 100], [line, line], linestyle='--', color=color, label=label)
            elif isinstance(line, tuple):
                min_a = min(min_a, line[0])
                # The first entry in the tuple is the line position, the second are the keyword args for the line
                color = 'C' + str(c)
                ax.plot([0, 100], [line[0], line[0]], **line[1])

    ax.legend(bbox_to_anchor=(1.03, 1))
    ax.set_ylim(min(a), max(a))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        nornir_imageregistration.ShowWithPassFail(plt.gcf())

    return
