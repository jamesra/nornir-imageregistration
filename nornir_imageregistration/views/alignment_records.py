'''
Created on Nov 2, 2018

@author: u0490822
'''

import numpy as np
import nornir_shared
import nornir_shared.plot
import nornir_shared.histogram



def PlotWeightHistogram(alignment_records, filename, cutoff):
    '''
    Plots the weights on the alignment records as a histogram
    :param list alignment_records: A list of EnhancedAlignmentRecords to render, uses a square to represent alignments in progress
    :param bool filename: Filename to save plot as
    :param float cutoff: Cutoff value, a vertical line is drawn on the plot at this value
    '''
    
    weights = np.asarray(list(map(lambda a: a.weight, alignment_records)))
    h = nornir_shared.histogram.Histogram.Init(np.min(weights), np.max(weights))
    h.Add(weights)
    nornir_shared.plot.Histogram(h, Title="Histogram of Weights", xlabel="Weight Value", ImageFilename=filename, MinCutoffPercent=cutoff, MaxCutoffPercent=1.0)

def PlotPeakList(new_alignment_records, finalized_alignment_records, filename, ylim=None, xlim=None, attrib=None):
    '''
    Converts a set of EnhancedAlignmentRecord peaks from the RefineTwoImages function into a transform
    :param list new_alignment_records: A list of EnhancedAlignmentRecords to render, uses a square to represent alignments in progress
    :param list finalized_alignment_records: A list of EnhancedAlignmentRecords to render, uses a circle to represent alignments are fixed
    :param bool filename: Filename to save plot as
    :param float ylim: y-limit of plot
    :param float xlim: x-limit of plot    
    '''
    
    if attrib is None:
        attrib = 'weight'
    
    all_records = new_alignment_records + finalized_alignment_records
    shapes = ['s' for a in new_alignment_records]
    shapes.extend(['.' for a in finalized_alignment_records])
                 
    TargetPoints = np.asarray(list(map(lambda a: a.TargetPoint, all_records)))
    #OriginalWarpedPoints = np.asarray(list(map(lambda a: a.SourcePoint, all_records)))
    AdjustedTargetPoints = np.asarray(list(map(lambda a: a.AdjustedTargetPoint, all_records)))
    #AdjustedWarpedPoints = np.asarray(list(map(lambda a: a.AdjustedWarpedPoint, all_records)))
    weights = np.asarray(list(map(lambda a: getattr(a, attrib), all_records)))
    percentile_prep = weights - weights.min()
    percentiles = (percentile_prep / np.max(percentile_prep)) * 100
    TargetPeaks = AdjustedTargetPoints - TargetPoints
    
    nornir_shared.plot.VectorField(TargetPoints.astype(np.float32), TargetPeaks, shapes, percentiles, filename, ylim, xlim )
     
    return