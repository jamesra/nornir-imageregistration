'''
Created on Feb 27, 2019

@author: u0490822
'''

#import numpy as np
import matplotlib.pyplot as plt
#import nornir_shared.plot

def PlotGridPositionsAndMask(source_coords, source_image=None, OutputFilename=None, ylim=None, xlim=None, attrib=None):
    """
    Converts a set of EnhancedAlignmentRecord peaks from the RefineTwoImages function into a transform
    :param source_coords:
    :param source_image:
    :param OutputFilename:
    :param attrib:
    """
    
    #if attrib is None:
    #    attrib = 'weight'
    
    #all_records = new_alignment_records + finalized_alignment_records
    #shapes = ['s' for a in new_alignment_records]
    #shapes.extend(['.' for a in finalized_alignment_records])
                 
    #TargetPoints = np.asarray(list(map(lambda a: a.TargetPoint, all_records)))
    #OriginalWarpedPoints = np.asarray(list(map(lambda a: a.SourcePoint, all_records)))
    #AdjustedTargetPoints = np.asarray(list(map(lambda a: a.AdjustedTargetPoint, all_records)))
    #AdjustedWarpedPoints = np.asarray(list(map(lambda a: a.AdjustedWarpedPoint, all_records)))
    #weights = np.asarray(list(map(lambda a: getattr(a, attrib), all_records)))
    #percentile_prep = weights - weights.min()
    #percentiles = (percentile_prep / np.max(percentile_prep)) * 100
    #TargetPeaks = AdjustedTargetPoints - TargetPoints
    plt.clf() 
    
    plt.scatter(source_coords[:, 1], source_coords[:, 0], color='r', marker='.', alpha=0.5)
    #nornir_shared.plot.VectorField(source_coords.astype(np.float32, copy=False), source_coords, shapes='.', OutputFilename=OutputFilename, ylim=ylim, xlim=xlim )
    
    if source_image is not None:
        plt.imshow(source_image, cmap=plt.gray(), aspect='equal')
    
    if OutputFilename is not None:
        plt.savefig(OutputFilename, dpi=300)
    else:
        plt.show()