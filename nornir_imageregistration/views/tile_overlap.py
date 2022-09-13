'''
Created on Apr 29, 2019

@author: u0490822
'''
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection 
import nornir_imageregistration


def _plot_mapped_space_overlap(ax, tile, overlap_rect, color=None):
        if color is None:
            color = 'red'
            
        patches = [mpatches.Rectangle(numpy.flip(overlap_rect.BottomLeft),
                                      overlap_rect.Width,
                                      overlap_rect.Height,
                                      color=color), mpatches.Rectangle(numpy.flip(tile.MappedBoundingBox.BottomLeft),
                                                                       tile.MappedBoundingBox.Width,
                                                                       tile.MappedBoundingBox.Height,
                                                                       color='gray',
                                                                       fill=False)]

        collection = PatchCollection(patches, True, alpha=0.3)
        ax.add_collection(collection)
        ax.axis('equal')

    
def _plot_fixed_space_overlap(ax, overlap, color=None):
    if color is None:
        color = 'red'
        
    patches = []
    patches.append(mpatches.Rectangle(numpy.flip(overlap.A.FixedBoundingBox.BottomLeft),
                                      overlap.A.FixedBoundingBox.Width,
                                      overlap.A.FixedBoundingBox.Height,
                                      color='green',
                                      label='A'))
    
    patches.append(mpatches.Rectangle(numpy.flip(overlap.B.FixedBoundingBox.BottomLeft),
                                      overlap.B.FixedBoundingBox.Width,
                                      overlap.B.FixedBoundingBox.Height,
                                      color='blue',
                                      label='B'))
    
    patches.append(mpatches.Rectangle(numpy.flip(overlap.overlapping_target_rect.BottomLeft),
                                      overlap.overlapping_target_rect.Width,
                                      overlap.overlapping_target_rect.Height,
                                      color='red'))
      
    collection = PatchCollection(patches, True, alpha=0.3)
    ax.add_collection(collection)
    ax.axis('equal')


def plot_tile_overlap(overlap, OutputFilename=None):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 2])
    
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    ax_tl = fig.add_subplot(gs[0, 0])
    _plot_mapped_space_overlap(ax_tl, overlap.A, overlap.overlapping_source_rect_A, color='green')
    ax_tl.set_title('A Source')
    ax_tr = fig.add_subplot(gs[0, 1])
    _plot_mapped_space_overlap(ax_tr, overlap.B, overlap.overlapping_source_rect_B, color='blue')
    ax_tr.set_title('B Source')
#         overlap_A = overlap.scaled_overlapping_source_rect_A
#         patches = []
#         patches.append(mpatches.Rectangle(overlap_A.BottomLeft,overlap_A.Width, overlap_A.Height, color='red'))
#         patches.append(mpatches.Rectangle(overlap.A.MappedBoundingBox.BottomLeft,overlap.A.MappedBoundingBox.Width, overlap.A.MappedBoundingBox.Height, color='gray', fill=False))
# 
#         collection = PatchCollection(patches, True, alpha=0.3)
#         ax[0,1].add_collection(collection)
#         ax[0,1].axis('equal')
    ax_target_space = fig.add_subplot(gs[1:, :])
    _plot_fixed_space_overlap(ax_target_space, overlap)
    ax_target_space.set_title('Target Space')
    # plt.axis('equal')
    # plt.tight_layout()
    if OutputFilename is not None:
        plt.savefig(OutputFilename)
    else:
        plt.show(block=True)

    
def _create_tile_target_space_patch(tile, **kwargs):
    return mpatches.Rectangle(numpy.flip(tile.FixedBoundingBox.BottomLeft),
                                  tile.FixedBoundingBox.Width,
                                  tile.FixedBoundingBox.Height,
                                  label='{0}'.format(tile.ID),
                                  **kwargs)

    
def plot_tile_overlaps(overlaps, colors=None, OutputFilename=None):
    
    fig = plt.figure(dpi=600)
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = fig.add_subplot(gs[0, 0])
    
    patches = []
    plotted_tiles = set()
    bbox = None
    if colors is None:
        colors = ['red'] * len(overlaps)
        
    for (i,overlap) in enumerate(overlaps):
        #label_str = f'{overlap.A.ID}-{overlap.B.ID} {overlap.overlap * 100:.1f}%  {overlap.offset[0]:.1f}y {overlap.offset[1]:.1f}x'
        label_str = f'{overlap.overlap * 100:.1f}%  {overlap.offset[0]:.1f}y {overlap.offset[1]:.1f}x'
        patches.append(mpatches.Rectangle(numpy.flip(overlap.overlapping_target_rect.BottomLeft),
                                      overlap.overlapping_target_rect.Width,
                                      overlap.overlapping_target_rect.Height,
                                      color=colors[i],
                                      edgecolor=None,
                                      linewidth=0,
                                      alpha=overlap.overlap,
                                      label=label_str))
        
        
        feature_scores = getattr(overlap, 'feature_scores', None)
        if feature_scores is not None and feature_scores[0] is not None:
            label_str = label_str + '\nScores: {0:.2f} {1:.2f}'.format(feature_scores[0], feature_scores[1])
            
        text_rotation = 0
        if overlap.overlapping_target_rect.Height > overlap.overlapping_target_rect.Width: 
            text_rotation = 90
            
        ax.text(overlap.overlapping_target_rect.Center[1],
                overlap.overlapping_target_rect.Center[0],
                label_str,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=text_rotation,
                fontsize=1)
        
        if overlap.A.ID not in plotted_tiles:
            plotted_tiles.union([overlap.A.ID])
            patches.append(_create_tile_target_space_patch(overlap.A,
                                                           # color='grey',
                                                           facecolor='grey',
                                                           alpha=0.15,
                                                           linewidth=0,
                                                           fill=True,
                                                           edgecolor='black'))
            ax.text(overlap.A.FixedBoundingBox.Center[1],
                overlap.A.FixedBoundingBox.Center[0],
                '{0}'.format(overlap.A.ID),
                horizontalalignment='center',
                verticalalignment='center',
                rotation=0,
                fontsize=2)
            
            if bbox is None:
                bbox = overlap.A.FixedBoundingBox
            else:
                bbox = nornir_imageregistration.Rectangle.Union(bbox, overlap.A.FixedBoundingBox)
                                                           
        if overlap.B.ID not in plotted_tiles:
            plotted_tiles.union([overlap.B.ID])
            patches.append(_create_tile_target_space_patch(overlap.B,
                                                           # color='grey',
                                                           facecolor='grey',
                                                           alpha=0.15,
                                                           linewidth=0,
                                                           fill=True,
                                                           edgecolor='black'))
            
            ax.text(overlap.B.FixedBoundingBox.Center[1],
                overlap.B.FixedBoundingBox.Center[0],
                '{0}'.format(overlap.B.ID),
                horizontalalignment='center',
                verticalalignment='center',
                rotation=0,
                fontsize=2)
            
            if bbox is None:
                bbox = overlap.B.FixedBoundingBox
            else:
                bbox = nornir_imageregistration.Rectangle.Union(bbox, overlap.B.FixedBoundingBox)
    
    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
    ax.axis('equal')
    
    ax.set_title('Overlaps')
    plt.tight_layout()
    ax.set_xlim(bbox.MinX, bbox.MaxX)
    ax.set_ylim(bbox.MinY, bbox.MaxY)
    ax.invert_yaxis()
    
    
    # 
    
    if OutputFilename is not None:
        plt.savefig(OutputFilename)
    else:
        plt.show(block=True)
        
    plt.close(fig)
         

if __name__ == '__main__':
    pass
