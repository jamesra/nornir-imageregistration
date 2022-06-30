'''
Created on Apr 30, 2019

@author: u0490822
'''

import numpy
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.colors as colors
import matplotlib.cm as cmx
import nornir_imageregistration
from nornir_imageregistration.spatial import iPoint

def __PlotVectorOriginShape(render_mask, shape, Points, weights=None, color=None, colormap=None):
    '''Plot a subset of the points that are True in the mask with the specified shape
    color is only used if weights is none.
    '''
     
    if weights is None:
        if color is None:
            color = 'red' 
        
        plt.scatter(Points[:, 1], Points[:, 0], color=color, marker=shape, alpha=0.5)
    else:
        if colormap is None:
            colormap = plt.get_cmap('jet')
        
        max_weight = None
        if weights is None or len(weights) == 0:
            max_weight = 1
        else:
            max_weight = max(weights)
        
        plt.scatter(Points[render_mask, 1],
                    Points[render_mask, 0],
                    c=weights[render_mask],
                    marker=shape,
                    vmin=0,
                    vmax=max_weight,
                    alpha=0.5,
                    cmap=colormap)
    
def __PlotLinkedNodes(layout_obj, ax, min_tension=None, max_tension=None):
    '''
    Plot a line between nodes that are linked in the layout
    '''
    
    pairs = layout_obj.linked_nodes
    
    if max_tension is None:
        max_tension = layout_obj.MaxTensionMagnitude
        if max_tension is None:
            max_tension = 0
        else:
            max_tension = max_tension[1]
    
    if min_tension is None:    
        min_tension = layout_obj.MinTensionMagnitude
        if min_tension is None:
            min_tension = 0
        else:
            min_tension = min_tension[1]
        
    cNorm  = colors.Normalize(vmin=0, vmax=max_tension)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('jet'))
                
    #Create a line for each pair 
    for (A_ID,B_ID) in pairs:
        A_pos = layout_obj.GetPosition(A_ID)
        B_pos = layout_obj.GetPosition(B_ID)
        xdata = [A_pos[iPoint.X], B_pos[iPoint.X]]
        ydata = [A_pos[iPoint.Y], B_pos[iPoint.Y]]
        
        weight = nornir_imageregistration.array_distance(layout_obj.PairTensionVector(A_ID,B_ID))           
        colorVal = scalarMap.to_rgba(weight)
        
        #alpha = layout_obj.
        line = lines.Line2D(xdata,ydata, color=colorVal, lw=1)
        ax.add_line(line)
    
    return max_tension

def __plot_nodes(layout_obj):
    
    for node in layout_obj.nodes:
        node_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(node.Position, node.dims)
        
        

def plot_layout(layout_obj, shapes=None, OutputFilename=None, ylim=None, xlim=None, max_tension=None, title=None, PassFail=False):
     
    plt.clf()
    if title is not None:
        plt.title(title)
    
    Points = layout_obj.GetPositions()
    Offsets = layout_obj.WeightedNetTensionVectors()
    weights=nornir_imageregistration.array_distance(layout_obj.WeightedNetTensionVectors()[:,1:])
    
    if max_tension is None:
        max_tension = layout_obj.MaxTensionMagnitude
        if max_tension is None:
            max_tension = 0
        else:
            max_tension = max_tension[1]
#         
#     if max_tension > 0:
#         weights /= max_tension
    
    if shapes is None:
        shapes = 's'
         
    if isinstance(shapes, str):
        shapes = shapes
        mask = numpy.ones(Points.shape[0], dtype=numpy.bool)
        __PlotVectorOriginShape(mask, shapes, Points, weights)
    else:
        try:
            _ = iter(shapes)
        except TypeError:
            raise ValueError("shapes must be None, string, or iterable type")
    
        #Iterable
        if len(shapes) != Points.shape[0]:
            raise ValueError("Length of shapes must match number of points")
        
        #Plot each batch of points with different shape
        all_shapes = set(shapes)
        
        for shape in all_shapes:
            mask = [s == shape for s in shapes]
            mask = numpy.asarray(mask, dtype=numpy.bool)
            
            __PlotVectorOriginShape(mask, shape, Points, weights)
            
     
            
    if ylim is not None:
        plt.ylim(ylim)
        
    if xlim is not None:
        plt.xlim(xlim)
             
    if weights is not None:
        plt.colorbar()
        
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.gcf().patch.set_facecolor((0.1,0.1,0.1))
    plt.gca().set_facecolor((0.5,0.5,0.5))
    plt.gcf().set_facecolor("w")
    
    new_max_tension = __PlotLinkedNodes(layout_obj, plt.gca(), max_tension=max_tension)
    
    assert(Points.shape[0] == Offsets.shape[0])
    for iRow in range(0, Points.shape[0]):
        Origin = Points[iRow, :]
        scaled_offset = Offsets[iRow, 1:]
         
        Destination = Origin + (scaled_offset * 20)
        
        if not numpy.array_equal(Origin, Destination):
            line = numpy.vstack((Origin, Destination))
            #line = lines.Line2D(line[:, 1], line[:, 0], color='blue', alpha=0.5, width=0.1)
#            This plot line isn't showing up.
            plt.plot(line[:, 1], line[:, 0], color='black', alpha=0.5, linewidth=1)
        
    if ylim is not None:
        plt.ylim(ylim)
        
    if xlim is not None:
        plt.xlim(xlim)
         
    if(OutputFilename is not None):
        plt.savefig(OutputFilename, dpi=300)
    else:
        if PassFail:
            return nornir_imageregistration.ShowWithPassFail(plt.gcf())
        else:
            plt.show(block=True)
        
    return new_max_tension
        
if __name__ == '__main__':
    pass