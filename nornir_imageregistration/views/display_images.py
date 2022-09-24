
import matplotlib.pyplot as plt 
import collections.abc
import nornir_imageregistration
import numpy as np
import math
import matplotlib.colors 
import matplotlib.gridspec

def ShowGrayscale(input_params, title=None, PassFail=False):
    '''
    :param PassFail:
    :param list input_params: A list or single ndimage to be displayed with imshow
    :param str title: Informative title for the figure, for example expected test results
    '''
 
    def set_title_for_single_image(title):
        if not title is None:
            plt.title(title)
        return 

    def set_title_for_multi_image(fig, title):
        if not title is None:
            fig.suptitle(title)
        return 
                
    fig = None
    axes = None
    
    image_data = _ConvertParamsToImageList(input_params)
    grid_dims = _GridLayoutDims(image_data)
    
    if grid_dims == (1, 1): 
        (fig, ax) = _DisplayImageSingle(image_data)
        set_title_for_single_image(title)
    elif grid_dims[1] == 1:
        (fig, ax) = _DisplayImageList1D(image_data)
        set_title_for_multi_image(fig, title)
    elif grid_dims[1] > 1:
        (fig, gs) = _DisplayImageList2D(image_data, grid_dims)
        set_title_for_multi_image(fig, title)
        
    elif isinstance(input_params, collections.abc.Iterable):
        # OK, we have a list of images or a list of lists
        # TODO: Why doesn't this use the DisplayImageList2D function?
        
        height, width = _GridLayoutDims(input_params)
        gs = matplotlib.gridspec.GridSpec(nrows=height, ncols=height)
        fig = plt.figure()
        set_title_for_multi_image(fig, title)
        
        for i, image in enumerate(input_params):
            # fig = figure()
            if isinstance(image, np.ndarray):
                # ax = fig.add_subplot(101 + ((len(input_params) - (i)) * 10))
                iRow = i // width
                iCol = (i - (iRow * width)) % width
        
                print("Row %d Col %d" % (iRow, iCol))
                
                ax = fig.add_subplot(gs[iRow, iCol])
        #                     if height > 1:
        #                         ax = axes[iRow, iCol ]
        #                     else:
        #                         ax = axes[iCol]
        
                ax.imshow(image, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm())  
    else:
        return
 
    fig.tight_layout()

    if PassFail == True:
        return nornir_imageregistration.ShowWithPassFail(fig)

    else:
        # plt.tight_layout(pad=1.0)  
        fig.show()
    # Do not call clf or we get two windows on the next call 
    # plt.clf()
 
    fig = None

    return 


def _ConvertParamsToImageList(param):
    output = None
    if isinstance(param, str):
        loaded_image = nornir_imageregistration.ImageParamToImageArray(param)
        output = nornir_imageregistration.core._Image_To_Uint8(loaded_image)
    elif isinstance(param, np.ndarray):
        output = nornir_imageregistration.core._Image_To_Uint8(param)
    elif isinstance(param, collections.abc.Iterable):
        output = [_ConvertParamsToImageList(item) for item in param]
        if len(output) == 1:
            output = output[0]
    
    return output
    

def _GridLayoutDims(imagelist):
    '''Given a list of N items, returns the number of rows & columns to display the list.  Dimensions will always be wider than they are tall or equal in dimension
    '''
    
    def _NumImages(param):
        if isinstance(param, np.ndarray):
            return 1
        else:
            return len(param)
                
    if isinstance(imagelist, np.ndarray):
        return (1, 1)
    elif isinstance(imagelist, collections.abc.Iterable):
        lengths = [_NumImages(p) for p in imagelist]
        max_len = np.max(lengths)
        return (len(imagelist), max_len)
    
def get_aspect(ax=None):
    remove_plot = False
    if ax is None:
        ax = plt.gca()
        remove_plot = True
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    axes_ratio = height / width
    aspect = axes_ratio / ax.get_data_ratio()

    if remove_plot:
        plt.close(fig)
        
    return aspect


def _ImageList1DGridDims(imagelist):
    # OK, a 1D list, so figure out how to spread the images across a grid
    numImages = len(imagelist)
    
    aspect_ratio = get_aspect()
    #Assume 
    # N = Num Images
    # R = Ratio
    # W = Width
    # H = Height
    #
    # W * H = N
    # W / H = R
    #
    # H = W / R
    # W * (W / R) = W^2 / R
    # sqrt(N * R) = W^2
    #
    
    A = math.ceil(math.sqrt(numImages * aspect_ratio))
    B = math.ceil(numImages / A)

    if len(imagelist) == 0:
        return (0,0)
    
    width = None
    height = None
    
    #Use the first image dimensions to determine if we have more rows or columns
    if A != B:
        imshapetotal = imagelist[0].shape
        for im in imagelist[1:]:
            imshapetotal = np.add(imshapetotal,im.shape)
        
        if imshapetotal[0] * aspect_ratio > imshapetotal[1]: #Images are taller than the are wide
            width = max(A,B)
            height = min(A,B)
        else:
            width = min(A,B)
            height = max(A,B)
            
    if width is None:
        if aspect_ratio > 1:    
            width = max(A,B)
            height = min(A,B)
        else:
            width = min(A,B)
            height = max(A,B)
         

    return (int(height), int(width))


def _DisplayImageSingle(input_param, title=None):
    fig, ax = plt.subplots()
    ax.imshow(input_param, cmap=plt.gray(), aspect='equal', norm=matplotlib.colors.NoNorm())
    if not title is None:
        ax.set_title(title)
    
    return (fig, ax)

    
def _DisplayImageList1D(input_params, title=None): 
        
    (height, width) = _ImageList1DGridDims(input_params)
    
    fig, axes = plt.subplots(height, width)
    
    total_plots = height * width
    
    ndim = len(axes.shape)
    
    for i, image in enumerate(input_params):
        
        ax = None
        
        if ndim == 1:
            ax = axes[i]
        elif ndim == 2:
            iRow = i // width
            iCol = (i - (iRow * width)) % width
            # print("Row %d Col %d" % (iRow, iCol))
    
            if height > 1:
                ax = axes[iRow, iCol ]
            else:
                ax = axes[iCol] 
        else:
            raise NotImplemented("What in the world?  3D Display or something?")
        
        ax.imshow(image, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm())
    
    i += 1
    while i < total_plots:
        iRow = i // width
        iCol = (i - (iRow * width)) % width
        
        if height > 1:
            ax = axes[iRow, iCol ]
        else:
            ax = axes[iCol]
            
        ax.remove()
            
        i += 1
            
    return (fig, axes)

    
def _DisplayImageList2D(input_params, grid_dims, title=None):
    (height, width) = grid_dims
    gs = matplotlib.gridspec.GridSpec(nrows=height, ncols=width)
    fig = plt.figure()
    # , axes = plt.subplots(height, width)

    for (iRow, row_list) in enumerate(input_params):
        
        if isinstance(row_list, np.ndarray):
            ax = fig.add_subplot(gs[iRow,:])  # axes[iRow, 0]
            ax.imshow(row_list, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm())
            continue 
        
        numCols = len(row_list)
        for iCol, image in enumerate(row_list):
            # print("Row %d Col %d" % (iRow, iCol))
            
            if iCol == numCols - 1 and numCols < width:
                ax = fig.add_subplot(gs[iRow, iCol:])
            else:
                ax = fig.add_subplot(gs[iRow, iCol])

 #           if height > 1:
#                ax = fig.add_subplot(gs[iRow, iCol]
#            else:
#                ax = fig.add_subplot([iCol]

            ax.imshow(image, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm()) 
    
    return (fig, gs)
    
