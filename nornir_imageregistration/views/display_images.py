
import matplotlib.pyplot as plt 
import collections
import nornir_imageregistration
import numpy as np
import math
import matplotlib.colors 

def ShowGrayscale(input_params, title=None,PassFail=False):
    '''
    :param list input_params: A list or single ndimage to be displayed with imshow
    :param str title: Informative title for the figure, for example expected test results
    '''
    from matplotlib.widgets import Button
 
    def set_title_for_single_image(title):
        if not title is None:
            plt.title(title)
        return 

    def set_title_for_multi_image(fig, title):
        if not title is None:
            fig.suptitle(title)
        return

    class PassFailInput(object):
        
        def __init__(self, fig):
            self.Pass = None
            self.fig = fig
            

        def OnPassButton(self, event):
            self.Pass = True   
            return

        def OnFailButton(self, event):
            self.Pass = False 
            return
                
    fig = None
    axes = None
    
    image_data = _ConvertParamsToImageList(input_params)
    grid_dims = _GridLayoutDims(image_data)
    
    if grid_dims == (1,1): 
        (fig, ax) = _DisplayImageSingle(image_data)
        set_title_for_single_image(title)
    elif grid_dims[1] == 1:
        (fig, ax) = _DisplayImageList1D(image_data)
        set_title_for_multi_image(fig, title)
    elif grid_dims[1] > 1:
        (fig, ax) = _DisplayImageList2D(image_data, grid_dims)
        set_title_for_multi_image(fig, title)
        
    elif isinstance(input_params, collections.Iterable):
        #OK, we have a list of images or a list of lists
           
            height, width = _GridLayoutDims(input_params)
            fig, axes = plt.subplots(height, width)
            set_title_for_multi_image(fig, title)

            for i, image in enumerate(input_params):
                # fig = figure()
                if isinstance(image, np.ndarray):
                    # ax = fig.add_subplot(101 + ((len(input_params) - (i)) * 10))
                    iRow = i // width
                    iCol = (i - (iRow * width)) % width

                    print("Row %d Col %d" % (iRow, iCol))

                    if height > 1:
                        ax = axes[iRow, iCol ]
                    else:
                        ax = axes[iCol]

                    ax.imshow(image, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm())  
    else:
        return

    callback = PassFailInput(fig)
    
    fig.tight_layout()

    if PassFail == True:
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

        bnext = Button(axnext, 'Pass', color='#00FF80')
        bnext.on_clicked(callback.OnPassButton)
        bprev = Button(axprev, 'Fail', color='#FF0000')
        bprev.on_clicked(callback.OnFailButton)
        #plt.tight_layout(pad=1.0)  
        fig.show()
        
        while callback.Pass is None:
            fig.waitforbuttonpress()
            
        plt.close(fig)

    else:
        #plt.tight_layout(pad=1.0)  
        fig.show()
    # Do not call clf or we get two windows on the next call 
    # plt.clf()
 
    fig = None

    return callback.Pass


def _ConvertParamsToImageList(param):
    output=None
    if isinstance(param, str):
        loaded_image = nornir_imageregistration.ImageParamToImageArray(param)
        output = nornir_imageregistration.core._Image_To_Uint8(loaded_image)
    elif isinstance(param, np.ndarray):
        output = nornir_imageregistration.core._Image_To_Uint8(param)
    elif isinstance(param, collections.Iterable):
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
        return (1,1)
    elif isinstance(imagelist, collections.Iterable):
        lengths = [_NumImages(p) for p in imagelist]
        max_len = np.max(lengths)
        return (len(imagelist), max_len)

def _ImageList1DGridDims(imagelist):
    #OK, a 1D list, so figure out how to spread the images across a grid
    numImages = len(imagelist)
    width = math.ceil(math.sqrt(numImages))
    height = math.ceil(numImages / width)

    if height > width:
        tempH = height
        height = width
        height = tempH

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

    for i, image in enumerate(input_params):
        iRow = i // width
        iCol = (i - (iRow * width)) % width

        #print("Row %d Col %d" % (iRow, iCol))

        if height > 1:
            ax = axes[iRow, iCol ]
        else:
            ax = axes[iCol]

        ax.imshow(image, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm())   
            
    return (fig, axes)
    
def _DisplayImageList2D(input_params, grid_dims, title=None):
    (height, width) = grid_dims
    fig, axes = plt.subplots(height, width)

    for iRow, row_list in enumerate(input_params):
        
        if isinstance(row_list, np.ndarray):
            ax = axes[iRow, 0]
            ax.imshow(row_list, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm())
            continue 
        
        for iCol, image in enumerate(row_list):
            #print("Row %d Col %d" % (iRow, iCol))

            if height > 1:
                ax = axes[iRow, iCol]
            else:
                ax = axes[iCol]

            ax.imshow(image, cmap=plt.gray(), figure=fig, aspect='equal', norm=matplotlib.colors.NoNorm())    
    
    return (fig, axes)
    