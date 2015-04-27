'''
Created on Apr 22, 2013


'''


import os

from matplotlib.pyplot import imsave
from scipy.ndimage import interpolation

from nornir_imageregistration.files.stosfile import StosFile
from   nornir_imageregistration.transforms import factory, triangulation
import nornir_imageregistration.transforms.base as transformbase
from   nornir_imageregistration.transforms.utils import InvalidIndicies
import nornir_pools as pools
import nornir_shared.images as images
import nornir_shared.prettyoutput as PrettyOutput
import numpy as np

from . import core


def GetROICoords(botleft, area):
    x_range = np.arange(botleft[1], botleft[1] + area[1], dtype=np.float32)
    y_range = np.arange(botleft[0], botleft[0] + area[0], dtype=np.float32)
    
    #Numpy arange sometimes accidentally adds an extra value to the array due to rounding error, remove the extra element if needed
    if len(x_range) > area[1]:
        x_range = x_range[:area[1]]
        
    if len(y_range) > area[0]:
        y_range = y_range[:area[0]]

    i_y, i_x = np.meshgrid(y_range, x_range, sparse=False, indexing='ij')

    coordArray = np.vstack((i_y.astype(np.float32).flat, i_x.astype(np.float32).flat)).transpose()
    
    del i_y
    del i_x
    del x_range
    del y_range

    return coordArray

def DestinationROI_to_SourceROI(transform, botleft, area, extrapolate=False):
    ''' 
    Apply a transform to a region of interest within an image. Center and area are in fixed space
    
    :param transform transform: The transform used to map points between fixed and mapped space
    :param 1x2_array botleft: The (Y,X) coordinates of the bottom left corner
    :param 1x2_array area: The (Height, Width) of the region of interest
    :param bool exrapolate: If true map points that fall outside the bounding box of the transform
    :return: Tuple of arrays.  First array is fixed space coordinates.  Second array is warped space coordinates.
    :rtype: tuple(Nx2 array,Nx2 array)
    '''

    DstSpace_coordArray = GetROICoords(botleft, area)

    SrcSpace_coordArray = transform.InverseTransform(DstSpace_coordArray, extrapolate=extrapolate).astype(np.float32)
    (valid_SrcSpace_coordArray, InvalidIndiciesList) = InvalidIndicies(SrcSpace_coordArray)

    del SrcSpace_coordArray

    valid_DstSpace_coordArray = np.delete(DstSpace_coordArray, InvalidIndiciesList, axis=0)
    valid_DstSpace_coordArray = valid_DstSpace_coordArray - botleft

    return (valid_DstSpace_coordArray, valid_SrcSpace_coordArray)


def SourceROI_to_DestinationROI(transform, botleft, area, extrapolate=False):
    '''
    Apply an inverse transform to a region of interest within an image. Center and area are in fixed space
    
    :param transform transform: The transform used to map points between fixed and mapped space
    :param 1x2_array botleft: The (Y,X) coordinates of the bottom left corner
    :param 1x2_array area: The (Height, Width) of the region of interest
    :param bool exrapolate: If true map points that fall outside the bounding box of the transform
    :return: Tuple of arrays.  First array is fixed space coordinates.  Second array is warped space coordinates.
    :rtype: tuple(Nx2 array,Nx2 array)
    '''

    SrcSpace_coordArray = GetROICoords(botleft, area)

    DstSpace_coordArray = transform.Transform(SrcSpace_coordArray, extrapolate=extrapolate).astype(np.float32)
    (valid_DstSpace_coordArray, InvalidIndiciesList) = InvalidIndicies(DstSpace_coordArray)

    del DstSpace_coordArray

    valid_SrcSpace_coordArray = np.delete(SrcSpace_coordArray, InvalidIndiciesList, axis=0)
    valid_SrcSpace_coordArray = valid_SrcSpace_coordArray - botleft

    return (valid_DstSpace_coordArray, valid_SrcSpace_coordArray)


def ExtractRegion(image, botleft=None, area=None, cval=0):
    '''
    Extract a region from an image
    
    :param ndarray image: Source image
    :param 1x2_array botleft: The (Y,X) coordinates of the bottom left corner
    :param 1x2_array area: The (Height, Width) of the region of interest
    :return: Image of requested region
    :rtype: ndarray
    
    '''
    
    raise DeprecationWarning("Deprecated __ExtractRegion call being used, use core.CropImage instead")
#     if botleft is None:
#         botleft = (0, 0)
# 
#     if area is None:
#         area = image.shape
# 
#     coords = GetROICoords(botleft, area)
# 
#     transformedImage = interpolation.map_coordinates(image, coords.transpose(), order=0, mode='constant', cval=cval)
# 
#     transformedImage = transformedImage.reshape(area)
#     return transformedImage
 

def __ExtractRegion(image, botleft, area):
    print("Deprecated __ExtractRegion call being used")
    return ExtractRegion(image, botleft, area)


def __CropImageToFitCoords(input_image, coordinates, cval=0):
    '''For large images we only need a specific range of coordinates from the image.  However Scipy calls such as map_coordinates will 
       send the entire image through a spline_filter first.  To avoid this we crop the image with a padding of one and adjust the 
       coordinates appropriately
       :param ndarray input_image: image we will be extracting data from at the specfied coordinates
       :param ndarray coordinates: Nx2 array of points indexing into the image
       :param float cval: Value to use for regions outside the existing image when padding
       :return: (cropped_image, translated_coordinates)
       '''
    minCoord = np.floor(np.min(coordinates, 0)) - np.array([1, 1])
    maxCoord = np.ceil(np.max(coordinates, 0)) + np.array([1, 1])
    
    if minCoord[0] < 0:
        minCoord[0] = 0
    if minCoord[1] < 0:
        minCoord[1] = 0
    
    if maxCoord[0] > input_image.shape[0]:
        maxCoord[0] = input_image.shape[0]
    if maxCoord[1] > input_image.shape[1]:
        maxCoord[1] = input_image.shape[1]

    cropped_image = core.CropImage(input_image, Xo=int(minCoord[1]), Yo=int(minCoord[0]), Width=int(maxCoord[1] - minCoord[1]), Height=int(maxCoord[0] - minCoord[0]), cval=cval)
    translated_coordinates = coordinates - minCoord
    
    return (cropped_image, translated_coordinates)

def __WarpedImageUsingCoords(fixed_coords, warped_coords, FixedImageArea, WarpedImage, area=None, cval=0):
    '''Use the passed coordinates to create a warped image
    :Param fixed_coords: 2D coordinates in fixed space
    :Param warped_coords: 2D coordinates in warped space
    :Param FixedImageArea: Dimensions of fixed space
    :Param WarpedImage: Image to read pixel values from while creating fixed space images
    :Param area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.'''

    if area is None:
        area = FixedImageArea

    if not isinstance(area, np.ndarray):
        area = np.asarray(area, dtype=np.uint64)

    if area.dtype != np.uint64:
        area = np.asarray(area, dtype=np.uint64)

    if(warped_coords.shape[0] == 0):
        # No points transformed into the requested area, return empty image
        transformedImage = np.full((area), cval, dtype=WarpedImage.dtype)
        return transformedImage

    subroi_warpedImage = None
    #For large images we only need a specific range of the image, but the entire image is passed through a spline filter by map_coordinates
    #In this case use only a subset of the warpedimage
    if np.prod(WarpedImage.shape) > warped_coords.shape[0]:
    #if not area[0] == FixedImageArea[0] and area[1] == FixedImageArea[1]:
        #if area[0] <= FixedImageArea[0] or area[1] <= FixedImageArea[1]:
        (subroi_warpedImage, warped_coords) = __CropImageToFitCoords(WarpedImage, warped_coords, cval=cval)
        del WarpedImage
    else:
        subroi_warpedImage = WarpedImage
    
    outputImage = interpolation.map_coordinates(subroi_warpedImage, warped_coords.transpose(), mode='constant', order=3, cval=cval)
    if fixed_coords.shape[0] == np.prod(area):
        # All coordinates mapped, so we can return the output warped image as is.
        outputImage = outputImage.reshape(area)
        return outputImage
    else:
        # Not all coordinates mapped, create an image of the correct size and place the warped image inside it.
        transformedImage = np.full((area), cval, dtype=outputImage.dtype)        
        fixed_coords_rounded = np.asarray(np.round(fixed_coords), dtype=np.int32)
        transformedImage[fixed_coords_rounded[:, 0], fixed_coords_rounded[:, 1]] = outputImage
        return transformedImage
    

def _LoadImageIfNeeded(value):
    if isinstance(value, str):
        return core.LoadImage(value)
    
    return value 
    

def _ReplaceFilesWithImages(listImages):
    '''Replace any filepath strings in the passed parameter with loaded images.'''
    
    if isinstance(listImages, list):
        for i, value in enumerate(listImages):
            listImages[i] = _LoadImageIfNeeded(value)
    else:
        listImages = _LoadImageIfNeeded(listImages)
        
    return listImages


def FixedImageToWarpedSpace(transform, WarpedImageArea, DataToTransform, botleft=None, area=None, cval=None, extrapolate=False):
    '''Warps every image in the DataToTransform list using the provided transform.
    :Param transform: transform to pass warped space coordinates through to obtain fixed space coordinates
    :Param FixedImageArea: Size of fixed space region to map pixels into
    :Param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :Param botleft: Origin of region to map
    :Param area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.
    :param bool exrapolate: If true map points that fall outside the bounding box of the transform
    '''
    
    if botleft is None:
        botleft = (0, 0)

    if area is None:
        area = WarpedImageArea

    if cval is None:
        cval = [0] * len(DataToTransform)

    if not isinstance(cval, list):
        cval = [cval] * len(DataToTransform)

    (DstSpace_coords, SrcSpace_coords) = SourceROI_to_DestinationROI(transform, botleft, area, extrapolate=extrapolate)
    
    ImagesToTransform = _ReplaceFilesWithImages(DataToTransform)  

    if isinstance(ImagesToTransform, list):
        FixedImageList = []
        for i, wi in enumerate(ImagesToTransform):
            fi = __WarpedImageUsingCoords(DstSpace_coords, SrcSpace_coords, WarpedImageArea, wi, area, cval=cval[i])
            FixedImageList.append(fi)
            
        del SrcSpace_coords
        del DstSpace_coords
        
        return FixedImageList
    else:
        return __WarpedImageUsingCoords(SrcSpace_coords, DstSpace_coords, WarpedImageArea, ImagesToTransform, area, cval=cval[0])

        

def WarpedImageToFixedSpace(transform, FixedImageArea, DataToTransform, botleft=None, area=None, cval=None, extrapolate=False):

    '''Warps every image in the DataToTransform list using the provided transform.
    :Param transform: transform to pass warped space coordinates through to obtain fixed space coordinates
    :Param FixedImageArea: Size of fixed space region to map pixels into
    :Param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :Param botleft: Origin of region to map
    :Param area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.
    :param bool exrapolate: If true map points that fall outside the bounding box of the transform
    '''

    if botleft is None:
        botleft = (0, 0)

    if area is None:
        area = FixedImageArea

    if cval is None:
        cval = [0] * len(DataToTransform)

    if not isinstance(cval, list):
        cval = [cval] * len(DataToTransform)

    (DstSpace_coords, SrcSpace_coords) = DestinationROI_to_SourceROI(transform, botleft, area, extrapolate=extrapolate)
    
    ImagesToTransform = _ReplaceFilesWithImages(DataToTransform)  

    if isinstance(ImagesToTransform, list):
        FixedImageList = []
        for i, wi in enumerate(ImagesToTransform):
            fi = __WarpedImageUsingCoords(DstSpace_coords, SrcSpace_coords, FixedImageArea, wi, area, cval=cval[i])
            FixedImageList.append(fi)
            
        del SrcSpace_coords
        del DstSpace_coords
        
        return FixedImageList
    else:
        return __WarpedImageUsingCoords(SrcSpace_coords, DstSpace_coords, FixedImageArea, ImagesToTransform, area, cval=cval[0])

def ParameterToStosTransform(transformData):
    '''
    :param object transformData: Either a full path to a .stos file, a stosfile, or a transform object
    :return: A transform
    '''
    stostransform = None 
    
    if isinstance(transformData, str):
        if not os.path.exists(transformData):
            raise ValueError("transformData is not a valid path to a .stos file %s" % transformData)
        stos = StosFile.Load(transformData)
        stostransform = factory.LoadTransform(stos.Transform)
    elif isinstance(transformData, StosFile):
        stos = transformData.Transform
        stostransform = factory.LoadTransform(stos.Transform)
    elif isinstance(transformData, transformbase.Base):
        stostransform = transformData
        
    return stostransform

def TransformStos(transformData, OutputFilename=None, fixedImageFilename=None, warpedImageFilename=None, scalar=1.0, CropUndefined=False):
    '''Assembles an image based on the passed transform.
    :param bool fixedImageFilename: Image describing the size we want the warped image to fill
    :param bool warpedImageFilename: Image we will warp into fixed space
    :param float scalar: Amount to scale the transform before passing the image through
    :param bool CropUndefined: If true do exclude areas outside the convex hull of the transform, if it exists
    :param bool Dicreet: True causes points outside the defined transform region to be clipped instead of interpolated
    :return: transformed image
    '''

    stos = None
    stostransform = ParameterToStosTransform(transformData)

    if CropUndefined:
        stostransform = triangulation.Triangulation(pointpairs=stostransform.points)

    if fixedImageFilename is None:
        if stos is None:
            return None

        fixedImageFilename = stos.ControlImageFullPath

    if warpedImageFilename is None:
        if stos is None:
            return None

        warpedImageFilename = stos.MappedImageFullPath

    fixedImageSize = core.GetImageSize(fixedImageFilename)
    fixedImageShape = np.array(fixedImageSize) * scalar
    warpedImage = core.LoadImage(warpedImageFilename)

    stostransform.points = stostransform.points * scalar

    warpedImage = TransformImage(stostransform, fixedImageShape, warpedImage)

    if not OutputFilename is None:
        imsave(OutputFilename, warpedImage, cmap='gray')

    return warpedImage


def TransformImage(transform, fixedImageShape, warpedImage):
    '''Cut image into tiles, assemble small chunks'''

    tilesize = [2048, 2048]

    height = int(fixedImageShape[0])
    width = int(fixedImageShape[1])
 
    # print('\nConverting image to ' + str(self.NumCols) + "x" + str(self.NumRows) + ' grid of OpenGL textures')

    tasks = []
 
    grid_shape = core.TileGridShape(warpedImage.shape, tilesize)
    
    if np.all(grid_shape == np.array([1,1])):
        #Single threaded
        return WarpedImageToFixedSpace(transform, fixedImageShape, warpedImage, botleft=np.array([0, 0]), area=fixedImageShape)
    else:
        outputImage = np.zeros(fixedImageShape, dtype=np.float32)
        sharedWarpedImage = core.npArrayToReadOnlySharedArray(warpedImage)
        mpool = pools.GetGlobalMultithreadingPool()
        
    
        for iY in range(0, height, int(tilesize[0])):
    
            end_iY = iY + tilesize[0]
            if end_iY > height:
                end_iY = height
    
            for iX in range(0, width, int(tilesize[1])):
    
                end_iX = iX + tilesize[1]
                if end_iX > width:
                    end_iX = width
    
                task = mpool.add_task(str(iX) + "x_" + str(iY) + "y", WarpedImageToFixedSpace, transform, fixedImageShape, sharedWarpedImage, botleft=[iY, iX], area=[end_iY - iY, end_iX - iX])
                task.iY = iY
                task.end_iY = end_iY
                task.iX = iX
                task.end_iX = end_iX
    
                tasks.append(task)
    
                # registeredTile = WarpedImageToFixedSpace(transform, fixedImageShape, warpedImage, botleft=[iY, iX], area=[end_iY - iY, end_iX - iX])
                # outputImage[iY:end_iY, iX:end_iX] = registeredTile
        mpool.wait_completion()
        
        for task in tasks:
            registeredTile = task.wait_return()
            outputImage[task.iY:task.end_iY, task.iX:task.end_iX] = registeredTile
        
        
        del sharedWarpedImage


    return outputImage
