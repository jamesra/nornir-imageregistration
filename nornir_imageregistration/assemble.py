'''
Created on Apr 22, 2013


'''


import core
import numpy as np
import nornir_shared.prettyoutput as PrettyOutput
import nornir_shared.images as images
from scipy.ndimage import interpolation
from   nornir_imageregistration.transforms import factory, triangulation
from   nornir_imageregistration.transforms.utils import InvalidIndicies
import nornir_imageregistration.transforms.base as transformbase
from nornir_imageregistration.files.stosfile import StosFile
import os
import nornir_pools as pools

from matplotlib.pyplot import imsave


def ROI(botleft, area):
    x_range = range(int(botleft[1]), int(botleft[1]) + int(area[1]))
    y_range = range(int(botleft[0]), int(botleft[0]) + int(area[0]))

    i_y, i_x = np.meshgrid(y_range, x_range, sparse=False, indexing='ij')

    coordArray = np.vstack((i_y.flat, i_x.flat)).transpose()

    return coordArray

def TransformROI(transform, botleft, area):
    '''
    Apply a transform to a region of interest within an image. Center and area are in fixed space
    
    :param transform transform: The transform used to map points between fixed and mapped space
    :param 1x2_array botleft: The (Y,X) coordinates of the bottom left corner
    :param 1x2_array area: The (Height, Width) of the region of interest
    :return: Tuple of arrays.  First array is fixed space coordinates.  Second array is warped space coordinates.
    :rtype: tuple(Nx2 array,Nx2 array)
    '''

    fixed_coordArray = ROI(botleft, area)

    warped_coordArray = transform.InverseTransform(fixed_coordArray)
    (valid_warped_coordArray, InvalidIndiciesList) = InvalidIndicies(warped_coordArray)

    del warped_coordArray

    valid_fixed_coordArray = np.delete(fixed_coordArray, InvalidIndiciesList, axis=0)
    valid_fixed_coordArray = valid_fixed_coordArray - botleft

    return (valid_fixed_coordArray, valid_warped_coordArray)


def ExtractRegion(image, botleft=None, area=None):
    '''
    Extract a region from an image
    
    :param ndarray image: Source image
    :param 1x2_array botleft: The (Y,X) coordinates of the bottom left corner
    :param 1x2_array area: The (Height, Width) of the region of interest
    :return: Image of requested region
    :rtype: ndarray
    
    '''
    if botleft is None:
        botleft = (0, 0)

    if area is None:
        area = image.shape

    coords = ROI(botleft, area)

    transformedImage = interpolation.map_coordinates(image, coords.transpose(), order=0, mode='constant')

    transformedImage = transformedImage.reshape(area)
    return transformedImage


def __ExtractRegion(image, botleft, area):
    print "Deprecated __ExtractRegion call being used"
    return ExtractRegion(image, botleft, area)


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
        area = area.asarray(dtype=np.uint64)

    if(warped_coords.shape[0] == 0):
        # No points transformed into the requested area, return empty image
        transformedImage = np.zeros((area), dtype=WarpedImage.dtype)
        return transformedImage

    subroi_warpedImage = WarpedImage
    if not area[0] == FixedImageArea[0] and area[1] == FixedImageArea[1]:
        if area[0] <= FixedImageArea[0] or area[1] <= FixedImageArea[1]:
            minCoord = np.floor(np.min(warped_coords, 0)) - np.array([1, 1])
            maxCoord = np.ceil(np.max(warped_coords, 0)) + np.array([1, 1])

            subroi_warpedImage = __ExtractRegion(WarpedImage, minCoord, (maxCoord - minCoord))
            warped_coords = warped_coords - minCoord

    warpedImage = interpolation.map_coordinates(subroi_warpedImage, warped_coords.transpose(), mode='constant', order=2, cval=cval)
    if fixed_coords.shape[0] == np.prod(area):
        # All coordinates mapped, so we can return the output warped image as is.
        warpedImage = warpedImage.reshape(area)
        return warpedImage
    else:
        # Not all coordinates mapped, create an image of the correct size and place the warped image inside it.
        transformedImage = np.zeros((area), dtype=WarpedImage.dtype)
        fixed_coords = np.asarray(np.round(fixed_coords), dtype=np.int64)
        transformedImage[fixed_coords[:, 0], fixed_coords[:, 1]] = warpedImage
        return transformedImage


def WarpedImageToFixedSpace(transform, FixedImageArea, WarpedImage, botleft=None, area=None, cval=None):

    '''Warps every image in the WarpedImageList using the provided transform.
    :Param transform: transform to pass warped space coordinates through to obtain fixed space coordinates
    :Param FixedImageArea: Size of fixed space region to map pixels into
    :Param WarpedImage: Image to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.
    :Param botleft: Origin of region to map
    :Param area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.
    '''

    if botleft is None:
        botleft = (0, 0)

    if area is None:
        area = FixedImageArea

    if cval is None:
        cval = [0] * len(WarpedImage)

    if not isinstance(cval, list):
        cval = [cval] * len(WarpedImage)

    (fixed_coords, warped_coords) = TransformROI(transform, botleft, area)

    if isinstance(WarpedImage, list):
        FixedImageList = []
        for i, wi in enumerate(WarpedImage):
            fi = __WarpedImageUsingCoords(fixed_coords, warped_coords, FixedImageArea, wi, area, cval=cval[i])
            FixedImageList.append(fi)
        return FixedImageList
    else:
        return __WarpedImageUsingCoords(fixed_coords, warped_coords, FixedImageArea, WarpedImage, area, cval=cval[0])


def TransformStos(transformData, OutputFilename=None, fixedImageFilename=None, warpedImageFilename=None, scalar=1.0, CropUndefined=False):
    '''Assembles an image based on the passed transform.
       Discreet = True causes points outside the defined transform region to be clipped instead of interpolated'''

    stos = None
    if isinstance(transformData, str):
        if not os.path.exists(transformData):
            return None;

        stos = StosFile.Load(transformData)
        stostransform = factory.LoadTransform(stos.Transform)
    elif isinstance(transformData, StosFile):
        stos = transformData.Transform
        stostransform = factory.LoadTransform(stos.Transform)
    elif isinstance(transformData, transformbase.Base):
        stostransform = transformData

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

    fixedImageSize = images.GetImageSize(fixedImageFilename)
    fixedImageShape = np.array([fixedImageSize[1], fixedImageSize[0]]) * scalar
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

    outputImage = np.zeros(fixedImageShape, dtype=np.float32)

    # print('\nConverting image to ' + str(self.NumCols) + "x" + str(self.NumRows) + ' grid of OpenGL textures')

    tasks = []

    mpool = pools.GetGlobalMultithreadingPool()

    for iY in range(0, height, int(tilesize[0])):

        end_iY = iY + tilesize[0]
        if end_iY > height:
            end_iY = height

        for iX in range(0, width, int(tilesize[1])):

            end_iX = iX + tilesize[1]
            if end_iX > width:
                end_iX = width

            task = mpool.add_task(str(iX) + "x_" + str(iY) + "y", WarpedImageToFixedSpace, transform, fixedImageShape, warpedImage, botleft=[iY, iX], area=[end_iY - iY, end_iX - iX])
            task.iY = iY
            task.end_iY = end_iY
            task.iX = iX
            task.end_iX = end_iX

            tasks.append(task)

            # registeredTile = WarpedImageToFixedSpace(transform, fixedImageShape, warpedImage, botleft=[iY, iX], area=[end_iY - iY, end_iX - iX])
            # outputImage[iY:end_iY, iX:end_iX] = registeredTile

    for task in tasks:
        registeredTile = task.wait_return()
        outputImage[task.iY:task.end_iY, task.iX:task.end_iX] = registeredTile

    return outputImage
