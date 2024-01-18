"""
Created on Apr 22, 2013


"""

import os
import warnings

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy
    import cupyx.scipy.ndimage
except ModuleNotFoundError:
    import cupy_thunk as cp
    import cupyx_thunk as cupyx
except ImportError:
    import cupy_thunk as cp
    import cupyx_thunk as cupyx
    
import numpy as np
from numpy.typing import NDArray
import scipy

import nornir_pools
import nornir_imageregistration
from nornir_imageregistration.transforms import ITransform, factory, triangulation
from nornir_imageregistration.transforms.utils import InvalidIndicies


def GetROICoords(botleft: tuple[float, float] | NDArray, area: tuple[float, float] | NDArray) -> NDArray[np.floating]:
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
    xp = cp if use_cp else np 
    
    x_range = xp.arange(botleft[1], botleft[1] + area[1], dtype=np.int32)
    y_range = xp.arange(botleft[0], botleft[0] + area[0], dtype=np.int32)

    # Numpy arange sometimes accidentally adds an extra value to the array due to rounding error, remove the extra element if needed
    if len(x_range) > area[1]:
        x_range = x_range[:int(area[1])]

    if len(y_range) > area[0]:
        y_range = y_range[:int(area[0])]

    i_y, i_x = xp.meshgrid(y_range, x_range, sparse=False, indexing='ij')
    
    if use_cp:
        coordArray = xp.vstack((i_y.ravel(), i_x.ravel())).transpose()
    else:
        coordArray = xp.vstack((i_y.flat, i_x.flat)).transpose()
        
    del i_y
    del i_x
    del x_range
    del y_range

    return coordArray
 

def write_to_source_roi_coords(transform: ITransform,
                               botleft: tuple[float, float] | NDArray,
                               area: tuple[float, float] | NDArray,
                               extrapolate: bool = False) -> tuple[NDArray, NDArray]:
    """
    This function is used to generate coordinates to transform image data in target space backwards into source space.


    Given a transform and a region in source space, create uniform integer coordinates over the region of interest in source space
    for each pixel.  Then run an forward transform to determine those coordinates in target space.  The target space
    coordinates will be used later to interpolate pixel values for each integer pixel valued destination space coordinates.

    :param extrapolate:
    :param transform transform: The transform used to map points between fixed and mapped space
    :param botleft: The (Y,X) coordinates of the bottom left corner in source space
    :param area: The (Height, Width) of the region of interest coordinates.
    :return: (read_space_coords, write_space_coords)
    """

    write_space_coords = GetROICoords(botleft, area)

    read_space_coords = transform.Transform(write_space_coords, extrapolate=extrapolate).astype(np.float32, copy=False)
    (valid_read_space_coords, invalid_coords_mask, valid_coords_mask) = InvalidIndicies(read_space_coords)

    del read_space_coords

    if nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy:
        valid_write_space_coords = write_space_coords[valid_coords_mask,:]
    else:
        valid_write_space_coords = np.delete(write_space_coords, invalid_coords_mask, axis=0)
    # valid_write_space_coords = valid_write_space_coords  # - botleft

    return valid_read_space_coords, valid_write_space_coords


def write_to_target_roi_coords(transform: ITransform,
                               botleft: tuple[float, float] | NDArray,
                               area: tuple[float, float] | NDArray,
                               extrapolate: bool = False) -> tuple[NDArray, NDArray]:
    """
    This function is used to generate coordinates to transform image data in source space forward into target space.

    Given a transform and a region in target space, create uniform integer coordinates over the region of interest in source
    space for each pixel.  Then run a inverse transform to map target coordinates back in source space.  The source
    space coordinates will be used later to interpolate pixel values for each integer pixel valued destination target
    coordinates.

    :param extrapolate:
    :param transform transform: The transform used to map points between fixed and mapped space
    :param botleft: The (Y,X) coordinates of the bottom left corner in target space
    :param area: The (Height, Width) of the region of interest
e coordinates.
    :return: (read_space_coords, write_space_coords)
    """

    write_space_coords = GetROICoords(botleft, area)

    read_space_coords = transform.InverseTransform(write_space_coords, extrapolate=extrapolate).astype(np.float32, copy=False)
    (valid_read_space_coords, invalid_coords_mask, valid_coords_mask) = InvalidIndicies(read_space_coords)

    del read_space_coords

    if nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy:
        valid_write_space_coords = write_space_coords[valid_coords_mask,:]
    else:
        valid_write_space_coords = np.delete(write_space_coords, invalid_coords_mask, axis=0)
    # valid_write_space_coords = valid_write_space_coords  # - botleft

    return valid_read_space_coords, valid_write_space_coords
 

def get_valid_coords(coords: NDArray, image_shape, origin=(0, 0), area=None) -> tuple[NDArray, NDArray]:
    """Given an Nx2 array off image coordinates, remove the coordinates that
    fall outside the image_shape boundaries.
    :param coords: Nx2 array of image coordinates
    :param image_shape: 1x2 array of image dimensions
    :param origin: 1x2 array with minimum valid coordinate
    :parm area: 1x2 array of expected area, which may exceed image_shape.  coords will be cropped to whatever is less
    :return: The coordinates greater than or equal to origin and less than origin + area and a mask indicating (== True) which coordinates met the criteria
    """

    xp = cp.get_array_module(coords)
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    if isinstance(origin, int):
        adjusted_origin = np.array((origin, origin), dtype=np.int32)
    elif isinstance(origin, tuple):
        adjusted_origin = np.array(origin)
    elif isinstance(origin, np.ndarray) or isinstance(origin, cp.ndarray):
        adjusted_origin = origin.copy()
    else:
        raise ValueError("Unexpected type passed to origin")

    if isinstance(area, int):
        adjusted_area = np.array((area, area), dtype=np.int32)
    elif isinstance(area, tuple):
        adjusted_area = np.array(area)
    elif isinstance(area, np.ndarray) or isinstance(area, cp.ndarray):
        adjusted_area = area.copy()
    elif area is None:
        adjusted_area = xp.copy(image_shape)
    else:
        raise ValueError("Unexpected type passed to area")

    origin = cp.asarray(origin) if use_cp and not isinstance(origin, cp.ndarray) else origin
    image_shape = cp.asarray(image_shape) if use_cp and not isinstance(image_shape, cp.ndarray) else image_shape

    adjusted_origin = cp.asarray(adjusted_origin) if use_cp and not isinstance(adjusted_origin,
                                                                               cp.ndarray) else adjusted_origin
    adjusted_area = cp.asarray(adjusted_area) if use_cp and not isinstance(adjusted_area,
                                                                           cp.ndarray) else adjusted_area

    adjust_area_mask = adjusted_origin < 0
    adjusted_area[adjust_area_mask] = adjusted_area[adjust_area_mask] + adjusted_origin[adjust_area_mask]
    adjusted_origin[adjust_area_mask] = 0
    # Warning - Clement: adjusted_area and adjusted_origin are not being used below...

    valid_coords_mask = xp.logical_and(xp.min(coords >= origin, 1), xp.min(coords < image_shape, 1))
    if use_cp:
        valid_adjusted_coords = coords[valid_coords_mask, :]
    else:
        valid_adjusted_coords = xp.delete(coords, xp.logical_not(valid_coords_mask), 0)  # coords[valid_coords_mask, :]

    return valid_adjusted_coords, valid_coords_mask


def _CropImageToFitCoords(input_image: NDArray, coordinates: NDArray, padding: int, cval=0) -> tuple[
    NDArray[np.floating], NDArray[np.integer]]:
    """For large images we only need a specific range of coordinates from the image.  However Scipy calls such as map_coordinates will
       send the entire image through a spline_filter first.  To avoid this we crop the image with a padding of one and adjust the
       coordinates appropriately
       :param ndarray input_image: image we will be extracting data from at the specified coordinates
       :param ndarray coordinates: Nx2 array of points indexing into the image
       :param float cval: Value to use for regions outside the existing image when padding
       :return: (cropped_image, translated_coordinates, coordinate_mask) Returns the cropped image, the coordinates translated into the cropped image, and a mask set to False for any coordinates that did not fit within the image boundaries
       """

    xp = cp.get_array_module(input_image)

    bottom_left = xp.floor(xp.min(coordinates, 0))
    # bottom_left[bottom_left < 0] = 0
    top_right = xp.ceil(xp.max(coordinates, 0))
    # top_right_out_of_bounds = top_right >= input_image.shape
    # top_right[top_right >= input_image.shape] = np.min(top_right, input_image.shape)

    filtered_coordinates, coord_mask = get_valid_coords(coordinates, input_image.shape, origin=(0, 0),
                                                        area=(top_right - bottom_left) + 1)

    if xp.all(coord_mask == False):
        # No mappable coords, just return an empty image
        return xp.empty((0, 0)), xp.empty((0, 2)), coord_mask

    # Recalculate boundaries to account for filtered coords
    filtered_bottom_left = xp.floor(xp.min(filtered_coordinates, 0))
    filtered_top_right = xp.ceil(xp.max(filtered_coordinates, 0))

    padded_bottom_left = filtered_bottom_left - padding
    # padded_top_right = filtered_top_right + padding

    Width = int(filtered_top_right[1] - filtered_bottom_left[1]) + 1 + (padding * 2)
    Height = int(filtered_top_right[0] - filtered_bottom_left[0]) + 1 + (padding * 2)

    cropped_image = nornir_imageregistration.CropImage(input_image, Xo=int(padded_bottom_left[1]),
                                                       Yo=int(padded_bottom_left[0]),
                                                       Width=Width, Height=Height, cval=cval)

    translated_coordinates = (filtered_coordinates - filtered_bottom_left) + padding

    return cropped_image, translated_coordinates, coord_mask


def my_cheesy_map_coordinates(image, coords):
    floor_coords = np.floor(coords).astype(int, copy=False)
    return image[floor_coords]


def _TransformImageUsingCoords(target_coords: NDArray,
                               source_coords: NDArray,
                               source_image: NDArray,
                               output_origin: NDArray[np.integer] | tuple[int, int] | None,
                               output_area: NDArray[np.integer] | tuple[float, float],
                               cval=0,
                               return_shared_memory: bool = False):
    """Use the passed coordinates to create a warped image
    :Param fixed_coords: 2D coordinates in fixed space
    :Param warped_coords: 2D coordinates in warped space
    :Param FixedImageArea: Dimensions of fixed space
    :Param WarpedImage: Image to read pixel values from while creating fixed space images
    :Param output_origin: Origin, in target coordinate space, of the output image.  Use this to translate the target_coords to the desired location in the output image.  If None, the origin is the minimum target_coord.
    :Param output_area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.
    :param use_shared_memory: If true, create and write output to a shared memory array
    :param use_cp: Use CuPy library for GPU processing
    """

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
    xp = cp if use_cp else np
    sp = cupyx.scipy if use_cp else scipy

    if output_origin is None:
        output_origin = target_coords.min(0)
    
    output_area = nornir_imageregistration.EnsurePointsAre1DNumpyArray(output_area, dtype=np.int32)
    output_origin = nornir_imageregistration.EnsurePointsAre1DArray(output_origin, dtype=np.int32)

    if use_cp: 
        target_coords = nornir_imageregistration.EnsurePointsAre2DArray(target_coords) 
        source_coords = nornir_imageregistration.EnsurePointsAre2DArray(source_coords)
        if not isinstance(source_image, cp.ndarray):
            source_image = cp.asarray(source_image)

    if output_origin.dtype != np.int32:
        output_origin = xp.asarray(output_origin, dtype=np.int32)
    if output_area.dtype != np.int32:
        output_area = xp.asarray(output_area, dtype=np.int32)

    if source_coords.shape[0] == 0:
        # No points transformed into the requested area, return empty image
        transformedImage = xp.full(output_area, cval, dtype=source_image.dtype)
        return transformedImage

    # Convert to a type the interpolation.map_coordinates supports
    original_dtype = source_image.dtype

    inbounds_target_coords = target_coords - output_origin  # If tempted to use target_coords -= output_origin, don't.  It modifies input parameter and if called in a loop the next iteration will fail
    del target_coords
    # Remove coordinates that fall outside the output region
    # inbounds_target_coords, inbounds_target_coord_mask = get_valid_coords(target_coords, output_area)
    # inbounds_source_coords = source_coords[inbounds_target_coord_mask]

    subroi_warpedImage = None
    # For large images we only need a specific range of the image, but the entire image is passed through a spline filter by map_coordinates
    # In this case use only a subset of the warpedimage
    if np.prod(source_image.shape) > source_coords.shape[0]:
        # if not area[0] == FixedImageArea[0] and area[1] == FixedImageArea[1]:
        # if area[0] <= FixedImageArea[0] or area[1] <= FixedImageArea[1]:
        (subroi_warpedImage, filtered_source_coords, source_coord_mask) = _CropImageToFitCoords(source_image,
                                                                                                source_coords,
                                                                                                padding=0, cval=cval)
        # subroi_warpedImage[] #Replace NaN entries with random values
        # Remove target coords that match removed source_coords
        inbounds_target_coords = inbounds_target_coords[source_coord_mask]

        # source_coords, source_coord_mask = get_valid_coords(coords=source_coords, image_shape=subroi_warpedImage.shape, origin=(1,1), area=(subroi_warpedImage.shape) - 1) #Remove one for padding
        if subroi_warpedImage.shape[0] == 0 or subroi_warpedImage.shape[1] == 0:
            # No points transformed into the requested area, return empty area
            if return_shared_memory:
                output_shared_mem_meta, outputImage = nornir_imageregistration.create_shared_memory_array(output_area,
                                                                                                          dtype=original_dtype)
                outputImage.fill(cval)
                return output_shared_mem_meta
            else:
                if use_cp:
                    return xp.full(output_area, cval, dtype=original_dtype).get()
                else:
                    return xp.full(output_area, cval, dtype=original_dtype)

        del source_image
    else:
        filtered_source_coords = source_coords
        # inbounds_target_coords = inbounds_target_coords
        subroi_warpedImage = source_image

    # del inbounds_target_coords

    # Use a dtype interpolation.map_coordinates supports
    if subroi_warpedImage.dtype == np.float16:
        subroi_warpedImage = subroi_warpedImage.astype(np.float32, copy=False)

    # Rounding helped solve a problem with image shift when using the CloughTocher interpolator with an identity function
    # filtered_source_coords = np.around(filtered_source_coords, 3)

    # TODO: Order appears to not matter so setting to zero may help
    # outputImage = interpolation.map_coordinates(subroi_warpedImage, warped_coords.transpose(), mode='constant', order=3, cval=cval)
    order = 1 if xp.any(xp.isnan(
        subroi_warpedImage)) or subroi_warpedImage.dtype == bool else 3  # Any interpolation of NaN returns NaN so ensure we use order=1 when using NaN as a fill value
    outputValues = sp.ndimage.map_coordinates(subroi_warpedImage,
                                              filtered_source_coords.transpose(),
                                              mode='constant',
                                              order=order,
                                              cval=cval,
                                              prefilter=True).astype(original_dtype, copy=False)

    del filtered_source_coords
    # outputvalaues = my_cheesy_map_coordinates(subroi_warpedImage, filtered_source_coords.transpose())

    # outputImage = np.full(output_area, cval, dtype=original_dtype) #Use same DType as source_image for output, we are past the call to map_coordinates that cannot handle float16
    output_shared_mem_meta = None
    if return_shared_memory:
        output_shared_mem_meta, outputImage = nornir_imageregistration.create_shared_memory_array(output_area if not use_cp else output_area.get(),
                                                                                                  dtype=original_dtype)
        outputImage.fill(cval)
    else: 
        outputImage = xp.full(output_area, cval, dtype=original_dtype) # Use same DType as source_image for output, we are past the call to map_coordinates that cannot handle float16
        
    target_coords_flat = nornir_imageregistration.ravel_index(inbounds_target_coords, outputImage.shape).astype(
        np.int32, copy=False)
    # del filtered_target_coords

    # Note - Clement: flat assignment doesn't work with cupy
    if cp.get_array_module(outputImage) == cp:
        start_shape = outputImage.shape
        outputImage = outputImage.ravel()
        outputImage[target_coords_flat] = outputValues
        outputImage = outputImage.reshape(start_shape)
    else:
        outputImage.flat[target_coords_flat] = outputValues
    # outputImage[fixed_coords] = outputValues

    # Scipy's interpolation can infer values slightly outside the source data's range.  We clip the result to fit in the original range of values
    xp.clip(outputImage, a_min=subroi_warpedImage.min(), a_max=subroi_warpedImage.max(), out=outputImage)

    # outputImage = outputImage.reshape(area)

    #     if fixed_coords.shape[0] == np.prod(area):
    #         # All coordinates mapped, so we can return the output warped image as is.
    #         outputImage = outputImage.reshape(area)
    #         return outputImage
    #     else:
    #         # Not all coordinates mapped, create an image of the correct size and place the warped image inside it.
    #         transformedImage = np.full((area), cval, dtype=outputImage.dtype)
    #         fixed_coords_rounded = np.round(fixed_coords).astype(dtype=np.int64)
    #         transformedImage[fixed_coords_rounded[:, 0], fixed_coords_rounded[:, 1]] = outputImage
    #         return transformedImage
    if return_shared_memory:
        return output_shared_mem_meta
    else:   
        return outputImage


def _ReplaceFilesWithImages(listImages: list[str] | list[NDArray] | NDArray | str):
    """Replace any filepath strings in the passed parameter with loaded images."""

    if isinstance(listImages, list):
        for i, value in enumerate(listImages):
            listImages[i] = nornir_imageregistration.ImageParamToImageArray(value)
    else:
        return nornir_imageregistration.ImageParamToImageArray(listImages)

    return listImages


def FixedImageToWarpedSpace(transform: ITransform, DataToTransform, botleft=None, area=None, cval=None,
                            extrapolate=False):
    warnings.warn("WarpedImageToFixedSpace should be replaced with SourceImageToTargetSpace", DeprecationWarning)
    return TargetImageToSourceSpace(transform, DataToTransform, output_botleft=None, output_area=None, cval=None,
                                    extrapolate=False)


def TargetImageToSourceSpace(transform: ITransform,
                             DataToTransform,
                             output_botleft: NDArray | tuple[float, float] | None = None,
                             output_area: NDArray | tuple[float, float] | None = None,
                             cval=None, extrapolate: bool = False, return_shared_memory: bool = False):
    """Warps every image in the DataToTransform list using the provided transform.
    :param transform: transform to pass fixed space coordinates through to obtain warped space coordinates
    :param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :param output_botleft: Origin of region to map data into, in source Space coordinates
    :param output_area: Area of region to map data into, in source space coordinates
    :param cval: Value to place in unmappable regions, defaults to zero.
    :param bool extrapolate: If true map points that fall outside the bounding box of the transform
    """

    ImagesToTransform = _ReplaceFilesWithImages(DataToTransform)

    if output_botleft is None:
        output_botleft = (0, 0)

    if output_area is None:
        firstImage = ImagesToTransform
        if isinstance(firstImage, list):
            firstImage = firstImage[0]
            raise ValueError("Area calculation is not implemented for lists of transforms, but could be")

        bounds = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), firstImage.shape)
        source_corners = transform.InverseTransform(bounds.Corners)
        output_area = np.ravel((np.max(source_corners, 0) - np.min(source_corners, 0)) + 1)

    if cval is None:
        cval = [0] * len(DataToTransform)

    if not isinstance(cval, list):
        cval = [cval] * len(DataToTransform)

    # This sometimes appears backwards, but what we are doing is defining the region in source space we want to obtain
    # values for, then determining the target space coordinates for each pixel to fill the source image region.  Then we map values
    # to each pixel using the target space coordinates
    (roi_read_coords, roi_write_coords) = write_to_source_roi_coords(transform, output_botleft, output_area,
                                                                     extrapolate=extrapolate)

    if isinstance(ImagesToTransform, list):
        if not isinstance(cval, list):
            cval = [cval] * len(DataToTransform)

        output_list = []
        for i, wi in enumerate(ImagesToTransform):
            fi = _TransformImageUsingCoords(roi_write_coords, roi_read_coords, wi, output_origin=output_botleft,
                                            output_area=output_area,
                                            cval=cval[i], return_shared_memory=return_shared_memory)
            output_list.append(fi)

        return output_list
    else:
        return _TransformImageUsingCoords(roi_write_coords, roi_read_coords, ImagesToTransform,
                                          output_origin=output_botleft, output_area=output_area,
                                          cval=cval[0], return_shared_memory=return_shared_memory)


def WarpedImageToFixedSpace(transform: ITransform, DataToTransform, botleft=None, area=None, cval=None,
                            extrapolate=False):
    warnings.warn("WarpedImageToFixedSpace should be replaced with TargetImageToSourceSpace", DeprecationWarning)
    return SourceImageToTargetSpace(transform, DataToTransform, output_botleft=None, output_area=None, cval=None,
                                    extrapolate=False)


def SourceImageToTargetSpace(transform: ITransform,
                             DataToTransform,
                             output_botleft: NDArray | tuple[float, float] | None = None,
                             output_area: NDArray | tuple[float, float] | None = None,
                             cval=None, extrapolate=False, return_shared_memory: bool = False):
    """Warps every image in the DataToTransform list using the provided transform.
    :param transform: transform to pass warped space coordinates through to obtain fixed space coordinates
    :param output_shape: shape of the output image
    :param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :param output_botleft: Origin of region to map data into, in target space coordinates
    :param output_area: Area of region to map data into, in target space coordinates
    :param cval: Value to place in unmappable regions, defaults to zero.
    :Param transform: transform to pass warped space coordinates through to obtain fixed space coordinates
    :Param FixedImageArea: Size of fixed space region to map pixels into
    :Param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :Param botleft: Origin of region to map
    :Param area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.
    :param bool extrapolate: If true map points that fall outside the bounding box of the transform
    """

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
    ImagesToTransform = _ReplaceFilesWithImages(DataToTransform)

    if output_botleft is None:
        output_botleft = (0, 0)

    if output_area is None:
        firstImage = ImagesToTransform
        if isinstance(firstImage, list):
            firstImage = firstImage[0]
            raise ValueError("Area calculation is not implemented for lists of transforms, but could be")

        bounds = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), firstImage.shape)
        target_corners = transform.Transform(bounds.Corners)
        target_bounds = nornir_imageregistration.Rectangle.CreateBoundingRectangleForPoints(target_corners)
        rounded_target_bounds = nornir_imageregistration.Rectangle.SafeRound(target_bounds)
        # output_area = np.ravel((np.max(target_corners, 0) - np.min(target_corners, 0)))
        # output_area = np.ceil(output_area)
        output_area = rounded_target_bounds.Dimensions

    if cval is None:
        cval = 0

    # output_botleft = cp.asarray(output_botleft) if use_cp and not isinstance(output_botleft,
    #                                                                          cp.ndarray) else output_botleft
    # output_area = cp.asarray(output_area) if use_cp and not isinstance(output_area, cp.ndarray) else output_area

    # This sometimes appears backwards, but what we are doing is defining the region in target space we want to obtain
    # values for, then determining the Source space coordinates for each pixel in the target image.  Then we map values
    # to each pixel using the source space coordinates

    # timer.Start('write_to_target_roi_coords')

    (roi_read_coords, roi_write_coords) = write_to_target_roi_coords(transform, output_botleft, output_area,
                                                                    extrapolate=extrapolate)
    # (roi_read_coords, roi_write_coords) = write_to_target_roi_coords(transform, output_botleft, output_area,
    #                                                                  extrapolate=extrapolate, use_cp=use_cp)

    # timer.End('write_to_target_roi_coords', True)

    # roi_read_coords = cp.asarray(roi_read_coords) if use_cp and not isinstance(roi_read_coords, cp.ndarray) else roi_read_coords
    # roi_write_coords = cp.asarray(roi_write_coords) if use_cp and not isinstance(roi_write_coords, cp.ndarray) else roi_write_coords

    if isinstance(ImagesToTransform, list):
        if not isinstance(cval, list):
            cval = [cval] * len(DataToTransform)

        output_list = []
        for i, wi in enumerate(ImagesToTransform):
            fi = _TransformImageUsingCoords(roi_write_coords, roi_read_coords, wi, output_origin=output_botleft,
                                            output_area=output_area, cval=cval[i],
                                            return_shared_memory=return_shared_memory)
            output_list.append(fi)
            # nornir_imageregistration.close_shared_memory(DataToTransform[i])

        return output_list
    else:
        result = _TransformImageUsingCoords(roi_write_coords, roi_read_coords, ImagesToTransform,
                                            output_origin=output_botleft, output_area=output_area, cval=cval,
                                            return_shared_memory=return_shared_memory)
        # nornir_imageregistration.close_shared_memory(DataToTransform)
        return result


def ParameterToStosTransform(transformData: str | NDArray | nornir_imageregistration.ITransform):
    """
    :param object transformData: Either a full path to a .stos file, a stosfile, or a transform object
    :param use_cp: Use CuPy library for GPU processing
    :return: A transform
    """
    stostransform = None

    if isinstance(transformData, str):
        if not os.path.exists(transformData):
            raise ValueError("transformData is not a valid path to a .stos file %s" % transformData)
        stos = nornir_imageregistration.StosFile.Load(transformData)
        stostransform = factory.LoadTransform(stos.Transform)
    elif isinstance(transformData, nornir_imageregistration.StosFile):
        stos = transformData.Transform
        stostransform = factory.LoadTransform(stos.Transform)
    elif isinstance(transformData, ITransform):
        stostransform = transformData

    return stostransform


def TransformStos(transformData, OutputFilename: str | None = None, fixedImage=None, warpedImage=None,
                  scalar: float = 1.0, CropUndefined: bool = False):
    """Assembles an image based on the passed transform.
    :param transformData:
    :param OutputFilename:
    :param str fixedImage: Image describing the size we want the warped image to fill, either a string or ndarray
    :param str warpedImage: Image we will warp into fixed space, either a string or ndarray
    :param float scalar: Amount to scale the transform before passing the image through
    :param bool CropUndefined: If true exclude areas outside the convex hull of the transform, if it exists
    """

    stos = None
    stostransform = ParameterToStosTransform(transformData)

    if fixedImage is None:
        if stos is None:
            return None

        fixedImage = stos.ControlImageFullPath

    if warpedImage is None:
        if stos is None:
            return None

        warpedImage = stos.MappedImageFullPath

    fixedImageSize = nornir_imageregistration.GetImageSize(fixedImage)
    fixedImageShape = np.array(fixedImageSize) * scalar
    warpedImage = nornir_imageregistration.ImageParamToImageArray(warpedImage)

    if isinstance(stostransform, nornir_imageregistration.transforms.ITransformScaling) is False:
        raise NotImplemented(f"Cannot scale transform that does not implement ITransformScaling {transformData}")

    stostransform.Scale(scalar)

    # warpedImage_shared_mem = TransformImage(stostransform, fixedImageShape, warpedImage, CropUndefined)

    if not OutputFilename is None:
        nornir_imageregistration.SaveImage(OutputFilename, warpedImage.get() if cp.get_array_module(warpedImage) == cp else warpedImage, cmap='gray', bpp=8)

    return warpedImage


def TransformImage(transform: ITransform,
                   fixedImageShape: tuple[float, float] | NDArray,
                   warpedImage: NDArray, CropUndefined: bool,
                   use_cp: bool = False) -> NDArray:
    """
    Cut image into tiles, assemble small chunks
    :param transform: Transform to apply to point to map from warped image to fixed space
    :param fixedImageShape: Width and Height of the image to create
    :param warpedImage: Image to transform to fixed space
    :param CropUndefined: If true exclude areas outside the convex hull of the transform, if it exists
    :param bool use_cp: Use CuPy library for GPU processing
    :return: An ndimage array of the transformed image
    """

    if CropUndefined:
        transform = triangulation.Triangulation(pointpairs=transform.points)

    tilesize = [2048, 2048]

    fixedImageShape = fixedImageShape.astype(dtype=np.int64, copy=False)
    height = int(fixedImageShape[0])
    width = int(fixedImageShape[1])

    # print('\nConverting image to ' + str(self.NumCols) + "x" + str(self.NumRows) + ' grid of OpenGL textures')

    tasks = []

    grid_shape = nornir_imageregistration.TileGridShape(warpedImage.shape, tilesize)

    if np.all(grid_shape == np.array([1, 1])):
        # Single threaded
        return SourceImageToTargetSpace(transform, warpedImage, output_botleft=np.array([0, 0]),
                                        output_area=fixedImageShape, extrapolate=not CropUndefined)
    else:
        outputImage = np.zeros(fixedImageShape, dtype=warpedImage.dtype)
        sharedwarpedimage_metadata, sharedWarpedImage = nornir_imageregistration.npArrayToSharedArray(warpedImage)
        mpool = nornir_pools.GetGlobalMultithreadingPool()

        try:

            for iY in range(0, height, int(tilesize[0])):

                end_iY = iY + tilesize[0]
                if end_iY > height:
                    end_iY = height

                for iX in range(0, width, int(tilesize[1])):

                    end_iX = iX + tilesize[1]
                    if end_iX > width:
                        end_iX = width

                    task = mpool.add_task(str(iX) + "x_" + str(iY) + "y", SourceImageToTargetSpace, transform,
                                          sharedwarpedimage_metadata, output_botleft=[iY, iX],
                                          output_area=[end_iY - iY, end_iX - iX], extrapolate=not CropUndefined,
                                          return_shared_memory=False)
                    task.iY = iY
                    task.end_iY = end_iY
                    task.iX = iX
                    task.end_iX = end_iX

                    tasks.append(task)

                    # registeredTile = WarpedImageToFixedSpace(transform, fixedImageShape, warpedImage, botleft=[iY, iX], area=[end_iY - iY, end_iX - iX])
                    # outputImage[iY:end_iY, iX:end_iX] = registeredTile
            mpool.wait_completion()

            for task in tasks:
                result = task.wait_return()
                registered_tile = nornir_imageregistration.ImageParamToImageArray(result)
                outputImage[task.iY:task.end_iY, task.iX:task.end_iX] = registered_tile
                nornir_imageregistration.unlink_shared_memory(result)
        finally:
            nornir_imageregistration.unlink_shared_memory(sharedwarpedimage_metadata)
            del sharedWarpedImage

    return outputImage
