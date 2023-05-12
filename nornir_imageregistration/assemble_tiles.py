'''
Created on Oct 28, 2013

Deals with assembling images composed of mosaics or dividing images into tiles
'''

import copy
import logging
import multiprocessing
import os
import tempfile
import threading
import time
import weakref
from typing import Iterable, Generator, Tuple, List
from numpy.typing import NDArray, DTypeLike

import nornir_imageregistration
import nornir_imageregistration.assemble as assemble
import nornir_imageregistration.transformed_image_data_temp_files
import nornir_pools
import nornir_shared.prettyoutput as prettyoutput
import numpy as np

import nornir_shared.tasktimer

# from nornir_imageregistration.files.mosaicfile import MosaicFile
# from nornir_imageregistration.mosaic import Mosaic
# import nornir_imageregistration.transforms.meshwithrbffallback as meshwithrbffallback
# import nornir_imageregistration.transforms.triangulation as triangulation
DistanceImageCache = {}

# TODO: Use atexit to delete the temporary files
# TODO: use_memmap does not work when assembling tiles on a cluster, disable for now.  Specific test is IDOCTests.test_AssembleTilesIDoc
def _use_memmap() -> bool:
    return False


nextNumpyMemMapFilenameIndex = 0


def GetProcessAndThreadUniqueString():
    '''We use the index because if the same thread makes a new tile of the same size and the original has not been garbage collected yet we get errors'''
    global nextNumpyMemMapFilenameIndex
    nextNumpyMemMapFilenameIndex += 1
    return "%d_%d_%d" % (os.getpid(), threading.get_ident(), nextNumpyMemMapFilenameIndex)


def CompositeImage(FullImage, SubImage, offset):
    minX = offset[1]
    minY = offset[0]
    maxX = minX + SubImage.shape[1]
    maxY = minY + SubImage.shape[0]

    iNonZero = SubImage > 0.0

    temp = FullImage[minY:maxY, minX:maxX]

    temp[iNonZero] = SubImage[iNonZero]

    # FullImage[minY:maxY, minX:maxX] += SubImage
    FullImage[minY:maxY, minX:maxX] = temp

    return FullImage


def CompositeImageWithZBuffer(FullImage, FullZBuffer, SubImage, SubZBuffer, offset):
    minX = int(offset[1])
    minY = int(offset[0])
    maxX = int(minX + SubImage.shape[1])
    maxY = int(minY + SubImage.shape[0])

    #print(f'CompositeImageWithZBuffer xo:{minX} yo:{minY} xMax:{maxX} yMax:{maxY}')

    if (np.array([maxY - minY, maxX - minX]) != SubZBuffer.shape).any():
        raise ValueError("Buffers do not have the same dimensions")

    if minY == maxY or minX == maxX:
        raise ValueError("Buffers have zero dimensions")

    iUpdate = FullZBuffer[minY:maxY, minX:maxX] >= SubZBuffer
    FullImage[minY:maxY, minX:maxX][iUpdate] = SubImage[iUpdate]
    FullZBuffer[minY:maxY, minX:maxX][iUpdate] = SubZBuffer[iUpdate]

    return


def CreateDistanceImage(shape, dtype=None):
    if dtype is None:
        dtype = nornir_imageregistration.default_image_dtype()

    center = [shape[0] / 2.0, shape[1] / 2.0]

    x_range = np.linspace(-center[1], center[1], shape[1])
    y_range = np.linspace(-center[0], center[0], shape[0])

    x_range = x_range * x_range
    y_range = y_range * y_range

    distance = np.empty(shape, dtype=dtype)

    for i in range(0, shape[0]):
        distance[i, :] = x_range + y_range[i]

    distance = np.sqrt(distance)

    return distance


def CreateDistanceImage2(shape, dtype=None):
    # TODO, this has some obvious optimizations available
    if dtype is None:
        dtype = nornir_imageregistration.default_image_dtype()

    # center = [shape[0] / 2.0, shape[1] / 2.0]
    shape = np.asarray(shape, dtype=np.int64)
    is_odd_shape = np.fmod(shape, 2) > 0

    half_shape = None
    if True:
        even_shape = shape.copy()
        even_shape[is_odd_shape] = even_shape[is_odd_shape] - 1
        half_shape = even_shape / 2
        half_shape = half_shape.astype(np.int64)

    y_range = None
    if not is_odd_shape[0]:
        y_range = np.linspace(0.5, half_shape[0] + 0.5, num=half_shape[0])
    else:
        half_shape[0] = half_shape[0] + 1
        y_range = np.linspace(0, half_shape[0] - 1, num=half_shape[0])

    x_range = None
    if not is_odd_shape[1]:
        x_range = np.linspace(0.5, half_shape[1] + 0.5, num=half_shape[1])
    else:
        half_shape[1] = half_shape[1] + 1
        x_range = np.linspace(0, half_shape[1] - 1, num=half_shape[1])

    x_range = x_range * x_range
    y_range = y_range * y_range

    distance = np.empty(half_shape, dtype=dtype)

    for i in range(0, half_shape[0]):
        distance[i, :] = x_range + y_range[i]

    distance = np.sqrt(distance)

    # OK, mirror the array as needed to build the final image
    if not is_odd_shape[1]:
        distance = np.hstack((np.fliplr(distance), distance))
    else:
        distance = np.hstack((np.fliplr(distance[:, 1:]), distance))

    if not is_odd_shape[0]:
        distance = np.vstack((np.flipud(distance), distance))
    else:
        distance = np.vstack((np.flipud(distance[1:, :]), distance))

    return distance


def __MaxZBufferValue(dtype):
    return np.finfo(dtype).max


def EmptyDistanceBuffer(shape, dtype: DTypeLike | None = None):
    global use_memmap

    dtype = np.float16 if dtype is None else dtype

    if _use_memmap():  # use_memmap:
        full_distance_image_array_path = os.path.join(tempfile.gettempdir(), 'distance_image_%dx%d_%s.npy' % (
            shape[0], shape[1], GetProcessAndThreadUniqueString()))
        fullImageZbuffer = np.memmap(full_distance_image_array_path, dtype=dtype, mode='w+', shape=shape)
        fullImageZbuffer.fill(__MaxZBufferValue(dtype))
        return fullImageZbuffer
        #fullImageZbuffer = np.memmap(full_distance_image_array_path, dtype=np.float16, mode='r+', shape=shape)
    else:
        return np.full(shape, __MaxZBufferValue(dtype), dtype=dtype)




#
# def __CreateOutputBufferForTransforms(transforms, target_space_scale=None):
#     '''Create output images using the passed rectangle
#     :param tuple rectangle: (minY, minX, maxY, maxX)
#     :return: (fullImage, ZBuffer)
#     '''
#     fullImage = None
#     fixed_bounding_box = tutils.FixedBoundingBox(transforms)
#     (maxY, maxX) = fixed_bounding_box.shape
#     fullImage_shape = (int(np.ceil(target_space_scale * maxY)), int(np.ceil(target_space_scale * maxX)))
# 
#     if use_memmap:
#         try:
#             fullimage_array_path = os.path.join(tempfile.gettempdir(), 'image_%dx%d_%s.npy' % (fullImage_shape[0], fullImage_shape[1], GetProcessAndThreadUniqueString()))
#             fullImage = np.memmap(fullimage_array_path, dtype=np.float16, mode='w+', shape=fullImage_shape)
#             fullImage[:] = 0
#             fullImage.flush()
#             del fullImage
#             fullImage = np.memmap(fullimage_array_path, dtype=np.float16, mode='r+', shape=fullImage_shape)
#         except: 
#             prettyoutput.LogErr("Unable to open memory mapped file %s." % (fullimage_array_path))
#             raise 
#     else:
#         fullImage = np.zeros(fullImage_shape, dtype=np.float16)
# 
#     fullImageZbuffer = EmptyDistanceBuffer(fullImage.shape, dtype=fullImage.dtype)
#     return (fullImage, fullImageZbuffer)


def __CreateOutputBufferForArea(Height: int, Width: int, dtype: DTypeLike):
    '''Create output images using the passed width and height
    '''
    global use_memmap
    fullImage = None
    fullImage_shape = (
        int(Height),
        int(Width))  # (int(np.ceil(target_space_scale * Height)), int(np.ceil(target_space_scale * Width)))

    if _use_memmap():  # use_memmap:
        try:
            fullimage_array_path = os.path.join(tempfile.gettempdir(), 'image_%dx%d_%s.npy' % (
                fullImage_shape[0], fullImage_shape[1], GetProcessAndThreadUniqueString()))
            # print("Open %s" % (fullimage_array_path))
            fullImage = np.memmap(fullimage_array_path, dtype=dtype, mode='w+', shape=fullImage_shape)
            fullImage.fill(0)
            finalizer = weakref.finalize(fullImage, os.remove, fullimage_array_path)
        except:
            prettyoutput.LogErr("Unable to open memory mapped file %s." % fullimage_array_path)
            raise
    else:
        fullImage = np.zeros(fullImage_shape, dtype=dtype)

    fullImageZbuffer = EmptyDistanceBuffer(fullImage.shape, dtype=np.float32)
    return fullImage, fullImageZbuffer


def __GetOrCreateCachedDistanceImage(imageShape):
    distance_array_path = os.path.join(tempfile.gettempdir(), 'distance%dx%d.npy' % (imageShape[0], imageShape[1]))

    distanceImage = None
  
    # distanceImage = nornir_imageregistration.LoadImage(distance_image_path)
    try:
        #             if use_memmap:
        #                 distanceImage = np.load(distance_array_path, mmap_mode='r')
        #             else:
        distanceImage = np.load(distance_array_path)
    except FileNotFoundError:
        #print("Distance_image %s does not exist" % distance_array_path)
        pass
    except:
        print("Invalid distance_image %s" % distance_array_path)
        try:
            os.remove(distance_array_path)
        except:
            print("Unable to delete invalid distance_image: %s" % distance_array_path)
            pass

        pass

    if distanceImage is None:
        distanceImage = CreateDistanceImage2(imageShape)
        try:
            np.save(distance_array_path, distanceImage)
        except:
            print("Unable to save invalid distance_image: %s" % distance_array_path)
            pass

    return distanceImage


def __GetOrCreateDistanceImage(distanceImage, imageShape):
    '''Determines size of the image.  Returns a distance image to match the size if the passed existing image is not the correct size.'''

    assert (len(imageShape) == 2)
    size = imageShape
    if distanceImage is not None:
        if np.array_equal(distanceImage.shape, size):
            return distanceImage

    return __GetOrCreateCachedDistanceImage(imageShape)


def TilesToImage(mosaic_tileset: nornir_imageregistration.MosaicTileset,
                 TargetRegion: nornir_imageregistration.Rectangle | List[float] = None,
                 target_space_scale: float | None = None) -> Tuple[NDArray | None, NDArray | None]:
    """
    Generate an image of the TargetRegion.
    :param MosaicTileset mosaic_tileset: Tileset to assemble
    :param tuple TargetRegion: (MinX, MinY, Width, Height) or Rectangle class.  Specifies the SourceSpace to render from
    :param float target_space_scale: Scalar for the target space coordinates.  Used to downsample or upsample the output image.  Changes the coordinates of the target space control points of the transform. 
    """

    if target_space_scale is not None and target_space_scale > 1.0:
        raise ValueError(
            "It isn't impossible this is what the caller requests, but this value expands the resulting image beyond full resolution of the transform.")

    # logger = logging.getLogger(__name__ + '.TilesToImage')
    source_space_scale = 1.0 / mosaic_tileset.image_to_source_space_scale
    if target_space_scale is None:
        target_space_scale = source_space_scale

    distanceImage = None
    original_fixed_rect_floats = None

    if TargetRegion is not None:
        if isinstance(TargetRegion, nornir_imageregistration.Rectangle):
            original_fixed_rect_floats = TargetRegion
        else:
            original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
                (TargetRegion[0], TargetRegion[1]),
                (TargetRegion[2] - TargetRegion[0], TargetRegion[3] - TargetRegion[1]))
    else:
        #We could use mosaic_tileset.TargetBoundingBox, but for mosaic-to-volume
        #transforms the non-zero origin is important, so we always use an origin
        #of 0, 0 and the max coordinates of the target bounding box 
        #original_fixed_rect_floats = mosaic_tileset.TargetBoundingBox #Breaks mosaic-to-volume image assembly
        original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), mosaic_tileset.TargetBoundingBox.TopRight)

    targetRect = nornir_imageregistration.Rectangle.SafeRound(original_fixed_rect_floats)
    scaled_targetRect = nornir_imageregistration.Rectangle.scale_on_origin(original_fixed_rect_floats,
                                                                           target_space_scale)
    scaled_targetRect = nornir_imageregistration.Rectangle.SafeRound(scaled_targetRect)
    # original_fixed_rect_floats#nornir_imageregistration.Rectangle.scale_on_origin(scaled_targetRect, 1.0 / target_space_scale)

    first_tile = next(iter(mosaic_tileset.values()))
    if first_tile is None:
        raise ValueError("Mosaic Tileset has no tiles.")
    output_dtype = first_tile.Image.dtype

    (fullImage, fullImageZbuffer) = __CreateOutputBufferForArea(scaled_targetRect.Height, scaled_targetRect.Width, 
                                                                dtype=output_dtype)

    for i, tile in enumerate(mosaic_tileset.values()):
        original_transform_fixed_rect = tile.TargetSpaceBoundingBox
        # transform_target_rect = nornir_imageregistration.Rectangle.SafeRound(original_transform_fixed_rect)

        # regionToRender = nornir_imageregistration.Rectangle.Intersect(targetRect, transform_target_rect)
        regionToRender = nornir_imageregistration.Rectangle.Intersect(targetRect, original_transform_fixed_rect)
        if regionToRender is None:
            continue

        if regionToRender.Area == 0:
            continue

        # Replaced by rendered_target_space_origin on TransformedImageData
        # scaled_region_rendered = nornir_imageregistration.Rectangle.scale_on_origin(regionToRender, target_space_scale)
        # scaled_region_rendered = nornir_imageregistration.Rectangle.SafeRound(scaled_region_rendered)

        distanceImage = __GetOrCreateDistanceImage(distanceImage, tile.ImageSize)

        transformedImageData = TransformTile(tile, distanceImage, target_space_scale=target_space_scale,
                                             TargetRegion=regionToRender, SingleThreadedInvoke=True)
        if transformedImageData.image is None:
            # logger = logging.getLogger('TilesToImageParallel')
            prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
            if transformedImageData.errormsg is not None:
                prettyoutput.LogErr(transformedImageData.errormsg)
                continue

        CompositeOffset = (transformedImageData.rendered_target_space_origin * transformedImageData.target_space_scale) - scaled_targetRect.BottomLeft
        CompositeOffset = CompositeOffset.astype(np.int64)

        CompositeImageWithZBuffer(fullImage, fullImageZbuffer,
                                  transformedImageData.image, transformedImageData.centerDistanceImage,
                                  CompositeOffset)

        del transformedImageData

    mask = np.less(fullImageZbuffer, __MaxZBufferValue(fullImageZbuffer.dtype))
    del fullImageZbuffer

    fullImage = np.maximum(fullImage, 0, out=fullImage)
    # Checking for > 1.0 makes sense for floating point images.  During the DM4 migration
    # I was getting images which used 0-255 values, and the 1.0 check set them to entirely black
    # fullImage[fullImage > 1.0] = 1.0

    if isinstance(fullImage, np.memmap):
        fullImage.flush()

    return fullImage, mask


def TilesToImageParallel(mosaic_tileset : nornir_imageregistration.MosaicTileset,
                         TargetRegion: nornir_imageregistration.Rectangle | List[float] = None,
                         target_space_scale: float | None = None,
                         pool=None) -> Tuple[NDArray | None, NDArray | None]:
    """Assembles a set of transforms and imagepaths to a single image using parallel techniques.
    :param pool:
    :param MosaicTileset mosaic_tileset: Tileset to assemble
    :param tuple TargetRegion: (MinX, MinY, Width, Height) or Rectangle class.  Specifies the SourceSpace to render from
    :param float target_space_scale: Scalar for the target space coordinates.  Used to downsample or upsample the output image.  Changes the coordinates of the target space control points of the transform. 
    :param float target_space_scale: Scalar for the source space coordinates.  Must match the change in scale of input images relative to the transform source space coordinates.  So if downsampled by
    4 images are used, this value should be 0.25.  Calculated to be correct if None.  Specifying is an optimization to reduce I/O of reading image files to calculate.
    """
    timer = nornir_shared.tasktimer.TaskTimer()

    timer.Start('Prep')

    logger = logging.getLogger('TilesToImageParallel')
    if pool is None:
        pool = nornir_pools.GetGlobalMultithreadingPool()
        # pool = nornir_pools.GetGlobalSerialPool()

    if target_space_scale is not None and target_space_scale > 1.0:
        raise ValueError(
            "It isn't impossible this is what the caller requests, but a target_space_scale value > 1 expands the resulting image beyond full resolution of the transform.")

    source_space_scale = 1.0 / mosaic_tileset.image_to_source_space_scale
    if target_space_scale is None:
        target_space_scale = source_space_scale

    original_fixed_rect_floats = None

    if TargetRegion is not None:
        if isinstance(TargetRegion, nornir_imageregistration.Rectangle):
            original_fixed_rect_floats = TargetRegion
        else:
            original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
                (TargetRegion[0], TargetRegion[1]),
                (TargetRegion[2] - TargetRegion[0], TargetRegion[3] - TargetRegion[1]))
    else:
        #We could use mosaic_tileset.TargetBoundingBox, but for mosaic-to-volume
        #transforms the non-zero origin is important, so we always use an origin
        #of 0, 0 and the max coordinates of the target bounding box 
        #original_fixed_rect_floats = mosaic_tileset.TargetBoundingBox #Breaks mosaic-to-volume image assembly
        original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), mosaic_tileset.TargetBoundingBox.TopRight)

    targetRect = nornir_imageregistration.Rectangle.SafeRound(original_fixed_rect_floats)
    scaled_targetRect = nornir_imageregistration.Rectangle.scale_on_origin(original_fixed_rect_floats,
                                                                           target_space_scale)
    scaled_targetRect = nornir_imageregistration.Rectangle.SafeRound(scaled_targetRect)
    #    targetRect = original_fixed_rect_floats#nornir_imageregistration.Rectangle.scale_on_origin(scaled_targetRect, 1.0 / target_space_scale)

    first_tile = next(iter(mosaic_tileset.values()))
    if first_tile is None:
        raise ValueError("Mosaic Tileset has no tiles.")

    output_dtype = nornir_imageregistration.default_image_dtype()
    (fullImage, fullImageZbuffer) = __CreateOutputBufferForArea(scaled_targetRect.Height, scaled_targetRect.Width,
                                                                dtype=output_dtype)

    timer.End('Prep')
    timer.Start('Task Queuing')
    timer.Start('Task Execution')
    CheckTaskInterval = multiprocessing.cpu_count() * 2
    tasks = []  # type: List[nornir_pools.Task]
    #Ensure the shared memory manager has been created so child processes can
    #access it
    #shared_memory_manager = nornir_pools.get_or_create_shared_memory_manager() 
    for i, tile in enumerate(mosaic_tileset.values()):
        # original_transform_target_rect = nornir_imageregistration.Rectangle(transform.FixedBoundingBox)
        original_transform_target_rect = tile.TargetSpaceBoundingBox
        transform_target_rect = nornir_imageregistration.Rectangle.SafeRound(original_transform_target_rect)

        regionToRender = nornir_imageregistration.Rectangle.Intersect(targetRect, transform_target_rect)
        if regionToRender is None:
            continue

        if regionToRender.Area == 0:
            continue

        #Replaced by rendered_target_space_origin on TransformedImageData
        #scaled_region_rendered = nornir_imageregistration.Rectangle.scale_on_origin(regionToRender, target_space_scale)
        #scaled_region_rendered = nornir_imageregistration.Rectangle.SafeRound(scaled_region_rendered)

        task = pool.add_task(f"TransformTile {tile.ImagePath}",
                             TransformTile, tile=tile,
                             distanceImage=None,
                             target_space_scale=target_space_scale, TargetRegion=regionToRender,
                             SingleThreadedInvoke=False)
        task.transform = tile.Transform
        task.regionToRender = regionToRender
        #task.scaled_region_rendered = scaled_region_rendered
        task.transform_fixed_rect = transform_target_rect
        tasks.append(task)

        if not i % CheckTaskInterval == 0:
            continue

        while len(tasks) > CheckTaskInterval:  # Don't bother cleaning completed tasks if we can still add to the queue
            iTask = len(tasks) - 1
            while iTask >= 0:
                t = tasks[iTask]
                if t.iscompleted:
                    transformed_image_data = t.wait_return()  # type: nornir_imageregistration.transformed_image_data.TransformedImageData
                    __AddTransformedTileTaskToComposite(t, transformed_image_data, fullImage, fullImageZbuffer,
                                                        scaled_targetRect)
                    transformed_image_data.Clear()
                    del transformed_image_data
                    del tasks[iTask]
    
                iTask -= 1
            
            if len(tasks) > CheckTaskInterval: # Sleep a while if we are still over the limit
                time.sleep(0.1)
    timer.End('Task Queuing')
    logger.info('All warps queued, integrating results into final image')

    while len(tasks) > 0:
        # Pass through the entire loop and eliminate completed tasks in case any finished out of order
        iTask = len(tasks) - 1
        while iTask >= 0:
            t = tasks[iTask]
            if t.iscompleted:
                transformed_image_data = t.wait_return()
                __AddTransformedTileTaskToComposite(t, transformed_image_data, fullImage, fullImageZbuffer,
                                                    scaled_targetRect)
                transformed_image_data.Clear()
                del transformed_image_data
                del tasks[iTask]

            iTask -= 1

        if len(tasks) > 0:
            time.sleep(0.1) #Give tasks some time to complete before we interrogate again

    timer.End('Task Execution')
    logger.info('Final image complete, building mask')

    mask = np.less(fullImageZbuffer, __MaxZBufferValue(fullImageZbuffer.dtype))
    del fullImageZbuffer

    #fullImage = np.clip(fullImage, 0, 1.0, out=fullImage)
    fullImage = np.maximum(fullImage, 0, out=fullImage)
    # Checking for > 1.0 makes sense for floating point images.  During the DM4 migration
    # I was getting images which used 0-255 values, and the 1.0 check set them to entirely black
    # fullImage[fullImage > 1.0] = 1.0

    logger.info('Assemble complete')

    if isinstance(fullImage, np.memmap):
        fullImage.flush()

    return fullImage, mask


def __AddTransformedTileTaskToComposite(task,
                                        transformedImageData: nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile,
                                        fullImage: NDArray,
                                        fullImageZBuffer: NDArray,
                                        scaled_target_rect: nornir_imageregistration.Rectangle | None = None):
    if transformedImageData is None:
        # logger = logging.getLogger('TilesToImageParallel')
        prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
        return

    if transformedImageData.image is None:
        # logger = logging.getLogger('TilesToImageParallel')
        prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
        if transformedImageData.errormsg is not None:
            prettyoutput.LogErr(transformedImageData.errormsg)
            return fullImage, fullImageZBuffer

    CompositeOffset = (transformedImageData.rendered_target_space_origin * transformedImageData.target_space_scale) - scaled_target_rect.BottomLeft
    CompositeOffset = CompositeOffset.astype(np.int32)

    try:
        CompositeImageWithZBuffer(fullImage, fullImageZBuffer,
                                  transformedImageData.image, transformedImageData.centerDistanceImage,
                                  CompositeOffset)
    except ValueError as e:
        # This is frustrating and usually indicates the input transform passed to assemble mapped to negative coordinates.
        # logger = logging.getLogger('TilesToImageParallel')
        prettyoutput.LogErr(f'Could not add tile to composite: {transformedImageData}\n{e}')
        pass
  
    return


def __CreateScalableTransformCopy(transform):
    if not isinstance(transform, nornir_imageregistration.transforms.ITransform):
        raise ValueError("Expected transform to be an ITransform type")

    if isinstance(transform, nornir_imageregistration.transforms.ITransformScaling):
        return copy.deepcopy(transform)

    raise ValueError("Transform does not support ITransformScaling and does not have a hand-coded mapping here")


def TransformTile(tile: nornir_imageregistration.Tile,
                  distanceImage: NDArray | None = None,
                  target_space_scale: float = None,
                  TargetRegion: nornir_imageregistration.Rectangle | Tuple[float] | NDArray | None = None,
                  SingleThreadedInvoke: bool = False) -> nornir_imageregistration.transformed_image_data.ITransformedImageData:
    """
       Transform the passed image.  DistanceImage is an existing image recording the distance to the center of the
       image for each pixel.  target_space_scale is used when the image size does not match the image size encoded in the
       transform.  A scale will be calculated in this case and if it does not match the required scale the tile will 
       not be transformed.
get_space_scale: Optional pre-calculated scalar to apply to the transforms target space control points.  If None the scale is calculated based on the difference
                                   between input image size and the image size of the transform. i.e.  If the source_space is downsampled by 4 then the target_space will be downsampled to match
       :param tile:
       :param SingleThreadedInvoke:
       :param TargetRegion: [MinY MinX MaxY MaxX] If specified only the specified region is populated.  Otherwise transform the entire image.'''
    """

    TargetRegionRect = None
    if TargetRegion is not None:
        if isinstance(TargetRegion, nornir_imageregistration.Rectangle):
            TargetRegionRect = TargetRegion.copy()
        elif isinstance(TargetRegion, Iterable):
            TargetRegionRect = nornir_imageregistration.Rectangle.CreateFromBounds(TargetRegion)
    else:
        TargetRegion = tile.TargetSpaceBoundingBox
        TargetRegionRect = TargetRegion

    del TargetRegion

    nornir_imageregistration.spatial.RaiseValueErrorOnInvalidBounds(TargetRegionRect)

    # if not os.path.exists(imagefullpath):
    #    return nornir_imageregistration.transformed_image_data.TransformedImageData(errorMsg='Tile does not exist ' + imagefullpath)

    # if isinstance(transform, meshwithrbffallback.MeshWithRBFFallback):
    # Don't bother mapping points falling outside the defined boundaries because we won't have image data for it
    #   transform = triangulation.Triangulation(transform.points)
    source_image = None
    try:
        # warpedImage = nornir_imageregistration.ImageParamToImageArray(tile.Image, dtype=np.float32)
        source_image = tile.Image
    except IOError:
        return nornir_imageregistration.transformed_image_data.TransformedImageData(
            errorMsg='Tile does not exist ' + tile.ImagePath)
    except ValueError as ve:
        return nornir_imageregistration.transformed_image_data.TransformedImageData(errorMsg=f'{ve}')

    source_image = nornir_imageregistration.ForceGrayscale(source_image)

    # Automatically scale the transform if the input image shape does not match the transform bounds
    source_space_scale = 1.0 / tile.image_to_source_space_scale  # Pass tile to this function and use the image_to_source_space attribute  #tiles.__DetermineTransformScale(transform, warpedImage.shape)

    if target_space_scale is None:
        target_space_scale = source_space_scale

    ########## Scale the transform output to fit the input image coordspace ####
    transform = tile.Transform
    if source_space_scale == target_space_scale:
        if source_space_scale != 1.0:
            scaledTransform = __CreateScalableTransformCopy(tile.Transform)

            scaledTransform.Scale(source_space_scale)
            transform = scaledTransform

    else:
        if source_space_scale != 1.0:
            scaledTransform = __CreateScalableTransformCopy(tile.Transform)
            scaledTransform.ScaleWarped(source_space_scale)
            transform = scaledTransform

        if target_space_scale != 1.0:
            scaledTransform = __CreateScalableTransformCopy(tile.Transform)
            scaledTransform.ScaleFixed(target_space_scale)
            transform = scaledTransform

    ############################################################################

    if target_space_scale != 1.0:
        # TargetRegion = np.array(TargetRegion) * target_space_scale
        Scaled_TargetRegionRect = nornir_imageregistration.Rectangle.scale_on_origin(TargetRegionRect,
                                                                                     target_space_scale)
        Scaled_rounded_TargetRegionRect = nornir_imageregistration.Rectangle.SnapRound(Scaled_TargetRegionRect)
    else:
        Scaled_TargetRegionRect = TargetRegionRect
        Scaled_rounded_TargetRegionRect = nornir_imageregistration.Rectangle.SnapRound(TargetRegionRect)

    (target_width, target_height, target_minX, target_minY) = (Scaled_rounded_TargetRegionRect.Width,
                                   Scaled_rounded_TargetRegionRect.Height,
                                   Scaled_rounded_TargetRegionRect.MinX,
                                   Scaled_rounded_TargetRegionRect.MinY)

    target_width = int(target_width)
    target_height = int(target_height)
    target_minX = int(target_minX)
    target_minY = int(target_minY)

    # if TargetRegion is None:
    #     TargetRegion = nornir_imageregistration.Rectangle.SafeRound(tile.TargetSpaceBoundingBox)
    #     Scaled_TargetRegionRect = nornir_imageregistration.Rectangle.scale_on_origin(tile.TargetSpaceBoundingBox, target_space_scale)
    #     Scaled_TargetRegionRect = nornir_imageregistration.Rectangle.SafeRound(Scaled_TargetRegionRect)
    #
    #     width = Scaled_TargetRegionRect.Width
    #     height = Scaled_TargetRegionRect.Height

    # if TargetRegion is None:
    #     if hasattr(transform, 'FixedBoundingBox'):
    #         width = transform.FixedBoundingBox.Width
    #         height = transform.FixedBoundingBox.Height
    #
    #         TargetRegionRect = transform.FixedBoundingBox
    #         TargetRegionRect = nornir_imageregistration.Rectangle.SafeRound(TargetRegionRect)
    #         Scaled_TargetRegionRect = TargetRegionRect
    #
    #         (minY, minX, maxY, maxX) = TargetRegionRect.ToTuple()
    #     else:
    #         width = warpedImage.shape[1]
    #         height = warpedImage.shape[0]
    #         TargetRegionRect = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), warpedImage.shape)
    #         Scaled_TargetRegionRect = TargetRegionRect
    # else:
    #     assert(len(TargetRegion) == 4)
    #     (minY, minX, maxY, maxX) = Scaled_TargetRegionRect.ToTuple()
    #     height = maxY - minY
    #     width = maxX - minX

    # Round up to the nearest integer value
    #height = np.ceil(height)
    #width = np.ceil(width)

    distanceImage = __GetOrCreateDistanceImage(distanceImage, source_image.shape[0:2])

    (fixedImage, centerDistanceImage) = assemble.SourceImageToTargetSpace(transform,
                                                                          [source_image, distanceImage],
                                                                          output_botleft=(target_minY, target_minX),
                                                                          output_area=(target_height, target_width),
                                                                          cval=[0, __MaxZBufferValue(distanceImage.dtype)],
                                                                          return_shared_memory=False)#not SingleThreadedInvoke)
    
    source_image_dtype = source_image.dtype
    del source_image
    del distanceImage

    return nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile.Create(fixedImage,
                                                                                       centerDistanceImage,
                                                                                       transform,
                                                                                       source_space_scale,
                                                                                       target_space_scale,
                                                                                       rendered_target_space_origin=(target_minY * (1.0 / target_space_scale), target_minX * (1.0 / target_space_scale)),
                                                                                       SingleThreadedInvoke=SingleThreadedInvoke)


if __name__ == '__main__':
    pass
