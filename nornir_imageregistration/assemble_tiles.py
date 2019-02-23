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

import nornir_imageregistration
import nornir_imageregistration.assemble  as assemble
import nornir_imageregistration.spatial as spatial
import nornir_imageregistration.tileset as tiles
import nornir_imageregistration.transforms.utils as tutils
import nornir_pools
import nornir_shared.prettyoutput as prettyoutput
import numpy as np


# from nornir_imageregistration.files.mosaicfile import MosaicFile
# from nornir_imageregistration.mosaic import Mosaic
# import nornir_imageregistration.transforms.meshwithrbffallback as meshwithrbffallback
# import nornir_imageregistration.transforms.triangulation as triangulation
DistanceImageCache = {}

# TODO: Use atexit to delete the temporary files
# TODO: use_memmap does not work when assembling tiles on a cluster, disable for now.  Specific test is IDOCTests.test_AssembleTilesIDoc
use_memmap = True
nextNumpyMemMapFilenameIndex = 0

def GetProcessAndThreadUniqueString():
    '''We use the index because if the same thread makes a new tile of the same size and the original has not been garbage collected yet we get errors'''
    global nextNumpyMemMapFilenameIndex
    nextNumpyMemMapFilenameIndex += 1
    return "%d_%d_%d" % (os.getpid(), threading.get_ident(), nextNumpyMemMapFilenameIndex)    


class TransformedImageData(object):
    '''
    Returns data from multiprocessing thread processes.  Uses memory mapped files when there is too much data for pickle to be efficient
    '''

    memmap_threshold = 64 * 64
#     
#     def __getstate__(self):
#         odict = {} 
#         odict["_image"] = self._image
#         odict["_centerDistanceImage"] = self._centerDistanceImage 
#         odict["_transformScale"] = self._transformScale
#         odict["_transform"] = self._transform
#         odict["_errmsg"] = self._errmsg
#         odict["_image_path"] = self._image_path
#         odict["_centerDistanceImage_path"] = self._centerDistanceImage_path
#         odict["_tempdir"] = self._tempdir
#         odict["_image_shape"] = self._image_shape
#         odict["_centerDistanceImage_shape"] = self._centerDistanceImage_shape
#         odict["_image_dtype"] = self._image_dtype
#         odict["_centerDistance_dtype"] = self._centerDistance_dtype
#         return odict
# 
#     def __setstate__(self, dictionary):
#         self.__dict__.update(dictionary)
#         
    
    @property
    def image(self):
        if self._image is None:
            if self._image_path is None:
                return None 
             
            self._image = np.load(self._image_path, mmap_mode='r')  #np.memmap(self._image_path, mode='c', shape=self._image_shape, dtype=self._image_dtype)
            
        return self._image
    
    @property
    def centerDistanceImage(self):
        if self._centerDistanceImage is None:
            if self._centerDistanceImage_path is None:
                return None
            
            self._centerDistanceImage = np.load(self._centerDistanceImage_path, mmap_mode='r') #np.memmap(self._centerDistanceImage_path, mode='c', shape=self._centerDistanceImage_shape, dtype=self._centerDistance_dtype)
            
        return self._centerDistanceImage

    @property
    def transformScale(self):
        return self._transformScale

    @property
    def transform(self):
        return self._transform

    @property
    def errormsg(self):
        return self._errmsg
    
    @property
    def tempfiledir(self):
        if self._tempdir is None:
            self._tempdir = tempfile.mkdtemp("_TransformedImageData")
            
        return self._tempdir

    @classmethod
    def Create(cls, image, centerDistanceImage, transform, scale, SingleThreadedInvoke):
        o = TransformedImageData()
        o._image = image
        o._centerDistanceImage = centerDistanceImage
        
        o._image_path = None
        o._centerDistanceImage_path = None
        o._transformScale = scale
        #o._transform = transform
        
        if not SingleThreadedInvoke:
            o.ConvertToMemmapIfLarge()
        
        return o
    
    def ConvertToMemmapIfLarge(self): 
        if np.prod(self._image.shape) > TransformedImageData.memmap_threshold:
            self._image_path = self.CreateMemoryMappedFilesForImage("Image", self._image)
            #self._image_shape = self._image.shape
            #self._image_dtype = self._image.dtype
            self._image = None
            
        if np.prod(self._centerDistanceImage.shape) > TransformedImageData.memmap_threshold:
            self._centerDistanceImage_path = self.CreateMemoryMappedFilesForImage("Distance", self._centerDistanceImage)
            #self._centerDistanceImage_shape = self._centerDistanceImage.shape
            #self._centerDistance_dtype = self._centerDistanceImage.dtype
            self._centerDistanceImage = None
            
        return
     
    def CreateMemoryMappedFilesForImage(self, name, image):
        if image is None:
            return None 
        
        tempfilename = os.path.join(self.tempfiledir, name + '.npy')
        np.save(tempfilename, image)
        #memmapped_image = np.memmap(tempfilename, dtype=image.dtype, mode='w+', shape=image.shape)
        #np.copyto(memmapped_image, image)
        #print("Write %s" % tempfilename)
        
        #del memmapped_image
        return tempfilename


    def Clear(self):
        '''Sets attributes to None to encourage garbage collection'''
        self._image = None
        self._centerDistanceImage = None
        self._transformScale = None
        self._transform = None
        
        if not self._centerDistanceImage_path is None or not self._image_path is None:
            pool = nornir_pools.GetGlobalThreadPool()
            pool.add_task(self._image_path, TransformedImageData._RemoveTempFiles, self._centerDistanceImage_path, self._image_path, self._tempdir)
        
        
    @staticmethod
    def _RemoveTempFiles(_centerDistanceImage_path, _image_path, _tempdir):
        try:
            if not _centerDistanceImage_path is None:
                os.remove(_centerDistanceImage_path)
        except IOError as E:
            pass
            
        try:
            if not _image_path is None:
                os.remove(_image_path)
        except IOError as E:
            pass
            
        try:
            if not _tempdir is None:
                os.rmdir(_tempdir)
        except IOError as E:
            pass
         

    def __init__(self, errorMsg=None):
        self._image = None
        self._centerDistanceImage = None
        self._transformScale = None
        self._transform = None
        self._errmsg = errorMsg
        self._image_path = None
        self._centerDistanceImage_path = None
        self._tempdir = None
        #self._image_shape = None
        #self._centerDistanceImage_shape = None
        #self._image_dtype = None
        #self._centerDistance_dtype = None

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
    
    if((np.array([maxY - minY, maxX - minX]) != SubZBuffer.shape).any()):
        raise ValueError("Buffers do not have the same dimensions")
     
    if minY == maxY or minX == maxX:
        raise ValueError("Buffers have zero dimensions")
    # iNewIndex = np.zeros(FullImage.shape, dtype=np.bool)

    # iUpdate = FullZBuffer[minY:maxY, minX:maxX] > SubZBuffer
    # iNewIndex[minY:maxY, minX:maxX] = iUpdate
    # FullImage[iNewIndex] = SubImage[iUpdate]
    # FullZBuffer[iNewIndex] = SubZBuffer[iUpdate]
    
    iUpdate = FullZBuffer[minY:maxY, minX:maxX] > SubZBuffer
    FullImage[minY:maxY, minX:maxX][iUpdate] = SubImage[iUpdate]
    FullZBuffer[minY:maxY, minX:maxX][iUpdate] = SubZBuffer[iUpdate] 

    return

    
def CreateDistanceImage(shape, dtype=None):

    if dtype is None:
        dtype = np.float32

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

    #TODO, this has some obvious optimizations available
    if dtype is None:
        dtype = np.float32
          
    #center = [shape[0] / 2.0, shape[1] / 2.0]
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
        y_range = np.linspace(0.5, half_shape[0]+0.5, num=half_shape[0])  
    else:
        half_shape[0] = half_shape[0] + 1
        y_range = np.linspace(0, half_shape[0]-1, num=half_shape[0])
        
    x_range = None
    if not is_odd_shape[1]:
        x_range = np.linspace(0.5, half_shape[1]+0.5, num=half_shape[1])  
    else:
        half_shape[1] = half_shape[1] + 1
        x_range = np.linspace(0, half_shape[1]-1, num=half_shape[1])
        
    
    x_range = x_range * x_range
    y_range = y_range * y_range

    distance = np.empty(half_shape, dtype=dtype)

    for i in range(0, half_shape[0]):
        distance[i, :] = x_range + y_range[i]

    distance = np.sqrt(distance)
    
    #OK, mirror the array as needed to build the final image
    if not is_odd_shape[1]:
        distance = np.hstack((np.fliplr(distance), distance))
    else:
        distance = np.hstack((np.fliplr(distance[:,1:]), distance))

    if not is_odd_shape[0]:
        distance = np.vstack((np.flipud(distance), distance))
    else:
        distance = np.vstack((np.flipud(distance[1:,:]),distance))
    
    return distance


def __MaxZBufferValue(dtype):
    return np.finfo(dtype).max


def EmptyDistanceBuffer(shape, dtype=np.float16):
    global use_memmap
    
    fullImageZbuffer = None
    
    if False: #use_memmap:
        full_distance_image_array_path = os.path.join(tempfile.gettempdir(), 'distance_image_%dx%d_%s.npy' % (shape[0], shape[1], GetProcessAndThreadUniqueString()))
        fullImageZbuffer = np.memmap(full_distance_image_array_path, dtype=np.float16, mode='w+', shape=shape)
        fullImageZbuffer[:] = __MaxZBufferValue(dtype)
        fullImageZbuffer.flush()
        del fullImageZbuffer
        fullImageZbuffer = np.memmap(full_distance_image_array_path, dtype=np.float16, mode='r+', shape=shape)
    else:
        fullImageZbuffer = np.full(shape, __MaxZBufferValue(dtype), dtype=dtype)
    
    return fullImageZbuffer
# 
# def __CreateOutputBufferForTransforms(transforms, requiredScale=None):
#     '''Create output images using the passed rectangle
#     :param tuple rectangle: (minY, minX, maxY, maxX)
#     :return: (fullImage, ZBuffer)
#     '''
#     fullImage = None
#     fixed_bounding_box = tutils.FixedBoundingBox(transforms)
#     (maxY, maxX) = fixed_bounding_box.shape
#     fullImage_shape = (int(np.ceil(requiredScale * maxY)), int(np.ceil(requiredScale * maxX)))
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


def __CreateOutputBufferForArea(Height, Width, requiredScale=None):
    '''Create output images using the passed width and height
    '''
    global use_memmap
    fullImage = None
    fullImage_shape = (int(Height), int(Width)) #(int(np.ceil(requiredScale * Height)), int(np.ceil(requiredScale * Width)))

    if False: #use_memmap:
        try:
            fullimage_array_path = os.path.join(tempfile.gettempdir(), 'image_%dx%d_%s.npy' % (fullImage_shape[0], fullImage_shape[1], GetProcessAndThreadUniqueString()))
            # print("Open %s" % (fullimage_array_path))
            fullImage = np.memmap(fullimage_array_path, dtype=np.float16, mode='w+', shape=fullImage_shape)
            fullImage[:] = 0
        except: 
            prettyoutput.LogErr("Unable to open memory mapped file %s." % (fullimage_array_path))
            raise 
    else:
        fullImage = np.zeros(fullImage_shape, dtype=np.float16)

    fullImageZbuffer = EmptyDistanceBuffer(fullImage.shape, dtype=fullImage.dtype) 
    return (fullImage, fullImageZbuffer)


def __GetOrCreateCachedDistanceImage(imageShape):
    distance_array_path = os.path.join(tempfile.gettempdir(), 'distance%dx%d.npy' % (imageShape[0], imageShape[1]))
    
    distanceImage = None 
    
    if os.path.exists(distance_array_path):
        # distanceImage = nornir_imageregistration.LoadImage(distance_image_path)
        try:
#             if use_memmap:
#                 distanceImage = np.load(distance_array_path, mmap_mode='r')
#             else:
                distanceImage = np.load(distance_array_path)
        except:
            print("Unable to load distance_image %s" % (distance_array_path))
            try:
                os.remove(distance_array_path)
            except:
                print("Unable to delete invalid distance_image: %s" % (distance_array_path))
                pass
            
            pass
    
    if distanceImage is None:
        distanceImage = CreateDistanceImage2(imageShape)
        try:
            np.save(distance_array_path, distanceImage)
        except:
            print("Unable to save invalid distance_image: %s" % (distance_array_path))
            pass
        
    return distanceImage
     

def __GetOrCreateDistanceImage(distanceImage, imageShape):
    '''Determines size of the image.  Returns a distance image to match the size if the passed existing image is not the correct size.'''

    assert(len(imageShape) == 2)
    size = imageShape
    if not distanceImage is None:
        if np.array_equal(distanceImage.shape, size):
            return distanceImage
                
    return __GetOrCreateCachedDistanceImage(imageShape)


def TilesToImage(transforms, imagepaths, FixedRegion=None, requiredScale=None):
    '''
    Generate an image of the FixedRegion.
    :param tuple FixedRegion: (MinX, MinY, Width, Height)

    '''

    assert(len(transforms) == len(imagepaths))

    # logger = logging.getLogger(__name__ + '.TilesToImage')

    if requiredScale is None:
        requiredScale = tiles.MostCommonScalar(transforms, imagepaths)

    distanceImage = None 
    original_fixed_rect_floats = None
    
    if not FixedRegion is None:
        if isinstance(FixedRegion, spatial.Rectangle):
            original_fixed_rect_floats = FixedRegion
        else: 
            original_fixed_rect_floats = spatial.Rectangle.CreateFromPointAndArea((FixedRegion[0], FixedRegion[1]), (FixedRegion[2] - FixedRegion[0], FixedRegion[3] - FixedRegion[1]))
    else:
        original_fixed_rect_floats = tutils.FixedBoundingBox(transforms)
        
    scaled_fixedRect = nornir_imageregistration.Rectangle.scale_on_origin(original_fixed_rect_floats, requiredScale)
    scaled_fixedRect = nornir_imageregistration.Rectangle.SafeRound(scaled_fixedRect)  
    fixedRect = nornir_imageregistration.Rectangle.scale_on_origin(scaled_fixedRect, 1.0/requiredScale)

    (fullImage, fullImageZbuffer) = __CreateOutputBufferForArea(scaled_fixedRect.Height, scaled_fixedRect.Width, requiredScale)

    for i, transform in enumerate(transforms): 
        regionToRender = None
        original_transform_fixed_rect = spatial.Rectangle(transform.FixedBoundingBox)
        transform_fixed_rect = nornir_imageregistration.Rectangle.SafeRound(original_transform_fixed_rect)
        
        regionToRender = nornir_imageregistration.Rectangle.Intersect(fixedRect, transform_fixed_rect)
        if regionToRender is None:
            continue
        
        if regionToRender.Area == 0:
            continue 
        
        scaled_region_rendered = nornir_imageregistration.Rectangle.scale_on_origin(regionToRender, requiredScale)
        scaled_region_rendered = nornir_imageregistration.Rectangle.SafeRound(scaled_region_rendered)
                      
        imagefullpath = imagepaths[i]
        
        distanceImage = __GetOrCreateDistanceImage(distanceImage, nornir_imageregistration.GetImageSize(imagefullpath))
        
        transformedImageData = TransformTile(transform, imagefullpath, distanceImage, requiredScale=requiredScale, FixedRegion=regionToRender, SingleThreadedInvoke=True)

        CompositeOffset = scaled_region_rendered.BottomLeft - scaled_fixedRect.BottomLeft
        CompositeOffset = CompositeOffset.astype(np.int64)
        
        CompositeImageWithZBuffer(fullImage, fullImageZbuffer,
                                  transformedImageData.image, transformedImageData.centerDistanceImage,
                                  CompositeOffset)

        del transformedImageData

    mask = fullImageZbuffer < __MaxZBufferValue(fullImageZbuffer.dtype)
    del fullImageZbuffer

    fullImage[fullImage < 0] = 0
    # Checking for > 1.0 makes sense for floating point images.  During the DM4 migration
    # I was getting images which used 0-255 values, and the 1.0 check set them to entirely black
    # fullImage[fullImage > 1.0] = 1.0
    
    if isinstance(fullImage, np.memmap):
        fullImage.flush()

    return (fullImage, mask)


def TilesToImageParallel(transforms, imagepaths, FixedRegion=None, requiredScale=None, pool=None):
    '''Assembles a set of transforms and imagepaths to a single image using parallel techniques'''

    assert(len(transforms) == len(imagepaths))

    logger = logging.getLogger('TilesToImageParallel')
  
    if pool is None:
        pool = nornir_pools.GetGlobalMultithreadingPool()
        
    # pool = nornir_pools.GetGlobalSerialPool()

    tasks = []
    if requiredScale is None:
        requiredScale = tiles.MostCommonScalar(transforms, imagepaths)

    distanceImage = None 
    original_fixed_rect_floats = None
    
    if not FixedRegion is None:
        if isinstance(FixedRegion, spatial.Rectangle):
            original_fixed_rect_floats = FixedRegion
        else: 
            original_fixed_rect_floats = spatial.Rectangle.CreateFromPointAndArea((FixedRegion[0], FixedRegion[1]), (FixedRegion[2] - FixedRegion[0], FixedRegion[3] - FixedRegion[1]))
    else:
        original_fixed_rect_floats = tutils.FixedBoundingBox(transforms)
        
    scaled_fixedRect = nornir_imageregistration.Rectangle.scale_on_origin(original_fixed_rect_floats, requiredScale)
    scaled_fixedRect = nornir_imageregistration.Rectangle.SafeRound(scaled_fixedRect)  
    fixedRect = nornir_imageregistration.Rectangle.scale_on_origin(scaled_fixedRect, 1.0/requiredScale)  
    (fullImage, fullImageZbuffer) = __CreateOutputBufferForArea(scaled_fixedRect.Height, scaled_fixedRect.Width, requiredScale)

    CheckTaskInterval = 16

    for i, transform in enumerate(transforms):
        regionToRender = None
        original_transform_fixed_rect = spatial.Rectangle(transform.FixedBoundingBox)
        transform_fixed_rect = nornir_imageregistration.Rectangle.SafeRound(original_transform_fixed_rect)
        
        regionToRender = nornir_imageregistration.Rectangle.Intersect(fixedRect, transform_fixed_rect)
        if regionToRender is None:
            continue
        
        if regionToRender.Area == 0:
            continue 
        
        scaled_region_rendered = nornir_imageregistration.Rectangle.scale_on_origin(regionToRender, requiredScale)
        scaled_region_rendered = nornir_imageregistration.Rectangle.SafeRound(scaled_region_rendered)
                      
        imagefullpath = imagepaths[i]

        task = pool.add_task("TransformTile" + imagefullpath,
                              TransformTile, transform=transform, 
                              imagefullpath=imagefullpath, distanceImage=None,
                              requiredScale=requiredScale, FixedRegion=regionToRender,
                              SingleThreadedInvoke=False)
        task.transform = transform
        task.regionToRender = regionToRender
        task.scaled_region_rendered = scaled_region_rendered
        task.transform_fixed_rect = transform_fixed_rect
        tasks.append(task)

        if not i % CheckTaskInterval == 0:
            continue
        
        if len(tasks) > multiprocessing.cpu_count():
            iTask = len(tasks) - 1
            while iTask >= 0:
                t = tasks[iTask]
                if t.iscompleted:
                    transformedImageData = t.wait_return()
                    __AddTransformedTileTaskToComposite(t, transformedImageData, fullImage, fullImageZbuffer, scaled_fixedRect)
                    del transformedImageData 
                    del tasks[iTask]
                
                iTask -= 1

    logger.info('All warps queued, integrating results into final image')

    while len(tasks) > 0:
        t = tasks.pop(0)
        transformedImageData = t.wait_return()     
        __AddTransformedTileTaskToComposite(t, transformedImageData, fullImage, fullImageZbuffer, scaled_fixedRect)
        del transformedImageData
        del t
        
        # Pass through the entire loop and eliminate completed tasks in case any finished out of order
        iTask = len(tasks) - 1
        while iTask >= 0:
            t = tasks[iTask]
            if t.iscompleted:
                transformedImageData = t.wait_return()
                __AddTransformedTileTaskToComposite(t, transformedImageData, fullImage, fullImageZbuffer, scaled_fixedRect)
                del transformedImageData 
                del tasks[iTask]
            
            iTask -= 1
            
    logger.info('Final image complete, building mask')

    mask = fullImageZbuffer < __MaxZBufferValue(fullImageZbuffer.dtype)
    del fullImageZbuffer

    fullImage[fullImage < 0] = 0
    # Checking for > 1.0 makes sense for floating point images.  During the DM4 migration
    # I was getting images which used 0-255 values, and the 1.0 check set them to entirely black
    # fullImage[fullImage > 1.0] = 1.0

    logger.info('Assemble complete')
    
    if isinstance(fullImage, np.memmap):
        fullImage.flush()

    return (fullImage, mask)


def __AddTransformedTileTaskToComposite(task, transformedImageData, fullImage, fullImageZBuffer, scaled_fixedRect=None):
    
    if transformedImageData is None:
            logger = logging.getLogger('TilesToImageParallel')
            logger.error('Convert task failed: ' + str(transformedImageData))
            return
        
    if transformedImageData.image is None:
        logger = logging.getLogger('TilesToImageParallel')
        logger.error('Convert task failed: ' + str(transformedImageData))
        if not transformedImageData.errormsg is None:
            logger.error(transformedImageData.errormsg)
            return (fullImage, fullImageZBuffer)

    CompositeOffset = task.scaled_region_rendered.BottomLeft - scaled_fixedRect.BottomLeft 
    CompositeOffset = CompositeOffset.astype(np.int64)
    
    try:
        CompositeImageWithZBuffer(fullImage, fullImageZBuffer,
                                  transformedImageData.image, transformedImageData.centerDistanceImage,
                                  CompositeOffset)
    except ValueError:
        # This is frustrating and usually indicates the input transform passed to assemble mapped to negative coordinates.
        logger = logging.getLogger('TilesToImageParallel')
        logger.error('Transformed tile mapped to negative coordinates ' + str(transformedImageData))
        pass
    
    transformedImageData.Clear()
    
    return 


def TransformTile(transform, imagefullpath, distanceImage=None, requiredScale=None, FixedRegion=None, SingleThreadedInvoke=False):
    '''Transform the passed image.  DistanceImage is an existing image recording the distance to the center of the
       image for each pixel.  requiredScale is used when the image size does not match the image size encoded in the
       transform.  A scale will be calculated in this case and if it does not match the required scale the tile will 
       not be transformed.
       :param transform transform: Transformation used to map pixels from source image to output image
       :param str imagefullpath: Full path to the image on disk
       :param ndarray distanceImage: Optional pre-allocated array to contain the distance of each pixel from the center for use as a depth mask
       :param float requiredScale: Optional pre-calculated scalar to apply to the transform.  If None the scale is calculated based on the difference
                                   between input image size and the image size of the transform
       :param array FixedRegion: [MinY MinX MaxY MaxX] If specified only the specified region is transformed.  Otherwise transform the entire image.'''

    FixedRegionRect = None
    if not FixedRegion is None:
        if isinstance(FixedRegion, nornir_imageregistration.Rectangle):
            FixedRegionRect = FixedRegion.copy()
            FixedRegion = list(FixedRegion.ToTuple())
        else:
            FixedRegionRect = nornir_imageregistration.Rectangle.CreateFromBounds(FixedRegion)
            
        spatial.RaiseValueErrorOnInvalidBounds(FixedRegion)

    if not os.path.exists(imagefullpath):
        return TransformedImageData(errorMsg='Tile does not exist ' + imagefullpath)
 
    # if isinstance(transform, meshwithrbffallback.MeshWithRBFFallback):
    # Don't bother mapping points falling outside the defined boundaries because we won't have image data for it
    #   transform = triangulation.Triangulation(transform.points)

    warpedImage = nornir_imageregistration.ImageParamToImageArray(imagefullpath, dtype=np.float32)

    # Automatically scale the transform if the input image shape does not match the transform bounds
    transformScale = tiles.__DetermineTransformScale(transform, warpedImage.shape)

    if not requiredScale is None:
        if not nornir_imageregistration.ApproxEqual(transformScale, requiredScale):
            return TransformedImageData(errorMsg="%g scale needed for %s is different than required scale %g used for mosaic" % (transformScale, imagefullpath, requiredScale))
        
        transformScale = requiredScale  # This is needed because we aren't specifying the size of the output tile like we should be

    Scaled_FixedRegionRect = FixedRegionRect
    if transformScale != 1.0:
        scaledTransform = copy.deepcopy(transform)
        scaledTransform.Scale(transformScale)
        transform = scaledTransform

        if not FixedRegion is None:
            FixedRegion = np.array(FixedRegion) * transformScale
            Scaled_FixedRegionRect = nornir_imageregistration.Rectangle.scale_on_origin(FixedRegionRect, transformScale)
            Scaled_FixedRegionRect = nornir_imageregistration.Rectangle.SafeRound(Scaled_FixedRegionRect)

    (width, height, minX, minY) = (0, 0, 0, 0)

    if FixedRegion is None:
        if hasattr(transform, 'FixedBoundingBox'):
            width = transform.FixedBoundingBox.Width
            height = transform.FixedBoundingBox.Height
            
            FixedRegionRect = transform.FixedBoundingBox
            FixedRegionRect = nornir_imageregistration.Rectangle.SafeRound(FixedRegionRect)
            Scaled_FixedRegionRect = FixedRegionRect
            
            (minY, minX, maxY, maxX) = FixedRegionRect.ToTuple()
        else:
            width = warpedImage.shape[1]
            height = warpedImage.shape[0]
            FixedRegionRect = nornir_imageregistration.CreateFromPointAndArea((0,0), warpedImage.shape)
            Scaled_FixedRegionRect = FixedRegionRect
    else:
        assert(len(FixedRegion) == 4)
        (minY, minX, maxY, maxX) = Scaled_FixedRegionRect.ToTuple()
        height = maxY - minY
        width = maxX - minX
        
    height = np.ceil(height)
    width = np.ceil(width)

    distanceImage = __GetOrCreateDistanceImage(distanceImage, warpedImage.shape)

    (fixedImage, centerDistanceImage) = assemble.WarpedImageToFixedSpace(transform,
                                                                         (height, width),
                                                                         [warpedImage, distanceImage],
                                                                         botleft=(minY, minX),
                                                                         area=(height, width),
                                                                         cval=[0, __MaxZBufferValue(np.float16)])

    del warpedImage
    del distanceImage

    return TransformedImageData.Create(fixedImage.astype(np.float16),
                                       centerDistanceImage.astype(np.float16),
                                       transform,
                                       transformScale,
                                       SingleThreadedInvoke=SingleThreadedInvoke)

if __name__ == '__main__':
    pass
