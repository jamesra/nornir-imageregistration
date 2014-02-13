'''
Created on Oct 28, 2013

Deals with assembling images composed of mosaics or dividing images into tiles
'''

import assemble
import tiles
import numpy as np

import scipy.spatial.distance
from scipy import stats
# from nornir_imageregistration.files.mosaicfile import MosaicFile
# from nornir_imageregistration.mosaic import Mosaic
import core
import os
import logging
import transforms.utils as tutils
import tiles
# import nornir_imageregistration.transforms.meshwithrbffallback as meshwithrbffallback
# import nornir_imageregistration.transforms.triangulation as triangulation
import nornir_pools as pools
import copy

class TransformedImageData(object):

    @property
    def image(self):
        return self._image

    @property
    def centerDistanceImage(self):
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

    @classmethod
    def Create(cls, image, centerDistanceImage, transform, scale):
        o = TransformedImageData()
        o._image = image
        o._centerDistanceImage = centerDistanceImage
        o._transformScale = scale
        o._transform = transform

        return o


    def Clear(self):
        '''Sets attributes to None to encourage garbase collection'''
        self._image = None
        self._centerDistanceImage = None
        self._transformScale = None
        self._transform = None


    def __init__(self, errorMsg=None):
        self._image = None
        self._centerDistanceImage = None
        self._transformScale = None
        self._transform = None
        self._errmsg = errorMsg

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

    minX = offset[1]
    minY = offset[0]
    maxX = minX + SubImage.shape[1]
    maxY = minY + SubImage.shape[0]

    tempFullImage = FullImage[minY:maxY, minX:maxX]
    tempZBuffer = FullZBuffer[minY:maxY, minX:maxX]

    iUpdate = tempZBuffer > SubZBuffer

    tempFullImage[iUpdate] = SubImage[iUpdate]
    tempZBuffer[iUpdate] = SubZBuffer[iUpdate]

    # FullImage[minY:maxY, minX:maxX] += SubImage
    FullImage[minY:maxY, minX:maxX] = tempFullImage
    FullZBuffer[minY:maxY, minX:maxX] = tempZBuffer

    return (FullImage, FullZBuffer)


def distFunc(i, j):
    print(str(i) + ", " + str(j))
    a = np.array(i, dtype=np.float)
    a = a * a
    b = np.array(j, dtype=np.float)
    b = b * b
    c = a + b
    c = np.sqrt(c)
    return c
    # return scipy.spatial.distance.cdist(np.dstack((i, j)), np.array([[5.0], [5.0]], dtype=np.float))


def CreateDistanceImage(shape, dtype=None):

    if dtype is None:
        dtype = np.float32

    center = [shape[0] / 2.0, shape[1] / 2.0]

    x_range = np.linspace(-center[1], center[1], shape[1])
    y_range = np.linspace(-center[0], center[0], shape[0])

    x_range = x_range * x_range
    y_range = y_range * y_range

    distance = np.zeros(shape, dtype=dtype)

    for i in range(0, shape[0]):
        distance[i, :] = x_range + y_range[i]

    distance = np.sqrt(distance)

    return distance


def __MaxZBufferValue(dtype):
    return np.finfo(dtype).max


def __CreateOutputBuffer(transforms, requiredScale=None):
    (minX, minY, maxX, maxY) = tutils.FixedBoundingBox(transforms)

    fullImage = np.zeros((np.ceil(requiredScale * maxY), np.ceil(requiredScale * maxX)), dtype=np.float16)
    fullImageZbuffer = np.ones(fullImage.shape, dtype=fullImage.dtype) * __MaxZBufferValue(fullImage.dtype)

    return (fullImage, fullImageZbuffer)


def __GetOrCreateDistanceImage(distanceImage, imagePath):
    '''Determines size of the image.  Returns a distance image to match the size if the passed existing image is not the correct size.'''

    size = core.GetImageSize(imagePath)
    if distanceImage is None:
        distanceImage = CreateDistanceImage(size)
    else:
        if not np.array_equal(distanceImage.shape, size):
            distanceImage = CreateDistanceImage(size)

    return distanceImage


def TilesToImage(transforms, imagepaths, requiredScale=None):

    assert(len(transforms) == len(imagepaths))

    logger = logging.getLogger('TilesToImage')

    if requiredScale is None:
        requiredScale = tiles.MostCommonScalar(transforms, imagepaths)

    (fullImage, fullImageZbuffer) = __CreateOutputBuffer(transforms, requiredScale)

    distanceImage = None

    for i, transform in enumerate(transforms):
        imagefullpath = imagepaths[i]

        distanceImage = __GetOrCreateDistanceImage(distanceImage, imagefullpath)

        transformedImageData = __TransformTile(transform, imagefullpath, distanceImage, requiredScale=requiredScale)

        (minX, minY, maxX, maxY) = transformedImageData.transform.FixedBoundingBox

        (fullImage, fullImageZbuffer) = CompositeImageWithZBuffer(fullImage, fullImageZbuffer, transformedImageData.image, transformedImageData.centerDistanceImage, (np.floor(minY), np.floor(minX)))

        del transformedImageData

    mask = fullImageZbuffer < __MaxZBufferValue(fullImageZbuffer.dtype)
    del fullImageZbuffer

    fullImage[fullImage < 0] = 0
    fullImage[fullImage > 1.0] = 1.0

    return (fullImage, mask)


def TilesToImageParallel(transforms, imagepaths, pool=None, requiredScale=None):
    '''Assembles a set of transforms and imagepaths to a single image using parallel techniques'''

    assert(len(transforms) == len(imagepaths))

    logger = logging.getLogger('TilesToImageParallel')

    distanceImage = None

    if requiredScale is None:
        requiredScale = tiles.MostCommonScalar(transforms, imagepaths)

    if pool is None:
        pool = pools.GetGlobalMultithreadingPool()

    tasks = []

    (fullImage, fullImageZbuffer) = __CreateOutputBuffer(transforms, requiredScale)

    CheckTaskInterval = 10

    for i, transform in enumerate(transforms):
        imagefullpath = imagepaths[i]
        task = pool.add_task("__TransformTile" + imagefullpath, __TransformTile, transform=transform, imagefullpath=imagefullpath, distanceImage=None, requiredScale=requiredScale)
        task.transform = transform
        tasks.append(task)

        if not i % CheckTaskInterval == 0:
            continue

        for iTask, t in enumerate(tasks):
            if t.iscompleted:
                transformedImageData = t.wait_return()
                (fullImage, fullImageZbuffer) = __AddTransformedTileToComposite(transformedImageData, fullImage, fullImageZbuffer)
                del transformedImageData
                del tasks[iTask]

    logger.info('All warps queued, integrating results into final image')

    while len(tasks) > 0:
        t = tasks.pop()
        transformedImageData = t.wait_return()

        if transformedImageData is None:
            logger = logging.getLogger('TilesToImageParallel')
            logger.error('Convert task failed: ' + str(t))
            continue

        (fullImage, fullImageZbuffer) = __AddTransformedTileToComposite(transformedImageData, fullImage, fullImageZbuffer)
        del transformedImageData
        del t

    mask = fullImageZbuffer < __MaxZBufferValue(fullImageZbuffer.dtype)
    del fullImageZbuffer

    fullImage[fullImage < 0] = 0
    fullImage[fullImage > 1.0] = 1.0

    logger.info('Final image complete')

    return (fullImage, mask)


def __AddTransformedTileToComposite(transformedImageData, fullImage, fullImageZBuffer):

    if transformedImageData.image is None:
        logger = logging.getLogger('TilesToImageParallel')
        logger.error('Convert task failed: ' + str(transformedImageData))
        if not transformedImageData.errormsg is None:
            logger.error(transformedImageData.errormsg)
            return (fullImage, fullImageZBuffer)

    (minX, minY, maxX, maxY) = transformedImageData.transform.FixedBoundingBox
    # print "%g %g" % (minX, minY)
    (fullImage, fullImageZBuffer) = CompositeImageWithZBuffer(fullImage, fullImageZBuffer, transformedImageData.image, transformedImageData.centerDistanceImage, (np.floor(minY), np.floor(minX)))
    transformedImageData.Clear()

    return (fullImage, fullImageZBuffer)


def __TransformTile(transform, imagefullpath, distanceImage=None, requiredScale=None):
    '''Transform the passed image.  DistanceImage is an existing image recording the distance to the center of the
       image for each pixel.  requiredScale is used when the image size does not match the image size encoded in the
       transform.  A scale will be calculated in this case and if it does not match the required scale the tile will 
       not be transformed.'''

    if not os.path.exists(imagefullpath):
        return TransformedImageData(errorMsg='Tile does not exist ' + imagefullpath)

    # if isinstance(transform, meshwithrbffallback.MeshWithRBFFallback):
       # Don't bother mapping points falling outside the defined boundaries because we won't have image data for it
    #   transform = triangulation.Triangulation(transform.points)

    warpedImage = core.LoadImage(imagefullpath)

    # Automatically scale the transform if the input image shape does not match the transform bounds
    transformScale = tiles.__DetermineScale(transform, warpedImage.shape)

    if not requiredScale is None:
        if not core.ApproxEqual(transformScale, requiredScale):
            return TransformedImageData(errorMsg="%g scale needed for %s is different than required scale %g used for mosaic" % (transformScale, imagefullpath, requiredScale))

    if transformScale != 1.0:
        scaledTransform = copy.deepcopy(transform)
        scaledTransform.Scale(transformScale)
        transform = scaledTransform

    width = transform.FixedBoundingBoxWidth
    height = transform.FixedBoundingBoxHeight

    (minX, minY, maxX, maxY) = transform.FixedBoundingBox

    if distanceImage is None:
        distanceImage = CreateDistanceImage(warpedImage.shape)
    else:
        if not np.array_equal(distanceImage.shape, warpedImage.shape):
            distanceImage = CreateDistanceImage(warpedImage.shape)

    (fixedImage, centerDistanceImage) = assemble.WarpedImageToFixedSpace(transform,
                                                                         (height, width),
                                                                         [warpedImage, distanceImage],
                                                                         botleft=(minY, minX),
                                                                         area=(height, width),
                                                                         cval=[0, __MaxZBufferValue(np.float16)])

    del warpedImage
    del distanceImage

    return TransformedImageData.Create(fixedImage, centerDistanceImage, transform, transformScale)




if __name__ == '__main__':
    pass