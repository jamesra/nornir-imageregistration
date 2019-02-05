'''
Created on Mar 29, 2013

@author: u0490822
'''

import copy
import os

import nornir_imageregistration
from nornir_imageregistration.files.mosaicfile import MosaicFile

import nornir_imageregistration.arrange_mosaic as arrange
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.transforms.factory as tfactory
import nornir_imageregistration.transforms.utils as tutils
import nornir_pools
import numpy as np

from . import spatial




class Mosaic(object):
    '''
    Maps images into a mosaic with a transform
    '''

    @classmethod
    def LoadFromMosaicFile(cls, mosaicfile):
        '''Return a dictionary mapping tiles to transform objects'''
        
        ImageToTransform = {}
        if isinstance(mosaicfile, str):
            print("Loading mosaic: " + mosaicfile)
            mosaicObj = MosaicFile.Load(mosaicfile)
            if mosaicObj is None:
                raise ValueError("Expected valid mosaic file path: {}".format(mosaicfile))
            
            # Don't copy, we throw away the mosaic object
            ImageToTransform = mosaicObj.ImageToTransformString
        elif isinstance(mosaicfile, MosaicFile):
            # Copy the transforms to ensure we don't break anything
            ImageToTransform = copy.deepcopy(mosaicfile.ImageToTransformString)
        else:
            raise ValueError("Expected valid mosaic file path or object")

        Mosaic._ConvertTransformStringsToTransforms(ImageToTransform)

        # keys = list(mosaicfile.ImageToTransformString.keys())
        # keys.sort()
        # for k, v in mosaicfile.ImageToTransformString.items():
            # print("Parsing transform for : " + k)
            # ImageToTransform[k] = tfactory.LoadTransform(v, pixelSpacing=1.0)

        return Mosaic(ImageToTransform)

    def ToMosaicFile(self):
        mfile = MosaicFile()

        for k, v in list(self.ImageToTransform.items()):
            mfile.ImageToTransformString[k] = tfactory.TransformToIRToolsString(v)

        return mfile

    def SaveToMosaicFile(self, mosaicfile):

        mfile = self.ToMosaicFile()
        mfile.Save(mosaicfile)

    @classmethod
    def TranslateMosaicFileToZeroOrigin(cls, path):
        '''Translate the origin to zero if needed.
        :return: True if translation was required.  False if the mosaic was already at zero
        '''
        mosaicObj = Mosaic.LoadFromMosaicFile(path)

        if mosaicObj.IsOriginAtZero():
            return False

        mosaicObj.TranslateToZeroOrigin()
        mosaicObj.SaveToMosaicFile(path)
        return True

    @property
    def ImageToTransform(self):
        return self._ImageToTransform

    def __init__(self, ImageToTransform=None):
        '''
        Constructor
        '''

        if ImageToTransform is None:
            ImageToTransform = dict()
        else:
            Mosaic._ConvertTransformStringsToTransforms(ImageToTransform)

        self._ImageToTransform = ImageToTransform
        self.ImageScale = 1

    @classmethod
    def _ConvertTransformStringsToTransforms(cls, image_to_transform):
        '''If the dictionary contains transforms in string format convert them to transform objects in place'''

        for (image, transform) in image_to_transform.items():
            if isinstance(transform, str):
                transform_object = tfactory.LoadTransform(transform)
                image_to_transform[image] = transform_object

        return

    @property
    def FixedBoundingBox(self):
        '''Calculate the bounding box of the warped position for a set of transforms
           (minX, minY, maxX, maxY)'''

        return tutils.FixedBoundingBox(list(self.ImageToTransform.values()))

    @property
    def MappedBoundingBox(self):
        '''Calculate the bounding box of the warped position for a set of transforms
           (minX, minY, maxX, maxY)'''

        return tutils.MappedBoundingBox(list(self.ImageToTransform.values()))

    @property
    def FixedBoundingBoxWidth(self):
        return tutils.FixedBoundingBoxWidth(list(self.ImageToTransform.values()))

    @property
    def FixedBoundingBoxHeight(self):
        return tutils.FixedBoundingBoxHeight(list(self.ImageToTransform.values()))

    @property
    def MappedBoundingBoxWidth(self):
        return tutils.MappedBoundingBoxWidth(list(self.ImageToTransform.values()))

    @property
    def MappedBoundingBoxHeight(self):
        return tutils.MappedBoundingBoxHeight(list(self.ImageToTransform.values()))


    def TileFullPaths(self, tilesDir):
        '''Return a list of full paths to the tile for each transform'''
        return [os.path.join(tilesDir, x) for x in list(self.ImageToTransform.keys())]


    def IsOriginAtZero(self):
        '''
        :return True if the mosaic origin is at (0,0).  Otherwise False
        :rtype: bool
        '''
        return tutils.IsOriginAtZero(self.ImageToTransform.values())

    def TranslateToZeroOrigin(self):
        '''Ensure that the transforms in the mosaic do not map to negative coordinates'''

        tutils.TranslateToZeroOrigin(list(self.ImageToTransform.values()))

    def TranslateFixed(self, offset):
        '''Translate the fixed space coordinates of all images in the mosaic'''
        for t in list(self.ImageToTransform.values()):
            t.TranslateFixed(offset)

    @classmethod
    def TranslateLayout(cls, Images, Positions, ImageScale=1):
        '''Creates a layout for the provided images at the provided
           It is assumed that Positions are not scaled, but the image size may be scaled'''

        raise Exception("Not implemented")

    def CreateTilesPathList(self, tilesPath, keys=None):
        if keys is None:
            keys = sorted(self.ImageToTransform.keys())
            
        if tilesPath is None:
            return keys
        else:
            return [os.path.join(tilesPath, x) for x in keys]


    def _TransformsSortedByKey(self):
        '''Return a list of transforms sorted according to the sorted key values'''
        
        values = []
        for k, item in sorted(self.ImageToTransform.items()):
            values.append(item)
            
        return values
         

    def ArrangeTilesWithTranslate(self, tiles_path,
                                   image_scale=None, 
                                   excess_scalar=1.5,
                                   min_overlap=None,
                                   feature_score_threshold=None,
                                   max_relax_iterations=None,
                                   max_relax_tension_cutoff=None):
        
        # We don't need to sort, but it makes debugging easier, and I suspect ensuring tiles are registered in the same order may increase reproducability
        (layout, tiles) = arrange.TranslateTiles2(transforms=self._TransformsSortedByKey(),
                                                 imagepaths=self.CreateTilesPathList(tiles_path),
                                                 excess_scalar=excess_scalar,
                                                 feature_score_threshold=feature_score_threshold,
                                                 image_scale=image_scale, 
                                                 max_relax_iterations=max_relax_iterations,
                                                 max_relax_tension_cutoff=max_relax_tension_cutoff,
                                                 min_overlap=min_overlap)
        return layout.ToMosaic(tiles)
    
    def RefineLayout(self, tilesPath):

        # We don't need to sort, but it makes debugging easier, and I suspect ensuring tiles are registered in the same order may increase reproducability
        (layout, tiles) = nornir_imageregistration.RefineGrid(self._TransformsSortedByKey(), self.CreateTilesPathList(tilesPath))
        return layout.ToMosaic(tiles)
    
    
    def QualityScore(self, tilesPath):
        
        score = arrange.ScoreMosaicQuality(self._TransformsSortedByKey(), self.CreateTilesPathList(tilesPath))
        return score


    def AssembleTiles(self, tilesPath, FixedRegion=None, usecluster=False, requiredScale=None):
        '''Create a single large mosaic.
        :param str tilesPath: Directory containing tiles referenced in our transform
        :param array FixedRegion: [MinY MinX MaxY MaxX] boundary of image to assemble
        :param boolean usecluster: Offload work to other threads or nodes if true
        :param float requiredScale: Optimization parameter, eliminates need for function to compare input images with transform boundaries to determine scale
        '''

        # Left off here, I need to split this function so that FixedRegion has a consistent meaning

        # Ensure that all transforms map to positive values
        # self.TranslateToZeroOrigin()

        if not FixedRegion is None:
            spatial.RaiseValueErrorOnInvalidBounds(FixedRegion)

        # Allocate a buffer for the tiles
        tilesPathList = self.CreateTilesPathList(tilesPath)

        if usecluster and len(tilesPathList) > 1:
            cpool = nornir_pools.GetGlobalMultithreadingPool()
            return at.TilesToImageParallel(self._TransformsSortedByKey(), tilesPathList, pool=cpool, FixedRegion=FixedRegion, requiredScale=requiredScale)
        else:
            # return at.TilesToImageParallel(self.ImageToTransform.values(), tilesPathList)
            return at.TilesToImage(self._TransformsSortedByKey(), tilesPathList, FixedRegion=FixedRegion, requiredScale=requiredScale)
