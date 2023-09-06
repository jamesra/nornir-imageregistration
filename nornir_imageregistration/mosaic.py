'''
Created on Mar 29, 2013

@author: u0490822
'''

import copy
import os
import numpy as np


import nornir_imageregistration
from nornir_imageregistration.files.mosaicfile import MosaicFile
  
import nornir_imageregistration.transforms.factory as tfactory
import nornir_imageregistration.transforms.utils as tutils
import nornir_pools 


class Mosaic(object):
    '''
    Maps image names into a mosaic with a transform.
    The image downsample is unknown to the mosaic.  Use
    Tiles when you are ready to arrange or assemble sets
    of tiles.
    '''

    @classmethod
    def LoadFromMosaicFile(cls, mosaicfile, use_cp: bool=False):
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

        Mosaic._ConvertTransformStringsToTransforms(ImageToTransform, use_cp=use_cp)

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
        self.ImageScale = 1 #All .mosaic file transforms are stored on disk at full resolution

    @classmethod
    def _ConvertTransformStringsToTransforms(cls, image_to_transform, use_cp: bool=False):
        '''If the dictionary contains transforms in string format convert them to transform objects in place'''

        for (image, transform) in image_to_transform.items():
            if isinstance(transform, str):
                transform_object = tfactory.LoadTransform(transform, use_cp=use_cp)
                image_to_transform[image] = transform_object

        return

    @property
    def FixedBoundingBox(self):
        '''Calculate the bounding box of the warped position for a set of transforms
           (minX, minY, maxX, maxY)'''
        DeprecationWarning("Use TargetBoundingBox of mosaic_tileset instead")
        return tutils.FixedBoundingBox(list(self.ImageToTransform.values()))

    @property
    def MappedBoundingBox(self):
        '''Calculate the bounding box of the warped position for a set of transforms
           (minX, minY, maxX, maxY)'''
        DeprecationWarning("Use SourceBoundingBox of mosaic_tileset instead")
        return tutils.MappedBoundingBox(list(self.ImageToTransform.values()))

    

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
        '''Ensure that the transforms in the mosaic do not map to negative coordinates
        :return: A Rectangle describing the new fixed space bounding box
        '''

        return tutils.TranslateToZeroOrigin(list(self.ImageToTransform.values()))

    def TranslateFixed(self, offset):
        '''Translate the fixed space coordinates of all images in the mosaic'''
        for t in list(self.ImageToTransform.values()):
            t.TranslateFixed(offset)
    
    def EnsureTransformsHaveMappedBoundingBoxes(self, image_scale, image_path, all_same_dims=True):
        '''
        If a transform does not have a mapped bounding box, define it using the image dimensions
        :param image_path:
        :param float image_scale: Downsample factor of image files
        :parma str image_path: Directory containing image files
        :param bool all_same_dims: If true, cache image dimensions and re-use for all transforms. 
        '''
        
        cached_tile_shape = None
        
        for (file, transform) in self.ImageToTransform.items(): 
             
            if transform.MappedBoundingBox is None:
                if all_same_dims == True: 
                    mapped_bbox_shape = cached_tile_shape
                
                if mapped_bbox_shape is None:
                    image_full_path = os.path.join(image_path, file)
                    mapped_bbox_shape = nornir_imageregistration.GetImageSize(image_full_path)
                    mapped_bbox_shape = np.array(mapped_bbox_shape, dtype=np.int32) * (1.0 / image_scale)
                    cached_tile_shape = mapped_bbox_shape
                    
                transform.MappedBoundingBox = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), mapped_bbox_shape)      
            
    

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

    
    
    def QualityScore(self, tilespath, downsample):
        tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(self, image_folder=tilespath, image_to_source_space_scale=downsample)
        score = nornir_imageregistration.arrange_mosaic.ScoreMosaicQuality(tileset)
        return score

