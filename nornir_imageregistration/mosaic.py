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
        '''Ensure that the transforms in the mosaic do not map to negative coordinates
        :return: A Rectangle describing the new fixed space bounding box
        '''

        return tutils.TranslateToZeroOrigin(list(self.ImageToTransform.values()))

    def TranslateFixed(self, offset):
        '''Translate the fixed space coordinates of all images in the mosaic'''
        for t in list(self.ImageToTransform.values()):
            t.TranslateFixed(offset)
            
            
    def CalculateGridDimensions(self, tile_dims, expected_scale=1):
        '''
        :param tuple tile_dims: (Height, Width) of tiles we are dividing the mosaic into
        :param float expected_scale: The scale factor applied to the mosaic before dividing it into tiles, default is 1
        '''
        tile_dims = np.asarray(tile_dims, dtype=np.int64)
        scaled_fixed_bounding_box_shape = np.ceil(self.FixedBoundingBox.shape / (1 / expected_scale)).astype(np.int64)
        return nornir_imageregistration.TileGridShape(scaled_fixed_bounding_box_shape, tile_size=tile_dims)

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
                                   min_translate_iterations=None,
                                   offset_acceptance_threshold=None,
                                   max_relax_iterations=None,
                                   max_relax_tension_cutoff=None):
        
        # We don't need to sort, but it makes debugging easier, and I suspect ensuring tiles are registered in the same order may increase reproducability
        (layout, tiles) = arrange.TranslateTiles2(transforms=self._TransformsSortedByKey(),
                                                 imagepaths=self.CreateTilesPathList(tiles_path),
                                                 excess_scalar=excess_scalar,
                                                 feature_score_threshold=feature_score_threshold,
                                                 image_scale=image_scale,
                                                 min_translate_iterations=min_translate_iterations,
                                                 offset_acceptance_threshold=offset_acceptance_threshold,
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

    def AssembleImage(self, tilesPath, FixedRegion=None, usecluster=False, requiredScale=None):
        '''Create a single image of the mosaic for the requested region.
        :param str tilesPath: Directory containing tiles referenced in our transform
        :param array FixedRegion: Rectangle object or [MinY MinX MaxY MaxX] boundary of image to assemble
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
        
    def GenerateOptimizedTiles(self, tilesPath, tile_dims=None,
                               max_temp_image_area=None, usecluster=True,
                               requiredScale=None):
        '''Divides the mosaic into a grid of smaller non-overlapping tiles.  Yields each tile along with their coordinates in the grid.
        :param str tilesPath: Directory containing tiles referenced in our transform
        :param tuple tile_dims: Size of the optimized tiles
        :param max_image_dims: The maximum size of image we will assemble at any time.  Smaller values consume less memory, larger values run faster
        :param boolean usecluster: Offload work to other threads or nodes if true
        :param float requiredScale: Optimization parameter, eliminates need for function to compare input images with transform boundaries to determine scale
        '''
        
        # TODO: Optimize how we search for transforms that overlap the working_image for small working image sizes 
        if tile_dims is None:
            tile_dims = (512, 512)
            
        tile_dims = np.asarray(tile_dims)
            
        if requiredScale is None:
            requiredScale = nornir_imageregistration.tileset.MostCommonScalar(self._TransformsSortedByKey(), self.CreateTilesPathList(tilesPath))
            
        scaled_tile_dims = tile_dims / requiredScale # tile_dims * ( 1 / requiredScale), The dimensions of the tile if assembled at full-resolution
        
        mosaic_fixed_bounding_box = self.FixedBoundingBox
        if not np.array_equal(mosaic_fixed_bounding_box.BottomLeft, np.asarray((0, 0))):
            self.TranslateToZeroOrigin()
            mosaic_fixed_bounding_box = self.FixedBoundingBox
            
        grid_dims = nornir_imageregistration.TileGridShape(mosaic_fixed_bounding_box.shape * requiredScale,
                                                           tile_size=tile_dims)
        
        # Lets build long vertical columns.  Figure out how many columns we can assemble at a time
        scaled_mosaic_fixed_bounding_box_shape = grid_dims * tile_dims
        
        if max_temp_image_area is None:
            max_temp_image_area = mosaic_fixed_bounding_box.Area
            
        template_image_shape = None  # Shape of the image we will assemble at each step
        template_image_grid_dims = None  # Dimensions of tiles contained within each working_image
        if max_temp_image_area >= np.prod(scaled_mosaic_fixed_bounding_box_shape):
            template_image_shape = mosaic_fixed_bounding_box.shape
            template_image_grid_dims = grid_dims
        else:
            num_rows = grid_dims[0]
            max_column_width = max_temp_image_area / scaled_mosaic_fixed_bounding_box_shape[0] 
            num_columns = int(np.floor(max_column_width / tile_dims[1]))
            if num_columns < 1:
                num_columns = 1
                
            template_image_grid_dims = np.asarray((num_rows, num_columns))
            template_image_shape = template_image_grid_dims * scaled_tile_dims
            
        working_image_origin = mosaic_fixed_bounding_box.BottomLeft
        assert(working_image_origin[0] == 0 and working_image_origin[1] == 0)
        
        iColumn = 0
        while iColumn < grid_dims[1]:
            # Assemble a strip of images, divide them up and save
            origin = (0, iColumn * scaled_tile_dims[1]) + working_image_origin
            
            working_image_shape = template_image_shape
            working_image_grid_dims = template_image_grid_dims
            # If we are on the final column don't make it larger than necessary
            if working_image_grid_dims[1] + iColumn > grid_dims[1]:
                working_image_grid_dims[1] = grid_dims[1] - iColumn
                working_image_shape[1] = working_image_grid_dims[1] * scaled_tile_dims[1]
                assert(working_image_shape[1] > 0)
            
            fixed_region = nornir_imageregistration.Rectangle.CreateFromPointAndArea(origin, working_image_shape)
                     
            (working_image, _mask ) = self.AssembleImage(tilesPath=tilesPath,
                                               FixedRegion=fixed_region,
                                               usecluster=usecluster,
                                               requiredScale=requiredScale)

            del _mask
            
            (yield from nornir_imageregistration.ImageToTilesGenerator(source_image=working_image,
                                                                       tile_size=tile_dims,
                                                                       grid_shape=working_image_grid_dims,
                                                                       coord_offset=(0, iColumn)))
        
            del working_image
            
            iColumn += working_image_grid_dims[1]
          
        return
