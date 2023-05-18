from __future__ import annotations

import copy
import logging
import os
import typing

import numpy as np

import nornir_imageregistration


def CreateFromMosaic(mosaic: str | nornir_imageregistration.mosaic.Mosaic, image_folder: str,
                     image_to_source_space_scale: float) -> MosaicTileset:
    """
    :param mosaic: The mosaic we are pulling transforms and image filenames from
    :param image_to_source_space_scale: Scalar for the transform source space coordinates.  Must match the change in scale of input images relative to the transform source space coordinates.  So if downsampled by
    4 images are used and the transform is at full-resolution as is customary this value should be 0.25.
    """

    if mosaic is None:
        raise ValueError("mosaic parameter must not be None")

    if image_to_source_space_scale is None:
        raise ValueError("image_to_source_space_scale parameter must not be None")

    if isinstance(mosaic, str):
        mosaic = nornir_imageregistration.mosaic.Mosaic.LoadFromMosaicFile(mosaic)
    elif not isinstance(mosaic, nornir_imageregistration.mosaic.Mosaic):
        raise ValueError("mosaic must be a file path or a Mosaic object")

    obj = MosaicTileset(image_to_source_space_scale=image_to_source_space_scale)
    for i, item in enumerate(mosaic.ImageToTransform.items()):
        filename = item[0]
        transfrom = item[1]

        tile_number = i
        try:
            (filename_root, _) = os.path.splitext(filename)
            tile_number = int(filename_root)
        except:
            pass

        if image_folder is not None:
            filename = os.path.join(image_folder, filename)

        if not os.path.exists(filename):
            log = logging.getLogger(__name__ + ".Create")
            log.error("Missing tile: " + filename)
            continue

        tile = nornir_imageregistration.tile.Tile(transfrom, filename,
                                                  image_to_source_space_scale=image_to_source_space_scale,
                                                  ID=tile_number)
        obj[tile_number] = tile

    return obj


def Create(transforms, imagepaths, image_to_source_space_scale: float) -> dict[int, nornir_imageregistration.tile.Tile]:
    """
    :param transforms:
    :param imagepaths:
    :param image_to_source_space_scale:
    """

    obj = MosaicTileset(image_to_source_space_scale=image_to_source_space_scale)
    for i, t in enumerate(transforms):

        if not os.path.exists(imagepaths[i]):
            log = logging.getLogger(__name__ + ".Create")
            log.error("Missing tile: " + imagepaths[i])
            continue

        tile_number = i

        filename = imagepaths[i]
        try:
            (filename_root, _) = os.path.splitext(os.path.basename(filename))
            tile_number = int(filename_root)
        except:
            pass

        tile = nornir_imageregistration.tile.Tile(t, imagepaths[i],
                                                  image_to_source_space_scale=image_to_source_space_scale,
                                                  ID=tile_number)
        obj[tile_number] = tile

    return obj


class MosaicTileset(typing.Dict[int, nornir_imageregistration.Tile]):
    """
    A MosaicTileset represents a set of transforms and images at a specific
    resolution.  It is a dictionary where each tile has a unique ID.
    A MosaicTileset can be used to arrange or assemble tiles
    into mosaics at specific resolutions
    """

    def __init__(self, image_to_source_space_scale: float):
        super().__init__()
        if image_to_source_space_scale is None:
            raise ValueError("image_to_source_space_scale is None")

        if image_to_source_space_scale < 1:
            raise ValueError(
                "This might be OK... but images are almost always downsampled.  This exception was added to migrate from old code to this class because at that time all scalars were positive.  For example a downsampled by 4 image must have coordinates multiplied by 4 to match the full-res source space of the transform.")

        self._source_space_bounding_box = None
        self._target_space_bounding_box = None
        self._image_to_source_space_scale = image_to_source_space_scale

    @property
    def image_to_source_space_scale(self) -> float:
        return self._image_to_source_space_scale

    @property
    def MappedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        DeprecationWarning("MappedBoundingBox")
        return self.SourceBoundingBox

    @property
    def SourceBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """
        The bounding box of the transform in source space
        """
        if self._source_space_bounding_box is not None:
            return self._source_space_bounding_box

        if len(self) == 0:
            raise ValueError(f"No transforms in {self.__class__}")

        transforms = self.values()
        self._source_space_bounding_box = nornir_imageregistration.Rectangle.Union(
            [t.SourceSpaceBoundingBox for t in transforms])
        return self._source_space_bounding_box

    @property
    def FixedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        DeprecationWarning("FixedBoundingBox")
        return self.TargetBoundingBox

    @property
    def TargetBoundingBox(self) -> nornir_imageregistration.Rectangle:
        """
        The bounding box of the transform in target space
        """
        if self._target_space_bounding_box is not None:
            return self._target_space_bounding_box

        if len(self) == 0:
            raise ValueError(f'No transforms in {self.__class__}')

        transforms = self.values()
        self._target_space_bounding_box = nornir_imageregistration.Rectangle.Union(
            [t.TargetSpaceBoundingBox for t in transforms])
        return self._target_space_bounding_box

    @property
    def FixedBoundingBoxWidth(self) -> float:
        return self.TargetBoundingBox.Width

    @property
    def FixedBoundingBoxHeight(self) -> float:
        return self.TargetBoundingBox.Height

    @property
    def MappedBoundingBoxWidth(self) -> float:
        return self.SourceBoundingBox.Width

    @property
    def MappedBoundingBoxHeight(self) -> float:
        return self.SourceBoundingBox.Height

    def TargetSpaceIntersections(self, rect):
        """
        :returns: tiles that intersect the provided rectangle in target space
        """
        tile_bbox_list = [(t, t.TargetSpaceBoundingBox) for t in self.values()]
        rset = nornir_imageregistration.RectangleSet.Create([bbox for (t, bbox) in tile_bbox_list])
        intersections = rset.Intersect(rect)

        return [tile_bbox_list[i][0] for i in intersections]

    def SourceSpaceIntersections(self, rect):
        """
        :returns: tiles that intersect the provided rectangle in source space
        """
        tile_bbox_list = [(t, t.SourceSpaceBoundingBox) for t in self.values()]
        rset = nornir_imageregistration.RectangleSet.Create([bbox for (t, bbox) in tile_bbox_list])
        intersections = rset.Intersect(rect)

        return [tile_bbox_list[i][0] for i in intersections]

    def CalculateGridDimensions(self, tile_dims, expected_scale=1):
        """
        :param tuple tile_dims: (Height, Width) of tiles we are dividing the mosaic into
        :param float expected_scale: The scale factor applied to the mosaic before dividing it into tiles, default is 1
        """
        tile_dims = np.asarray(tile_dims, dtype=np.int64)
        scaled_fixed_bounding_box_shape = np.ceil(self.TargetBoundingBox.shape / (1 / expected_scale)).astype(np.int64,
                                                                                                              copy=False)
        return nornir_imageregistration.TileGridShape(scaled_fixed_bounding_box_shape, tile_size=tile_dims)

    def TranslateToZeroOrigin(self):
        """Translate the origin to zero if needed.
        :return: True if translation was required.  False if the mosaic was already at zero
        """

        current_origin = self.TargetBoundingBox.BottomLeft
        if np.array_equal(current_origin, np.array((0, 0), dtype=current_origin.dtype)):
            return self

        # OK, walk each tile and adjust the bounding box
        self.TranslateTargetSpace(-current_origin)
        assert (np.array_equal(self.TargetBoundingBox.BottomLeft, np.array((0, 0), dtype=current_origin.dtype)))
        return self

    def TranslateTargetSpace(self, offset):
        """
        Adjust our target space coordinates by the provided offset.
        Often used to adjust a set of tiles so the target space bounding box has
        an origin at (0,0) for image generation
        """

        for tile in self.values():
            tile.TranslateTargetSpace(offset)

        if self._target_space_bounding_box is not None:
            self._target_space_bounding_box = nornir_imageregistration.Rectangle.translate(
                self._target_space_bounding_box, offset)

    def AssembleImage(self, FixedRegion: nornir_imageregistration.Rectangle | None = None,
                      usecluster: bool = False,
                      target_space_scale: float | None = None) -> np.typing.NDArray:
        """Create a single image of the mosaic for the requested region.
        :param array FixedRegion: Rectangle object or [MinY MinX MaxY MaxX] boundary of image to assemble
        :param boolean usecluster: Offload work to other threads or nodes if true
        :param float target_space_scale: Scalar for target space, used to adjust size of assembled image
        """

        # Left off here, I need to split this function so that FixedRegion has a consistent meaning

        # Ensure that all transforms map to positive values
        # self.TranslateToZeroOrigin()

        if not FixedRegion is None:
            nornir_imageregistration.spatial.RaiseValueErrorOnInvalidBounds(FixedRegion)

        # Allocate a buffer for the tiles
        # tilesPathList = self.CreateTilesPathList(tilesPath)
        tilesPathList = sorted(self)
        # transformList = [self[path] for path in tilesPathList]

        if usecluster and len(tilesPathList) > 1:
            # cpool = nornir_pools.GetGlobalMultithreadingPool()
            return nornir_imageregistration.assemble_tiles.TilesToImageParallel(self,
                                                                                pool=None,
                                                                                TargetRegion=FixedRegion,
                                                                                target_space_scale=target_space_scale)
            # source_space_scale=self._image_to_source_space_scale)
        else:
            # return at.TilesToImageParallel(self.ImageToTransform.values(), tilesPathList)
            return nornir_imageregistration.assemble_tiles.TilesToImage(self,
                                                                        TargetRegion=FixedRegion,
                                                                        target_space_scale=target_space_scale)

    def GenerateOptimizedTiles(self, tile_dims=None, max_temp_image_area=None, usecluster=True, target_space_scale=None,
                               source_space_scale=None):
        """
        Divides the mosaic into a grid of smaller non-overlapping tiles.  Yields each tile along with their coordinates in the grid.
        :param max_temp_image_area:
        :param tuple tile_dims: Size of the optimized tiles
        :param boolean usecluster: Offload work to other threads or nodes if true
        :param float target_space_scale: Scalar for target space, used to adjust size of assembled image
        :param float source_space_scale: Optimization parameter, eliminates need for function to compare input images with transform boundaries to determine scale
        """

        # TODO: Optimize how we search for transforms that overlap the working_image for small working image sizes 
        if tile_dims is None:
            tile_dims = (512, 512)

        tile_dims = np.asarray(tile_dims)

        if source_space_scale is None:
            source_space_scale = self.image_to_source_space_scale  # nornir_imageregistration.tileset.MostCommonScalar(self._TransformsSortedByKey(), self.CreateTilesPathList(tilesPath))

        if target_space_scale is None:
            target_space_scale = source_space_scale

        scaled_tile_dims = tile_dims / target_space_scale  # tile_dims * ( 1 / target_space_scale), The dimensions of the tile if assembled at full-resolution

        mosaic_fixed_bounding_box = self.TargetBoundingBox
        if not np.array_equal(mosaic_fixed_bounding_box.BottomLeft, np.asarray((0, 0))):
            self.TranslateToZeroOrigin()
            mosaic_fixed_bounding_box = self.TargetBoundingBox

        grid_dims = nornir_imageregistration.TileGridShape(mosaic_fixed_bounding_box.shape * target_space_scale,
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
        if working_image_origin[0] != 0 or working_image_origin[1] != 0:
            raise ValueError(f"Expected working_image_origin of (0,0) for assemble {working_image_origin}")

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
                assert (working_image_shape[1] > 0)

            fixed_region = nornir_imageregistration.Rectangle.CreateFromPointAndArea(origin, working_image_shape)

            (working_image, _mask) = self.AssembleImage(
                FixedRegion=fixed_region,
                usecluster=usecluster,
                target_space_scale=target_space_scale)
            # source_space_scale=source_space_scale)

            del _mask

            (yield from nornir_imageregistration.ImageToTilesGenerator(source_image=working_image,
                                                                       tile_size=tile_dims,
                                                                       grid_shape=working_image_grid_dims,
                                                                       coord_offset=(0, iColumn)))

            del working_image

            iColumn += working_image_grid_dims[1]

        return

    def ArrangeTilesWithTranslate(self,
                                  config: nornir_imageregistration.settings.TranslateSettings):

        # We don't need to sort, but it makes debugging easier, and I suspect ensuring tiles are registered in the same order may increase reproducability
        (layout, tiles) = nornir_imageregistration.TranslateTiles2(self, config=config)
        return layout.ToMosaicTileset(tiles)

    def RefineLayout(self):

        # We don't need to sort, but it makes debugging easier, and I suspect ensuring tiles are registered in the same order may increase reproducability
        (layout, tiles) = nornir_imageregistration.RefineGrid(self)
        return layout.ToMosaic(tiles)

    def QualityScore(self):
        score = nornir_imageregistration.arrange_mosaic.ScoreMosaicQuality(self)
        return score

    def ToMosaic(self):
        '''Return a mosaic object for this mosaic set'''
        output = {}
        for (ID, tile) in self.items():
            output[os.path.basename(tile.ImagePath)] = copy.deepcopy(tile.Transform)

        return nornir_imageregistration.Mosaic(output)

    def SaveMosaic(self, output_path):
        created_mosaic = self.ToMosaic()
        created_mosaic.SaveToMosaicFile(output_path)
