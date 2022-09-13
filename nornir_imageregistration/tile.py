'''
Created on Feb 21, 2014

@author: u0490822
'''
 
import os 
import nornir_imageregistration
import numpy as np
from nornir_imageregistration.transforms.base import IDiscreteTransform
from nornir_shared import prettyoutput

class Tile(object):
    '''
    A combination of a transform and a path to an image on disk.  Image will be loaded on demand.
    When serialized with __getstate__ the image and any large data objects will not be serialized
    to facilitate marshaling the object to another processs in an efficient manner.
     
    ''' 
    
    __nextID = 0
    
    def TryEstimateImageToSourceSpaceScalar(self):
        '''
        Calculate image_to_source_space_scale if it was not passed.
        TODO: This may be a function that is no longer needed or should 
        not exist with the refactor of tile and mosaic_tileset.  Probably
        better to require passing image_to_source_space in constructor
        '''
        t_dim = self.SourceSpaceBoundingBox.Dimensions
        i_dim = self.ImageSize
         
        if np.allclose(t_dim, i_dim):
            return 1.0
        else:
            scales = i_dim / t_dim
    
            if np.all(scales == scales[0]): #Make sure both values are equal
                return scales[0]
            else:
                raise ValueError(f"Mismatch between heightScale and widthScale. {scales}")
         
    @property
    def MappedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        '''
        The bounding rectangle of the source space mapped area.
        Limited to the full-resolution image dimensions if it is a continuous transform
        '''
        if self._source_bounding_box is None:
            self._source_bounding_box = self._GetOrCalculateSourceBoundingBox()
            
        return self._source_bounding_box
    
    @property
    def FixedBoundingBox(self) -> nornir_imageregistration.Rectangle:
        '''
        The bounding rectangle of the target space mapped area.
        Limited to the full-resolution image dimensions if it is a continuous transform
        '''
        if self._target_bounding_box is None:
            self._target_bounding_box = self._GetOrCalculateTargetBoundingBox()
            
        return self._target_bounding_box 
    
    @property
    def SourceSpaceBoundingBox(self) -> nornir_imageregistration.Rectangle:
        return self.MappedBoundingBox
    
    @property
    def TargetSpaceBoundingBox(self) -> nornir_imageregistration.Rectangle: 
        return self.FixedBoundingBox
 
    @property
    def FullResolutionImageSize(self):
        dims = self.MappedBoundingBox
        return (dims[nornir_imageregistration.iRect.MaxY] - dims[nornir_imageregistration.iRect.MinY],
                dims[nornir_imageregistration.iRect.MaxX] - dims[nornir_imageregistration.iRect.MinY])

    @property
    def WarpedImageSize(self):
        dims = self.FixedBoundingBox
        return (dims[nornir_imageregistration.iRect.MaxY] - dims[nornir_imageregistration.iRect.MinY], dims[nornir_imageregistration.iRect.MaxX] - dims[nornir_imageregistration.iRect.MinY])

    @property
    def Transform(self):
        '''A string encoding our tile's transform'''
        return self._transform
    
    @Transform.setter
    def Transform(self,val):
        '''A string encoding our tile's transform'''
        self._transform = val
        #Reset the bounding box of the target and source space
        self._target_bounding_box = None 
        self._source_bounding_box = None

    @property
    def Image(self):
        if self._image is None:
            try:
                self._image = nornir_imageregistration.LoadImage(self._imagepath, dtype=np.float16)
            except IOError:
                prettyoutput.LogErr(f'Unable to load {self._imagepath}')
                raise
        
        return self._image
    
    @property
    def ImageSize(self):
        '''
        Size of the image.  It may not match the dimensions of the Source Space
        if the image is downsampled.  Use image_to_source_space_scale to correct.
        '''
        
        if self._image_size is None:
            if self._image is None:
                self._image_size = np.array(nornir_imageregistration.GetImageSize(self._imagepath), np.int64)
            else:
                self._image_size = np.array(self._image.shape, np.int64)
        
        assert(isinstance(self._image_size, np.ndarray))
        return self._image_size
        
    @property
    def PaddedImage(self):
        if self._paddedimage is None:
            self._paddedimage = nornir_imageregistration.PadImageForPhaseCorrelation(self.Image)

        return self._paddedimage

    @property
    def ImagePath(self):
        '''
        Path to the image data on disk.  This should be populated, but is rarely
        None for some unit tests if an image array is passed to the constructor
        '''
        return self._imagepath

    @property
    def FFTImage(self):
        if self._fftimage is None:
            self._fftimage = np.fft.rfft2(self.PaddedImage)

        return self._fftimage
    
    def PrecalculateImages(self):
        temp = self.FFTImage.shape
        
    def Assemble(self, distanceImage=None, target_space_scale=None, TargetRegion=None, SingleThreadedInvoke=False):
        '''Returns the source image tranformed into the target space'''
        if TargetRegion is None:
            TargetRegion = self.TargetSpaceBoundingBox
            
        return nornir_imageregistration.assemble_tiles.TransformTile(self,
                                                                     distanceImage=distanceImage,
                                                                     target_space_scale=target_space_scale,
                                                                     TargetRegion=TargetRegion,
                                                                     SingleThreadedInvoke=SingleThreadedInvoke)

    @property
    def ID(self):
        return self._ID
    
    def __str__(self):
        return f"{self.ID} : {self.ImagePath}"
    
    def TranslateTargetSpace(self, offset):
        '''
        Adjust our target space coordinates by the provided offset.
        Often used to adjust a set of tiles so the target space bounding box has
        an origin at (0,0) for image generation
        '''
        
        self.Transform.TranslateFixed(offset)
        if self._target_bounding_box is not None:
            self._target_bounding_box = nornir_imageregistration.Rectangle.translate(self._target_bounding_box, offset)
        
    
    def Get_Overlapping_Source_Rect(self, overlapping_target_rect):
        ''':return: Rectangle describing which region of the tile_obj image is contained in the overlapping_rect from volume space'''
        source_space_points = self.Transform.InverseTransform(overlapping_target_rect.Corners)    
        return nornir_imageregistration.BoundingPrimitiveFromPoints(source_space_points)
 
    def _GetOrCalculateSourceBoundingBox(self) -> nornir_imageregistration.Rectangle:
        '''
        Returns the bounding rectangle of the source space mapped area.
        Limited to the full-resolution image dimensions if it is a continuous transform
        '''
        if isinstance(self.Transform, IDiscreteTransform):
            return self._transform.MappedBoundingBox
        
        adjusted_image_size = self.ImageSize * self.image_to_source_space_scale
        #image_size = np.array(image_size, dtype=np.int32) * (1.0 / self.source_space_scale)
        return nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), adjusted_image_size)
     
    def _GetOrCalculateTargetBoundingBox(self) -> nornir_imageregistration.Rectangle:
        '''
        Returns the bounding rectangle of the target space mapped area.
        Limited to the full-resolution image dimensions if it is a continuous transform
        '''
        if isinstance(self.Transform, IDiscreteTransform):
            return self._transform.FixedBoundingBox
        
        source_bbox = self.MappedBoundingBox
        target_bbox_corners = self.Transform.Transform(source_bbox.Corners)
        target_bbox = nornir_imageregistration.Rectangle.CreateBoundingRectangleForPoints(target_bbox_corners)
                    
        return target_bbox
        
    
    # def EnsureTileHasMappedBoundingBox(self, cached_tile_shape = None):
    #     '''
    #     If our tile does not have a mapped bounding box, define it using the tile's image dimensions
    #     :param float image_scale: Downsample factor of image files
    #     :param tuple cached_tile_shape: If not None, use these dimensions for the mapped bounding box instead of loading the image to obtain the shape
    #     '''
    #
    #     if self._transform.MappedBoundingBox is None:
    #         if cached_tile_shape is not None: 
    #             mapped_bbox_shape = cached_tile_shape
    #         else:
    #             mapped_bbox_shape = nornir_imageregistration.GetImageSize(self._imagepath)
    #             mapped_bbox_shape = np.array(mapped_bbox_shape, dtype=np.int32) * (1.0 / self.source_space_scale)
    #
    #         self._transform.MappedBoundingBox = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), mapped_bbox_shape)
    #         return mapped_bbox_shape
    #
    #     return None

    def __init__(self, transform, imagepath, image_to_source_space_scale, ID):
        '''
        :param transform: The transform object
        :param imagepath: Full path to the image to be transformed.  This can also be an ndarray for testing purposes, but the tile will not marshall across process boundaries.
        :param float image_to_source_space_scale: Scalar for the transform source space coordinates.  Must match the change in scale of input images relative to the transform source space coordinates.  So if downsampled by
        4 images are used and the transform is at full-resolution as is customary this value should be 0.25.
        Calculated to be correct if None.  Specifying is an optimization to reduce I/O of reading image files to calculate.
        '''

        if transform is None: 
            raise ValueError("transform is None")
        if imagepath is None: 
            raise ValueError("imagepath is None")
        if image_to_source_space_scale is None: 
            raise ValueError("image_to_source_space_scale is None")
        if image_to_source_space_scale < 1:
            raise ValueError("This might be OK... but images are almost always downsampled.  This exception was added to migrate from old code to this class because at that time all scalars were positive.  For example a downsampled by 4 image must have coordinates multiplied by 4 to match the full-res source space of the transform.")

        self._transform = transform
        
        if isinstance(imagepath, str):
            self._imagepath = imagepath
            self._image = None
        elif isinstance(imagepath, np.ndarray):
            self._image = imagepath
            self._imagepath = None
        else:
            raise ValueError("imagepath must be str or ndarray type")
            
        self._paddedimage = None
        self._fftimage = None
        
        self._source_bounding_box = None
        self._target_bounding_box = None
        
        self._image_size = None
        
        self.image_to_source_space_scale = image_to_source_space_scale
        #if image_to_source_space_scale is None:
        #    self.image_to_source_space_scale = self.TryEstimateImageToSourceSpaceScalar()

        if ID is None:
            self._ID = Tile.__nextID
            Tile.__nextID += 1
        else:
            self._ID = ID
         
        if not isinstance(self._ID, int):
            raise TypeError("Tile ID must be an integer: {0}".format(ID))

            
    def __getstate__(self):
        odict = {'_transform': self._transform, '_imagepath': self._imagepath,
                 '_source_bounding_box': self._source_bounding_box, '_target_bounding_box': self._target_bounding_box,
                 'image_to_source_space_scale': self.image_to_source_space_scale, '_image_size': self._image_size,
                 '_ID': self._ID}

        return odict

    def __setstate__(self, dictionary):         
        self.__dict__.update(dictionary)
        self._image = None
        self._paddedimage = None
        self._fftimage = None

    def __repr__(self):
        return "%d: %s" % (self._ID, self._imagepath)
