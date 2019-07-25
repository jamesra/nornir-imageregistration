'''
Created on Jul 18, 2019

@author: u0490822

A helper class to marshal large images using the file system instead of in-memory. 
'''
import os
import tempfile
import numpy as np
import nornir_pools


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
#         odict["_source_space_scale"] = self._source_space_scale
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
    def source_space_scale(self):
        return self._source_space_scale
    
    @property
    def target_space_scale(self):
        return self._target_space_scale

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
    def Create(cls, image, centerDistanceImage, transform, source_space_scale, target_space_scale, SingleThreadedInvoke):
        o = TransformedImageData()
        o._image = image
        o._centerDistanceImage = centerDistanceImage
        
        o._image_path = None
        o._centerDistanceImage_path = None
        o._source_space_scale = source_space_scale
        o._target_space_scale = target_space_scale
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
        self._source_space_scale = None
        self._target_space_scale = None
        self._transform = None
        
        if not self._centerDistanceImage_path is None or not self._image_path is None:
            pool = nornir_pools.GetGlobalMultithreadingPool()
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
        self._source_space_scale = None
        self._target_space_scale = None
        self._transform = None
        self._errmsg = errorMsg
        self._image_path = None
        self._centerDistanceImage_path = None
        self._tempdir = None
        #self._image_shape = None
        #self._centerDistanceImage_shape = None
        #self._image_dtype = None
        #self._centerDistance_dtype = None
        