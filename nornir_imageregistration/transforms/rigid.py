
from typing import *
import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import base, triangulation

from nornir_imageregistration.spatial import Rectangle
import numpy as np


class RigidNoRotation(base.ITransform, base.ITransformTranslation, base.DefaultTransformChangeEvents):
    '''This class is legacy and probably needs a deprecation warning'''
    
    def _transform_rectangle(self, rect):
        if rect is None:
            return None
        
        #Transform the other bounding box
        mapped_corners = self.Transform(rect.Corners)
        return Rectangle.CreateBoundingRectangleForPoints(mapped_corners)
        
    def _inverse_transform_rectangle(self, rect):
        if rect is None:
            return None
        
        #Transform the other bounding box
        mapped_corners = self.InverseTransform(rect.Corners)
        return Rectangle.CreateBoundingRectangleForPoints(mapped_corners)
      
    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''
        self.target_offset  = self.target_offset + offset
        self.OnTransformChanged()

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self.target_offset  = self.target_offset - offset
        self.OnTransformChanged()
        
    def Scale(self, value):
        '''Scale both warped and control space by scalar'''
        self.source_space_center_of_rotation = self.source_space_center_of_rotation * value 
        self.target_offset = self.target_offset * value
        self.OnTransformChanged()
    
    def __init__(self, target_offset = Tuple[float, float], source_rotation_center: Tuple[float, float] | None = None, angle: float | None = None, **kwargs):
        '''
        Creates a Rigid Transformation.  If used only one BoundingBox parameter needs to be specified
        :param tuple target_offset:  The amount to offset points in mapped (source) space to translate them to fixed (target) space
        :param tuple source_rotation_center: The (Y,X) center of rotation in mapped space
        :param float angle: The angle to rotate, in radians
        :param Rectangle FixedBoundingBox:  Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.  
        :param Rectangle MappedBoundingBox: Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.  
        '''
        super(RigidNoRotation, self).__init__()
    
        if angle is None:
            angle = 0.0
        
        if source_rotation_center is None:
            source_rotation_center = (0.0, 0.0)
            
        self.target_offset = nornir_imageregistration.EnsurePointsAre1DNumpyArray(target_offset)
        self.source_space_center_of_rotation = nornir_imageregistration.EnsurePointsAre1DNumpyArray(source_rotation_center)
        self._angle = angle  # type: float
        
        
    def __getstate__(self):
        odict = {'_angle': self._angle, 'target_offset': (self.target_offset[0], self.target_offset[1]),
                 'source_space_center_of_rotation': (self.source_space_center_of_rotation[0],
                                                     self.source_space_center_of_rotation[1])}
        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)
        
        self.target_offset = np.asarray((self.target_offset[0], self.target_offset[1]), dtype=np.float64)
        self.source_space_center_of_rotation = np.asarray((self.source_space_center_of_rotation[0],
                                                           self.source_space_center_of_rotation[1]), dtype=np.float64)
        
        self.OnChangeEventListeners = []
        self.OnTransformChanged()
        
    def __repr__(self):
        return f"Offset: {self.target_offset[0]:03g}y,{self.target_offset[1]:03g}x"

        
    @staticmethod
    def Load(TransformString,pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString, pixelSpacing)
        
    def ToITKString(self):
        #TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return f"Rigid2DTransform_double_2_2 vp 3 {self._angle} {self.target_offset[1]} {self.target_offset[0]} fp 2 {self.source_space_center_of_rotation[1]} {self.source_space_center_of_rotation[0]}"
    
    def Transform(self, points, **kwargs):
        
        
        if not (self._angle is None or self._angle == 0):
            #Look at GetTransformedRigidCornerPoints for a possible implementation
            raise NotImplemented("Rotation is not implemented")
        
        return points + self.target_offset

    def InverseTransform(self, points, **kwargs):
        
        
        if not (self._angle is None or self._angle == 0):
            #Look at GetTransformedRigidCornerPoints for a possible implementation
            raise NotImplemented("Rotation is not implemented")
        
        return points - self.target_offset
    
class Rigid(RigidNoRotation):
    '''
    Applies a rotation+translation transform
    '''
    def __getstate__(self):
        return super(Rigid, self).__getstate__()
    
    def __setstate__(self, dictionary):
        super(Rigid, self).__setstate__(dictionary)
        self.forward_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(self.angle)
        self.inverse_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(-self.angle)
    
    def __init__(self, target_offset, source_rotation_center=None, angle=None, **kwargs):
        '''
        Creates a Rigid Transformation.  If used only one BoundingBox parameter needs to be specified
        :param tuple target_offset:  The amount to offset points in mapped (source) space to translate them to fixed (target) space
        :param tuple source_rotation_center: The (Y,X) center of rotation in mapped space
        :param float angle: The angle to rotate, in radians
        :param Rectangle FixedBoundingBox:  Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.  
        :param Rectangle MappedBoundingBox: Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        '''
        super(Rigid, self).__init__(target_offset, source_rotation_center, angle, **kwargs)
        
        self.forward_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(self.angle)
        self.inverse_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(-self.angle)
    
    @property
    def angle(self):
        return self._angle
    
    @staticmethod
    def Load(TransformString):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString)
    
    def ToITKString(self):
        #TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return "Rigid2DTransform_double_2_2 vp 3 {0} {1} {2} fp 2 {3} {4}".format(self.angle, self.target_offset[1], self.target_offset[0], self.source_space_center_of_rotation[1], self.source_space_center_of_rotation[0])
    
    def Transform(self, points, **kwargs):
        
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        
        if self.angle == 0:
            return points + self.target_offset
        
        numPoints = points.shape[0]
        
        centered_points = points - self.source_space_center_of_rotation
        centered_points = np.transpose(centered_points)
        centered_points = np.vstack((centered_points, np.ones((1, numPoints)))) #Add a row so we can multiply the matrix
        
        
        centered_rotated_points = self.forward_rotation_matrix * centered_points
        centered_rotated_points = np.transpose(centered_rotated_points)
        centered_rotated_points = centered_rotated_points[:,0:2]
          
        rotated_points = centered_rotated_points + self.source_space_center_of_rotation
        output_points = rotated_points + self.target_offset
        return np.asarray(output_points)

    def InverseTransform(self, points, **kwargs):
        
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        
        if self.angle == 0:
            return points - self.target_offset
    
        numPoints = points.shape[0]
        
        input_points = points - self.target_offset
        centered_points = input_points - self.source_space_center_of_rotation
        
        centered_points = np.transpose(centered_points)
        centered_points = np.vstack((centered_points, np.ones((1, numPoints))))
        rotated_points = self.inverse_rotation_matrix * centered_points
        rotated_points = np.transpose(rotated_points)
        rotated_points = rotated_points[:,0:2]
        
        output_points = rotated_points + self.source_space_center_of_rotation
        return np.asarray(output_points)
    
    def __repr__(self):
        return f"Offset: {self.target_offset[0]:03g}y,{self.target_offset[1]:03g}x Angle: {self.angle:03g}deg Rot Center: {self.source_space_center_of_rotation[0]:03g}y,{self.source_space_center_of_rotation[1]:03g}x"
 

class CenteredSimilarity2DTransform(Rigid, base.ITransformScaling):
    '''
    Applies a scale+rotation+translation transform
    '''
    def __getstate__(self):
        odict = super(CenteredSimilarity2DTransform, self).__getstate__()
        odict['_scalar'] = self._scalar
        return odict
    
    def __setstate__(self, dictionary):
        super(CenteredSimilarity2DTransform, self).__setstate__(dictionary)
        self._scalar = dictionary['_scalar']
        
    @property
    def scalar(self):
        return self._scalar
     
    def __init__(self, target_offset: Tuple[float,float], source_rotation_center: Tuple[float, float] = None, angle: float = None, scalar: float = None, **kwargs):
        '''
        Creates a Rigid Transformation.  If used only one BoundingBox parameter needs to be specified
        :param tuple target_offset:  The amount to offset points in mapped (source) space to translate them to fixed (target) space
        :param tuple source_rotation_center: The (Y,X) center of rotation in mapped space
        :param float angle: The angle to rotate, in radians
        :param Rectangle FixedBoundingBox:  Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.  
        :param Rectangle MappedBoundingBox: Optional, the boundaries of points expected to be mapped.  Used for informational purposes only.
        '''
        super(CenteredSimilarity2DTransform, self).__init__(target_offset, source_rotation_center, angle, **kwargs)
        
        self._scalar = scalar
        
    @staticmethod
    def Load(TransformString):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString)
    
    def ToITKString(self):
        #TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return "CenteredSimilarity2DTransform_double_2_2 vp 6 {0} {1} {2} {3} {4} {5} fp 0".format(self._scalar,
                                                                                       self.angle,
                                                                                       self.source_space_center_of_rotation[1],
                                                                                       self.source_space_center_of_rotation[0],
                                                                                       self.target_offset[1],
                                                                                       self.target_offset[0])
            
    def ScaleWarped(self, scalar):
        '''Scale source space control points by scalar'''
        self.source_space_center_of_rotation = self.source_space_center_of_rotation / scalar
        self._scalar = self._scalar / scalar
        self.OnTransformChanged()
        
    def ScaleFixed(self, scalar):
        '''Scale target space control points by scalar'''
        self._scalar = self._scalar * scalar
        self.OnTransformChanged()                                                                                   
                                                                                       
    
    def Transform(self, points, **kwargs):
        
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        
        if self.angle == 0 and self._scalar == 1.0:
            return points + self.target_offset
        
        numPoints = points.shape[0]
        
        centered_points = points - self.source_space_center_of_rotation
        centered_points = np.transpose(centered_points)
        centered_points = np.vstack((centered_points, np.ones((1, numPoints)))) #Add a row so we can multiply the matrix
        
        if self._scalar != 1.0:
            centered_points = centered_points * self._scalar
            
        centered_rotated_points = self.forward_rotation_matrix * centered_points
        centered_rotated_points = np.transpose(centered_rotated_points)
        centered_rotated_points = centered_rotated_points[:,0:2]
          
        rotated_points = centered_rotated_points + self.source_space_center_of_rotation
        output_points = rotated_points + self.target_offset
        return np.asarray(output_points)

    def InverseTransform(self, points, **kwargs):
        
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        
        if self.angle == 0 and self._scalar == 1.0:
            return points - self.target_offset
    
        numPoints = points.shape[0]
        
        input_points = points - self.target_offset
        centered_points = input_points - self.source_space_center_of_rotation
        
        centered_points = np.transpose(centered_points)
        centered_points = np.vstack((centered_points, np.ones((1, numPoints))))
        
        if self._scalar != 1.0:
            centered_points = centered_points / self._scalar
            
        rotated_points = self.inverse_rotation_matrix * centered_points
        rotated_points = np.transpose(rotated_points)
        rotated_points = rotated_points[:,0:2]
        
        output_points = rotated_points + self.source_space_center_of_rotation
        return np.asarray(output_points)
    
    def __repr__(self):
        return super(CenteredSimilarity2DTransform, self).__repr__() + " scale: {0}:04g".format(self.scalar)