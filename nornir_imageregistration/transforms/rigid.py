
from nornir_imageregistration.transforms import base, triangulation
import nornir_imageregistration.transforms.utils
import nornir_imageregistration.transforms.factory
from nornir_imageregistration.spatial import Rectangle
import numpy as np


class RigidNoRotation(base.Base):
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
    
    @property
    def MappedBoundingBox(self):
        if self._mapped_bounding_box is None:
            self._mapped_bounding_box = self._inverse_transform_rectangle(self._fixed_bounding_box)
        
        return self._mapped_bounding_box
 
    @property
    def FixedBoundingBox(self):
        if self._fixed_bounding_box is None:
            self._fixed_bounding_box = self._transform_rectangle(self._mapped_bounding_box)
        
        return self._fixed_bounding_box
    
    @MappedBoundingBox.setter
    def MappedBoundingBox(self, val):
        self._fixed_bounding_box = None
        self._mapped_bounding_box = val
 
    @FixedBoundingBox.setter
    def FixedBoundingBox(self, val):
        self._mapped_bounding_box = None
        self._fixed_bounding_box = val
      
    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''
        self.target_offset  = self.target_offset + offset
        self._fixed_bounding_box = None
        self.OnTransformChanged()

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self.target_offset  = self.target_offset - offset
        self._fixed_bounding_box = None
        self._mapped_bounding_box = self._mapped_bounding_box.translate(offset)
        self.OnTransformChanged()
        
    def Scale(self, value):
        self.source_space_center_of_rotation = self.source_space_center_of_rotation * value 
        self.target_offset = self.target_offset * value
        if self.MappedBoundingBox is not None:
            self._fixed_bounding_box = None
            self._mapped_bounding_box = scaled_targetRect = nornir_imageregistration.Rectangle.scale_on_origin(self._mapped_bounding_box, value)
        self.OnTransformChanged()
    
    def __init__(self, target_offset, source_rotation_center=None, angle=None, **kwargs):
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
            angle = 0
        
        if source_rotation_center is None:
            source_rotation_center = [0,0]
            
        self.target_offset = nornir_imageregistration.EnsurePointsAre1DNumpyArray(target_offset)
        self.source_space_center_of_rotation = nornir_imageregistration.EnsurePointsAre1DNumpyArray(source_rotation_center)
        self._angle = angle
        
        self._fixed_bounding_box = kwargs.get('FixedBoundingBox', None)
        self._mapped_bounding_box = kwargs.get('MappedBoundingBox', None)
        
        if self._fixed_bounding_box is not None:
            if not isinstance(self._fixed_bounding_box, Rectangle):
                raise ValueError("FixedBoundingBox must be a Rectangle")
            
        if self._mapped_bounding_box is not None:
            if not isinstance(self._mapped_bounding_box, Rectangle):
                raise ValueError("MappedBoundingBox must be a Rectangle") 
            
        
    def __getstate__(self):
        odict = {}
        odict['_angle'] = self._angle
        odict['target_offset'] = (self.target_offset[0], self.target_offset[1])
        odict['source_space_center_of_rotation'] = (self.source_space_center_of_rotation[0],
                                                    self.source_space_center_of_rotation[1])
        odict['_mapped_bounding_box'] = self._mapped_bounding_box
        odict['_fixed_bounding_box'] = self._fixed_bounding_box

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)
        
        self.target_offset = np.asarray((self.target_offset[0], self.target_offset[1]), dtype=np.float64)
        self.source_space_center_of_rotation = np.asarray((self.source_space_center_of_rotation[0],
                                                           self.source_space_center_of_rotation[1]), dtype=np.float64)
        
        self.OnChangeEventListeners = []
        self.OnTransformChanged()
        
    @staticmethod
    def Load(TransformString):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString)
        
    def ToITKString(self):
        #TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return "Rigid2DTransform_double_2_2 vp 3 {0} {1} {2} fp 2 {3} {4}".format(self._angle, self.target_offset[1], self.target_offset[0], self.source_space_center_of_rotation[1], self.source_space_center_of_rotation[0])
    
    def Transform(self, points, **kwargs):
        
        
        if not (self.angle is None or self.angle == 0):
            #Look at GetTransformedRigidCornerPoints for a possible implementation
            raise NotImplemented("Rotation is not implemented")
        
        return points + self.target_offset

    def InverseTransform(self, points, **kwargs):
        
        
        if not (self.angle is None or self.angle == 0):
            #Look at GetTransformedRigidCornerPoints for a possible implementation
            raise NotImplemented("Rotation is not implemented")
        
        return points - self.target_offset
    
    def __repr__(self):
        return "Offset: {0}y,{1}x Angle: {2}deg Rot Center: {3}y,{4}x".format(self.target_offset[0], self.target_offset[1], self.angle, self.source_space_center_of_rotation[0], self.source_space_center_of_rotation[1])


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
        return super(Rigid, self).__repr__()

class CenteredSimilarity2DTransform(Rigid):
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
     
    def __init__(self, target_offset, source_rotation_center=None, angle=None, scalar=None, **kwargs):
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
        self._scalar = self._scalar / scalar
        self._mapped_bounding_box = nornir_imageregistration.Rectangle.scale_on_origin(self._mapped_bounding_box, scalar) #self._inverse_transform_rectangle(self._fixed_bounding_box)
        self._fixed_bounding_box = None
        self.OnTransformChanged()
        
    def ScaleFixed(self, scalar):
        '''Scale target space control points by scalar'''
        self._scalar = self._scalar * scalar
        #self._fixed_bounding_box = None nornir_imageregistration.Rectangle.scale_on_center(self._fixed_bounding_box, scalar) #self._inverse_transform_rectangle(self._fixed_bounding_box)
        self._fixed_bounding_box = None
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
        return super(CenteredSimilarity2DTransform, self).__repr__() + " scale: {0}".format(self.scalar)