

from nornir_imageregistration.transforms import base, triangulation
import nornir_imageregistration.transforms.utils
import nornir_imageregistration.transforms.factory
import numpy as np


class RigidNoRotation(base.Base):
    '''This class is legacy and probably needs a deprecation warning'''
    
    def __init__(self, target_offset, source_rotation_center=None, angle=None):
        
        if angle is None:
            angle = 0
        
        if source_rotation_center is None:
            source_rotation_center = [0,0]
            
        self.target_offset = nornir_imageregistration.EnsurePointsAre2DNumpyArray(target_offset)
        self.source_space_center_of_rotation = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_rotation_center)
        self.angle = angle
        
    @classmethod
    def Load(cls, TransformString):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString)
    
    def ToITKString(self):
        #TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return "Rigid2DTransform_double_2_2 vp 3 {0} {1} {2} fp 2 {3} {4}".format(self.angle, self.target_offset[1], self.target_offset[0], self.source_space_center_of_rotation[1], self.source_space_center_of_rotation[0])
    
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
        return "Offset: {0}y,{1}x Angle: {2} deg Rot Center: {3}y,{4}x".format(self.target_offset[0], self.target_offset[1], self.angle, self.source_space_center_of_rotation[0], self.source_space_center_of_rotation[1])


class Rigid(RigidNoRotation):
    '''
    Applies a rotation+translation transform
    '''
    
    def __init__(self, target_offset, source_rotation_center, angle):
        '''
        :param point target_offset: Offset from source to target space
        :param poitn source_rotation_center: 
        :param float angle: angle in radians
        '''
        super(Rigid, self).__init__(target_offset, source_rotation_center, angle)
        
        self.forward_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(self.angle)
        self.inverse_rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(-self.angle)
        
    @classmethod
    def Load(cls, TransformString):
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