

from nornir_imageregistration.transforms import base, triangulation
import nornir_imageregistration.transforms.utils
import nornir_imageregistration.transforms.factory


class Rigid(base):
    
    def __init__(self, target_offset, source_center, angle=None):
        self.target_offset = nornir_imageregistration.transforms.utils.EnsurePointsAre2DNumpyArray(target_offset)
        self.source_center = nornir_imageregistration.transforms.utils.EnsurePointsAre2DNumpyArray(source_center)
        self.angle = angle
        
        if not (self.angle is None or self.angle == 0):
            #Look at GetTransformedRigidCornerPoints for a possible implementation
            raise NotImplemented("Rotation is not implemented")
        
    @classmethod
    def Load(cls, TransformString):
        return nornir_imageregistration.transforms.factory.ParseRigid2DTransform(TransformString)
    
    def ToITKString(self):
        #TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return "Rigid2DTransform_double_2_2 vp 3 {0} {1} {2} fp 2 {3} {4}".format(self.angle, self.target_offset[1], self.target_offset[0], self.source_center[1], self.source_center[0])
    
    def Transform(self, point, **kwargs):
        return point + self.target_offset

    def InverseTransform(self, point, **kwargs):
        return point - self.target_offset
    
        
    