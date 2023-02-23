'''
Created on Sep 4, 2018

@author: u0490822
'''

import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.interpolate
import nornir_shared.histogram
import nornir_shared.plot


import nornir_imageregistration
from nornir_imageregistration.transforms import factory, ITriangulatedTargetSpace

class TransformWarpView:
    
    def __init__(self, transform):
        '''
        Calculates various metrics of the warp introduced by a transform 
        :param object transform: A full path to a .stos file, a triangulation transform, or a .stos file object
        '''
        
        if isinstance(transform, str):
            transform = nornir_imageregistration.StosFile.Load(transform)
        
        if isinstance(transform, nornir_imageregistration.StosFile):
            transform = factory.LoadTransform(transform.Transform,1)
            
        if not isinstance(transform, ITriangulatedTargetSpace):
            raise ValueError("TransformWarpView requires a TriangulationTransform or a .stos file")
        
        self._angle_delta = None
        self.__transform = transform 
        
        
    @property
    def transform(self) -> ITriangulatedTargetSpace: 
        return self.__transform
    
    @transform.setter
    def transform(self, val: ITriangulatedTargetSpace):
        if self.__transform != val:
            self._angle_delta = None
            
            if not isinstance(val, ITriangulatedTargetSpace):
                raise ValueError("TransformWarpView.transform requires an object derived from TriangulationTransform")
            
            self.__transform = val
            
    
    @property
    def angle_delta(self):
        if self._angle_delta is None:
            self._angle_delta = nornir_imageregistration.transforms.metrics.TriangleVertexAngleDelta(self.transform)
            
        return self._angle_delta
    
    
        
        
    def GenerateWarpImage(self, RenderToSourceSpace: bool = True,
                           title: str | None = None,
                           outputfullpath: str | None = None,
                           maxAngle: float | None = None):
        '''Generate a plot for the change in angles for verticies
        :param RenderToSourceSpace:
        :param title:
        :param str outputfullpath: Filename to save, if None then display interactively
        :param float maxAngle: The angle that maps to the highest value in the temperature map of the output.  Defaults to maximum angle found
        '''
        
        
        if maxAngle is None:
            maxAngle = numpy.max(list(map(numpy.max, self.angle_delta)))
        else:
            maxAngle = (maxAngle / 180) * numpy.pi
             
        #maxVal = numpy.pi / 12.0 #Maximum angle change is 60
        measurement = numpy.asarray(list(map(numpy.max, self.angle_delta)))

        if RenderToSourceSpace:
            points = self.transform.SourcePoints
        else:
            points = self.transform.TargetPoints

        #plotTriangulation = mtri.Triangulation(points[:,1], points[:,0], self.transform.FixedTriangles)

        #interp_cubic_min_E = mtri.CubicTriInterpolator(plotTriangulation, measurement, kind='min_E')
        #zi_cubic_min_E = interp_cubic_min_E(xi, yi)

        fig1, ax1 = plt.subplots(figsize=(8,8), dpi=150 ) 
        ax1.set_aspect('equal')
        ax1.axis('off')
        #ax1.triplot(points[:,1], points[:,0], transform.FixedTriangles)
        tpc = ax1.tripcolor(points[:,1], points[:,0], self.transform.FixedTriangles, measurement, vmin=0, vmax=maxAngle, shading='gouraud')

        cbar = fig1.colorbar(tpc)
        cbar.set_label('Angle delta (radians)')
        #ax1.plot(points[:,1], points[:,0], 'o')
        if title is None:
            title = 'Normalized mean difference in vertex triangle angles\nVolume vs Section space'
        ax1.set_title(title)
        #ax1.invert_yaxis()
        
        if(outputfullpath is not None):
            # plt.show()
            plt.savefig(outputfullpath, bbox_inches='tight', figure=(1280,1280))
            plt.close()
        else:
            plt.show() 
            
            
    def GenerateWarpHistogram(self, outputfullpath=None):
        '''
        :return: A histogram object representing the maximum angle change for each vertex
        '''
        
        #maxVal = numpy.max(list(map(numpy.max, self.angle_delta)))
        measurement = numpy.asarray(list(map(numpy.max, self.angle_delta)))
        
        #allMeasures = numpy.concatenate(self.angle_delta)
                
        #std = numpy.std(allMeasures)
        
        #numbins = int(numpy.abs(maxVal / (std/4.0)))
        
        
        
#        if numbins > 500:
            #numbins = int(numpy.sqrt(self.transform.NumControlPoints) * numpy.log(self.transform.NumControlPoints))
            
        maxVal = numpy.pi / 12.0 #Maximum angle change is 60
        numbins = 120 #A bin every 1/2 degree
                
        h = nornir_shared.histogram.Histogram.Init(0, maxVal, numbins)
        h.Add(measurement)
        return h
        

class StosTransformWarpView(TransformWarpView):
    
    @property
    def stos(self): 
        return self.__stos
    
    @stos.setter
    def stos(self, val):
        if self.__stos != val:
            if val is not None:
                if not isinstance(val, nornir_imageregistration.StosFile):
                    raise ValueError("stos attribute must be a StosFile object")
                
                self.transform = factory.LoadTransform(val.Transform,1)
            else:
                self.transform = None
            
            self.__stos = val
    
    def __init__(self, transform):
        
        self.__stos = None
        
        if isinstance(transform, str):
            transform = nornir_imageregistration.StosFile.Load(transform)
        
        if isinstance(transform, nornir_imageregistration.StosFile):
            self.stos = transform
            
    def GenerateWarpOverlayForFixedImage(self, outputfullpath=None):
        
        maxVal = numpy.max(list(map(numpy.max, self.angle_delta)))
        measurement = numpy.asarray(list(map(numpy.max, self.angle_delta)))

        points = self.transform.FixedPoints
        

if __name__ == '__main__':
    pass