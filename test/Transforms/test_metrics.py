'''
Created on Aug 31, 2018

@author: u0490822
'''
import unittest
import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.interpolate

import nornir_imageregistration.files
from nornir_imageregistration.transforms import * 
import test.setup_imagetest
import nornir_shared.histogram
import nornir_shared.plot

class TestTransformMetrics(test.setup_imagetest.TestBase):

    def setUp(self):

        super(TestTransformMetrics, self).setUp()

        self.stosFilePath = os.path.join(self.TestInputPath, 'Transforms','0216-0215_grid_16.stos')
        self.stos = nornir_imageregistration.files.stosfile.StosFile.Load(self.stosFilePath)
        self.transform = factory.LoadTransform(self.stos.Transform,1)


    def test_TransformTriangleAngles(self):
        #stosFile = os.path.join(self.TestInputPath, 'Transforms','1153-1152_TT.stos')
         
        transform = self.transform

        FixedTriAngles = metrics.TriangleAngles(transform.FixedTriangles, transform.FixedPoints)

        CheckTriAngleSum180 = numpy.sum(FixedTriAngles, 1)
        assert(numpy.alltrue(CheckTriAngleSum180 == numpy.pi))

    def test_TriangleAnglesAndView(self):
        #stosFile = os.path.join(self.TestInputPath, 'Transforms','1153-1152_TT.stos')
        transform = self.transform

        angledelta = metrics.TriangleAngleDelta(transform)
        #order the angles from low to high
        angledelta = numpy.sort(angledelta, 1)
        maxdelta = numpy.max(angledelta[:,2])

        triangleTension = angledelta[:,2] / maxdelta

        triColor = numpy.swapaxes(numpy.vstack((triangleTension, triangleTension, triangleTension)),0,1)
        points = transform.WarpedPoints
        
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        #ax1.triplot(points[:,1], points[:,0], transform.FixedTriangles)
        tpc = ax1.tripcolor(points[:,1], points[:,0], transform.FixedTriangles, facecolors=triangleTension, shading='flat')
        fig1.colorbar(tpc)
        #ax1.plot(points[:,1], points[:,0], 'o')
        ax1.set_title('Normalized maximum difference in internal triangle angle from control to warped space')
        ax1.invert_yaxis()
        plt.show()

        pass

    def test_TriangleVertexAnglesAndView(self):
        #stosFile = os.path.join(self.TestInputPath, 'Transforms','1153-1152_TT.stos')
        transform = self.transform

        angledelta = metrics.TriangleVertexAngleDelta(transform)

        maxVal = numpy.max(list(map(numpy.max, angledelta)))
        measurement = numpy.asarray(list(map(numpy.max, angledelta)))

        #measurement = measurement / maxVal

        #order the angles from low to high
        #angledelta = numpy.sort(angledelta, 1)
        #maxdelta = numpy.max(angledelta[:,2])

        #triangleTension = angledelta[:,2] / maxdelta

        #triColor = numpy.swapaxes(numpy.vstack((triangleTension, triangleTension, triangleTension)),0,1)
        points = transform.WarpedPoints

        plotTriangulation = mtri.Triangulation(points[:,1], points[:,0], transform.FixedTriangles)

        #interp_cubic_min_E = mtri.CubicTriInterpolator(plotTriangulation, measurement, kind='min_E')
        #zi_cubic_min_E = interp_cubic_min_E(xi, yi)

        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        #ax1.triplot(points[:,1], points[:,0], transform.FixedTriangles)
        tpc = ax1.tripcolor(points[:,1], points[:,0], transform.FixedTriangles, measurement, vmin=0, vmax=maxVal, shading='gouraud')

        fig1.colorbar(tpc)
        #ax1.plot(points[:,1], points[:,0], 'o')
        ax1.set_title('Normalized mean difference in internal triangle angles for each vertex from control to warped space')
        #ax1.invert_yaxis()
        plt.show()
        
        h = nornir_shared.histogram.Histogram.Init(0, maxVal, int(numpy.sqrt(transform.NumControlPoints) * numpy.log(transform.NumControlPoints)))
        h.Add(measurement)
        
        nornir_shared.plot.Histogram(h)
        
        pass
    
    
    def test_FixedImageOverlay(self):

        dims = self.stos.ControlImageDim[2:]

        angledelta = metrics.TriangleVertexAngleDelta(self.transform)

        maxVal = numpy.max(list(map(numpy.max, angledelta)))
        measurement = numpy.asarray(list(map(numpy.max, angledelta)))

        grid_x, grid_y = numpy.mgrid[0:int(dims[1]), 0:int(dims[0])]

        img = scipy.interpolate.griddata(self.transform.WarpedPoints, measurement, (grid_x, grid_y), method='cubic')

        self.assertTrue(nornir_imageregistration.ShowGrayscale([img], Title='An image showing transform warp metric', PassFail=True))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_LargestAnglePerTriangle']
    unittest.main()