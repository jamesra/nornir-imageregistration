# -*- coding: utf-8 -*-

import unittest

import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
from hypothesis.extra.numpy import arrays
# import cv2
from scipy.stats import linregress

import nornir_imageregistration.transforms


class TestLinearFit(unittest.TestCase):

    def test_linearFit1D(self):
        points1 = arrays(np.float64, 10, elements=st.floats(0, 10), unique=True).example()
        points2 = arrays(np.float64, 10, elements=st.floats(0, 10), unique=True).example()

        x = points1
        y = 5.5 * x - points2

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        distanceVector = (-1 * y + slope * x + intercept) / np.sqrt(-1 ** 2 + slope ** 2)
        print(distanceVector)
        sumOfDistances = np.sum(distanceVector)

        # Testing if linear fit has all the distances of points to itself = 0
        np.testing.assert_allclose(sumOfDistances, 0, atol=1e-5)
        # =============================================================================

    #         print("SLOPE: ",slope)
    #         print("INTERCEPT: ",intercept)
    #         print(slope*0 + intercept)
    #         print("R-Squared: ",r_value**2)
    #         plt.plot(x,y,'o',label="original data")
    #         plt.plot(x,intercept+slope*x,'r',label="fitted data")
    #         plt.legend()
    #         plt.show()
    #
    # =============================================================================

    def test_linearFit2D(self):
        points2D = arrays(np.float64, (20, 2), elements=st.floats(0, 30), unique=True).example()
        xv = np.arange(-5, 6)
        yv = np.arange(-5, 6)
        xx, yy = np.meshgrid(xv, yv)
        xx = xx.flatten()
        yy = yy.flatten()
        gridPoints = np.transpose(np.vstack((xx, yy)))
        print(gridPoints)
        plt.scatter(xx, yy)
        plt.show()
        print(points2D[:, 0])
        print(points2D[:, 1])
        numPoints = len(gridPoints)

        # numPoints = len(points2D)
        print(numPoints)
        # slope1,intercept1,r_value1,p_value1,std_err1 = linregress(points2D[:,0],points2D[:,1])
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(xx, yy)

        translate = np.random.randint(1, 10, size=2)
        # translate =np.array([0,0])
        print("Translate: ", translate)
        rotate = -np.pi / 2 + np.pi * np.random.rand()
        print("Rotate ", rotate)
        scale = np.random.randint(1, 10)
        print('Scaling by: ', scale)
        # center_rotation = np.random.randint(-10,10,size=2)
        center_rotation = np.array([0, 0])
        print("Center of rotation: ", center_rotation)

        points2D1 = gridPoints - center_rotation
        points2D1 = np.transpose(points2D1)
        points2D1 = np.vstack((points2D1, np.ones((1, numPoints))))

        # points2D1 = points2D1 * scale
        rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(rotate)
        print(rotation_matrix)
        points2D1_rotated = rotation_matrix @ points2D1
        print(points2D1_rotated)
        points2D1_rotated = np.transpose(points2D1_rotated)
        print(points2D1_rotated)

        points2D1_rotated = points2D1_rotated[:, 0:2]
        points2D1_rotated = points2D1_rotated + center_rotation

        output_points2D = points2D1_rotated + translate

        output_x = np.zeros(len(gridPoints))
        output_y = np.zeros(len(gridPoints))

        for i in range(len(points2D)):
            output_x[i] = output_points2D[i, 0]
            output_y[i] = output_points2D[i, 1]
        # print(output_x)
        # print(output_points2D)
        # print(output_points2D[:,0])
        # print(output_points2D.shape)

        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(output_x, output_y)

        # Took four sample points at different x coordinates.
        sample_x1 = 0
        sample_x2 = 1
        sample_x3 = -1
        sample_x4 = 2
        beforeFit1 = slope1 * sample_x1 + intercept1
        afterFit1 = slope2 * sample_x1 + intercept2
        beforeFit2 = slope1 * sample_x2 + intercept1
        afterFit2 = slope2 * sample_x2 + intercept2
        beforeFit3 = slope1 * sample_x3 + intercept1
        afterFit3 = slope2 * sample_x3 + intercept2
        beforeFit4 = slope1 * sample_x4 + intercept1
        afterFit4 = slope2 * sample_x4 + intercept2

        beforeVector = np.array([[0, 1, -1], [beforeFit1, beforeFit2, beforeFit3]])
        afterVector = np.array([[0, 1, -1], [afterFit1, afterFit2, afterFit3]])
        # A = np.dot(afterVector,np.linalg.inv(beforeVector))
        # print(A)

        centroid_before = np.mean(beforeVector, axis=1).reshape(-1, 1)
        centroid_after = np.mean(afterVector, axis=1).reshape(-1, 1)

        center_centroid_before = beforeVector - centroid_before
        center_centroid_after = afterVector - centroid_after

        H = np.matmul(center_centroid_after, np.transpose(center_centroid_before))
        # print(H)
        U, S, VH = np.linalg.svd(H)
        R = np.matmul(U, VH.T)
        # print(VH)
        # print(np.linalg.det(R))
        if np.linalg.det(R) < 0:
            print("Correcting for reflection!...")
            VH[:, 1] *= -1
            R = U @ VH.T
        print(R)
        # fit_rotation = np.arctan(R[1,1]/R[0,0])
        # print(fit_rotation)
        t = -R @ centroid_before + centroid_after
        print(t)

    # =============================================================================
    #         fit_translation = t
    #         print(fit_translation)
    #         matrix = cv2.estimateAffinePartial2D(beforeVector.T,afterVector.T)
    #         print(matrix)
    # =============================================================================
    # =============================================================================
    #         plt.plot(points2D[0],points2D[1],'o')
    #         plt.plot(points2D[0],intercept+slope*points2D[0],'r',label="fitted line")
    #         plt.show()
    # =============================================================================
    def test_linearFit2DKabschUmeyama(self):
        # warpPoints = arrays(np.float64, (20,2), elements=st.floats(0, 30),unique=True).example()
        warpPoints = np.random.randint(-50, 50, size=(100, 2))
        # refPoints = arrays(np.float64, (20,2), elements=st.floats(0, 30),unique=True).example()
        '''
        Originally, a 11x11 grid of points is created and scipy's linregress function is used
        to get the linear fit for those points (The range values for the points are -5 to 5).
        This is commented out and then a randomly generated array of size 20x2 is created
        with the first column containing x values and second containing y values. 
        
        
        After applying rotation, translation and scaling to those points (randomly generated transformation components), 
        linear fit is then generated on the new set of points. Later, taking a sample set of x coordinates, 
        the corresponding y coordinates are found for both the fits. Now, with a set of points (warp: before and fixed: after), 
        the Kabsch Umeyama algorithm is used to find optimal rotation, translation and scaling between the fits. 
        
        Testing is then done to see if the factors obtained are the same as the initial randomly generated ones. 
        
        '''
        # =============================================================================
        #         xv = np.arange(-5,6)
        #         yv = np.arange(-5,6)
        #         xx,yy = np.meshgrid(xv,yv)
        #         xx = xx.flatten()
        #         yy = yy.flatten()
        # =============================================================================
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(warpPoints[:, 0], warpPoints[:, 1])
        # slope1,intercept1,r_value1,p_value1,std_err1 = linregress(xx,yy)

        # grid = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
        # print(grid)
        print("\n\nWarp points: ", warpPoints)
        # n,m = grid.shape
        n, m = warpPoints.shape

        translate = np.random.randint(1, 10, size=2)
        # translate =np.array([0,0])
        print("Translate: ", translate)
        rotate = -np.pi / 2 + np.pi * np.random.rand()
        print("Rotate ", rotate)
        scale = np.random.randint(1, 10)
        print('Scaling by: ', scale)
        # center_rotation = np.random.randint(-10,10,size=2)
        center_rotation = np.array([0, 0])
        print("Center of rotation: ", center_rotation)

        # points2D1 = np.transpose(grid)
        points2D1 = np.transpose(warpPoints)
        points2D1 = np.vstack((points2D1, np.ones((1, n))))

        # points2D1 = points2D1 * scale
        # rotation_matrix = nornir_imageregistration.transforms.utils.IdentityMatrix()#nornir_imageregistration.transforms.utils.RotationMatrix(rotate)
        rotation_matrix = nornir_imageregistration.transforms.utils.RotationMatrix(rotate)
        print(rotation_matrix)
        points2D1_rotated = rotation_matrix @ points2D1
        print(points2D1_rotated)
        points2D1_rotated = np.transpose(points2D1_rotated)
        print(points2D1_rotated)

        points2D1_rotated = points2D1_rotated[:, 0:2]

        # output_points2D = points2D1_rotated
        output_points2D = points2D1_rotated + translate

        print(output_points2D)
        output_x = np.zeros(len(warpPoints))
        output_y = np.zeros(len(warpPoints))

        for i in range(len(output_points2D)):
            output_x[i] = output_points2D[i, 0]
            output_y[i] = output_points2D[i, 1]

        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(output_x, output_y)

        # Generating points on each of the linear fits in order to pass them to algorithm.
        sample_x_vector = np.arange(-100, 100)
        beforeFit = slope1 * sample_x_vector + intercept1
        afterFit = slope2 * sample_x_vector + intercept2
        beforeVector = np.hstack((sample_x_vector.reshape(-1, 1), beforeFit.reshape(-1, 1)))
        afterVector = np.hstack((sample_x_vector.reshape(-1, 1), afterFit.reshape(-1, 1)))

        # =============================================================================
        #         sample_x1 = 10
        #         sample_x2 = 20
        #         sample_x3 = -10
        #         beforeFit1 = slope1*sample_x1 + intercept1
        #         afterFit1 = slope2*sample_x1 + intercept2
        #         beforeFit2 = slope1*sample_x2 + intercept1
        #         afterFit2 = slope2*sample_x2 + intercept2
        #         beforeFit3 = slope1*sample_x3 + intercept1
        #         afterFit3 = slope2*sample_x3 + intercept2
        #
        #         beforeVector = np.array([[sample_x1,beforeFit1],[sample_x2,beforeFit2],[sample_x3,beforeFit3]])
        #         afterVector = np.array([[sample_x1,afterFit1],[sample_x2,afterFit2],[sample_x3,afterFit3]])
        # =============================================================================
        print(beforeVector)
        print(afterVector)

        centroid_before = np.mean(beforeVector, axis=0)
        centroid_after = np.mean(afterVector, axis=0)

        varianceAfter = np.mean(np.linalg.norm(afterVector - centroid_after, axis=1) ** 2)

        H = ((afterVector - centroid_after).T @ (beforeVector - centroid_before)) / n

        U, D, VT = np.linalg.svd(H)

        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))

        S = np.diag([1] * (m - 1) + [d])

        rotateResult = U @ S @ VT

        scaleResult = varianceAfter / np.trace(np.diag(D) @ S)

        translateResult = centroid_after - scaleResult * rotateResult @ centroid_before

        print("\n\nResulting rotation matrix from fit: ", rotateResult)
        print("\nRotation angle: ", np.arctan(rotateResult[1, 0] / rotateResult[0, 0]))
        print("\nResulting scaling factor from fit: ", scaleResult)
        print("\nResulting translation vector from fit: ", translateResult)

        print("\n\nOriginal rotation angle: ", rotate)
        print("\nOriginal rotation matrix: ", rotation_matrix)
        print("\nOriginal scaling value:  ", scale)
        print("\nOriginal translation vector: ", translate)


# =============================================================================
#         centroidRef = np.mean(refPoints, axis=0)
#         centroidWarp = np.mean(warpPoints, axis=0)
#         varianceRef= np.mean(np.linalg.norm(refPoints - centroidRef, axis=1) ** 2)
#         
#         H = ((refPoints - centroidRef).T @ (warpPoints - centroidWarp)) / n
#         
#         U, D, VT = np.linalg.svd(H)
#         
#         d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
#         
#         #S matrix to prevent reflection
#         S = np.diag([1] * (m - 1) + [d])
#         
#         R = U @ S @ VT
#         scale = varianceRef / np.trace(np.diag(D) @ S)
#         t = centroidRef - scale * R @ centroidWarp
# 
#         print("\n\nRotation component: ",R)
#         print("\n\nScaling Componenet: ",scale)
#         print("\n\ntranslation component: ",t)
#         
#         points2D1 = np.transpose(warpPoints)
#         points2D1 = np.vstack((points2D1, np.ones((1, n))))
# 
#         points2D1 = points2D1 * scale
#         R = np.hstack((R,np.zeros((2,1))))
#         R = np.vstack((R,np.ones((1,3))))
#         print(R)
#         points2D1_rotated = R @ points2D1
#         print(points2D1_rotated)
#         points2D1_rotated = np.transpose(points2D1_rotated)
#         print(points2D1_rotated)
#         
#         points2D1_rotated = points2D1_rotated[:,0:2]
#         
#         output_points2D = points2D1_rotated + t
# =============================================================================

# =============================================================================
#         print(np.sqrt(np.sum(np.abs(output_points2D-refPoints))))
#         print(output_points2D)
#         print(refPoints)
# 
# =============================================================================


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
