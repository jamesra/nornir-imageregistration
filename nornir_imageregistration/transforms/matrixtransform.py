'''
Created on Sep 28, 2022

@author: u0490822
'''

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import base
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents


class AffineMatrixTransform(base.ITransform, base.ITransformTranslation, DefaultTransformChangeEvents):
    '''
    classdocs
    '''

    @property
    def matrix(self) -> NDArray:
        return self._matrix

    @property
    def inverse_matrix(self) -> NDArray:
        if self._inverse_matrix is None:
            self._inverse_matrix = np.linalg.inv(self._matrix)

        return self._inverse_matrix

    @property
    def pre_transform_translation(self) -> NDArray:
        return self._pre_transform_translation

    @property
    def post_transform_translation(self) -> NDArray:
        return self._post_transform_translation

    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''
        self._post_transform_translation += offset
        self.OnTransformChanged()
        raise NotImplemented("This implementation is untested")

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self._pre_transform_translation -= offset
        self.OnTransformChanged()
        raise NotImplemented("This implementation is untested")

    def __init__(self,
                 matrix: NDArray,
                 pre_transform_translation: NDArray,
                 post_transform_translation: NDArray):
        '''
        :param matrix: The matrix to apply after pre-transform-translation
        :param pre_transform_translation: The amount to translate coordinates before applying the matrix.  To  rotate
        about the center, this should be half of the source space image dimensions.
        :param post_transform_translation: The translation applied to coordinates after the matrix is applied.
        '''

        self._matrix = matrix
        self._inverse_matrix = None  # type: NDArray
        self._pre_transform_translation = pre_transform_translation
        self._post_transform_translation = post_transform_translation

        super(AffineMatrixTransform, self).__init__()

    def __getstate__(self):
        odict = {'_matrix': self._matrix,
                 '_pre_transform_translation': (self._pre_transform_translation[0], self._pre_transform_translation[1]),
                 '_post_transform_translation': (
                     self._post_transform_translation[0], self._post_transform_translation[1])}
        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)

        self._pre_transform_translation = np.asarray(self._pre_transform_translation, dtype=np.float64)
        self._post_transform_translation = np.asarray(self._post_transform_translation, dtype=np.float64)

        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    @staticmethod
    def Load(TransformString: str, pixelSpacing: float = None):
        return nornir_imageregistration.transforms.factory.ParseFixedCenterOfRotationAffineTransform(TransformString,
                                                                                                     pixelSpacing)

    def ToITKString(self):
        # TODO look at using CenteredRigid2DTransform_double_2_2 to make rotation more straightforward
        return f"FixedCenterOfRotationAffineTransform_double_2_2 vp 8 {self._matrix[0, 1]} {self._matrix[0, 0]} {self._matrix[1, 1]} {self._matrix[1, 0]} fp 2 {pre_transform_translation[1]} {pre_transform_translation[0]}"

    def Transform(self, points, **kwargs):
        p1 = points + self._pre_transform_translation
        p2 = np.matmul(p1, self._matrix)
        p3 = p2 + self._post_transform_translation
        p4 = p3 - self._pre_transform_translation

        return p4

    def InverseTransform(self, points, **kwargs):
        p4 = points + self._pre_transform_translation
        p3 = p4 - self._post_transform_translation
        p2 = np.matmul(p3, self.inverse_matrix)
        p1 = p2 - self._pre_transform_translation

        return p1
