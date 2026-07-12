'''
Created on Sep 28, 2022

@author: u0490822
'''

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
from numpy.typing import NDArray

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import base
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents


class AffineMatrixTransform(base.ITransform, base.ITransformTranslation, DefaultTransformChangeEvents):
    '''
    classdocs
    '''

    @property
    def type(self):
        """Nearest editable family; flipped rigids decompose to CS2D on load."""
        return nornir_imageregistration.transforms.TransformType.RIGID

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
        raise NotImplementedError("This implementation is untested")

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self._pre_transform_translation -= offset
        self.OnTransformChanged()
        raise NotImplementedError("This implementation is untested")

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
        self._inverse_matrix: NDArray | None = None
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
        self.__dict__.update(dictionary)  # type: ignore[attr-defined]

        self._pre_transform_translation = np.asarray(self._pre_transform_translation, dtype=np.float64)
        self._post_transform_translation = np.asarray(self._post_transform_translation, dtype=np.float64)

        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    @staticmethod
    def Load(TransformString: str, pixelSpacing: float | None = None):
        return nornir_imageregistration.transforms.factory.ParseFixedCenterOfRotationAffineTransform(TransformString,  # type: ignore[arg-type]
                                                                                                     pixelSpacing)

    def ToITKString(self):
        """Serialize as ITK FixedCenterOfRotationAffineTransform (8 vp + center fp)."""
        # vp: m01 m00 m11 m10 post_x post_y pad pad; fp: center_x center_y
        # pre_transform_translation is (-center_y, -center_x).
        pre = self._pre_transform_translation
        post = self._post_transform_translation
        cx = -float(pre[1])
        cy = -float(pre[0])
        m = self._matrix
        return (
            f"FixedCenterOfRotationAffineTransform_double_2_2 vp 8 "
            f"{m[0, 1]} {m[0, 0]} {m[1, 1]} {m[1, 0]} "
            f"{post[1]} {post[0]} 0 0 "
            f"fp 2 {cx} {cy}"
        )

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


class AffineMatrixTransform_GPU(base.ITransform, base.ITransformTranslation, DefaultTransformChangeEvents):
    '''
    classdocs
    '''

    @property
    def type(self):
        """Nearest editable family; flipped rigids decompose to CS2D on load."""
        return nornir_imageregistration.transforms.TransformType.RIGID

    @property
    def matrix(self) -> NDArray:
        return self._matrix

    @property
    def inverse_matrix(self) -> NDArray:
        if self._inverse_matrix is None:
            self._inverse_matrix = cp.linalg.inv(self._matrix)

        return self._inverse_matrix

    @property
    def pre_transform_translation(self) -> NDArray:
        return self._pre_transform_translation

    @property
    def post_transform_translation(self) -> NDArray:
        return self._post_transform_translation

    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''
        self._post_transform_translation = self._post_transform_translation + cp.array(offset)
        self.OnTransformChanged()
        raise NotImplementedError("This implementation is untested")

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self._pre_transform_translation = self._pre_transform_translation - cp.array(offset)
        self.OnTransformChanged()
        raise NotImplementedError("This implementation is untested")

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
        self._inverse_matrix: NDArray | None = None
        self._pre_transform_translation = pre_transform_translation
        self._post_transform_translation = post_transform_translation

        super(AffineMatrixTransform_GPU, self).__init__()

    def __getstate__(self):
        odict = {'_matrix': self._matrix,
                 '_pre_transform_translation': (self._pre_transform_translation[0], self._pre_transform_translation[1]),
                 '_post_transform_translation': (
                 self._post_transform_translation[0], self._post_transform_translation[1])}
        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)  # type: ignore[attr-defined]

        self._pre_transform_translation = cp.asarray(self._pre_transform_translation, dtype=np.float64)
        self._post_transform_translation = cp.asarray(self._post_transform_translation, dtype=np.float64)

        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    @staticmethod
    def Load(TransformString: str, pixelSpacing: float | None = None):
        return nornir_imageregistration.transforms.factory.ParseFixedCenterOfRotationAffineTransform(TransformString,  # type: ignore[arg-type]
                                                                                                     pixelSpacing)

    def ToITKString(self):
        """Serialize as ITK FixedCenterOfRotationAffineTransform (8 vp + center fp)."""
        # vp: m01 m00 m11 m10 post_x post_y pad pad; fp: center_x center_y
        # pre_transform_translation is (-center_y, -center_x).
        pre = self._pre_transform_translation
        post = self._post_transform_translation
        cx = -float(pre[1])
        cy = -float(pre[0])
        m = self._matrix
        return (
            f"FixedCenterOfRotationAffineTransform_double_2_2 vp 8 "
            f"{m[0, 1]} {m[0, 0]} {m[1, 1]} {m[1, 0]} "
            f"{post[1]} {post[0]} 0 0 "
            f"fp 2 {cx} {cy}"
        )

    def Transform(self, points, **kwargs):
        points = cp.array(points) if not isinstance(points, cp.ndarray) else points

        p1 = points + self._pre_transform_translation
        p2 = cp.matmul(p1, self._matrix)
        p3 = p2 + self._post_transform_translation
        p4 = p3 - self._pre_transform_translation

        return p4

    def InverseTransform(self, points, **kwargs):
        points = cp.array(points) if not isinstance(points, cp.ndarray) else points

        p4 = points + self._pre_transform_translation
        p3 = p4 - self._post_transform_translation
        p2 = cp.matmul(p3, self.inverse_matrix)
        p1 = p2 - self._pre_transform_translation

        return p1