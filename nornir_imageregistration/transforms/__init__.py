__all__ = ['base', 'triangulation', "meshwithrbffallback", "factory", 'metrics', 'rigid', "registrationtree", "utils",
           "one_way_rbftransform", "two_way_rbftransform", "defaulttransformchangeevents", "controlpointbase",
           "transform_type", 'ITransform', 'ITransformChangeEvents', 'ITransformTranslation', 'IDiscreteTransform',
           'ITransformScaling', 'IControlPoints', 'ITransformTargetRotation', 'ITransformSourceRotation',
           'gridwithrbffallback', 'gridtransform']

import numpy as np
from numpy.typing import NDArray

# if __name__ == "__main__":

NumberOfControlPointsToTriggerMultiprocessing = 20

def distance(A: NDArray[float], B: NDArray[float]) -> NDArray[float]:
    """Distance between two arrays of points with equal numbers"""
    return np.sqrt(np.sum(np.square(A - B), 1))

import nornir_imageregistration.transforms.transform_type as transform_type
from nornir_imageregistration.transforms.transform_type import TransformType

import nornir_imageregistration.transforms.base as base
from nornir_imageregistration.transforms.base import Base, ITransform, ITransformChangeEvents, ITransformTranslation, \
    IDiscreteTransform, ITransformScaling, IControlPoints, ITransformTargetRotation, ITransformSourceRotation

import nornir_imageregistration.transforms.defaulttransformchangeevents as defaulttransformchangeevents
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents

import nornir_imageregistration.transforms.controlpointbase as controlpointbase
from nornir_imageregistration.transforms.controlpointbase import ControlPointBase

import nornir_imageregistration.transforms.triangulation as triangulation
from nornir_imageregistration.transforms.triangulation import Triangulation

import nornir_imageregistration.transforms.rigid as rigid
from nornir_imageregistration.transforms.rigid import Rigid, RigidNoRotation, CenteredSimilarity2DTransform

import nornir_imageregistration.transforms.metrics as metrics

import nornir_imageregistration.transforms.factory as factory
from nornir_imageregistration.transforms.factory import TransformToIRToolsString, LoadTransform

import nornir_imageregistration.transforms.one_way_rbftransform as one_way_rbftransform
from nornir_imageregistration.transforms.one_way_rbftransform import OneWayRBFWithLinearCorrection

import nornir_imageregistration.transforms.two_way_rbftransform as two_way_rbftransform
from nornir_imageregistration.transforms.two_way_rbftransform import TwoWayRBFWithLinearCorrection

import nornir_imageregistration.transforms.gridtransform as gridtransform
from nornir_imageregistration.transforms.gridtransform import GridTransform

import nornir_imageregistration.transforms.meshwithrbffallback as meshwithrbffallback
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback

import nornir_imageregistration.transforms.gridwithrbffallback as gridwithrbffallback
from nornir_imageregistration.transforms.gridwithrbffallback import GridWithRBFFallback

import nornir_imageregistration.transforms.discretewithcontinuousfallback as discretewithcontinuousfallback
from nornir_imageregistration.transforms.discretewithcontinuousfallback import DiscreteWithContinuousFallback

import nornir_imageregistration.transforms.matrixtransform as matrixtransform
from nornir_imageregistration.transforms.matrixtransform import AffineMatrixTransform

import nornir_imageregistration.transforms.converters as converters
from nornir_imageregistration.transforms.converters import ConvertTransform, ConvertTransformToGridTransform, ConvertTransformToMeshTransform, ConvertTransformToRigidTransform

import nornir_imageregistration.transforms.addition as addition
from nornir_imageregistration.transforms.addition import AddTransforms

