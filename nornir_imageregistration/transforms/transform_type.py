from enum import Enum


class TransformType(Enum):
    """Used by Pyre to determine transform type"""
    RIGID = 'Rigid'
    GRID = 'Grid'
    MESH = 'Mesh'
    RBF = 'RBF'
