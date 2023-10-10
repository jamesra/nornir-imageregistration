#
# if six.PY2:
#     from nornir_imageregistration.nornir_enum import enum as Enum
#
#     iBox = Enum(MinZ=0,
#                     MinY=1,
#                     MinX=2,
#                     MaxZ=3,
#                     MaxY=4,
#                     MaxX=5)
#
#     iRect = Enum(MinY=0,
#              MinX=1,
#              MaxY=2,
#              MaxX=3)
#
#     iPoint = Enum(Y=0,
#               X=1)
#
#     iPoint3 = Enum(Z=0,
#                    Y=1,
#                    X=2)
#
#     iArea = Enum(Height=0,
#              Width=1)
#
#     iVolume = Enum(Depth=0,
#                    Height=1,
#                    Width=2)
# else:
from enum import IntEnum


class iBox(IntEnum):
    MinZ = 0
    MinY = 1
    MinX = 2
    MaxZ = 3
    MaxY = 4
    MaxX = 5


class iRect(IntEnum):
    MinY = 0
    MinX = 1
    MaxY = 2
    MaxX = 3


class iPoint(IntEnum):
    Y = 0
    X = 1


class iPoint3(IntEnum):
    Z = 0
    Y = 1
    X = 2


class iArea(IntEnum):
    Height = 0
    Width = 1


class iVolume(IntEnum):
    Depth = 0
    Height = 1
    Width = 2
