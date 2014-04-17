import six


if six.PY2:
    from nornir_imageregistration.nornir_enum import enum as Enum


    iRect = Enum(MinY=0,
             MinX=1,
             MaxY=2,
             MaxX=3)

    iPoint = Enum(Y=0,
              X=1)

    iArea = Enum(Height=0,
             Width=1)
else:
    from enum import IntEnum

    class iRect(IntEnum):
        MinY = 0
        MinX = 1
        MaxY = 2
        MaxX = 3

    class iPoint(IntEnum):
        Y = 0
        X = 1

    class iArea(IntEnum):
        Height = 0
        Width = 1
