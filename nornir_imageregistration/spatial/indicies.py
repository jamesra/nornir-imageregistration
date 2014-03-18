from nornir_imageregistration.enum import enum

iRect = enum(MinY=0,
             MinX=1,
             MaxY=2,
             MaxX=3)

iPoint = enum(Y=0,
              X=1)

iArea = enum(Height=0,
             Width=1)