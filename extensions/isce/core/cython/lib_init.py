#!/usr/bin/env python

# Note: We can use the __init__ to protect the "py-tagged" namespace of the C++-to-Python
#       objects, while still being able to call 'from iscecore import Orbit' e.g. instead
#       of 'from iscecore import pyOrbit'

def Ellipsoid(a=0., e2=0.):
    from .iscecore import pyEllipsoid
    return pyEllipsoid(a,e2)

def Interpolator():
    from .iscecore import pyInterpolator
    return pyInterpolator()

def Peg(lat=0., lon=0., hdg=0.):
    from .iscecore import pyPeg
    return pyPeg(lat,lon,hdg)

def Pegtrans():
    from .iscecore import pyPegtrans
    return pyPegtrans()

def Position():
    from .iscecore import pyPosition
    return pyPosition()

def LinAlg():
    from .iscecore import pyLinAlg
    return pyLinAlg()

def Poly1d(order=-1, mean=0., norm=1.):
    from .iscecore import pyPoly1d
    return pyPoly1d(order,mean,norm)

def Poly2d(azimuthOrder=-1, rangeOrder=-1, azimuthMean=0., rangeMean=0., azimuthNorm=1., 
           rangeNorm=1.):
    from .iscecore import pyPoly2d
    return pyPoly2d(azimuthOrder,rangeOrder,azimuthMean,rangeMean,azimuthNorm,rangeNorm)

def Orbit(basis=1, nVectors=0):
    from .iscecore import pyOrbit
    return pyOrbit(basis,nVectors)
