#!/usr/bin/env python

# Note: We can use the __init__ to protect the "Py-tagged" namespace of the C++-to-Python
#       objects, while still being able to call 'from iscecore import Orbit' e.g. instead
#       of 'from iscecore import PyOrbit'

def Ellipsoid(a=0., e2=0.):
    from .iscecore import PyEllipsoid
    return PyEllipsoid(a,e2)

def Interpolator():
    from .iscecore import PyInterpolator
    return PyInterpolator()

def Peg(lat=0., lon=0., hdg=0.):
    from .iscecore import PyPeg
    return PyPeg(lat,lon,hdg)

def Pegtrans():
    from .iscecore import PyPegtrans
    return PyPegtrans()

def Position():
    from .iscecore import PyPosition
    return PyPosition()

def LinAlg():
    from .iscecore import PyLinAlg
    return PyLinAlg()

def Poly1d(order=-1, mean=0., norm=1.):
    from .iscecore import PyPoly1d
    return PyPoly1d(order,mean,norm)

def Poly2d(azimuthOrder=-1, rangeOrder=-1, azimuthMean=0., rangeMean=0., azimuthNorm=1., rangeNorm=1.):
    from .iscecore import PyPoly2d
    return PyPoly2d(azimuthOrder,rangeOrder,azimuthMean,rangeMean,azimuthNorm,rangeNorm)

def Orbit(basis=1, nVectors=0):
    from .iscecore import PyOrbit
    return PyOrbit(basis,nVectors)
