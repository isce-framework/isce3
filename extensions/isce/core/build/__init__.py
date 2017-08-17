#!/usr/bin/env python

# Note: We can use the __init__ to protect the "Py-tagged" namespace of the C++-to-Python
#       objects, while still being able to call 'from isceLib import Orbit' e.g. instead
#       of 'from isceLib import PyOrbit'

def Ellipsoid(a=0., e2=0.):
    from .isceLib import PyEllipsoid
    return PyEllipsoid(a,e2)

def Interpolator():
    from .isceLib import PyInterpolator
    return PyInterpolator()

def Peg(lat=0., lon=0., hdg=0.):
    from .isceLib import PyPeg
    return PyPeg(lat,lon,hdg)

def Pegtrans():
    from .isceLib import PyPegtrans
    return PyPegtrans()

def Position():
    from .isceLib import PyPosition
    return PyPosition()

def LinAlg():
    from .isceLib import PyLinAlg
    return PyLinAlg()

def Poly1d(order=-1, mean=0., norm=1.):
    from .isceLib import PyPoly1d
    return PyPoly1d(order,mean,norm)

def Poly2d(azimuthOrder=-1, rangeOrder=-1, azimuthMean=0., rangeMean=0., azimuthNorm=1., rangeNorm=1.):
    from .isceLib import PyPoly2d
    return PyPoly2d(azimuthOrder,rangeOrder,azimuthMean,rangeMean,azimuthNorm,rangeNorm)

def Orbit(basis=1, nVectors=0):
    from .isceLib import PyOrbit
    return PyOrbit(basis,nVectors)
