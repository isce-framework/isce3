#-*- coding: utf-8 -*-

def basis(**kwds):
    
    from .Basis import Basis

    return Basis(**kwds)

def dateTime(**kwds):

    from .DateTime import DateTime

    return DateTime(**kwds)

def dopplerEuler(**kwds):

    from .DopplerEuler import DopplerEuler

    return DopplerEuler(**kwds)

def ellipsoid(**kwds):

    from .Ellipsoid import Ellipsoid

    return Ellipsoid(**kwds)

def interpolator(**kwds):

    from .Interpolator import Interpolator

    return Interpolator(**kwds)

def linAlg(**kwds):
    
    from .LinAlg import LinAlg

    return LinAlg(**kwds)

def lut1d(**kwds):
    
    from .LUT1d import LUT1d

    return LUT1d(**kwds)

def lut2d(**kwds):

    from .LUT2d import LUT2d

    return LUT2d(**kwds)

def metadata(**kwds):
    
    from .Metadata import Metadata

    return Metadata(**kwds)

def orbit(**kwds):

    from .Orbit import Orbit

    return Orbit(**kwds)

def peg(**kwds):
    
    from .Peg import Peg

    return Peg(**kwds)

def pegtrans(**kwds):

    from .Pegtrans import Pegtrans

    return Peg(**kwds)

def poly1d(**kwds):

    from .Poly1d import Poly1d

    return Poly1d(**kwds)

def poly1d(**kwds):

    from .Poly2d import Poly2d

    return Poly2d(**kwds)

def timeDelta(**kwds):

    from .TimeDelta import TimeDelta

    return TimeDelta(**kwds)

