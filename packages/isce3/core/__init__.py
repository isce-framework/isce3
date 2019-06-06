#-*- coding: utf-8 -*-

def basis(**kwds):
    """A factory for Basis"""
    from .Basis import Basis
    
    return Basis(**kwds)

def dateTime(**kwds):
    """A factory for DateTime"""
    from .DateTime import DateTime
    return DateTime(**kwds)

def dopplerEuler(**kwds):
    """A factory for DopplerEuler"""
    from .DopplerEuler import DopplerEuler

    return DopplerEuler(**kwds)

def ellipsoid(**kwds):
    """A factory for Ellipsoid"""
    from .Ellipsoid import Ellipsoid

    return Ellipsoid(**kwds)

def interpolator(**kwds):
    """A factory for Interpolator"""
    from .Interpolator import Interpolator

    return Interpolator(**kwds)

def linAlg(**kwds):
    """A factory for LinAlg"""
    from .LinAlg import LinAlg

    return LinAlg(**kwds)

def lut1d(**kwds):
    """A factory for LUT1d"""
    from .LUT1d import LUT1d

    return LUT1d(**kwds)

def lut2d(**kwds):
    """A factory for LUT2d"""
    from .LUT2d import LUT2d

    return LUT2d(**kwds)

def metadata(**kwds):
    """A factory for Metadata"""
    from .Metadata import Metadata

    return Metadata(**kwds)

def orbit(**kwds):
    """A factory for Orbit"""
    from .Orbit import Orbit

    return Orbit(**kwds)

def peg(**kwds):
    """A factory for Peg"""
    from .Peg import Peg

    return Peg(**kwds)

def pegtrans(**kwds):
    """A factory for Pegtrans"""
    from .Pegtrans import Pegtrans

    return Peg(**kwds)

def poly1d(**kwds):
    """A factory for Poly1d"""
    from .Poly1d import Poly1d

    return Poly1d(**kwds)

def poly2d(**kwds):
    """A factory for Poly2d"""
    from .Poly2d import Poly2d

    return Poly2d(**kwds)

def timeDelta(**kwds):
    """A factory for TimeDelta"""
    from .TimeDelta import TimeDelta

    return TimeDelta(**kwds)

