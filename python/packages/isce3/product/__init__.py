#-*- coding: utf-8 -*-

# Import the wrappers
def swath(**kwds):
    """A factory for swath"""
    from .Swath import Swath

    return Swath(**kwds)

def radarGridParameters(**kwds):
    """A factory for radar grid parameters"""
    from .RadarGridParameters import RadarGridParameters

    return RadarGridParameters(**kwds)

# end of file
