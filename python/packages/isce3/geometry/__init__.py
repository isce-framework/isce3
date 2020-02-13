#-*- coding: utf-8 -*-

# Import the wrappers
from .Rdr2geo import rdr2geo_point, rdr2geo_cone
from .Geo2rdr import geo2rdr_point
from .geometry import getGeoPerimeter

def rdr2geo(**kwds):
    """A factory for Rdr2geo"""
    from .Rdr2geo import Rdr2geo

    return Rdr2geo(**kwds)


def geo2rdr(**kwds):
    """A factory for Geo2rdr"""
    from .Geo2rdr import Geo2rdr

    return Geo2rdr(**kwds)

def geocode(**kwds):
    """A factory for Geocode"""
    from .Geocode import Geocode

    return Geocode(**kwds)

def deminterpolator(**kwds):
    """A factory for DEMInterpolator"""
    from .DEMInterpolator import DEMInterpolator

    return DEMInterpolator(**kwds)
# end of file

