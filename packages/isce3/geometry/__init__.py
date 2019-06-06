#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.isceextension as isceextension

# Import the wrappers
from .Rdr2geo import rdr2geo_point
from .Geo2rdr import geo2rdr_point
from .geometry import getBoundsOnGround

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


# end of file

