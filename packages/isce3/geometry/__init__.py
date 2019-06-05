#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.isceextension as isceextension

# Import the wrappers
from .Rdr2geo import rdr2geo_point
from .Geo2rdr import geo2rdr_point
from .geometry import getBoundsOnGround

def rdr2geo(**kwds):

    from .Rdr2geo import Rdr2geo

    return Rdr2geo(**kwds)


def geo2rdr(**kwds):

    from .Geo2rdr import Geo2rdr

    return Geo2rdr(**kwds)

def geocode(**kwds):

    from .Geocode import Geocode

    return Geocode(**kwds)


# end of file

