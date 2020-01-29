#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.iscecudaextension as iscecudaextension

def rdr2geo(**kwds):
    """A factory for Rdr2geo"""
    from .Rdr2geo import Rdr2geo

    return Rdr2geo(**kwds)

def geo2rdr(**kwds):
    """A factory for Geo2rdr"""
    from .Geo2rdr import Geo2rdr

    return Geo2rdr(**kwds)

# end of file
