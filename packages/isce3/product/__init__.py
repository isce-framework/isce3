#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.isceextension as isceextension

# Import the wrappers
def swath(**kwds):
    """A factory for swath"""
    from .Swath import Swath

    return Swath(**kwds)

# end of file
