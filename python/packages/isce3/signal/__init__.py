#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.isceextension as isceextension

# Import the wrappers
def crossmul(**kwds):
    """A factory for Crossmul"""
    from .Crossmul import Crossmul

    return Crossmul(**kwds)

# end of file
