#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.iscecudaextension as iscecudaextension

# Import the wrappers
def crossmul(**kwds):
    """A factory for Crossmul"""
    from .Crossmul import Crossmul

    return Crossmul(**kwds)

# end of file
