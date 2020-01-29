#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.iscecudaextension as iscecudaextension

# Import the wrappers
def resampSlc(**kwds):
    """A factory for ResampSlc"""
    from .ResampSlc import ResampSlc

    return ResampSlc(**kwds)

# end of file
