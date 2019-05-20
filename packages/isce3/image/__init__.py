#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.isceextension as isceextension

# Import the wrappers
def resampSlc(**kwds):

    from .ResampSlc import ResampSlc

    return ResampSlc(**kwds)

# end of file
