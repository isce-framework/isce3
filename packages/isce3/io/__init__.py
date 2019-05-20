#-*- coding: utf-8 -*-

# Import the extension
import isce3.extensions.isceextension as isceextension

# Import the wrappers
def ih5File(**kwds):

    from .IH5File import IH5File

    return IH5File(**kwds)

def raster(**kwds):

    from .Raster import Raster 

    return Raster(**kwds)

# end of file
