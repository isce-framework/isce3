#-*- coding: utf-8 -*-
import numpy as np

# The extensions
import isce3.extensions.isceextension as isceextension

def getBoundsOnGround(orbit,
                     ellipsoid,
                     doppler,
                     lookSide,
                     radarGrid,
                     xoff=0,
                     yoff=0,
                     xsize=None,
                     ysize=None,
                     margin=np.radians(0.15)):
    """
    Wrapper for py_computeDEMBounds function.
    """
    # Properly set radar bounds
    xsize = xsize or radarGrid.width
    ysize = ysize or radarGrid.length

    # Call function
    bounds = isceextension.py_computeDEMBounds(orbit, ellipsoid, doppler,
                                lookSide, radarGrid, xoff,
                                 yoff, xsize, ysize, margin)

    # Return result
    return bounds
