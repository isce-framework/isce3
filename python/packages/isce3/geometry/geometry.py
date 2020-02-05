#-*- coding: utf-8 -*-
#
# Heresh Fattahi, Bryan Riel
# Copyright 2019-

import numpy as np

# The extensions
from .. import isceextension
from isceextension import getGeoPerimeter


def getBoundsOnGround(orbit=None,
                     ellipsoid=None,
                     doppler=None,
                     lookSide=None,
                     radarGrid=None,
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


