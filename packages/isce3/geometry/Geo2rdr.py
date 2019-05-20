#-*- coding: utf-8 -*-
#import numpy as np

# The extensions
import isce3.extensions.isceextension as isceextension

class Geo2rdr(isceextension.pyGeo2rdr):
    """
    Wrapper for Geo2rdr
    """
    pass

def geo2rdr(llh,
            ellipsoid,
            orbit,
            doppler,
            wavelength=0.24,
            threshold=0.05,
            maxiter=50,
            dR=1.0e-8):

    """
    Wrapper for py_geo2rdr standalone function.
    """
    azimuthTime, slantRange = isceextension.py_geo2rdr(
        llh, ellipsoid, orbit, doppler, wvl, threshold=threshold,
        maxiter=maxiter, dR=dR
    )
    return azimuthTime, slantRange
