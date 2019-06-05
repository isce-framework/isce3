#-*- coding: utf-8 -*-
#
# Heresh Fattahi, Bryan Riel
# Copyright 2019-

# The extensions
import isce3.extensions.isceextension as isceextension

class Geo2rdr(isceextension.pyGeo2rdr):
    """
    Wrapper for Geo2rdr
    """
    pass

def geo2rdr_point(lonlatheight=None,
            ellipsoid=None,
            orbit=None,
            doppler=None,
            wavelength=0.24,
            threshold=0.05,
            maxiter=50,
            dR=1.0e-8):

    """
    Wrapper for py_geo2rdr standalone function.
    """
    azimuthTime, slantRange = isceextension.py_geo2rdr(
        lonlatheight, ellipsoid, orbit, doppler, 
        wavelength, threshold=threshold,
        maxiter=maxiter, dR=dR
    )
    return azimuthTime, slantRange


