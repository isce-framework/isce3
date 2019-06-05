#-*- coding: utf-8 -*-
# Heresh Fattahi
# Copyright 2019-

# The extensions
import isce3.extensions.isceextension as isceextension

class Rdr2geo(isceextension.pyTopo):
    """
    Wrapper for Topo
    """
    pass

def radar2geo_point(azimuthTime=None, 
            slantRange=None,
            ellipsoid=None,
            orbit=None,
            side=None,
            doppler = 0,
            wvl = 0.24,
            threshold = 0.05,
            maxIter = 50,
            extraIter = 50,
            orbitMethod = 'hermite',
            demInterpolatorHeight = 0):

    """
    Wrapper for py_rdr2geo standalone function.
    """


    llh = isceextension.py_rdr2geo(
            orbit, ellipsoid,
            azimuthTime, slantRange, side,
            doppler, wvl,
            threshold, maxIter, extraIter,
            orbitMethod, demInterpolatorHeight
            )

    return llh

