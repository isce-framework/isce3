#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from cython.operator cimport dereference as deref
from geometry cimport *

def py_geo2rdr(list llh, pyEllipsoid ellps, pyOrbit orbit, pyLUT1d doppler,
               pyImageMode mode, double threshold=0.05, int maxiter=50, double dR=1.0e-8):

    # Transfer llh to a cartesian_t
    cdef int i
    cdef cartesian_t cart_llh
    for i in range(3):
        cart_llh[i] = llh[i]

    # Call C++ geo2rdr
    cdef double azimuthTime = 0.0
    cdef double slantRange = 0.0
    geo2rdr(cart_llh, deref(ellps.c_ellipsoid), deref(orbit.c_orbit),
            deref(doppler.c_lut), deref(mode.c_imagemode), azimuthTime, slantRange,
            threshold, maxiter, dR)

    # All done 
    return azimuthTime, slantRange 

# end of file 
