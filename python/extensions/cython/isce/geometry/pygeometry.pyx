#cython: language_level=3
#
# Author: Bryan Riel, Tamas Gal
# Copyright 2017-2019
#

from cython.operator cimport dereference as deref
from geometry cimport *


def py_geo2rdr(list llh, pyEllipsoid ellps, pyOrbit orbit, pyLUT2d doppler,
               double wvl, int side, double threshold = 0.05, int maxiter = 50,
               double dR = 1.0e-8):

    # Transfer llh to a cartesian_t
    cdef int i
    cdef cartesian_t cart_llh
    for i in range(3):
        cart_llh[i] = llh[i]

    # Call C++ geo2rdr
    cdef double azimuthTime = 0.0
    cdef double slantRange = 0.0
    geo2rdr(cart_llh,
            deref(ellps.c_ellipsoid),
            orbit.c_orbit,
            deref(doppler.c_lut),
            azimuthTime, slantRange, wvl, side, threshold, maxiter, dR)

    # All done
    return azimuthTime, slantRange



def py_rdr2geo(pyOrbit orbit,  pyEllipsoid ellps,
               double aztime, double slantRange, int side,
               double doppler         = 0.0,
               double wvl             = 0.24,
               double threshold         = 0.05,
               int      maxIter         = 50,
               int      extraIter         = 50,
               demInterpolatorHeight = 0):

    cdef cartesian_t targ_llh

    cdef DEMInterpolator demInterpolator = DEMInterpolator(demInterpolatorHeight)

    # Call C++ rdr2geo
    rdr2geo(aztime, slantRange, doppler,
            orbit.c_orbit,
            deref(ellps.c_ellipsoid),
            demInterpolator,
            targ_llh, wvl, side, threshold, maxIter, extraIter)

    llh = [0, 0, 0]

    for i in range(3):
        llh[i] = targ_llh[i]

    return llh

def py_computeDEMBounds(pyOrbit orbit,
                        pyEllipsoid ellps,
                        pyLUT2d doppler,
                        int lookSide,
                        pyRadarGridParameters radarGrid,
                        unsigned int xoff,
                        unsigned int yoff,
                        unsigned int xsize,
                        unsigned int ysize,
                        double margin):

    # Initialize outputs
    cdef double min_lon = 0.0
    cdef double min_lat = 0.0
    cdef double max_lon = 0.0
    cdef double max_lat = 0.0

    # Call C++ computeDEMBounds
    computeDEMBounds(orbit.c_orbit, deref(ellps.c_ellipsoid), deref(doppler.c_lut),
                     lookSide, deref(radarGrid.c_radargrid), xoff, yoff, xsize, ysize,
                     margin, min_lon, min_lat, max_lon, max_lat)

    # Return GDAL projwin-like list [ulx, uly, lrx, lry]
    return [min_lon, max_lat, max_lon, min_lat]

# end of file
