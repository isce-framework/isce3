#cython: language_level=3
#
# Author: Bryan Riel, Tamas Gal
# Copyright 2017-2019
#

from cython.operator cimport dereference as deref
from geometry cimport *
from Cartesian cimport cartesian_t
from Basis cimport Basis
from libc.math cimport sin
from libcpp.string cimport string
# NOTE get toVec3 and pyDEMInterpolator defs due to isceextension include order.


def py_geo2rdr(list llh, pyEllipsoid ellps, pyOrbit orbit, pyLUT2d doppler,
               double wvl, _side, double threshold = 0.05, int maxiter = 50,
               double dR = 1.0e-8):

    cdef LookSide side = pyParseLookSide(_side)

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
               double aztime, double slantRange, _side,
               double doppler         = 0.0,
               double wvl             = 0.24,
               double threshold         = 0.05,
               int      maxIter         = 50,
               int      extraIter         = 50,
               demInterpolatorHeight = 0):

    cdef cartesian_t targ_llh

    cdef DEMInterpolator demInterpolator = DEMInterpolator(demInterpolatorHeight)

    cdef LookSide side = pyParseLookSide(_side)

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


def py_rdr2geo_cone(
        position = None,
        axis = None,
        double angle = 0.0,
        double slantRange = 0.0,
        pyEllipsoid ellipsoid = None,
        pyDEMInterpolator demInterp = None,
        side = None,
        threshold = 0.05,
        maxIter = 50,
        extraIter = 50):
    """See isce3.geometry.rdr2geo_cone for documentation.
    """
    # Handle duck typing of inputs.
    cdef cartesian_t p = toVec3(position)
    cdef cartesian_t v = toVec3(axis).normalized()
    assert slantRange > 0.0, "Require slantRange > 0"
    assert side is not None, "Must specify 'left' or 'right' side"
    cdef LookSide _side = pyParseLookSide(side)
    cdef DEMInterpolator dem
    if demInterp is not None:
        dem = deref(demInterp.c_deminterp)
    # NOTE The C++ interface is kind of busted since DEMInterpolator has its own
    # implicit ellipsoid (but only when its loadDEM method has been called) and
    # the Ellipsoid argument is assumed to describe the same one.  Usually
    # everything is WGS84, at least.
    cdef Ellipsoid ell
    if ellipsoid is not None:
        ell = deref(ellipsoid.c_ellipsoid)

    # Generate TCN basis using the given axis as the velocity.
    cdef Basis tcn = Basis(p, v)
    # Using this "doppler factor" accomplishes the desired task.
    cdef Pixel pix = Pixel(slantRange, slantRange*sin(angle), 0)

    # Call C++ and return a Python array.
    cdef cartesian_t llh
    rdr2geo(pix, tcn, p, v, ell, dem, llh, _side,threshold, maxIter, extraIter)
    return [llh[i] for i in range(3)]


def py_computeDEMBounds(pyOrbit orbit,
                        pyEllipsoid ellps,
                        pyLUT2d doppler,
                        lookSide,
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

    cdef LookSide side = pyParseLookSide(lookSide)

    # Call C++ computeDEMBounds
    computeDEMBounds(orbit.c_orbit, deref(ellps.c_ellipsoid), deref(doppler.c_lut),
                     side, deref(radarGrid.c_radargrid), xoff, yoff, xsize, ysize,
                     margin, min_lon, min_lat, max_lon, max_lat)

    # Return GDAL projwin-like list [ulx, uly, lrx, lry]
    return [min_lon, max_lat, max_lon, min_lat]

# end of file
