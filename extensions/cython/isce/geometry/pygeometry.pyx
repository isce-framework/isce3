#cython: language_level=3
#
# Author: Bryan Riel, Tamas Gal
# Copyright 2017-2019
#

from cython.operator cimport dereference as deref
from geometry cimport *
from Orbit cimport orbitInterpMethod


def py_geo2rdr(list llh, pyEllipsoid ellps, pyOrbit orbit, pyLUT2d doppler,
			   double wvl, double threshold = 0.05, int maxiter = 50,
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
			deref(orbit.c_orbit),
			deref(doppler.c_lut),
			azimuthTime, slantRange, wvl, threshold, maxiter, dR)

	# All done
	return azimuthTime, slantRange



def py_rdr2geo(pyOrbit orbit,  pyEllipsoid ellps,
			   double aztime, double slantRange, int side,
			   double doppler		 = 0.0,
			   double wvl			 = 0.24,
			   double threshold		 = 0.05,
			   int	  maxIter		 = 50,
			   int	  extraIter		 = 50,
			   orbitMethod			 = 'hermite',
			   demInterpolatorHeight = 0):

	# Orbit interpolation methods
	orbitInterpMethods = {
		'hermite':	orbitInterpMethod.HERMITE_METHOD,
		'sch' :		orbitInterpMethod.SCH_METHOD,
		'legendre': orbitInterpMethod.LEGENDRE_METHOD
	}

	cdef cartesian_t targ_llh

	cdef DEMInterpolator demInterpolator = DEMInterpolator(demInterpolatorHeight)

	cdef orbitInterpMethod orbitMethodCpp = orbitInterpMethods[orbitMethod]

	# Call C++ rdr2geo
	rdr2geo(aztime, slantRange, doppler,
			deref(orbit.c_orbit),
			deref(ellps.c_ellipsoid),
			demInterpolator,
			targ_llh, wvl, side, threshold, maxIter, extraIter,
			orbitMethodCpp)

	llh = [0, 0, 0]

	for i in range(3):
		llh[i] = targ_llh[i]

	return llh



# end of file
