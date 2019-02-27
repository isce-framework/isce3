#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram, Tamas Gal
# Copyright 2017-2019
#

from DEMInterpolator cimport DEMInterpolator
from Orbit cimport Orbit, orbitInterpMethod
from Ellipsoid cimport Ellipsoid
from Cartesian cimport cartesian_t
from LUT2d cimport LUT2d

cdef extern from "isce/geometry/geometry.h" namespace "isce::geometry":

    int geo2rdr(const cartesian_t &,
                const Ellipsoid &,
                const Orbit &,
                const LUT2d[double] &,
                double &, double &,
                double, double, int, double)

    int rdr2geo(double, double, double,
                const Orbit &, const Ellipsoid &, const DEMInterpolator &,
                cartesian_t &,
                double, int, double, int, int, orbitInterpMethod)


# end of file
