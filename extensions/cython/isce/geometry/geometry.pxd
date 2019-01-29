#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram, Tamas Gal
# Copyright 2017-2019
#

from DEMInterpolator cimport DEMInterpolator
from Orbit cimport Orbit, orbitInterpMethod
from Ellipsoid cimport Ellipsoid
from Cartesian cimport cartesian_t
from LUT1d cimport LUT1d
from ImageMode cimport ImageMode

cdef extern from "isce/geometry/geometry.h" namespace "isce::geometry":

    int geo2rdr(cartesian_t &,
                Ellipsoid &,
                Orbit &,
                LUT1d[double] &,
                ImageMode &,
                double &, double &,
                double, int, double)



    int rdr2geo(double, double, double,
                Orbit &, Ellipsoid &, DEMInterpolator &,
                cartesian_t &,
                double, int, double, int, int, orbitInterpMethod)


# end of file