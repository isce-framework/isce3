#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram
# Copyright 2017-2018
#

from Orbit cimport Orbit
from Ellipsoid cimport Ellipsoid
from Cartesian cimport cartesian_t
from Metadata cimport Metadata
from Poly2d cimport Poly2d

cdef extern from "isce/geometry/geometry.h" namespace "isce::geometry":

    int geo2rdr(cartesian_t &,
                Ellipsoid &,
                Orbit &,
                Poly2d &,
                Metadata &,
                double &, double &,
                double, int, double)
        

# end of file
