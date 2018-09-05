#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram
# Copyright 2017-2018
#

from Orbit cimport Orbit
from Ellipsoid cimport Ellipsoid
from Cartesian cimport cartesian_t
from Poly2d cimport Poly2d
from ImageMode cimport ImageMode

cdef extern from "isce/geometry/geometry.h" namespace "isce::geometry":

    int geo2rdr(cartesian_t &,
                Ellipsoid &,
                Orbit &,
                Poly2d &,
                ImageMode &,
                double &, double &,
                double, int, double)
        

# end of file
