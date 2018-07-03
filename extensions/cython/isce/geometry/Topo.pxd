#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string

# Cython declarations for isce::core objects
from Ellipsoid cimport Ellipsoid
from Orbit cimport Orbit
from Metadata cimport Metadata
from Raster cimport Raster
from Poly2d cimport Poly2d

cdef extern from "isce/geometry/Topo.h" namespace "isce::geometry":
    cdef cppclass Topo:

        # Constructor
        Topo(Ellipsoid, Orbit, Metadata) except +
        
        # Main topo entrypoint
        void topo(Raster &, Poly2d &, string)
        
# end of file
