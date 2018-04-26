#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string

# Cython declarations for isce::core objects
from Ellipsoid cimport Ellipsoid
from Orbit cimport Orbit, orbitInterpMethod
from Metadata cimport Metadata
from Raster cimport Raster
from Poly2d cimport Poly2d

cdef extern from "isce/geometry/Geo2rdr.h" namespace "isce::geometry":

    # Geo2rdr class
    cdef cppclass Geo2rdr:

        # Constructor
        Geo2rdr(Ellipsoid, Orbit, Metadata) except +

        # Set options
        void threshold(double)
        void numiter(int);
        void orbitMethod(orbitInterpMethod)

        # Run geo2rdr - main entrypoint
        void geo2rdr(Raster &, Raster &, Raster &, Poly2d &, const string &,
                     double, double)
        
# end of file
