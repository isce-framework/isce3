#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string

# Cython declarations for isce::core objects
from Raster cimport Raster

# Cython declarations for isce::product objects
from Product cimport Product

cdef extern from "isce/cuda/geometry/Geo2rdr.h" namespace "isce::cuda::geometry":

    # Geo2rdr class
    cdef cppclass Geo2rdr:

        # Constructor
        Geo2rdr(Product) except +

        # Run geo2rdr with no constant offsets and internally created offset rasters
        void geo2rdr(Raster &, const string &, double, double)

        # Run geo2rdr with offsets and externally created offset rasters
        void geo2rdr(Raster &, Raster &, Raster &, double, double)

# end of file
