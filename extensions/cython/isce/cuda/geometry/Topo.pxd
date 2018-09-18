#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string

# Cython declaration for isce::io objects
from Raster cimport Raster

# Cython declarations for isce::product objects
from Product cimport Product

cdef extern from "isce/cuda/geometry/Topo.h" namespace "isce::cuda::geometry":
    cdef cppclass Topo:

        # Constructor
        Topo(Product &) except +
        
        # Main topo entrypoint
        void topo(Raster &, string)
        
# end of file
