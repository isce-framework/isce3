#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from libcpp cimport bool

# Cython declaration for isce::io objects
from isceextension cimport Raster
from isceextension cimport Product
from isceextension cimport orbitInterpMethod
from isceextension cimport dataInterpMethod

cdef extern from "isce/cuda/geometry/Topo.h" namespace "isce::cuda::geometry":
    cdef cppclass Topo:

        # Constructor
        Topo(Product &) except +
        
        # Main topo entrypoint; internal construction of topo rasters
        void topo(Raster &, const string)
        
        # Run topo with externally created topo rasters
        void topo(Raster &, Raster &, Raster &, Raster &, Raster &,
                  Raster &, Raster &, Raster &, Raster &)

        # Run topo with externally created topo rasters (plus mask raster)
        void topo(Raster &, Raster &, Raster &, Raster &, Raster &,
                  Raster &, Raster &, Raster &, Raster &, Raster &)

        # Setting processing options
        void threshold(double)
        void numiter(int)
        void extraiter(int)
        void orbitMethod(orbitInterpMethod)
        void demMethod(dataInterpMethod)
        void epsgOut(int)
        void computeMask(bool)
        
# end of file
