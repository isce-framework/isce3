#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from libcpp cimport bool

# Cython declaration for isce3::io objects
from isceextension cimport dataInterpMethod
from isceextension cimport Ellipsoid
from isceextension cimport Product
from isceextension cimport Orbit
from isceextension cimport Raster
from LUT2d cimport LUT2d
from RadarGridParameters cimport RadarGridParameters

cdef extern from "isce3/cuda/geometry/Topo.h" namespace "isce3::cuda::geometry":
    cdef cppclass Topo:

        # Constructor
        Topo(Product & product, char frequency, bool nativeDoppler) except +

        Topo(RadarGridParameters & radarGrid, Orbit & orbit,
                Ellipsoid & ellipsoid, LUT2d[double] & doppler) except+

        Topo(RadarGridParameters & radarGrid, Orbit & orbit,
                Ellipsoid & ellipsoid) except+

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
        void demMethod(dataInterpMethod)
        void epsgOut(int)
        void computeMask(bool)
        void minimumHeight(double)
        void maximumHeight(double)
        void decimaldegMargin(double)

# end of file
