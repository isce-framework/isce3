#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from libcpp cimport bool

# Cython declaration for isce3::io objects
from Raster cimport Raster

# Cython declarations for isce3::product objects
from Product cimport Product

# Interpolation methods and Orbit
from Orbit cimport Orbit
from Interpolator cimport dataInterpMethod

from RadarGridParameters cimport RadarGridParameters
from Ellipsoid cimport Ellipsoid

from LUT2d cimport LUT2d

cdef extern from "isce3/geometry/Topo.h" namespace "isce3::geometry":
    cdef cppclass Topo:

        # Constructor
        Topo(Product & product, char frequency, bool nativeDoppler,
             size_t numberAzimuthLooks, size_t numberRangeLooks) except +
        Topo(RadarGridParameters & radarGrid, Orbit & orbit,
                Ellipsoid & ellipsoid,
                LUT2d[double] & doppler) except +
        Topo(RadarGridParameters & radarGrid, Orbit & orbit,
                Ellipsoid & ellipsoid) except +
        # Main topo entrypoint; internal construction of topo rasters
        void topo(Raster &, const string)

        # Run topo with externally created topo rasters (plus mask raster)
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
