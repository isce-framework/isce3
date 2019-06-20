#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from libcpp cimport bool

# Cython declaration for isce::io objects
from Raster cimport Raster

# Cython declarations for isce::product objects
from Product cimport Product

# Interpolation methods and Orbit
from Orbit cimport Orbit, orbitInterpMethod
from Interpolator cimport dataInterpMethod

from RadarGridParameters cimport RadarGridParameters
from Ellipsoid cimport Ellipsoid

from LUT2d cimport LUT2d

cdef extern from "isce/geometry/Topo.h" namespace "isce::geometry":
    cdef cppclass Topo:

        # Constructor
        Topo(Product & product, char frequency, bool nativeDoppler,
             size_t numberAzimuthLooks, size_t numberRangeLooks) except +
        Topo(RadarGridParameters & radarGrid, Orbit & orbit, 
                Ellipsoid & ellipsoid,
                int lookSide, LUT2d[double] & doppler) except +
        Topo(RadarGridParameters & radarGrid, Orbit & orbit,
                Ellipsoid & ellipsoid,
                int lookSide) except +
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
        void orbitMethod(orbitInterpMethod)
        void demMethod(dataInterpMethod)
        void epsgOut(int)
        void computeMask(bool)
        
# end of file
