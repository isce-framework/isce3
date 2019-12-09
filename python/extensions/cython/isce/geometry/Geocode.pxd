#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from libcpp.string cimport string

# Cython declarations for isce::core objects
from Raster cimport Raster

# Cython declarations for isce::product objects
from Product cimport Product

# Core
from DateTime cimport DateTime
from Ellipsoid cimport Ellipsoid
from Orbit cimport Orbit
from LUT2d cimport LUT2d
from Interpolator cimport dataInterpMethod

cdef extern from "isce/geometry/Geocode.h" namespace "isce::geometry":

    # Geo2rdr class
    cdef cppclass Geocode[T]:

        # Constructor
        Geocode() except +

        # Set options
        void orbit(Orbit & orbit)
        void ellipsoid(Ellipsoid & ellipsoid)
        void thresholdGeo2rdr(double threshold)
        void numiterGeo2rdr(int numiter)
        void linesPerBlock(size_t lines)
        void demBlockMargin(double margin)
        void radarBlockMargin(int margin)
        void interpolator(dataInterpMethod method)

        # Set the geographic grid geometry
        void geoGrid(double geoGridStartX,
                     double geoGridStartY,
                     double geoGridSpacingX,
                     double geoGridSpacingY,
                     int width,
                     int length,
                     int epsgcode)
       
        # Set the radar grid geometry 
        void radarGrid(LUT2d[double] doppler,
                       DateTime refEpoch,
                       double azimuthStartTime,
                       double azimuthTimeInterval,
                       int radarGridLength,
                       double startingRange,
                       double rangeSpacing,
                       double wavelength,
                       int radarGridWidth,
                       int lookSide,
                       int numberAzimuthLooks,
                       int numberRangeLooks)

        # Run geocoding
        void geocode(Raster & inputRaster,
                     Raster & outputRaster,
                     Raster & demRaster) 

# end of file
