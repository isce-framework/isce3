#cython: language_level=3
#
# Author: Tamas Gal
# Copyright 2019
#

from Interpolator cimport dataInterpMethod
from Raster cimport Raster
from libcpp cimport bool

# DEMInterpolator
cdef extern from "isce3/geometry/DEMInterpolator.h" namespace "isce3::geometry":
    cdef cppclass DEMInterpolator:

        # Constructor
        DEMInterpolator() except +
        DEMInterpolator(float height) except +
        DEMInterpolator(float height, dataInterpMethod method) except +

        #Read in a subset
        void loadDEM(Raster &demRaster, double minX, double maxX, double minY, double maxY) except +
        void loadDEM(Raster &demRaster) except +

        #Interpolation methods
        double interpolateLonLat(double lon, double lat) except +
        double interpolateXY(double x, double y) except +

        #Utility functions
        double xStart()
        double yStart()
        double deltaX()
        double deltaY()
        double midX()
        double midY()
        bool haveRaster()
        double refHeight()
        void refHeight(double h)
        size_t width()
        size_t length()
        int epsgCode()

        #Acces to DEM
        float* data()
