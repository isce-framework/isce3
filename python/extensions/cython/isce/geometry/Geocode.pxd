#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

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
from RTC cimport rtcInputRadiometry, rtcAlgorithm

from LookSide cimport LookSide

# RadarGridParameters
from RadarGridParameters cimport RadarGridParameters


cdef extern from "isce3/geometry/Geocode.h" namespace "isce::geometry":

    cdef enum geocodeOutputMode:
        INTERP = 0
        AREA_PROJECTION = 1
        AREA_PROJECTION_GAMMA_NAUGHT = 2

    cdef enum geocodeMemoryMode:
        AUTO = 0
        SINGLE_BLOCK = 1
        BLOCKS_GEOGRID = 2
        BLOCKS_GEOGRID_AND_RADARGRID = 3

    # Geocode class
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

        # Set the geogrid geometry
        void geoGrid(double geoGridStartX,
                     double geoGridStartY,
                     double geoGridSpacingX,
                     double geoGridSpacingY,
                     int width,
                     int length,
                     int epsgcode) except +

        # Update geogrid
        void updateGeoGrid(RadarGridParameters& radar_grid, 
                           Raster & dem_raster) except +

        # Run geocoding
        void geocode(RadarGridParameters& radar_grid, 
                     Raster & inputRaster,
                     Raster & outputRaster,
                     Raster & demRaster,
                     geocodeOutputMode output_mode_enum,
                     double upsampling, 
                     rtcInputRadiometry input_radiometry,
                     int exponent,
                     float rtc_min_value_db,
                     double rtc_upsampling,
                     rtcAlgorithm rtc_algorithm,
                     double abs_cal_factor,
                     float clip_min,
                     float clip_max,
                     float min_nlooks,
                     float radar_grid_nlooks,
                     Raster * out_geo_vertices,
                     Raster * out_dem_vertices,
                     Raster * out_geo_nlooks,
                     Raster * out_geo_rtc,
                     Raster * input_rtc,
                     Raster * output_rtc,
                     geocodeMemoryMode memory_mode_enum) except +

        double geoGridStartX()
        double geoGridStartY()
        double geoGridSpacingX()
        double geoGridSpacingY()
        int geoGridWidth()
        int geoGridLength()
        
    vector[float] getGeoAreaElementMean(
        vector[double] & x_vect,
        vector[double] & y_vect,
        RadarGridParameters& radar_grid, 
        Orbit& orbit,
        LUT2d[double]& dop,
        Raster& input_raster,
        Raster& dem_raster, 
        rtcInputRadiometry inputRadiometry,
        int exponent,
        geocodeOutputMode output_mode,
        double dem_upsampling,
        float rtc_min_value_db,
        double abs_cal_factor,
        float radar_grid_nlooks) except +
        
 # end of file
