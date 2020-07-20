#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.complex cimport complex as complex_t
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref
cimport cython
from libc.math cimport NAN
from GDAL cimport GDALDataType as GDT

from Raster cimport Raster
from RTC cimport rtcInputRadiometry

from LookSide cimport LookSide
from Geocode cimport *


rtc_input_radiometry_dict = {'BETA_NAUGHT': rtcInputRadiometry.BETA_NAUGHT,
                             'SIGMA_NAUGHT_ELLIPSOID': rtcInputRadiometry.SIGMA_NAUGHT_ELLIPSOID}

geocode_output_mode_dict = {'INTERP': geocodeOutputMode.INTERP,
                            'AREA_PROJECTION': geocodeOutputMode.AREA_PROJECTION,
                            'AREA_PROJECTION_GAMMA_NAUGHT': geocodeOutputMode.AREA_PROJECTION_GAMMA_NAUGHT}

geocode_memory_mode_dict = {'AUTO': geocodeMemoryMode.AUTO,
                            'SINGLE_BLOCK': geocodeMemoryMode.SINGLE_BLOCK,
                            'BLOCKS_GEOGRID': geocodeMemoryMode.BLOCKS_GEOGRID,
                            'BLOCKS_GEOGRID_AND_RADARGRID': geocodeMemoryMode.BLOCKS_GEOGRID_AND_RADARGRID}

rtc_algorithm_dict = {'RTC_DAVID_SMALL': rtcAlgorithm.RTC_DAVID_SMALL,
                      'RTC_AREA_PROJECTION': rtcAlgorithm.RTC_AREA_PROJECTION}

def enum_dict_decorator(enum_dict, default_key):
    def decorated(f):
        def wrapper(input_key):
            input_enum = None
            if input_key is None:
                dict_key=default_key
            elif isinstance(input_key, numbers.Number):
                input_enum = input_key
            else:
                dict_key = input_key.upper().replace('-', '_')
            if input_enum is None:
                input_enum = enum_dict[dict_key]
            return input_enum
        return wrapper
    return decorated

@enum_dict_decorator(rtc_input_radiometry_dict, 'SIGMA_NAUGHT_ELLIPSOID')
def getRtcInputRadiometry(*args, **kwargs):
    pass

@enum_dict_decorator(geocode_output_mode_dict, 'INTERP')
def getOutputMode(*args, **kwargs):
    pass

@enum_dict_decorator(geocode_memory_mode_dict, 'AUTO')
def getMemoryMode(*args, **kwargs):
    pass

@enum_dict_decorator(rtc_algorithm_dict, 'RTC_AREA_PROJECTION')
def getRtcAlgorithm(*args, **kwargs):
    pass



cdef class pyGeocodeBase:
    """
    Cython wrapper for isce3::geometry::Geocode.

    Args:
        orbit (pyOrbit):                    Orbit instance.
        ellps (pyEllipsoid):                Ellipsoid instance.
        threshold (Optional[float]):        Threshold for underlying geo2rdr function calls.
        numiter (Optional[int]):            Max number of iterations for underlying geo2rdr.
        linesPerBlock (Optional[int]):      Number of lines per input radar block.
        demBlockMargin (Optional[float]):   DEM block margin in degrees.
        radarBlockMargin (Optional[int]):   Radar block margin.
        interpMethod (Optional[str]):       Image interpolation method
                                                ('sinc', 'bilinear', 'bicubic', 'nearest',
                                                 'biquintic')

    Return:
        None
    """
    cdef Orbit c_orbit
    cdef Ellipsoid * c_ellipsoid
    cdef LUT2d[double] * c_doppler
    cdef string refepoch_string

    # Processing options
    cdef double threshold
    cdef int numiter
    cdef int linesPerBlock
    cdef double demBlockMargin
    cdef int radarBlockMargin
    cdef dataInterpMethod interpMethod

    # Radar grid parameters
    cdef double azimuthStartTime
    cdef double azimuthTimeInterval
    cdef int radarGridLength
    cdef double startingRange
    cdef double rangeSpacing
    cdef double wavelength
    cdef int radarGridWidth
    cdef LookSide lookSide

    # Geographic grid parameters
    cdef int epsgcode

    # DEM interpolation methods
    demInterpMethods = {
        'sinc': dataInterpMethod.SINC_METHOD,
        'bilinear': dataInterpMethod.BILINEAR_METHOD,
        'bicubic': dataInterpMethod.BICUBIC_METHOD,
        'nearest': dataInterpMethod.NEAREST_METHOD,
        'biquintic': dataInterpMethod.BIQUINTIC_METHOD
    }


cdef class pyGeocodeFloat(pyGeocodeBase):

    # Create Geocoding object
    cdef Geocode[float] c_geocode
    
    def __cinit__(self,
                  pyOrbit orbit,
                  pyEllipsoid ellps,
                  double threshold=1.0e-8,
                  int numiter=25,
                  int linesPerBlock=1000,
                  double demBlockMargin=0.1,
                  int radarBlockMargin=10,
                  interpMethod='biquintic'):

        # Save pointers to ISCE objects
        self.c_orbit = orbit.c_orbit
        self.c_ellipsoid = ellps.c_ellipsoid

        # Save geocoding properties
        self.threshold = threshold
        self.numiter = numiter
        self.linesPerBlock = linesPerBlock
        self.demBlockMargin = demBlockMargin
        self.radarBlockMargin = radarBlockMargin
        self.interpMethod = self.demInterpMethods[interpMethod]
        self.c_geocode = Geocode[float]()

        # Set properties
        self.c_geocode.orbit(self.c_orbit)
        self.c_geocode.ellipsoid(deref(self.c_ellipsoid))
        self.c_geocode.thresholdGeo2rdr(self.threshold)
        self.c_geocode.numiterGeo2rdr(self.numiter)
        self.c_geocode.linesPerBlock(self.linesPerBlock)
        self.c_geocode.demBlockMargin(self.demBlockMargin)
        self.c_geocode.radarBlockMargin(self.radarBlockMargin)
        self.c_geocode.interpolator(self.interpMethod)

    def geoGrid(self,
                double geoGridStartX,
                double geoGridStartY,
                double geoGridSpacingX,
                double geoGridSpacingY,
                int width,
                int length,
                int epsgcode):
        """
        Saves parameters for output geographic grid.
        """
        self.epsgcode = epsgcode
        # Set geo grid
        self.c_geocode.geoGrid(
            geoGridStartX, geoGridStartY, geoGridSpacingX,
            geoGridSpacingY, width, length, self.epsgcode)

    def updateGeoGrid(self,
                      pyRadarGridParameters radarGrid,
                      pyRaster demRaster):
        """
        Update geogrid with radar grid and DEM raster
        """
        self.c_geocode.updateGeoGrid(deref(radarGrid.c_radargrid), 
                                     deref(demRaster.c_raster))

    def geocode(self,
                pyRadarGridParameters radarGrid,
                pyRaster inputRaster,
                pyRaster outputRaster,
                pyRaster demRaster,
                int inputBand = 1,
                output_mode = None,
                double upsampling = 1,
                input_radiometry = None,
                int exponent = 0,
                rtc_min_value_db = NAN,
                double rtc_upsampling = NAN,
                rtc_algorithm = None,
                double abs_cal_factor = 1,
                float clip_min = NAN,
                float clip_max = NAN,
                float min_nlooks = NAN,
                float radar_grid_nlooks = 1,
                out_geo_vertices = None,
                out_dem_vertices = None,
                out_geo_nlooks = None,
                out_geo_rtc = None,
                input_rtc = None,
                output_rtc = None,
                memory_mode = 'AUTO'):
        """
        Run geocoding.
        """

        output_mode_enum = getOutputMode(output_mode)
        rtc_input_radiometry = getRtcInputRadiometry(input_radiometry)

        # RTC algorithm
        rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

        out_geo_vertices_raster = _getRaster(out_geo_vertices)
        out_dem_vertices_raster = _getRaster(out_dem_vertices)
        out_geo_nlooks_raster = _getRaster(out_geo_nlooks)
        out_geo_rtc_raster = _getRaster(out_geo_rtc)
        input_rtc_raster = _getRaster(input_rtc)
        output_rtc_raster = _getRaster(output_rtc)
        memory_mode_enum = getMemoryMode(memory_mode)

        # Run geocoding
        self.c_geocode.geocode(deref(radarGrid.c_radargrid),
                               deref(inputRaster.c_raster), 
                               deref(outputRaster.c_raster),
                               deref(demRaster.c_raster), 
                               output_mode_enum, 
                               upsampling,
                               rtc_input_radiometry, 
                               exponent,
                               rtc_min_value_db,
                               rtc_upsampling,
                               rtc_algorithm_obj,
                               abs_cal_factor,
                               clip_min,
                               clip_max,
                               min_nlooks,
                               radar_grid_nlooks,
                               out_geo_vertices_raster,
                               out_dem_vertices_raster,
                               out_geo_nlooks_raster,
                               out_geo_rtc_raster,
                               input_rtc_raster,
                               output_rtc_raster,
                               memory_mode_enum)
    
    @property
    def geoGridStartX(self):
        return self.c_geocode.geoGridStartX()
    @property
    def geoGridStartY(self):
        return self.c_geocode.geoGridStartY()
    @property
    def geoGridSpacingX(self):
        return self.c_geocode.geoGridSpacingX()
    @property
    def geoGridSpacingY(self):
        return self.c_geocode.geoGridSpacingY()
    @property
    def geoGridWidth(self):
        return self.c_geocode.geoGridWidth()
    @property
    def geoGridLength(self):
        return self.c_geocode.geoGridLength()


cdef class pyGeocodeDouble(pyGeocodeBase):

    # Create Geocoding object
    cdef Geocode[double] c_geocode
    
    def __cinit__(self,
                  pyOrbit orbit,
                  pyEllipsoid ellps,
                  double threshold=1.0e-8,
                  int numiter=25,
                  int linesPerBlock=1000,
                  double demBlockMargin=0.1,
                  int radarBlockMargin=10,
                  interpMethod='biquintic'):

        # Save pointers to ISCE objects
        self.c_orbit = orbit.c_orbit
        self.c_ellipsoid = ellps.c_ellipsoid

        # Save geocoding properties
        self.threshold = threshold
        self.numiter = numiter
        self.linesPerBlock = linesPerBlock
        self.demBlockMargin = demBlockMargin
        self.radarBlockMargin = radarBlockMargin
        self.interpMethod = self.demInterpMethods[interpMethod]
        self.c_geocode = Geocode[double]()

        # Set properties
        self.c_geocode.orbit(self.c_orbit)
        self.c_geocode.ellipsoid(deref(self.c_ellipsoid))
        self.c_geocode.thresholdGeo2rdr(self.threshold)
        self.c_geocode.numiterGeo2rdr(self.numiter)
        self.c_geocode.linesPerBlock(self.linesPerBlock)
        self.c_geocode.demBlockMargin(self.demBlockMargin)
        self.c_geocode.radarBlockMargin(self.radarBlockMargin)
        self.c_geocode.interpolator(self.interpMethod)

    def geoGrid(self,
                double geoGridStartX,
                double geoGridStartY,
                double geoGridSpacingX,
                double geoGridSpacingY,
                int width,
                int length,
                int epsgcode):
        """
        Saves parameters for output geographic grid.
        """
        self.epsgcode = epsgcode
        # Set geo grid
        self.c_geocode.geoGrid(
            geoGridStartX, geoGridStartY, geoGridSpacingX,
            geoGridSpacingY, width, length, self.epsgcode)

    def updateGeoGrid(self,
                      pyRadarGridParameters radarGrid,
                      pyRaster demRaster):
        """
        Update geogrid with radar grid and DEM raster
        """
        self.c_geocode.updateGeoGrid(deref(radarGrid.c_radargrid), 
                                     deref(demRaster.c_raster))

    def geocode(self,
                pyRadarGridParameters radarGrid,
                pyRaster inputRaster,
                pyRaster outputRaster,
                pyRaster demRaster,
                int inputBand = 1,
                output_mode = None,
                double upsampling = 1,
                input_radiometry = None,
                int exponent = 0,
                rtc_min_value_db = NAN,
                double rtc_upsampling = NAN,
                rtc_algorithm = None,
                double abs_cal_factor = 1,
                float clip_min = NAN,
                float clip_max = NAN,
                float min_nlooks = NAN,
                float radar_grid_nlooks = 1,
                out_geo_vertices = None,
                out_dem_vertices = None,
                out_geo_nlooks = None,
                out_geo_rtc = None,
                input_rtc = None,
                output_rtc = None,
                memory_mode = 'AUTO'):
        """
        Run geocoding.
        """

        output_mode_enum = getOutputMode(output_mode)
        rtc_input_radiometry = getRtcInputRadiometry(input_radiometry)

        # RTC algorithm
        rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

        out_geo_vertices_raster = _getRaster(out_geo_vertices)
        out_dem_vertices_raster = _getRaster(out_dem_vertices)
        out_geo_nlooks_raster = _getRaster(out_geo_nlooks)
        out_geo_rtc_raster = _getRaster(out_geo_rtc)
        input_rtc_raster = _getRaster(input_rtc)
        output_rtc_raster = _getRaster(output_rtc)
        memory_mode_enum = getMemoryMode(memory_mode)

        # Run geocoding
        self.c_geocode.geocode(deref(radarGrid.c_radargrid),
                               deref(inputRaster.c_raster), 
                               deref(outputRaster.c_raster),
                               deref(demRaster.c_raster), 
                               output_mode_enum, 
                               upsampling,
                               rtc_input_radiometry, 
                               exponent,
                               rtc_min_value_db,
                               rtc_upsampling,
                               rtc_algorithm_obj,
                               abs_cal_factor,
                               clip_min,
                               clip_max,
                               min_nlooks,
                               radar_grid_nlooks,
                               out_geo_vertices_raster,
                               out_dem_vertices_raster,
                               out_geo_nlooks_raster,
                               out_geo_rtc_raster,
                               input_rtc_raster,
                               output_rtc_raster,
                               memory_mode_enum)
    
    @property
    def geoGridStartX(self):
        return self.c_geocode.geoGridStartX()
    @property
    def geoGridStartY(self):
        return self.c_geocode.geoGridStartY()
    @property
    def geoGridSpacingX(self):
        return self.c_geocode.geoGridSpacingX()
    @property
    def geoGridSpacingY(self):
        return self.c_geocode.geoGridSpacingY()
    @property
    def geoGridWidth(self):
        return self.c_geocode.geoGridWidth()
    @property
    def geoGridLength(self):
        return self.c_geocode.geoGridLength()



cdef class pyGeocodeComplexFloat(pyGeocodeBase):

    # Create Geocoding object
    cdef Geocode[complex_t[float]] c_geocode
    
    def __cinit__(self,
                  pyOrbit orbit,
                  pyEllipsoid ellps,
                  double threshold=1.0e-8,
                  int numiter=25,
                  int linesPerBlock=1000,
                  double demBlockMargin=0.1,
                  int radarBlockMargin=10,
                  interpMethod='biquintic'):

        # Save pointers to ISCE objects
        self.c_orbit = orbit.c_orbit
        self.c_ellipsoid = ellps.c_ellipsoid

        # Save geocoding properties
        self.threshold = threshold
        self.numiter = numiter
        self.linesPerBlock = linesPerBlock
        self.demBlockMargin = demBlockMargin
        self.radarBlockMargin = radarBlockMargin
        self.interpMethod = self.demInterpMethods[interpMethod]
        self.c_geocode = Geocode[complex_t[float]]()
        
        # Set properties
        self.c_geocode.orbit(self.c_orbit)
        self.c_geocode.ellipsoid(deref(self.c_ellipsoid))
        self.c_geocode.thresholdGeo2rdr(self.threshold)
        self.c_geocode.numiterGeo2rdr(self.numiter)
        self.c_geocode.linesPerBlock(self.linesPerBlock)
        self.c_geocode.demBlockMargin(self.demBlockMargin)
        self.c_geocode.radarBlockMargin(self.radarBlockMargin)
        self.c_geocode.interpolator(self.interpMethod)

    def geoGrid(self,
                double geoGridStartX,
                double geoGridStartY,
                double geoGridSpacingX,
                double geoGridSpacingY,
                int width,
                int length,
                int epsgcode):
        """
        Saves parameters for output geographic grid.
        """
        self.epsgcode = epsgcode
        # Set geo grid
        self.c_geocode.geoGrid(
            geoGridStartX, geoGridStartY, geoGridSpacingX,
            geoGridSpacingY, width, length, self.epsgcode)

    def updateGeoGrid(self,
                      pyRadarGridParameters radarGrid,
                      pyRaster demRaster):
        """
        Update geogrid with radar grid and DEM raster
        """
        self.c_geocode.updateGeoGrid(deref(radarGrid.c_radargrid), 
                                     deref(demRaster.c_raster))

    def geocode(self,
                pyRadarGridParameters radarGrid,
                pyRaster inputRaster,
                pyRaster outputRaster,
                pyRaster demRaster,
                int inputBand = 1,
                output_mode = None,
                double upsampling = 1,
                input_radiometry = None,
                int exponent = 0,
                rtc_min_value_db = NAN,
                double rtc_upsampling = NAN,
                rtc_algorithm = None,
                double abs_cal_factor = 1,
                float clip_min = NAN,
                float clip_max = NAN,
                float min_nlooks = NAN,
                float radar_grid_nlooks = 1,
                out_geo_vertices = None,
                out_dem_vertices = None,
                out_geo_nlooks = None,
                out_geo_rtc = None,
                input_rtc = None,
                output_rtc = None,
                memory_mode = 'AUTO'):
        """
        Run geocoding.
        """

        output_mode_enum = getOutputMode(output_mode)
        rtc_input_radiometry = getRtcInputRadiometry(input_radiometry)

        # RTC algorithm
        rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

        out_geo_vertices_raster = _getRaster(out_geo_vertices)
        out_dem_vertices_raster = _getRaster(out_dem_vertices)
        out_geo_nlooks_raster = _getRaster(out_geo_nlooks)
        out_geo_rtc_raster = _getRaster(out_geo_rtc)
        input_rtc_raster = _getRaster(input_rtc)
        output_rtc_raster = _getRaster(output_rtc)
        memory_mode_enum = getMemoryMode(memory_mode)

        # Run geocoding
        self.c_geocode.geocode(deref(radarGrid.c_radargrid),
                               deref(inputRaster.c_raster), 
                               deref(outputRaster.c_raster),
                               deref(demRaster.c_raster), 
                               output_mode_enum, 
                               upsampling,
                               rtc_input_radiometry, 
                               exponent,
                               rtc_min_value_db,
                               rtc_upsampling,
                               rtc_algorithm_obj,
                               abs_cal_factor,
                               clip_min,
                               clip_max,
                               min_nlooks,
                               radar_grid_nlooks,
                               out_geo_vertices_raster,
                               out_dem_vertices_raster,
                               out_geo_nlooks_raster,
                               out_geo_rtc_raster,
                               input_rtc_raster,
                               output_rtc_raster,
                               memory_mode_enum)
    
    @property
    def geoGridStartX(self):
        return self.c_geocode.geoGridStartX()
    @property
    def geoGridStartY(self):
        return self.c_geocode.geoGridStartY()
    @property
    def geoGridSpacingX(self):
        return self.c_geocode.geoGridSpacingX()
    @property
    def geoGridSpacingY(self):
        return self.c_geocode.geoGridSpacingY()
    @property
    def geoGridWidth(self):
        return self.c_geocode.geoGridWidth()
    @property
    def geoGridLength(self):
        return self.c_geocode.geoGridLength()


cdef class pyGeocodeComplexDouble(pyGeocodeBase):

    # Create Geocoding object
    cdef Geocode[complex_t[double]] c_geocode
    
    def __cinit__(self,
                  pyOrbit orbit,
                  pyEllipsoid ellps,
                  double threshold=1.0e-8,
                  int numiter=25,
                  int linesPerBlock=1000,
                  double demBlockMargin=0.1,
                  int radarBlockMargin=10,
                  interpMethod='biquintic'):

        # Save pointers to ISCE objects
        self.c_orbit = orbit.c_orbit
        self.c_ellipsoid = ellps.c_ellipsoid

        # Save geocoding properties
        self.threshold = threshold
        self.numiter = numiter
        self.linesPerBlock = linesPerBlock
        self.demBlockMargin = demBlockMargin
        self.radarBlockMargin = radarBlockMargin
        self.interpMethod = self.demInterpMethods[interpMethod]
        self.c_geocode = Geocode[complex_t[double]]()

        # Set properties
        self.c_geocode.orbit(self.c_orbit)
        self.c_geocode.ellipsoid(deref(self.c_ellipsoid))
        self.c_geocode.thresholdGeo2rdr(self.threshold)
        self.c_geocode.numiterGeo2rdr(self.numiter)
        self.c_geocode.linesPerBlock(self.linesPerBlock)
        self.c_geocode.demBlockMargin(self.demBlockMargin)
        self.c_geocode.radarBlockMargin(self.radarBlockMargin)
        self.c_geocode.interpolator(self.interpMethod)

    def geoGrid(self,
                double geoGridStartX,
                double geoGridStartY,
                double geoGridSpacingX,
                double geoGridSpacingY,
                int width,
                int length,
                int epsgcode):
        """
        Saves parameters for output geographic grid.
        """
        self.epsgcode = epsgcode
        # Set geo grid
        self.c_geocode.geoGrid(
            geoGridStartX, geoGridStartY, geoGridSpacingX,
            geoGridSpacingY, width, length, self.epsgcode)

    def updateGeoGrid(self,
                      pyRadarGridParameters radarGrid,
                      pyRaster demRaster):
        """
        Update geogrid with radar grid and DEM raster
        """
        self.c_geocode.updateGeoGrid(deref(radarGrid.c_radargrid), 
                                     deref(demRaster.c_raster))

    def geocode(self,
                pyRadarGridParameters radarGrid,
                pyRaster inputRaster,
                pyRaster outputRaster,
                pyRaster demRaster,
                int inputBand = 1,
                output_mode = None,
                double upsampling = 1,
                input_radiometry = None,
                int exponent = 0,
                rtc_min_value_db = NAN,
                double rtc_upsampling = NAN,
                rtc_algorithm = None,
                double abs_cal_factor = 1,
                float clip_min = NAN,
                float clip_max = NAN,
                float min_nlooks = NAN,
                float radar_grid_nlooks = 1,
                out_geo_vertices = None,
                out_dem_vertices = None,
                out_geo_nlooks = None,
                out_geo_rtc = None,
                input_rtc = None,
                output_rtc = None,
                memory_mode = 'AUTO'):
        """
        Run geocoding.
        """

        output_mode_enum = getOutputMode(output_mode)
        rtc_input_radiometry = getRtcInputRadiometry(input_radiometry)

        # RTC algorithm
        rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

        out_geo_vertices_raster = _getRaster(out_geo_vertices)
        out_dem_vertices_raster = _getRaster(out_dem_vertices)
        out_geo_nlooks_raster = _getRaster(out_geo_nlooks)
        out_geo_rtc_raster = _getRaster(out_geo_rtc)
        input_rtc_raster = _getRaster(input_rtc)
        output_rtc_raster = _getRaster(output_rtc)
        memory_mode_enum = getMemoryMode(memory_mode)

        # Run geocoding
        self.c_geocode.geocode(deref(radarGrid.c_radargrid),
                               deref(inputRaster.c_raster), 
                               deref(outputRaster.c_raster),
                               deref(demRaster.c_raster), 
                               output_mode_enum, 
                               upsampling,
                               rtc_input_radiometry, 
                               exponent,
                               rtc_min_value_db,
                               rtc_upsampling,
                               rtc_algorithm_obj,
                               abs_cal_factor,
                               clip_min,
                               clip_max,
                               min_nlooks,
                               radar_grid_nlooks,
                               out_geo_vertices_raster,
                               out_dem_vertices_raster,
                               out_geo_nlooks_raster,
                               out_geo_rtc_raster,
                               input_rtc_raster,
                               output_rtc_raster,
                               memory_mode_enum)
    
    @property
    def geoGridStartX(self):
        return self.c_geocode.geoGridStartX()
    @property
    def geoGridStartY(self):
        return self.c_geocode.geoGridStartY()
    @property
    def geoGridSpacingX(self):
        return self.c_geocode.geoGridSpacingX()
    @property
    def geoGridSpacingY(self):
        return self.c_geocode.geoGridSpacingY()
    @property
    def geoGridWidth(self):
        return self.c_geocode.geoGridWidth()
    @property
    def geoGridLength(self):
        return self.c_geocode.geoGridLength()


def pyGetGeoAreaElementMean(
        vector[double] x_vect,
        vector[double] y_vect,
        pyRadarGridParameters radarGrid,
        pyOrbit orbit,
        pyLUT2d doppler,
        pyRaster input_raster,
        pyRaster dem_raster, 
        input_radiometry=None,
        int exponent = 0,
        output_mode = None,
        dem_upsampling = NAN,
        rtc_min_value_db = NAN,
        abs_cal_factor = 1,
        radar_grid_nlooks = 1):

    # input radiometry
    rtc_input_radiometry = getRtcInputRadiometry(input_radiometry)

    # output mode
    output_mode_enum = getOutputMode(output_mode)

    ret = getGeoAreaElementMean(        
        x_vect,
        y_vect,
        deref(radarGrid.c_radargrid),
        orbit.c_orbit,
        deref(doppler.c_lut),
        deref(input_raster.c_raster),
        deref(dem_raster.c_raster),
        rtc_input_radiometry,
        exponent,
        output_mode_enum,
        dem_upsampling,
        rtc_min_value_db,
        abs_cal_factor,
        radar_grid_nlooks)

    return ret

# end of file
