# cython: language_level=3

from Product cimport Product
from RadarGridParameters cimport RadarGridParameters

from Orbit cimport Orbit
from LUT2d cimport LUT2d

from Ellipsoid cimport Ellipsoid

from Raster  cimport Raster
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "isce3/geometry/RTC.h" namespace "isce::geometry":

    cdef enum rtcInputRadiometry:
        BETA_NAUGHT = 0
        SIGMA_NAUGHT_ELLIPSOID = 1

    cdef enum rtcAreaMode:
        AREA = 0
        AREA_FACTOR = 1
    
    cdef enum rtcAlgorithm:
        RTC_DAVID_SMALL = 0
        RTC_AREA_PROJECTION = 1

    cdef enum rtcMemoryMode:
        RTC_AUTO = 0
        RTC_SINGLE_BLOCK = 1
        RTC_BLOCKS_GEOGRID = 2

    void applyRTC(RadarGridParameters& radar_grid, 
                  Orbit& orbit,
                  LUT2d[double]& dop,
                  Raster& input_raster,
                  Raster& dem_raster, 
                  Raster& out_raster,
                  rtcInputRadiometry inputRadiometry,
                  int exponent,
                  rtcAreaMode rtc_area_mode,
                  rtcAlgorithm rtc_algorithm,
                  double dem_upsampling,
                  float rtc_min_value_db,
                  double abs_cal_factor,
                  float radar_grid_nlooks,
                  Raster * out_nlooks_raster,
                  Raster * input_rtc_raster,
                  Raster * output_rtc_raster,
                  rtcMemoryMode memory_mode_enum)  except +

    void facetRTC(RadarGridParameters& radar_grid, 
                  Orbit& orbit,
                  LUT2d[double]& dop,
                  Raster& dem_raster, 
                  Raster& out_raster,
                  rtcInputRadiometry inputRadiometry,
                  rtcAreaMode rtc_area_mode,
                  rtcAlgorithm rtc_algorithm,
                  double dem_upsampling,
                  float rtc_min_value_db,
                  float radar_grid_nlooks,
                  Raster * out_nlooks_raster,
                  rtcMemoryMode memory_mode_enum)  except +

    void facetRTC(Raster& dem_raster, 
                  Raster& out_raster,
                  RadarGridParameters& radar_grid, 
                  Orbit& orbit,
                  LUT2d[double]& dop,
                  double y0,
                  double dy,
                  double x0,
                  double dx,
                  int geogrid_length,
                  int geogrid_width,
                  int epsg,
                  rtcInputRadiometry inputRadiometry,
                  rtcAreaMode rtc_area_mode,
                  rtcAlgorithm rtc_algorithm,
                  double dem_upsampling,
                  float rtc_min_value_db,
                  float radar_grid_nlooks,
                  Raster * out_geo_vertices_raster,
                  Raster * out_geo_grid_raster,
                  Raster * out_nlooks_raster,
                  rtcMemoryMode memory_mode_enum)  except +

    void facetRTC(Product& product, 
                  Raster& dem_raster, 
                  Raster& out_raster,
                  char frequency,
                  bool native_doppler,
                  rtcInputRadiometry inputRadiometry,
                  rtcAreaMode rtc_area_mode,
                  rtcAlgorithm rtc_algorithm,
                  double dem_upsampling,
                  float rtc_min_value_db,
                  size_t nlooks_az,
                  size_t nlooks_rg,
                  Raster * out_nlooks_raster,
                  rtcMemoryMode memory_mode_enum)  except +

