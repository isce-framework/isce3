import numbers
from cython.operator cimport dereference as deref
from RTC cimport *
from libc.math cimport NAN

rtc_input_terrain_radiometry_dict = {'BETA_NAUGHT': rtcInputRadiometry.BETA_NAUGHT,
                             'SIGMA_NAUGHT_ELLIPSOID': rtcInputRadiometry.SIGMA_NAUGHT_ELLIPSOID}
        
rtc_area_mode_dict = {'AREA': rtcAreaMode.AREA,
                      'AREA_FACTOR': rtcAreaMode.AREA_FACTOR}

rtc_algorithm_dict = {'RTC_BILINEAR_DISTRIBUTION': rtcAlgorithm.RTC_BILINEAR_DISTRIBUTION,
                      'RTC_AREA_PROJECTION': rtcAlgorithm.RTC_AREA_PROJECTION}

rtc_memory_mode_dict = {'AUTO': rtcMemoryMode.RTC_AUTO,
                        'SINGLE_BLOCK': rtcMemoryMode.RTC_SINGLE_BLOCK,
                        'BLOCKS_GEOGRID': rtcMemoryMode.RTC_BLOCKS_GEOGRID}


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

@enum_dict_decorator(rtc_input_terrain_radiometry_dict, 'SIGMA_NAUGHT_ELLIPSOID')
def getRtcInputTerrainRadiometry(*args, **kwargs):
    pass

@enum_dict_decorator(rtc_area_mode_dict, 'RTC_AREA_FACTOR')
def getRtcAreaMode(*args, **kwargs):
    pass

@enum_dict_decorator(rtc_algorithm_dict, 'RTC_AREA_PROJECTION')
def getRtcAlgorithm(*args, **kwargs):
    pass

@enum_dict_decorator(rtc_memory_mode_dict, 'AUTO')
def getMemoryMode(*args, **kwargs):
    pass

def pyApplyRTC(pyRadarGridParameters radarGrid,
               pyOrbit orbit,
               pyLUT2d doppler,
               pyRaster input_raster,
               pyRaster dem_raster, 
               out_rtc, 
               input_terrain_radiometry=None,
               int exponent = 0,
               rtc_area_mode = None,
               rtc_algorithm = None,
               dem_upsampling = NAN,
               rtc_min_value_db = NAN,
               double abs_cal_factor = 1,
               float radar_grid_nlooks = 1,
               out_nlooks = None,
               input_rtc = None,
               output_rtc = None,
               memory_mode = 'AUTO'):

    # input radiometry
    rtc_input_terrain_radiometry = getRtcInputTerrainRadiometry(input_terrain_radiometry)

    # RTC area mode
    rtc_area_mode_obj = getRtcAreaMode(rtc_area_mode)

    # RTC algorithm
    rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

    # other attributes
    out_nlooks_raster = _getRaster(out_nlooks)
    input_rtc_raster = _getRaster(input_rtc)
    output_rtc_raster = _getRaster(output_rtc)

    out_raster = _getRaster(out_rtc)
    if out_raster == NULL:
        print('ERROR invalid output raster')
        return

    memory_mode_enum = getMemoryMode(memory_mode)
  
    applyRTC(deref(radarGrid.c_radargrid),
             orbit.c_orbit,
             deref(doppler.c_lut),
             deref(input_raster.c_raster),
             deref(dem_raster.c_raster),
             deref(out_raster),
             rtc_input_terrain_radiometry,
             exponent,
             rtc_area_mode_obj,
             rtc_algorithm_obj,
             dem_upsampling,
             rtc_min_value_db,
             abs_cal_factor,
             radar_grid_nlooks,
             out_nlooks_raster,
             input_rtc_raster,
             output_rtc_raster,
             memory_mode_enum)

def pyRTC(pyRadarGridParameters radarGrid,
          pyOrbit orbit,
          pyLUT2d doppler,
          pyRaster dem_raster, 
          out_rtc,
          input_terrain_radiometry = None,
          rtc_area_mode = None,
          rtc_algorithm = None,
          dem_upsampling = NAN,
          rtc_min_value_db = NAN,
          float radar_grid_nlooks = 1,
          out_nlooks = None,
          memory_mode = 'AUTO'):

    # input radiometry
    rtc_input_terrain_radiometry = getRtcInputTerrainRadiometry(input_terrain_radiometry)

    # RTC area mode
    rtc_area_mode_obj = getRtcAreaMode(rtc_area_mode)

    # RTC algorithm
    rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

    # other attributes
    width = dem_raster.width
    length = dem_raster.length 
    out_nlooks_raster = _getRaster(out_nlooks)

    out_raster = _getRaster(out_rtc)
    if out_raster == NULL:
        print('ERROR invalid output raster')
        return
    
    memory_mode_enum = getMemoryMode(memory_mode)
  
    computeRtc(deref(radarGrid.c_radargrid),
             orbit.c_orbit,
             deref(doppler.c_lut),
             deref(dem_raster.c_raster),
             deref(out_raster),
             rtc_input_terrain_radiometry,
             rtc_area_mode_obj,
             rtc_algorithm_obj,
             dem_upsampling,
             rtc_min_value_db,
             radar_grid_nlooks,
             out_nlooks_raster,
             memory_mode_enum)

def pyRTCBBox(pyRadarGridParameters radarGrid,
              pyOrbit orbit,
              pyLUT2d doppler,
              pyRaster dem_raster, 
              out_rtc,
              double y0,
              double dy,
              double x0,
              double dx,
              int geogrid_length,
              int geogrid_width,
              int epsg,
              input_terrain_radiometry = None,
              rtc_area_mode = None,
              rtc_algorithm = None,
              dem_upsampling = NAN,
              rtc_min_value_db = NAN,
              float radar_grid_nlooks = 1,
              out_geo_vertices = None,
              out_geo_grid = None,
              out_nlooks = None,
              memory_mode = 'AUTO'):

    # input radiometry
    rtc_input_terrain_radiometry = getRtcInputTerrainRadiometry(input_terrain_radiometry)

    # RTC area mode
    rtc_area_mode_obj = getRtcAreaMode(rtc_area_mode)

    # RTC algorithm
    rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

    # other attributes
    width = dem_raster.width
    length = dem_raster.length 
    out_geo_vertices_raster = _getRaster(out_geo_vertices)
    out_geo_grid_raster = _getRaster(out_geo_grid)
    out_nlooks_raster = _getRaster(out_nlooks)
    out_raster = _getRaster(out_rtc)
    if out_raster == NULL:
        print('ERROR invalid output raster')
        return

    memory_mode_enum = getMemoryMode(memory_mode)

    computeRtc(deref(dem_raster.c_raster),
             deref(out_raster),
             deref(radarGrid.c_radargrid),
             orbit.c_orbit,
             deref(doppler.c_lut),
             y0,
             dy,
             x0,
             dx,
             geogrid_length,
             geogrid_width,
             epsg,
             rtc_input_terrain_radiometry,
             rtc_area_mode_obj,
             rtc_algorithm_obj,
             dem_upsampling,
             rtc_min_value_db,
             radar_grid_nlooks,
             out_geo_vertices_raster,
             out_geo_grid_raster,
             out_nlooks_raster,
             memory_mode_enum)


def pyRTCProd(pyProduct prod, 
              pyRaster dem_raster, 
              out_rtc,
              char frequency = b'A',
              bool native_doppler = False,
              input_terrain_radiometry = None,
              rtc_area_mode = None,
              rtc_algorithm = None,
              dem_upsampling = NAN,
              rtc_min_value_db = NAN,
              size_t nlooks_az = 1,
              size_t nlooks_rg = 1,
              out_nlooks = None,
              memory_mode = 'AUTO'):

    # input radiometry
    rtc_input_terrain_radiometry = getRtcInputTerrainRadiometry(input_terrain_radiometry)

    # RTC area mode
    rtc_area_mode_obj = getRtcAreaMode(rtc_area_mode)

    # RTC algorithm
    rtc_algorithm_obj = getRtcAlgorithm(rtc_algorithm)

    # other attributes
    width = dem_raster.width
    length = dem_raster.length 
    out_nlooks_raster = _getRaster(out_nlooks)
    out_raster = _getRaster(out_rtc)

    if out_raster == NULL:
        print('ERROR invalid output raster')
        return

    memory_mode_enum = getMemoryMode(memory_mode) 

    computeRtc(deref(prod.c_product),
             deref(dem_raster.c_raster),
             deref(out_raster),
             frequency,
             native_doppler,
             rtc_input_terrain_radiometry,
             rtc_area_mode_obj,
             rtc_algorithm_obj,
             dem_upsampling,
             rtc_min_value_db,
             nlooks_az,
             nlooks_rg,
             out_nlooks_raster,
             memory_mode_enum)
