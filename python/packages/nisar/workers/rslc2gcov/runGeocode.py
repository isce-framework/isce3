#!/usr/bin/env python3

import os
import gdal
import osr
import time
import h5py
import numpy as np
from collections import defaultdict
import isce3.extensions.isceextension as isce3

def runGeocode(self, frequency):
    '''
    This step maps locations on a DEM to slant range and azimuth time.
    '''

    # only execute worker if frequency is listed in subset_dict
    if frequency not in self.state.subset_dict.keys():
        self._print(f'skipping frequency {frequency} because it'
                    '  is not in input parameters:'
                    f'  {[str(f) for f in self.state.subset_dict.keys()]}')

        # 1. indicates that the worker was not executed
        return 1
    _runGeocodeFrequency(self, frequency)


def _runGeocodeFrequency(self, frequency):

    self._print(f'starting geocode module for frequency: {frequency}')

    state = self.state
    pol_list = state.subset_dict[frequency]
    radar_grid = self.radar_grid_list[frequency]
    orbit = self.orbit

    raster_ref_list = []
    for pol in pol_list:
        h5_ds = f'//science/LSAR/SLC/swaths/frequency{frequency}/{pol}'
        raster_ref = f'HDF5:"{state.input_hdf5}":{h5_ds}'
        raster_ref_list.append(raster_ref)

    self._print('raster list:', raster_ref_list)
    self._print('pol list: ', pol_list)

    # set temporary files
    time_id = str(time.time())
    input_temp = os.path.join(state.scratch_path,
        f'temp_rslc2gcov_{frequency}_{time_id}.vrt')
    output_file = os.path.join(state.scratch_path,
        f'temp_rslc2gcov_{frequency}_{time_id}.bin')
    output_off_diag_file = os.path.join(state.scratch_path,
        f'temp_rslc2gcov_{frequency}_{time_id}_off_diag.bin')

    out_geo_nlooks = os.path.join(state.scratch_path,
        f'temp_geo_nlooks_{time_id}.bin')
    out_geo_rtc = os.path.join(state.scratch_path,
        f'temp_geo_rtc_{time_id}.bin')
    out_dem_vertices = os.path.join(state.scratch_path,
        f'temp_dem_vertices_{time_id}.bin')
    out_geo_vertices = os.path.join(state.scratch_path,
        f'temp_geo_vertices_{time_id}.bin')

    # build input VRT
    gdal.BuildVRT(input_temp, raster_ref_list, separate=True)
    input_raster_obj = isce3.pyRaster(input_temp) 
    ellps = isce3.pyEllipsoid()

    # Reading processing parameters
    flag_fullcovariance = self.get_value(['processing',
        'input_subset', 'fullcovariance'])

    # RTC
    rtc_dict = self.get_value(['processing', 'rtc'])
    rtc_output_type = rtc_dict['output_type']
    rtc_geogrid_upsampling = rtc_dict['geogrid_upsampling'] 
    rtc_algorithm_type = rtc_dict['algorithm_type']
    input_terrain_radiometry = rtc_dict[
        'input_terrain_radiometry']
    rtc_min_value_db = rtc_dict['rtc_min_value_db']

    # Geocode
    geocode_dict = self.get_value(['processing', 'geocode'])

    geocode_algorithm_type = geocode_dict['algorithm_type']
    memory_mode = geocode_dict['memory_mode']
    geogrid_upsampling = geocode_dict['geogrid_upsampling']
    abs_cal_factor = geocode_dict['abs_rad_cal']

    clip_min = geocode_dict['clip_min']
    clip_max = geocode_dict['clip_max']
    min_nlooks = geocode_dict['min_nlooks']

    flag_upsample_radar_grid = geocode_dict['upsample_radargrid']
    flag_save_nlooks = geocode_dict['save_nlooks']
    flag_save_rtc = geocode_dict['save_rtc']
    flag_save_dem_vertices = geocode_dict['save_dem_vertices']
    flag_save_geo_vertices = geocode_dict['save_geo_vertices']
    
    # Geogrid
    state.output_epsg = geocode_dict['outputEPSG']
    y_snap = geocode_dict['y_snap']
    x_snap = geocode_dict['x_snap']

    y_max = geocode_dict['top_left']['y_abs']
    x_min = geocode_dict['top_left']['x_abs']

    y_min = geocode_dict['bottom_right']['y_abs']
    x_max = geocode_dict['bottom_right']['x_abs']
    step = geocode_dict['output_posting']

    # fix types
    rtc_min_value_db = self.cast_input(rtc_min_value_db, dtype=float,
                                       frequency=frequency)
    state.output_epsg = self.cast_input(state.output_epsg, dtype=int,
                                        frequency=frequency)
    geogrid_upsampling = self.cast_input(geogrid_upsampling, dtype=float,
                                         frequency=frequency)
    rtc_geogrid_upsampling = self.cast_input(rtc_geogrid_upsampling,
                                             dtype=float, frequency=frequency)
    abs_cal_factor = self.cast_input(abs_cal_factor, dtype=float, 
                                     frequency=frequency)

    clip_min = self.cast_input(clip_min, dtype=float, default=0,
                                     frequency=frequency)
    clip_max = self.cast_input(clip_max, dtype=float,  default=2,
                                     frequency=frequency)
    min_nlooks = self.cast_input(min_nlooks, dtype=float,
                                     frequency=frequency)
    y_snap = self.cast_input(y_snap, dtype=float, 
                             default=np.nan, frequency=frequency)
    x_snap = self.cast_input(x_snap, dtype=float, 
                             default=np.nan, frequency=frequency)

    y_max = self.cast_input(y_max, dtype=float, default=np.nan, 
                            frequency=frequency)
    x_min = self.cast_input(x_min, dtype=float, default=np.nan, 
                            frequency=frequency)
    y_min = self.cast_input(y_min, dtype=float,
                                       default=np.nan, frequency=frequency)
    x_max = self.cast_input(x_max, dtype=float, 
                                       default=np.nan, frequency=frequency)

    step_x = self.cast_input(step, dtype=float, default=np.nan, 
                             frequency=frequency)
    step_y = -step_x if _is_valid(step_x) else None

    # prepare parameters
    zero_doppler = isce3.pyLUT2d()

    # Instantiate Geocode object according to the raster type
    if input_raster_obj.getDatatype() == gdal.GDT_Float32:
        geo = isce3.pyGeocodeFloat(orbit, ellps)
    elif input_raster_obj.getDatatype() == gdal.GDT_Float64:
        geo = isce3.pyGeocodeDouble(orbit, ellps)
    elif input_raster_obj.getDatatype() == gdal.GDT_CFloat32:
        geo = isce3.pyGeocodeComplexFloat(orbit, ellps)
    elif input_raster_obj.getDatatype() == gdal.GDT_CFloat64:
        geo = isce3.pyGeocodeComplexDouble(orbit, ellps)
    else:
        raise NotImplementedError('Unsupported raster type for geocoding')

    dem_raster = isce3.pyRaster(state.dem_file)

    if state.output_epsg is None:
        state.output_epsg = dem_raster.EPSG

    if state.geotransform_dict is None:
        state.geotransform_dict = {}

    if (_is_valid(y_min) and _is_valid(y_max) and 
            _is_valid(step_y)):
        size_y = int(np.round((y_min - y_max)/step_y))
    else:
        size_y = -32768
    if (_is_valid(x_max) and _is_valid(x_min) and
            _is_valid(step_x)):
        size_x = int(np.round((x_max - x_min)/step_x))
    else:
        size_x = -32768

    # if Geogrid is not fully determined, let Geocode find the missing values
    if (size_x == -32768 or size_y == -32768):
        geo.geoGrid(x_min, y_max, step_x, step_y,
                    size_x, size_y, state.output_epsg)
        geo.updateGeoGrid(radar_grid, dem_raster)

        # update only missing values
        if not _is_valid(x_min):
            x_min = geo.geoGridStartX
        if not _is_valid(y_max):
            y_max = geo.geoGridStartY
        if not _is_valid(step_x):
            step_x = geo.geoGridSpacingX
        if not _is_valid(step_y):
            step_y = geo.geoGridSpacingY

        if not _is_valid(x_max):
            x_max = geo.geoGridStartX + geo.geoGridSpacingX * geo.geoGridWidth 

        if not _is_valid(y_min):
            y_min = geo.geoGridStartY + geo.geoGridSpacingY * geo.geoGridLength 

    x_min = _snap_coordinate(x_min, x_snap, np.floor)
    y_max = _snap_coordinate(y_max, y_snap, np.ceil)
    x_max = _snap_coordinate(x_max, x_snap, np.ceil)
    y_min = _snap_coordinate(y_min, y_snap, np.floor)

    size_y = int(np.round((y_min - y_max)/step_y))
    size_x = int(np.round((x_max - x_min)/step_x))

    geo.geoGrid(x_min, y_max, step_x, step_y,
                size_x, size_y, state.output_epsg)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dtype = gdal.GDT_Float32
    output_dtype_off_diag_terms = gdal.GDT_CFloat32
    exponent = 2
    nbands = input_raster_obj.numBands

    if geogrid_upsampling is None:
        geogrid_upsampling = 1

    self._print(f'creating temporary output raster: {output_file}')

    geocoded_dict = defaultdict(lambda: None)
    output_raster_obj = isce3.pyRaster(output_file,
                                       gdal.GA_Update,
                                       output_dtype,
                                       size_x,
                                       size_y,
                                       nbands,
                                       "ENVI")
    geocoded_dict['output_file'] = output_file

    nbands_off_diag_terms = 0
    out_off_diag_terms_obj = None
    if flag_fullcovariance:
        nbands_off_diag_terms = (nbands**2 - nbands) // 2
        if nbands_off_diag_terms > 0:
            out_off_diag_terms_obj =  isce3.pyRaster(
                output_off_diag_file,
                gdal.GA_Update,
                output_dtype_off_diag_terms,
                size_x,
                size_y,
                nbands_off_diag_terms,
                "ENVI")
            geocoded_dict['output_off_diag_file'] = \
                output_off_diag_file

    if flag_save_nlooks:
        out_geo_nlooks_obj = isce3.pyRaster(out_geo_nlooks,
                                            gdal.GA_Update,
                                            gdal.GDT_Float32,
                                            size_x,
                                            size_y,
                                            1,
                                            "ENVI")
        geocoded_dict['out_geo_nlooks'] = out_geo_nlooks
    else:
        out_geo_nlooks_obj = None

    if flag_save_rtc:
        out_geo_rtc_obj = isce3.pyRaster(out_geo_rtc,
                                         gdal.GA_Update,
                                         gdal.GDT_Float32,
                                         size_x,
                                         size_y,
                                         1,
                                         "ENVI")
        geocoded_dict['out_geo_rtc'] = out_geo_rtc
    else:
        out_geo_rtc_obj = None

    if flag_save_dem_vertices:
        out_dem_vertices_obj = isce3.pyRaster(out_dem_vertices,
                                              gdal.GA_Update,
                                              gdal.GDT_Float32,
                                              size_x + 1,
                                              size_y + 1,
                                              1,
                                              "ENVI")
        geocoded_dict['out_dem_vertices'] = out_dem_vertices
    else:
        out_dem_vertices_obj = None

    if flag_save_geo_vertices:
        out_geo_vertices_obj = isce3.pyRaster(out_geo_vertices,
                                            gdal.GA_Update,
                                            gdal.GDT_Float32,
                                            size_x + 1,
                                            size_y + 1,
                                            2,
                                            "ENVI")
        geocoded_dict['out_geo_vertices'] = out_geo_vertices
    else:
        out_geo_vertices_obj = None
    

    # Run geocoding
    flag_apply_rtc =  (rtc_output_type and
                       rtc_output_type != input_terrain_radiometry and
                       'gamma' in rtc_output_type)
    if flag_apply_rtc is None:
        flag_apply_rtc = False
    geotransform = [x_min, step_x, 0, y_max, 0, step_y]
    state.geotransform_dict[frequency] = geotransform

    # output mode
    if ('interp' in geocode_algorithm_type and flag_apply_rtc):
        raise NotImplementedError('ERROR interp algorithm does not provide'
                                  ' RTC correction')
    elif 'interp' in geocode_algorithm_type:
        output_mode = 'interp'
    elif not flag_apply_rtc:
        output_mode = 'area-projection'
    else:
        output_mode = 'area-projection-gamma_naught'

    # input terrain radiometry
    if (input_terrain_radiometry is not None and
           'sigma' in input_terrain_radiometry):
        input_terrain_radiometry = 'sigma-naught-ellipsoid'
    else:
        input_terrain_radiometry = 'beta-naught'

    if flag_apply_rtc:
        output_radiometry_str = 'gamma-naught'
    else:
        output_radiometry_str = input_terrain_radiometry 

    # number of looks
    radar_grid_nlooks = state.nlooks_az * state.nlooks_rg

    # rtc min value
    kwargs = {}
    if rtc_min_value_db is not None:
        kwargs['rtc_min_value_db'] = rtc_min_value_db

    # absolute calibration factor
    if abs_cal_factor is not None:
        kwargs['abs_cal_factor'] = abs_cal_factor

    # memory mode
    if memory_mode is not None:
        kwargs['memory_mode'] = memory_mode

    if (rtc_algorithm_type is not None and
        'BILINEAR' in rtc_algorithm_type.upper()):
        kwargs['rtc_algorithm'] = 'RTC_BILINEAR_DISTRIBUTION'
    elif rtc_algorithm_type is not None:
        kwargs['rtc_algorithm'] = 'RTC_AREA_PROJECTION'

    if (rtc_geogrid_upsampling is not None and 
            np.isfinite(rtc_geogrid_upsampling)):
        kwargs['rtc_upsampling'] = rtc_geogrid_upsampling 

    if clip_min is not None:
        kwargs['clip_min'] = clip_min 

    if clip_max is not None:
        kwargs['clip_max'] = clip_max

    if min_nlooks is not None:
        kwargs['min_nlooks'] = min_nlooks

    # call the geocode module
    geo.geocode(radar_grid,
                input_raster_obj,
                output_raster_obj,
                dem_raster,
                flag_upsample_radar_grid=flag_upsample_radar_grid,
                output_mode=output_mode,
                upsampling=geogrid_upsampling,
                input_terrain_radiometry=input_terrain_radiometry,
                exponent=exponent,
                radar_grid_nlooks=radar_grid_nlooks,
                out_off_diag_terms=out_off_diag_terms_obj,
                out_geo_nlooks=out_geo_nlooks_obj,
                out_geo_rtc=out_geo_rtc_obj,
                out_dem_vertices=out_dem_vertices_obj,
                out_geo_vertices=out_geo_vertices_obj,
                **kwargs)

    del output_raster_obj

    if flag_save_nlooks:
        del out_geo_nlooks_obj
    
    if flag_save_rtc:
        del out_geo_rtc_obj

    if flag_fullcovariance:
        del out_off_diag_terms_obj

    if flag_save_dem_vertices:
        del out_dem_vertices_obj

    if flag_save_geo_vertices:
        del out_geo_vertices_obj

    self._print(f'removing temporary file: {input_temp}')
    _remove(input_temp)
    output_hdf5 = state.output_hdf5

    h5_ds_list = []

    with h5py.File(output_hdf5, 'a') as hdf5_obj:
        hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
        root_ds = os.path.join('//', 'science', 'LSAR', 'GCOV', 'grids',
                               f'frequency{frequency}')

        # radiometricTerrainCorrectionFlag
        h5_ds = os.path.join(root_ds, 'listOfPolarizations')
        if h5_ds in hdf5_obj:
            del hdf5_obj[h5_ds] 
        pol_list_s2 = np.array(pol_list, dtype='S2')
        dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
        h5_ds_list.append(h5_ds)
        dset.attrs['description'] = np.string_(
            'List of processed polarization layers with frequency ' + 
            frequency)

        h5_ds = os.path.join(root_ds, 'radiometricTerrainCorrectionFlag')
        if h5_ds in hdf5_obj:
            del hdf5_obj[h5_ds]
        dset = hdf5_obj.create_dataset(h5_ds, data=np.string_(str(flag_apply_rtc)))
        h5_ds_list.append(h5_ds)

        # X and Y coordinates
        geotransform = self.state.geotransform_dict[frequency]
        dx = geotransform[1]
        dy = geotransform[5]
        x0 = geotransform[0] + 0.5 * dx
        y0 = geotransform[3] + 0.5 * dy
        xf = x0 + (size_x - 1) * dx
        yf = y0 + (size_y - 1) * dy

        # xCoordinates
        h5_ds = os.path.join(root_ds, 'xCoordinates') # float64
        x_vect = np.linspace(x0, xf, size_x, dtype=np.float64)
        if h5_ds in hdf5_obj:
            del hdf5_obj[h5_ds]
        xds = hdf5_obj.create_dataset(h5_ds, data=x_vect)
        h5_ds_list.append(h5_ds)
        try:	
            xds.make_scale()	
        except AttributeError:	
            pass

        # yCoordinates
        h5_ds = os.path.join(root_ds, 'yCoordinates') # float64
        y_vect = np.linspace(y0, yf, size_y, dtype=np.float64)
        if h5_ds in hdf5_obj:
            del hdf5_obj[h5_ds]
        yds = hdf5_obj.create_dataset(h5_ds, data=y_vect)
        h5_ds_list.append(h5_ds)
        try:	
            yds.make_scale()	
        except AttributeError:	
            pass

        #Associate grid mapping with data - projection created later
        h5_ds = os.path.join(root_ds, "projection")

        #Set up osr for wkt
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(self.state.output_epsg)

        ###Create a new single int dataset for projections
        if h5_ds in hdf5_obj:
            del hdf5_obj[h5_ds]
        projds = hdf5_obj.create_dataset(h5_ds, (), dtype='i')
        projds[()] = self.state.output_epsg

        h5_ds_list.append(h5_ds)

        ##WGS84 ellipsoid
        projds.attrs['semi_major_axis'] = 6378137.0
        projds.attrs['inverse_flattening'] = 298.257223563
        projds.attrs['ellipsoid'] = np.string_("WGS84")

        ##Additional fields
        projds.attrs['epsg_code'] = self.state.output_epsg

        ##CF 1.7+ requires this attribute to be named "crs_wkt"
        ##spatial_ref is old GDAL way. Using that for testing only. 
        ##For NISAR replace with "crs_wkt"
        projds.attrs['spatial_ref'] = np.string_(srs.ExportToWkt())

        ##Here we have handcoded the attributes for the different cases
        ##Recommended method is to use pyproj.CRS.to_cf() as shown above
        ##To get complete set of attributes.

        ###Geodetic latitude / longitude
        if self.state.output_epsg == 4326:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_('latitude_longitude')
            projds.attrs['longitude_of_prime_meridian'] = 0.0

            #Setup units for x and y 
            xds.attrs['standard_name'] = np.string_("longitude")
            xds.attrs['units'] = np.string_("degree_east")

            yds.attrs['standard_name'] = np.string_("latitude")
            yds.attrs['units'] = np.string_("degree_north")

        ### UTM zones
        elif ((self.state.output_epsg > 32600 and 
               self.state.output_epsg < 32661) or
              (self.state.output_epsg > 32700 and 
               self.state.output_epsg < 32761)):
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_('universal_transverse_mercator')
            projds.attrs['utm_zone_number'] = self.state.output_epsg % 100

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")
    
        ### Polar Stereo North
        elif self.state.output_epsg == 3413:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("polar_stereographic")
            projds.attrs['latitude_of_projection_origin'] = 90.0
            projds.attrs['standard_parallel'] = 70.0
            projds.attrs['straight_vertical_longitude_from_pole'] = -45.0
            projds.attrs['false_easting'] = 0.0
            projds.attrs['false_northing'] = 0.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

        ### Polar Stereo south
        elif self.state.output_epsg == 3031:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("polar_stereographic")
            projds.attrs['latitude_of_projection_origin'] = -90.0
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0
            projds.attrs['false_easting'] = 0.0
            projds.attrs['false_northing'] = 0.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

        ### EASE 2 for soil moisture L3
        elif self.state.output_epsg == 6933:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("lambert_cylindrical_equal_area")
            projds.attrs['longitude_of_central_meridian'] = 0.0
            projds.attrs['standard_parallel'] = 30.0
            projds.attrs['false_easting'] = 0.0
            projds.attrs['false_northing'] = 0.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

        ### Europe Equal Area for Deformation map (to be implemented in isce3)
        elif self.state.output_epsg == 3035:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("lambert_azimuthal_equal_area")
            projds.attrs['longitude_of_projection_origin']= 10.0
            projds.attrs['latitude_of_projection_origin'] = 52.0
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0
            projds.attrs['false_easting'] = 4321000.0
            projds.attrs['false_northing'] = 3210000.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

        else:
            raise NotImplementedError('Waiting for implementation / Not supported in ISCE3')

        # save GCOV diagonal elements
        diag_terms_list = [p.upper()+p.upper() for p in pol_list]
        _save_hdf5_dataset(self, 'output_file', hdf5_obj, root_ds,
                           h5_ds_list, geocoded_dict, frequency, yds, xds,
                           diag_terms_list,
                           standard_name = output_radiometry_str,
                           long_name = output_radiometry_str, 
                           units = 'unitless',
                           fill_value = np.nan, 
                           valid_min = clip_min, 
                           valid_max = clip_max)


        # save GCOV off-diagonal elements
        if flag_fullcovariance:
            off_diag_terms_list = []
            for b1, p1 in enumerate(pol_list):
                for b2, p2 in enumerate(pol_list):
                    if (b2 <= b1):
                        continue
                    off_diag_terms_list.append(p1.upper()+p2.upper())

            _save_hdf5_dataset(self, 'output_off_diag_file', hdf5_obj, root_ds,
                               h5_ds_list, geocoded_dict, frequency, yds, xds,
                               off_diag_terms_list,
                               standard_name = output_radiometry_str,
                               long_name = output_radiometry_str, 
                               units = 'unitless',
                               fill_value = np.nan, 
                               valid_min = clip_min, 
                               valid_max = clip_max)

        # save nlooks
        _save_hdf5_dataset(self, 'out_geo_nlooks', hdf5_obj, root_ds, 
                           h5_ds_list, geocoded_dict, frequency, yds, xds, 
                           'numberOfLooks',
                           standard_name = 'numberOfLooks',
                           long_name = 'number of looks', 
                           units = 'looks',
                           fill_value = np.nan, 
                           valid_min = 0)

        # save rtc
        if flag_apply_rtc:
            _save_hdf5_dataset(self, 'out_geo_rtc', hdf5_obj, root_ds, 
                            h5_ds_list, geocoded_dict, frequency, yds, xds, 
                            'areaNormalizationFactor',
                            standard_name = 'areaNormalizationFactor',
                            long_name = 'RTC area factor', 
                            units = 'unitless',
                            fill_value = np.nan, 
                            valid_min = 0,
                            valid_max = 2)

        if ('out_dem_vertices' in geocoded_dict or 
            'out_geo_vertices' in  geocoded_dict):

            # X and Y coordinates
            geotransform = self.state.geotransform_dict[frequency]
            dx = geotransform[1]
            dy = geotransform[5]
            x0 = geotransform[0] 
            y0 = geotransform[3]
            xf = xf + size_x * dx
            yf = yf + size_y * dy

            # xCoordinates
            h5_ds = os.path.join(root_ds, 'xCoordinatesVertices') # float64
            x_vect_vertices = np.linspace(x0, xf, size_x + 1, dtype=np.float64)
            if h5_ds in hdf5_obj:
                del hdf5_obj[h5_ds]
            xds_vertices = hdf5_obj.create_dataset(h5_ds, data=x_vect_vertices)
            h5_ds_list.append(h5_ds)
            try:	
                xds_vertices.make_scale()	
            except AttributeError:	
                pass

            # yCoordinates
            h5_ds = os.path.join(root_ds, 'yCoordinatesVertices') # float64
            y_vect_vertices = np.linspace(y0, yf, size_y + 1, dtype=np.float64)
            if h5_ds in hdf5_obj:
                del hdf5_obj[h5_ds]
            yds_vertices = hdf5_obj.create_dataset(h5_ds, data=y_vect_vertices)
            h5_ds_list.append(h5_ds)
            try:	
                yds_vertices.make_scale()	
            except AttributeError:	
                pass

            # save geo grid
            _save_hdf5_dataset(self, 'out_dem_vertices', hdf5_obj, root_ds, 
                            h5_ds_list, geocoded_dict, frequency, 
                            yds_vertices, xds_vertices, 
                            'interpolatedDem',
                            standard_name = 'interpolatedDem',
                            long_name = 'interpolated dem', 
                            units = 'meters',
                            fill_value = np.nan, 
                            valid_min = -500,
                            valid_max = 9000)

            # save geo vertices
            _save_hdf5_dataset(self, 'out_geo_vertices', hdf5_obj, root_ds, 
                            h5_ds_list, geocoded_dict, frequency, 
                            yds_vertices, xds_vertices,
                            ['vertices_a', 'vertices_r'])

    for h5_ds_str in h5_ds_list:
        h5_ref = f'HDF5:{output_hdf5}:{h5_ds_str}'
        state.outputList[frequency].append(h5_ref)

def _snap_coordinate(val, snap, round_function, dtype=float):
    if np.isnan(snap):
        return dtype(val)
    new_val = round_function(float(val) / snap) * snap
    return dtype(new_val)

def _is_valid(data):
    return data is not None and np.isfinite(data)

def _remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    header_file = filename.replace('.bin', '.hdr')
    try:
        os.remove(header_file)
    except FileNotFoundError:
        pass

def _save_hdf5_dataset(self, name, hdf5_obj, root_ds, h5_ds_list, geocoded_dict,
                       frequency, yds, xds, ds_name=None, standard_name=None,
                       long_name=None, units=None, fill_value=None,
                       valid_min=None, valid_max=None):
                       
    ds_filename = geocoded_dict[name]
    if not ds_filename:
        return

    gdal_ds = gdal.Open(ds_filename)
    if gdal_ds is None:
        print(f'ERROR opening {ds_filename}')
        return
    nbands = gdal_ds.RasterCount
    for band in range(nbands):
        gdal_band = gdal_ds.GetRasterBand(band+1)
        data = gdal_band.ReadAsArray()
        if ds_name is None:
            h5_ds = os.path.join(root_ds, name)
        elif isinstance(ds_name, str):
            h5_ds = os.path.join(root_ds, ds_name)
        else:
            h5_ds = os.path.join(root_ds, ds_name[band])
        if h5_ds in hdf5_obj:
            del hdf5_obj[h5_ds]
        dset = hdf5_obj.create_dataset(h5_ds, data=data)
        h5_ds_list.append(h5_ds)
        dset.dims[0].attach_scale(yds)
        dset.dims[1].attach_scale(xds)
        dset.attrs['grid_mapping'] = np.string_("projection") 

        if standard_name is not None:
            dset.attrs['standard_name'] = np.string_(standard_name)

        if long_name is not None:
            dset.attrs['long_name'] = np.string_(long_name)

        if units is not None:
            dset.attrs['units'] = np.string_(units)

        if fill_value is not None:
            dset.attrs.create('_FillValue', data = fill_value)

        if valid_min is not None:
            dset.attrs.create('valid_min', data = valid_min)

        if valid_max is not None:
            dset.attrs.create('valid_max', data = valid_max)

    del gdal_ds
    self._print(f'removing temporary file: {ds_filename}')
    _remove(ds_filename)


# end of file
