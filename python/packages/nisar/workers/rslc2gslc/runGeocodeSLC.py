# -*- coding: utf-8 -*-

#!/usr/bin/env python3

import os
import gdal
import osr
import time
import h5py
import numpy as np
from collections import defaultdict
import isce3

#this is a temporary import. Needs to be removed
import isce3.extensions.isceextension as temp_isce3


def runGeocodeSLC(self, frequency):
    '''
    This step maps the SLC to a geographic map grid 
    '''
    
    # only execute worker if frequency is listed in subset_dict
    if frequency not in self.state.subset_dict.keys():
        self._print(f'skipping frequency {frequency} because it'
                      '  is not in input parameters:'
                      f'  {[str(f) for f in self.state.subset_dict.keys()]}')

          # 1. indicates that the worker was not executed
        return 1
    _runGeocodeSLC(self, frequency)

def _runGeocodeSLC(self, frequency):

    self._print(f'starting geocode module for frequency: {frequency}')

    state = self.state
    pol_list = state.subset_dict[frequency]
    radar_grid = self.radar_grid_list[frequency]
    orbit = self.orbit


    raster_ref_list = []
    for pol in pol_list:
        h5_ds = f'//science/LSAR/SLC/swaths/frequency{frequency}/{pol}'
        raster_ref = f'HDF5:{state.input_hdf5}:{h5_ds}'
        raster_ref_list.append(raster_ref)

    self._print('raster list:', raster_ref_list)
    self._print('pol list: ', pol_list)

    time_id = str(time.time())
    
    '''
    geocode_dict = self.get_value(['parameters', 'geocode'])
    clip_min = geocode_dict['clip_min']
    clip_max = geocode_dict['clip_max']

    #Geogrid parameters
    state.output_epsg = geocode_dict['outputEPSG']
    y_max = geocode_dict['top_left']['y_abs']
    x_min = geocode_dict['top_left']['x_abs']
    top_left_x_snap = geocode_dict['top_left']['x_snap']
    top_left_y_snap = geocode_dict['top_left']['y_snap']

    y_min = geocode_dict['bottom_right']['y_abs']
    x_max = geocode_dict['bottom_right']['x_abs']
    bottom_right_x_snap = geocode_dict['bottom_right']['x_snap']
    bottom_right_y_snap = geocode_dict['bottom_right']['y_snap']
    step = geocode_dict['output_posting']

    #fix types
    state.output_epsg = self.cast_input(state.output_epsg, dtype=int,
                                          frequency=frequency)
    clip_min = self.cast_input(clip_min, dtype=float, default=0,
                                       frequency=frequency)
    clip_max = self.cast_input(clip_max, dtype=float,  default=2,
                                       frequency=frequency)
    y_max = self.cast_input(y_max, dtype=float, default=np.nan,
                              frequency=frequency)
    x_min = self.cast_input(x_min, dtype=float, default=np.nan,
                              frequency=frequency)
    top_left_x_snap = self.cast_input(top_left_x_snap, dtype=float,
                                        default=np.nan, frequency=frequency)
    top_left_y_snap = self.cast_input(top_left_y_snap, dtype=float,
                                        default=np.nan, frequency=frequency)
    y_min = self.cast_input(y_min, dtype=float,
                                         default=np.nan, frequency=frequency)
    x_max = self.cast_input(x_max, dtype=float,
                                         default=np.nan, frequency=frequency)
    bottom_right_x_snap = self.cast_input(bottom_right_x_snap, dtype=float,
                                            default=np.nan, frequency=frequency)
    bottom_right_snap = self.cast_input(bottom_right_y_snap, dtype=float,
                                          default=np.nan, frequency=frequency)

    step_x = self.cast_input(step, dtype=float, default=np.nan,
                               frequency=frequency)

    step_y = -step_x if _is_valid(step_x) else None
    '''
    dem_raster = isce3.io.raster(filename=state.dem_file)
   
    '''
    if state.output_epsg is None:
        state.output_epsg = dem_raster.EPSG

    if state.geotransform_dict is None:
        state.geotransform_dict = {}

    if (_is_valid(y_min) and _is_valid(y_max) and _is_valid(step_y)):
        size_y = int(np.round((y_min - y_max)/step_y))
    else:
        size_y = -32768

    if (_is_valid(x_max) and _is_valid(x_min) and _is_valid(step_x)):
        size_x = int(np.round((x_max - x_min)/step_x))
    else:
        size_x = -32768

    x_min = _snap_coordinate(x_min, top_left_x_snap, step_x, np.floor)
    y_max = _snap_coordinate(y_max, top_left_y_snap, step_y, np.ceil)
    x_max = _snap_coordinate(x_max, bottom_right_x_snap, step_x, np.ceil)
    y_min = _snap_coordinate(y_min, bottom_right_y_snap, step_y, np.floor)

    size_y = int(np.round((y_min - y_max)/step_y))
    size_x = int(np.round((x_max - x_min)/step_x))
    '''
    x_start = self.userconfig['processing']['geocode']['top_left']['x_abs']
    y_start = self.userconfig['processing']['geocode']['top_left']['y_abs']
    x_end = self.userconfig['processing']['geocode']['bottom_right']['x_abs']
    y_end = self.userconfig['processing']['geocode']['bottom_right']['y_abs']
    x_step = self.userconfig['processing']['geocode']['output_posting']['x_posting']
    y_step = -1.0*self.userconfig['processing']['geocode']['output_posting']['y_posting']
    epsg_code = self.userconfig['processing']['geocode']['outputEPSG']
    print(x_start, y_start, x_end, y_end, x_step, y_step, epsg_code)
    y_size = int(np.round((y_end-y_start)/y_step))
    x_size = int(np.round((x_end-x_start)/x_step))
    print(x_size, y_size)
    
    # Construct GeoGridParameters 
    # this should be isce3.product.GeoGridParameters()
    #print(self.userconfig)
    geo_grid = temp_isce3.pyGeoGridParameters()
    geo_grid.startX = x_start
    geo_grid.startY = y_start
    geo_grid.spacingX = x_step
    geo_grid.spacingY = y_step
    geo_grid.width = x_size
    geo_grid.length = y_size
    geo_grid.epsg = epsg_code #state.output_epsg 

    # construct ellipsoid which is by default WGS84
    ellipsoid = isce3.core.ellipsoid()

    # get doppler centroid
    print(self.state.input_hdf5)
    from nisar.products.readers import SLC
    slc = SLC(hdf5file=self.state.input_hdf5)
    native_doppler = slc.getDopplerCentroid()

    #native_doppler = self.doppler() #slc.getDopplerCentroid()

    # Doppler of the image grid (Zero for NISAR)
    #image_grid_doppler = self.doppler() #isce3.core.LUT2d()
    image_grid_doppler = slc.getDopplerCentroid()

    dem_raster = isce3.io.raster(filename=state.dem_file)

    output_dir = os.path.dirname(self.state.output_hdf5)
    #os.makedirs(output_dir, exist_ok=True)

    #
    polarization = pol_list[0]
    slc_dataset = self.slc_obj.getSlcDataset(frequency, polarization)
    slc_raster = isce3.io.raster(filename='', h5=slc_dataset)

    # Needs to be constructed from HDF5 
    driver = gdal.GetDriverByName('ENVI')
    output = os.path.join(output_dir, "gslc.bin")
    gslc_dataset = driver.Create(output, 
                                x_size, 
                                y_size, 
                                1, 
                                gdal.GDT_CFloat32)

    gslc_raster = isce3.io.raster(filename='', 
                                dataset=gslc_dataset)
    
    # Needs to be determined somewhere else
    thresholdGeo2rdr = 1.0e-9 ;
    numiterGeo2rdr = 25;
    linesPerBlock = 1000;
    demBlockMargin = 0.1;

    sincLength = 9;
    flatten = True;

    

    # run geocodeSlc : This should become isce3.geocode.geocodeSlc(...)
    temp_isce3.pygeocodeSlc(gslc_raster, slc_raster, dem_raster,
                    radar_grid, geo_grid,
                    orbit,
                    native_doppler, image_grid_doppler,
                    ellipsoid,
                    thresholdGeo2rdr, numiterGeo2rdr,
                    linesPerBlock, demBlockMargin,
                    sincLength, flatten)
    

