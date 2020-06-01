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
from nisar.products.readers import SLC

def runGeocodeSLC(self):
    self._print('starting geocode module')

    state = self.state

    time_id = str(time.time())
    
    orbit = self.orbit

    dem_raster = isce3.io.raster(filename=state.dem_file)
    
    # construct ellipsoid which is by default WGS84
    ellipsoid = isce3.core.ellipsoid()

    
    slc = SLC(hdf5file=self.state.input_hdf5)
    for freq in state.subset_dict.keys():
        frequency = "frequency{}".format(freq)
        pol_list = state.subset_dict[freq]
        radar_grid = self.radar_grid_list[freq]
        geo_grid = self.geogrid_dict[frequency]
        for polarization in pol_list: 
            self._print(f'working on frequency: {freq}, polarization: {polarization}')
            # get doppler centroid
            native_doppler = slc.getDopplerCentroid(frequency=freq)

            # Doppler of the image grid (Zero for NISAR)
            image_grid_doppler = isce3.core.lut2d()

            output_dir = os.path.dirname(os.path.abspath(self.state.output_hdf5))
            os.makedirs(output_dir, exist_ok=True)
            
            slc_dataset = self.slc_obj.getSlcDataset(freq, polarization)
            slc_raster = isce3.io.raster(filename='', h5=slc_dataset)

            # access the HDF5 dataset for a given frequency and polarization
            dst_h5 = h5py.File(state.output_hdf5, 'a')
            #dataset_path = 'science/LSAR/GSLC/grids/{frequency}/{polarization}'.format(
            #        frequency=frequency, polarization=polarization)
            dataset_path = f'science/LSAR/GSLC/grids/{frequency}/{polarization}'
            gslc_dataset = dst_h5[dataset_path]

            # Construct the output ratster directly from HDF5 dataset
            gslc_raster = isce3.io.raster(filename='', h5=gslc_dataset, 
                                access=gdal.GA_Update)
            
            # Needs to be determined somewhere else
            thresholdGeo2rdr = 1.0e-9 ;
            numiterGeo2rdr = 25;
            linesPerBlock = 1000;
            demBlockMargin = 0.1;

            flatten = True;

            # run geocodeSlc : 
            isce3.geocode.geocodeSlc(gslc_raster, slc_raster, dem_raster,
                    radar_grid, geo_grid,
                    orbit,
                    native_doppler, image_grid_doppler,
                    ellipsoid,
                    thresholdGeo2rdr, numiterGeo2rdr,
                    linesPerBlock, demBlockMargin,
                    flatten)

            # the rasters need to be deleted
            del gslc_raster
            del slc_raster

            dst_h5.close()

