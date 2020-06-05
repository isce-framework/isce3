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
            dataset_path = f'science/LSAR/GSLC/grids/{frequency}/{polarization}'
            gslc_dataset = dst_h5[dataset_path]

            # Construct the output ratster directly from HDF5 dataset
            gslc_raster = isce3.io.raster(filename='', h5=gslc_dataset, 
                                access=gdal.GA_Update)
            
            # This whole section requires better sanity check and handling defaults
            threshold_geo2rdr = self.userconfig['runconfig']['groups']['processing']['geo2rdr']['threshold']
            iteration_geo2rdr = self.userconfig['runconfig']['groups']['processing']['geo2rdr']['maxiter']
            lines_per_block = self.userconfig['runconfig']['groups']['processing']['blocksize']['y']
            dem_block_margin = self.userconfig['runconfig']['groups']['processing']['dem_margin']
            flatten = self.userconfig['runconfig']['groups']['processing']['flatten']

            # this may not be the best way. needs to be revised
            if flatten:
                self._print("flattening is True")
            else:
                self._print("flattening is False")

            if np.isnan(threshold_geo2rdr):
                threshold_geo2rdr = 1.0e-9 ;

            if np.isnan(iteration_geo2rdr):
                iteration_geo2rdr = 25;

            if np.isnan(lines_per_block):
                lines_per_block = 1000;

            if np.isnan(dem_block_margin):
                dem_block_margin = 0.1;

            # run geocodeSlc : 
            isce3.geocode.geocodeSlc(gslc_raster, slc_raster, dem_raster,
                    radar_grid, geo_grid,
                    orbit,
                    native_doppler, image_grid_doppler,
                    ellipsoid,
                    threshold_geo2rdr, iteration_geo2rdr,
                    lines_per_block, dem_block_margin,
                    flatten)

            # the rasters need to be deleted
            del gslc_raster
            del slc_raster

            dst_h5.close()

