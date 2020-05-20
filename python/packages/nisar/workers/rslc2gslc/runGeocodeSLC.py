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

#this is a temporary import. Needs to be remove when all functionalities exist with isce3
import isce3.extensions.isceextension as temp_isce3


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
            self._print("working on frequency: {} and polarization: {}".format(freq, polarization))
            # get doppler centroid
            native_doppler = slc.getDopplerCentroid(frequency=freq)

            # Doppler of the image grid (Zero for NISAR)
            image_grid_doppler = isce3.core.lut2d()

            output_dir = os.path.dirname(os.path.abspath(self.state.output_hdf5))
            os.makedirs(output_dir, exist_ok=True)
            
            slc_dataset = self.slc_obj.getSlcDataset(freq, polarization)
            slc_raster = isce3.io.raster(filename='', h5=slc_dataset)

            # Needs to be constructed from HDF5 
            #dst_h5 = h5py.File(state.output_hdf5, 'a')
            #dataset_path = 'science/LSAR/GSLC/grids/{frequency}/{polarization}'.format(frequency="frequencyA", polarization="HH")
            #gslc_dataset = dst_h5[dataset_path]
            #gslc_raster = isce3.io.raster(filename='', h5=gslc_dataset, access=gdal.GA_Update)

            driver = gdal.GetDriverByName('ENVI')
            output = os.path.join(output_dir, "gslc.bin")
            gslc_dataset = driver.Create(output, 
                                geo_grid.width, 
                                geo_grid.length, 
                                1, 
                                gdal.GDT_CFloat32)

            gslc_raster = isce3.io.raster(filename='', 
                                dataset=gslc_dataset)
    
            # Needs to be determined somewhere else
            thresholdGeo2rdr = 1.0e-9 ;
            numiterGeo2rdr = 25;
            linesPerBlock = 1000;
            demBlockMargin = 0.1;

            flatten = True;

            # run geocodeSlc : This should become isce3.geocode.geocodeSlc(...)
            temp_isce3.pygeocodeSlc(gslc_raster, slc_raster, dem_raster,
                    radar_grid, geo_grid,
                    orbit,
                    native_doppler, image_grid_doppler,
                    ellipsoid,
                    thresholdGeo2rdr, numiterGeo2rdr,
                    linesPerBlock, demBlockMargin,
                    flatten)

            #save the output gslc to the HDF5
            # This is not needed when we figure out the pyraster construction from H5 itself
            # or if we use a buffer and not a rster like SLC workflow
            dst_h5 = h5py.File(state.output_hdf5, 'a')
            dataset_path = 'science/LSAR/GSLC/grids/{frequency}/{polarization}'.format(
                            frequency=frequency, polarization=polarization)
            dst_h5[dataset_path][:] = gslc_dataset.GetRasterBand(1).ReadAsArray()
            dst_h5.close()

