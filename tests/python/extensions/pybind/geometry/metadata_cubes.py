#!/usr/bin/env python3
import os

import h5py
from osgeo import gdal

import numpy as np
import iscetest
import pybind_isce3 as isce3
from nisar.products.readers import SLC
from nisar.workflows.h5_prep import add_geolocation_grid_cubes_to_hdf5

# run tests
def test_run():
    # load parameters shared across all test runs
    # init geocode object and populate members
    rslc = SLC(hdf5file=os.path.join(iscetest.data, "envisat.h5"))
    orbit = rslc.getOrbit()
    native_doppler = rslc.getDopplerCentroid()
    native_doppler.bounds_error = False
    grid_doppler = native_doppler
    threshold_geo2rdr = 1e-8
    numiter_geo2rdr = 25
    delta_range = 1e-6
    epsg = 4326

    # get radar grid from HDF5
    radar_grid = isce3.product.RadarGridParameters(
        os.path.join(iscetest.data, "envisat.h5"))
    radar_grid = radar_grid[::10, ::10]
    
    heights = [0.0, 1000.0]

    output_h5 = 'envisat_geolocation_cube.h5'
    fid = h5py.File(output_h5, 'w')

    cube_group_name = '/science/LSAR/SLC/metadata/radarGrid'

    add_geolocation_grid_cubes_to_hdf5(fid, cube_group_name, radar_grid, 
                                       heights, orbit, native_doppler, 
                                       grid_doppler, epsg, threshold_geo2rdr,
                                       numiter_geo2rdr, delta_range)

    print('saved file:', output_h5)
