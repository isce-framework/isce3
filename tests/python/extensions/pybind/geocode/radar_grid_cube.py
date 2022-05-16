#!/usr/bin/env python3
import os

import h5py
from osgeo import gdal

import numpy as np
import iscetest
import isce3.ext.isce3 as isce3
from nisar.products.readers import SLC
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5

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
    delta_range = 1e-8

    # prepare geogrid
    geogrid = isce3.product.GeoGridParameters(start_x=-115.65,
        start_y=34.84,
        spacing_x=0.0002,
        spacing_y=-8.0e-5,
        width=500,
        length=500,
        epsg=4326)

    # get radar grid from HDF5
    radar_grid = isce3.product.RadarGridParameters(os.path.join(iscetest.data, "envisat.h5"))

    heights = [0.0, 1000.0]

    output_h5 = 'envisat_radar_grid_cube.h5'
    fid = h5py.File(output_h5, 'w')

    cube_group_name = '/science/LSAR/GCOV/metadata/radarGrid'

    add_radar_grid_cubes_to_hdf5(fid, cube_group_name, geogrid, heights, radar_grid,
                                 orbit, native_doppler, grid_doppler,
                                 threshold_geo2rdr, numiter_geo2rdr, delta_range)

    print('saved file:', output_h5)
