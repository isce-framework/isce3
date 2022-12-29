#!/usr/bin/env python3

import numpy.testing as npt

import isce3.ext.isce3 as isce
import iscetest

def test_radargridparameters():

    # Create Grid object
    gcov_crop_path = iscetest.data + "nisar_129_gcov_crop.h5"
    grid = isce.product.Grid(gcov_crop_path, 'A')

    # Check its values
    npt.assert_equal(grid.wavelength, 0.24118460016090104)
    npt.assert_equal(grid.range_bandwidth, 20000000.0)
    npt.assert_equal(grid.azimuth_bandwidth, 19.100906906009193)
    npt.assert_equal(grid.center_frequency, 1243000000.0)
    npt.assert_equal(grid.slant_range_spacing, 6.245676208)
    npt.assert_equal(grid.zero_doppler_time_spacing, 0.022481251507997513)
    npt.assert_equal(grid.spacing_x, 0.0002)
    npt.assert_equal(grid.spacing_y, -0.0002)
    npt.assert_equal(grid.start_x, -83.45989999999999 - grid.spacing_x / 2)
    npt.assert_equal(grid.start_y, 35.179899999999996 - grid.spacing_y / 2)
    npt.assert_equal(grid.width, 250)
    npt.assert_equal(grid.length, 250)
    npt.assert_equal(grid.epsg, 4326)

# end of file
