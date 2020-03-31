#!/usr/bin/env python3

import numpy.testing as npt

import pybind_isce3 as isce
import iscetest

def test_radargridparameters():
    # Create RadarGridParameters object
    grid = isce.product.RadarGridParameters(iscetest.data + "envisat.h5")

    # Check its values
    npt.assert_equal(grid.lookside, isce.core.LookSide.Right)
    npt.assert_almost_equal(grid.starting_range, 826988.6900674499)
    npt.assert_almost_equal(grid.sensing_start, 237330.843491759)
    c = 299792458.0
    npt.assert_almost_equal(grid.wavelength, c/5.331004416e9)
    npt.assert_almost_equal(grid.range_pixel_spacing, 7.803973670948287)
    npt.assert_almost_equal(grid.az_time_interval, 6.051745968279355e-4)
    
# end of file
