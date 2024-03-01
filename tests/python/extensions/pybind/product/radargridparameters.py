#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
import pytest

import isce3.ext.isce3 as isce
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

    # Python-only utility functions.
    assert str(grid).startswith("RadarGrid")
    assert grid.shape == (grid.length, grid.width)
    assert grid.copy() is not grid
    assert grid[:,:] is not grid
    with pytest.raises(IndexError):
        grid[::-1,:]
    npt.assert_almost_equal(grid[:,1:].starting_range,
        grid.starting_range + grid.range_pixel_spacing)
    npt.assert_almost_equal(grid[1:,:].sensing_start,
        grid.sensing_start + 1.0 / grid.prf)
    assert grid[1:,:].prf == grid.prf
    assert grid[1:,:].length == grid.length - 1
    assert grid[:,1:].range_pixel_spacing == grid.range_pixel_spacing
    assert grid[:,1:].width == grid.width - 1
    assert grid.slant_ranges[0] == grid.starting_range
    assert grid.slant_ranges.size == grid.width
    npt.assert_almost_equal(grid.range_pixel_spacing,
        set(np.diff(grid.slant_ranges)).pop())
    assert grid.sensing_times[0] == grid.sensing_start
    assert grid.sensing_times.size == grid.length
    npt.assert_almost_equal(1.0 / grid.prf,
        set(np.diff(grid.sensing_times)).pop())
    # Check case of length != width.
    rect = grid[:10, :11]
    assert rect[:,10:11].shape == (10, 1)

def test_radargridparameters_contains():
    # Create RadarGridParameters object
    grid = isce.product.RadarGridParameters(iscetest.data + "envisat.h5")

    # Check if center of grid is in bounds
    assert grid.contains(grid.sensing_mid, grid.mid_range)

    # Check all permutations of out of bounds
    az_too_short = grid.sensing_start - grid.az_time_interval
    az_too_long = grid.sensing_stop + grid.az_time_interval
    rg_too_short = grid.starting_range - grid.range_pixel_spacing
    rg_too_long = grid.end_range + grid.range_pixel_spacing
    assert grid.contains(az_too_short, rg_too_short) == False
    assert grid.contains(az_too_short, grid.mid_range) == False
    assert grid.contains(az_too_short, rg_too_long) == False
    assert grid.contains(az_too_long, rg_too_short) == False
    assert grid.contains(az_too_long, grid.mid_range) == False
    assert grid.contains(az_too_long, rg_too_long) == False
    assert grid.contains(grid.sensing_mid, rg_too_short) == False
    assert grid.contains(grid.sensing_mid, rg_too_long) == False

def test_radargridparameters_resize():

    # Create RadarGridParameters object
    grid = isce.product.RadarGridParameters(iscetest.data + "envisat.h5")

    # Test the resize and keep start and stop
    grid_resize = grid.resize_and_keep_startstop(20, 20)
    assert grid_resize.length == 20
    assert grid_resize.width == 20
    assert grid_resize.sensing_start == grid.sensing_start
    npt.assert_almost_equal(grid_resize.sensing_stop, grid.sensing_stop)
    npt.assert_almost_equal(grid_resize.prf,
                            grid.prf * (grid_resize.length - 1.0) / (grid.length - 1))

    assert grid_resize.starting_range == grid.starting_range
    npt.assert_almost_equal(grid_resize.range_pixel_spacing,
                            grid.range_pixel_spacing * \
                            (grid.width - 1.0) / (grid_resize.width - 1))
    npt.assert_almost_equal(grid_resize.end_range, grid.end_range)

def test_radargridparameters_add_margin():
    # Create RadarGridParameters object
    grid = isce.product.RadarGridParameters(iscetest.data + "envisat.h5")

    # Test adding the margin to azimuth and slant range
    grid_margin = grid.add_margin(2,2)
    assert grid_margin.length == grid.length + 4
    assert grid_margin.width == grid.width + 4
    npt.assert_almost_equal(grid_margin.sensing_start,
                            grid.sensing_start - 2.0 / grid.prf)
    assert grid_margin.prf == grid.prf
    npt.assert_almost_equal(grid_margin.starting_range,
                            grid.starting_range - grid.range_pixel_spacing * 2.0)
    assert grid_margin.range_pixel_spacing == grid.range_pixel_spacing

    # Test adding the margin to the azimuth only
    grid_margin = grid.add_margin(2,0)
    assert grid_margin.length == grid.length + 4
    assert grid_margin.width == grid.width
    npt.assert_almost_equal(grid_margin.sensing_start,
                            grid.sensing_start - 2.0 / grid.prf)
    assert grid_margin.prf == grid.prf
    assert grid_margin.starting_range, grid.starting_range
    assert grid_margin.range_pixel_spacing == grid.range_pixel_spacing

    # Test adding the margin to the slant range only
    grid_margin = grid.add_margin(0,2)
    assert grid_margin.length == grid.length
    assert grid_margin.width == grid.width + 4
    npt.assert_almost_equal(grid_margin.starting_range,
                            grid.starting_range - 2.0 * grid.range_pixel_spacing)
    assert grid_margin.prf == grid.prf
    assert grid_margin.sensing_start, grid.sensing_start
    assert grid_margin.range_pixel_spacing == grid.range_pixel_spacing