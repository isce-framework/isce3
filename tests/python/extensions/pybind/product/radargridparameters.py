#!/usr/bin/env python3

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
    # Check case of length != width.
    rect = grid[:10, :11]
    assert rect[:,10:11].shape == (10, 1)
