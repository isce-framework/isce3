#!/usr/bin/env python3

import numpy.testing as npt
import pytest

import isce3.ext.isce3 as isce

def test_geogridparameters():
    # Create GeoGridParameters object
    grid = isce.product.GeoGridParameters(
        start_x=100,
        start_y=500,
        spacing_x=10,
        spacing_y=-5,
        width=5000,
        length=5000,
        epsg=4326,
    )

    # Check its values
    npt.assert_equal(grid.epsg, 4326)
    npt.assert_almost_equal(grid.start_x, 100)
    npt.assert_almost_equal(grid.start_y, 500)
    npt.assert_almost_equal(grid.spacing_x, 10)
    npt.assert_almost_equal(grid.spacing_y, -5)
    npt.assert_equal(grid.width, 5000)
    npt.assert_equal(grid.length, 5000)

    sliced_grid = grid[:100, :100]

    # Python-only utility functions.
    assert grid[:,:] is not grid

    with pytest.raises(IndexError):
        grid[::-1,:]

    # EPSG
    npt.assert_equal(grid[1:,:].epsg, grid.epsg)

    # Sliced grid length/widths
    npt.assert_equal(grid[1:,:].length, grid.length - 1)
    npt.assert_equal(grid[:,1:].width, grid.width - 1)
    npt.assert_equal(sliced_grid.length, 100)
    npt.assert_equal(sliced_grid.width, 100)

    # Sliced grid spacings
    npt.assert_almost_equal(grid[1:,:].spacing_y, grid.spacing_y)
    npt.assert_almost_equal(grid[:,1:].spacing_x, grid.spacing_x)
    npt.assert_almost_equal(grid[::2,:].spacing_y, grid.spacing_y * 2)
    npt.assert_almost_equal(grid[:,::2].spacing_x, grid.spacing_x * 2)

    # Sliced grid starts
    npt.assert_almost_equal(grid[1:,:].start_y, grid.start_y + grid.spacing_y)
    npt.assert_almost_equal(grid[:,1:].start_x, grid.start_x + grid.spacing_x)
