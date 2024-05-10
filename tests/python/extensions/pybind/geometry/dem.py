#!/usr/bin/env python3
import pytest
import isce3.ext.isce3.geometry as m
from isce3.ext.isce3.core import DataInterpMethod
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
import iscetest

import os
import collections as cl
import numpy.testing as npt
import numpy as np

from osgeo import gdal


def dem_info_from_gdal(file_raster: str) -> cl.namedtuple:
    """Get shape, min, max, mean of dem"""
    dset = gdal.Open(file_raster, gdal.GA_ReadOnly)
    band = dset.GetRasterBand(1)
    dem = band.ReadAsArray()
    return cl.namedtuple('dem_info', 'shape min max mean')(
        dem.shape, dem.min(), dem.max(), dem.mean())


def test_constructor_ref_height():
    href = 10.

    dem = DEMInterpolator()
    dem.ref_height = href
    assert dem.ref_height == href

    dem = DEMInterpolator(href)
    assert dem.ref_height == href

    assert dem.interpolate_xy(0, 0) == href
    assert dem.interpolate_lonlat(0, 0) == href

    npt.assert_equal(dem.have_raster, False)


def test_constructor_raster_obj():
    # filename of the DEM ratster
    filename_dem = 'dem_himalayas_E81p5_N28p3_short.tiff'
    file_raster = os.path.join(iscetest.data, filename_dem)
    # get some DEM info via gdal to be used as a reference for V&V
    dem_info = dem_info_from_gdal(file_raster)
    # build DEM object
    raster_obj = Raster(file_raster)
    dem_obj = DEMInterpolator(raster_obj)
    # validate existence and details of DEM data
    npt.assert_equal(dem_obj.have_raster, True, err_msg='No DEM ratser data')
    npt.assert_equal(dem_obj.data.shape, dem_info.shape,
                     err_msg='Wrong shape of DEM ratser data')
    npt.assert_allclose(dem_obj.data.min(), dem_info.min,
                        err_msg='Wrong min DEM height')
    npt.assert_allclose(dem_obj.data.max(), dem_info.max,
                        err_msg='Wrong max DEM height')
    npt.assert_allclose(dem_obj.data.mean(), dem_info.mean,
                        err_msg='Wrong mean DEM height')

def test_compute_min_max_mean_height():
    # filename of the DEM ratster
    filename_dem = 'dem_himalayas_E81p5_N28p3_short.tiff'
    file_raster = os.path.join(iscetest.data, filename_dem)

    # Default ctor has stats because it's zero everywhere
    dem = DEMInterpolator()
    npt.assert_(dem.have_stats)
    npt.assert_(dem.min_height == dem.max_height == dem.mean_height == 0.0)

    # Updating reference height of constant DEM should update stats accordingly
    href = 1.
    dem.ref_height = href
    npt.assert_(dem.min_height == dem.max_height == dem.mean_height == href)

    # Loading data from file invalidates the stats.
    dem.load_dem(Raster(file_raster))
    npt.assert_(not dem.have_stats)

    # compute min/max/mean heights
    dem.compute_min_max_mean_height()
    npt.assert_(dem.have_stats)

    # Computing stats should update reference height so that it's in bounds.
    # Note that this test file has (min, max) = (90.7, 355.3) m.
    npt.assert_(dem.min_height <= dem.ref_height <= dem.max_height)

    # validate computed values
    npt.assert_allclose(np.nanmin(dem.data), dem.min_height,
                        err_msg='Wrong computed min DEM height')
    npt.assert_allclose(np.nanmax(dem.data), dem.max_height,
                        err_msg='Wrong computed max DEM height')
    npt.assert_allclose(np.nanmean(dem.data), dem.mean_height,
                        err_msg='Wrong computed mean DEM height')

def test_methods():
    # pybind11::enum_ is not iterable
    for name in "SINC BILINEAR BICUBIC NEAREST BIQUINTIC".split():
        # enum constructor
        method = getattr(DataInterpMethod, name)
        dem = DEMInterpolator(method=method)
        assert dem.interp_method == method
        # string constructor
        dem = DEMInterpolator(method=name)
        assert dem.interp_method == method

    dem = DEMInterpolator(method="bicubic")
    assert dem.interp_method == DataInterpMethod.BICUBIC

    dem = DEMInterpolator(method="biCUBic")
    assert dem.interp_method == DataInterpMethod.BICUBIC

    with pytest.raises(ValueError):
        dem = DEMInterpolator(method="TigerKing")
