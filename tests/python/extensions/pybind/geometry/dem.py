#!/usr/bin/env python3
import pytest
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
from pybind_isce3.core import DataInterpMethod
import iscetest

import os
import collections as cl
import numpy.testing as npt

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
