#!/usr/bin/env python3

import numpy.testing as npt
from pathlib import Path

import isce3.ext.isce3 as isce
import iscetest

def test_default_driver():
    driver = isce.io.gdal.Dataset.default_driver()
    print(driver)

# init from path
# (access defaults to read-only)
def test_open1():
    path = Path(iscetest.data) / "io" / "gdal" / "ENVIRaster-dem"
    dataset = isce.io.gdal.Dataset(str(path))

    assert( dataset.access == isce.io.gdal.GDALAccess.GA_ReadOnly )

# init from path + access mode enum
def test_open2():
    path = Path(iscetest.data) / "io" / "gdal" / "ENVIRaster-dem"
    access = isce.io.gdal.GDALAccess.GA_ReadOnly
    dataset = isce.io.gdal.Dataset(str(path), access)

    assert( dataset.access == access )

# init from path + access mode char
# (valid access modes are 'r', 'w')
def test_open3():
    path = Path(iscetest.data) / "io" / "gdal" / "ENVIRaster-dem"
    dataset = isce.io.gdal.Dataset(str(path), 'r')

    assert( dataset.access == isce.io.gdal.GDALAccess.GA_ReadOnly )

    # not mappable to GDALAccess
    with npt.assert_raises(RuntimeError):
        dataset = isce.io.gdal.Dataset(str(path), 'z')

# create new dataset using default driver
def test_create1():
    path = "Dataset-create1"
    width = 4
    length = 8
    bands = 3
    datatype = isce.io.gdal.GDALDataType.GDT_Float32
    dataset = isce.io.gdal.Dataset(path, width, length, bands, datatype)

    assert( dataset.access == isce.io.gdal.GDALAccess.GA_Update )
    assert( dataset.width == width )
    assert( dataset.length == length )
    assert( dataset.bands == bands )
    assert( dataset.driver == isce.io.gdal.Dataset.default_driver() )

def test_create2():
    path = "Dataset-create2"
    width = 4
    length = 8
    bands = 3
    datatype = isce.io.gdal.GDALDataType.GDT_Float32
    driver = "GTiff"
    dataset = isce.io.gdal.Dataset(path, width, length, bands, datatype, driver)

    assert( dataset.driver == driver )

# datatype can be anything convertible to numpy.dtype
# (as long as it's mappable to GDALDataType)
def test_create3():
    import numpy as np

    path = "Dataset-create3"
    width = 4
    length = 8
    bands = 3

    dataset = isce.io.gdal.Dataset(path, width, length, bands, float)
    dataset = isce.io.gdal.Dataset(path, width, length, bands, "i4")
    dataset = isce.io.gdal.Dataset(path, width, length, bands, np.complex64)

    # not mappable to GDALDataType
    with npt.assert_raises(RuntimeError):
        dataset = isce.io.gdal.Dataset(path, width, length, bands, "int64")

def test_get_raster():
    path = Path(iscetest.data) / "io" / "gdal" / "ENVIRaster-dem"
    dataset = isce.io.gdal.Dataset(str(path), 'r')
    raster = dataset.get_raster(1)

    assert( raster.access == dataset.access )
    assert( raster.width == dataset.width )
    assert( raster.length == dataset.length )
    assert( raster.band == 1 )
    assert( raster.x0 == dataset.x0 )
    assert( raster.y0 == dataset.y0 )
    assert( raster.dx == dataset.dx )
    assert( raster.dy == dataset.dy )
