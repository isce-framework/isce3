#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt

import os
from osgeo import gdal, gdal_array

import isce3.ext.isce3 as isce


class commonClass:
    def __init__(self):
        self.nc = 100
        self.nl = 200
        self.nbx = 5
        self.nby = 7
        self.lat_file = 'lat.tif'
        self.lon_file = 'lon.vrt'
        self.inc_file = 'inc.bin'
        self.msk_file = 'msk.bin'
        self.vrt_file = 'topo.vrt'


def test_create_geotiff_float():
    # load shared params and clean up if necessary
    cmn = commonClass()
    if os.path.exists(cmn.lat_file):
        os.remove(cmn.lat_file)

    # create raster object
    raster = isce.io.Raster(path=cmn.lat_file,
            width=cmn.nc, length=cmn.nl, num_bands=1,
            dtype=gdal.GDT_Float32, driver_name='GTiff')

    # check generated raster
    assert( os.path.exists(cmn.lat_file) )
    assert( raster.width == cmn.nc )
    assert( raster.length == cmn.nl )
    assert( raster.num_bands == 1 )
    assert( raster.datatype() == gdal.GDT_Float32 )
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nc)[None,:]

    ds = gdal.Open(cmn.lat_file, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None


def test_create_vrt_double():
    # load shared params and clean up if necessary
    cmn = commonClass()
    if os.path.exists(cmn.lon_file):
        os.remove(cmn.lon_file)

    # create raster object
    raster = isce.io.Raster(path=cmn.lon_file,
            width=cmn.nc, length=cmn.nl, num_bands=1,
            dtype=gdal.GDT_Float64, driver_name='VRT')

    # check generated raster
    assert( os.path.exists(cmn.lon_file) )
    assert( raster.datatype() == gdal.GDT_Float64 )
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nl)[:,None]

    # open and populate
    ds = gdal.Open(cmn.lon_file, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    arr = ds.GetRasterBand(1).ReadAsArray()
    npt.assert_array_equal(data, arr, err_msg='RW in Update mode')
    ds = None

    # read array
    ds = gdal.Open(cmn.lon_file, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    npt.assert_array_equal(data, arr, err_msg='Readonly mode')


def test_create_2band_envi():
    # load shared params and clean up if necessary
    cmn = commonClass()
    if os.path.exists(cmn.inc_file):
        os.remove(cmn.inc_file)

    # create raster object
    raster = isce.io.Raster(path=cmn.inc_file,
            width=cmn.nc, length=cmn.nl, num_bands=2,
            dtype=gdal.GDT_Int16, driver_name='ENVI')

    # check generated raster
    assert( os.path.exists(cmn.inc_file) )
    assert( raster.width == cmn.nc )
    assert( raster.length == cmn.nl )
    assert( raster.num_bands == 2 )
    assert( raster.datatype() == gdal.GDT_Int16 )
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nl)[:,None]

    # open and populate
    ds = gdal.Open(cmn.lon_file, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    arr = ds.GetRasterBand(1).ReadAsArray()
    npt.assert_array_equal(data, arr, err_msg='RW in Update mode')
    ds = None

    # read array
    ds = gdal.Open(cmn.lon_file, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    npt.assert_array_equal(data, arr, err_msg='Readonly mode')


def test_create_multiband_vrt():
    # load shared params and clean up if necessary
    cmn = commonClass()

    lat = isce.io.Raster(cmn.lat_file)
    lon = isce.io.Raster(cmn.lon_file)
    inc = isce.io.Raster(cmn.inc_file)

    if os.path.exists( cmn.vrt_file):
        os.remove(cmn.vrt_file)

    vrt = isce.io.Raster(cmn.vrt_file, raster_list=[lat,lon,inc])

    assert( vrt.width == cmn.nc )
    assert( vrt.length == cmn.nl )
    assert( vrt.num_bands == 4 )
    assert( vrt.datatype(1) == gdal.GDT_Float32 )
    assert( vrt.datatype(2) == gdal.GDT_Float64 )
    assert( vrt.datatype(3) == gdal.GDT_Int16 )
    assert( vrt.datatype(4) == gdal.GDT_Int16 )

    vrt = None



def test_createNumpyDataset():
    ny, nx = 200, 100
    data = np.random.randn(ny, nx).astype(np.float32)

    dset = gdal_array.OpenArray(data)
    raster = isce.io.Raster(np.uintp(dset.this))

    assert( raster.width == nx )
    assert( raster.length == ny )
    assert( raster.datatype() == gdal.GDT_Float32 )

    dset = None
    del raster


if __name__ == "__main__":
    test_create_geotiff_float()
    test_create_vrt_double()
    test_create_2band_envi()
    test_create_multiband_vrt()
    test_createNumpyDataset()

# end of file
