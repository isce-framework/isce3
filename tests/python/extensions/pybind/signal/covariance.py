#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
from osgeo import gdal

import isce3.ext.isce3 as isce

def make_slc_data(shape):
    # create indices and reshape to raster shape
    size = shape[0]*shape[1]
    i = np.arange(size).reshape(shape)

    # create hh and hv np arrays in a dict to match cxx tests
    arrs = {'hh':i + 2j*i, 'hv':0.1*i + 1j*(i+0.3)}

    return arrs


def create_test_vrts(shape):
    arrs = make_slc_data(shape)

    # create vrt test files holding test data
    for pol in ['hh', 'hv']:
        fname = f'{pol}.vrt'

        # create raster object
        raster = isce.io.Raster(path=fname,
                width=shape[0], length=shape[1], num_bands=1,
                dtype=gdal.GDT_CFloat32, driver_name='VRT')
        del raster

        ds = gdal.Open(fname, gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(arrs[pol])
        ds = None


def test_dualpol_run():
    length = width = 10
    create_test_vrts((length, width))

    slc_list = {'hh':isce.io.Raster('hh.vrt'), 'hv':isce.io.Raster('hv.vrt')}
    cov_list = {}
    datatype = gdal.GDT_CFloat32
    cov_list[('hh','hh')] = isce.io.Raster('cov_hh_hh.vrt', width, length, 1, datatype, 'VRT')
    cov_list[('hh','hv')] = isce.io.Raster('cov_hh_hv.vrt', width, length, 1, datatype, 'VRT')
    cov_list[('hv','hv')] = isce.io.Raster('cov_hv_hv.vrt', width, length, 1, datatype, 'VRT')

    cov64 = isce.signal.CovarianceComplex64()
    cov64.covariance(slc_list, cov_list)


def test_dualpol_check():
    length = width = 10
    slc = make_slc_data((length, width))

    for pols in  [('hh', 'hh'), ('hh', 'hv'), ('hv', 'hv')]:
        pol0 = pols[0]
        pol1 = pols[1]

        # calculate expected
        expected = slc[pol0] * np.conj(slc[pol1])

        # read generated
        ds = gdal.Open(f'cov_{pol0}_{pol1}.vrt', gdal.GA_ReadOnly)
        generated = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        # XXX unsure why results aren't spot on match
        npt.assert_array_almost_equal(expected, generated, decimal=2)


if __name__ == "__main__":
    test_dualpol_run()
    test_dualpol_check()

# end of file
