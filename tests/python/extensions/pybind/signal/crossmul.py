#!/usr/bin/env python3
'''
unit tests for CPU pybind Crossmul
'''

import os

import numpy as np
import numpy.testing as npt

from osgeo import gdal

import iscetest
import pybind_isce3 as isce
from pybind_nisar.products.readers import SLC


def common_crossmul_obj():
    '''
    instantiate and return common crossmul object for both run tests
    '''
    # make SLC object and extract parameters
    slc_obj = SLC(hdf5file=os.path.join(iscetest.data, 'envisat.h5'))
    dopp = isce.core.avg_lut2d_to_lut1d(slc_obj.getDopplerCentroid())
    prf = slc_obj.getRadarGrid().prf

    crossmul = isce.signal.Crossmul()
    crossmul.set_dopplers(dopp, dopp)
    crossmul.set_az_filter(prf, 2000.0, 0.25)

    return crossmul


def test_run_no_filter():
    '''
    run pybind CPU crossmul module without azimuth filtering
    '''
    ref_slc_raster = isce.io.Raster(os.path.join(iscetest.data, 'warped_envisat.slc.vrt'))

    crossmul = common_crossmul_obj()

    crossmul.filter_az = False

    # prepare output rasters
    width = ref_slc_raster.width
    length = ref_slc_raster.length
    igram = isce.io.Raster('igram.int', width, length, 1, gdal.GDT_CFloat32, "ISCE")
    coherence = isce.io.Raster('coherence.bin', width, length, 1, gdal.GDT_Float32, "ISCE")

    crossmul.crossmul(ref_slc_raster, ref_slc_raster, igram, coherence)


def test_validate_no_filter():
    '''
    make sure pybind CPU crossmul results have zero phase
    '''
    # convert complex test data to angle
    data = np.angle(np.fromfile('igram.int', dtype=np.complex64))

    # check if interferometric phase is very small (should be zero)
    npt.assert_array_less(data, 1.0e-6)


def test_run_az_filter():
    '''
    run pybind CPU crossmul module with azimuth filtering
    '''
    ref_slc_raster = isce.io.Raster(os.path.join(iscetest.data, 'warped_envisat.slc.vrt'))

    crossmul = common_crossmul_obj()

    # prepare output rasters
    width = ref_slc_raster.width
    length = ref_slc_raster.length
    igram = isce.io.Raster('igram_az.int', width, length, 1, gdal.GDT_CFloat32, "ISCE")
    coherence = isce.io.Raster('coherence_az.bin', width, length, 1, gdal.GDT_Float32, "ISCE")

    crossmul.crossmul(ref_slc_raster, ref_slc_raster, igram, coherence)


def test_validate_az_filter():
    '''
    make sure pybind CPU crossmul results have zero phase
    '''
    # convert complex test data to angle
    data = np.angle(np.fromfile('igram_az.int', dtype=np.complex64))

    # check if interferometric phase is very small (should be zero)
    npt.assert_array_less(data, 1.0e-6)


if __name__ == '__main__':
    test_init()
    test_run_no_filter()
    test_validate_no_filter()
    test_run_az_filter()
    test_validate_az_filter()
