#!/usr/bin/env python3
'''
unit tests for CPU pybind Crossmul
'''

import os

import numpy as np
import numpy.testing as npt

from osgeo import gdal

import iscetest
import isce3.ext.isce3 as isce3
from nisar.products.readers import SLC


def common_crossmul_obj():
    '''
    instantiate and return common crossmul object for both run tests
    '''
    # make SLC object and extract parameters
    slc_obj = SLC(hdf5file=os.path.join(iscetest.data, 'envisat.h5'))
    dopp = isce3.core.avg_lut2d_to_lut1d(slc_obj.getDopplerCentroid())
    prf = slc_obj.getRadarGrid().prf

    crossmul = isce3.signal.Crossmul()
    crossmul.set_dopplers(dopp, dopp)

    return crossmul


def test_run_no_filter():
    '''
    run pybind CPU crossmul module without azimuth filtering
    '''
    ref_slc_raster = isce3.io.Raster(os.path.join(iscetest.data, 'warped_envisat.slc.vrt'))

    crossmul = common_crossmul_obj()

    # prepare output rasters
    width = ref_slc_raster.width
    length = ref_slc_raster.length
    igram = isce3.io.Raster(
        'igram.int', width, length, 1, gdal.GDT_CFloat32, "ISCE")
    coherence = isce3.io.Raster(
        'coherence.bin', width, length, 1, gdal.GDT_Float32, "ISCE")

    crossmul.crossmul(ref_slc_raster, ref_slc_raster, igram, coherence)


def test_validate_no_filter():
    '''
    make sure pybind CPU crossmul results have zero phase
    '''
    # convert complex test data to angle
    data = np.angle(np.fromfile('igram.int', dtype=np.complex64))

    # check if interferometric phase is very small (should be zero)
    npt.assert_array_less(data, 1.0e-6)


if __name__ == '__main__':
    test_run_no_filter()
    test_validate_no_filter()
