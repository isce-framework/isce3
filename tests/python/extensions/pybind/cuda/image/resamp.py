#!/usr/bin/env python3

import os

from osgeo import gdal
import numpy as np

import iscetest
import isce3.ext.isce3 as isce3
from nisar.products.readers import SLC


def test_run():
    '''
    check if resamp runs
    '''
    # init params
    h5_path = os.path.join(iscetest.data, "envisat.h5")
    grid = isce3.product.RadarGridParameters(h5_path)
    slc = SLC(hdf5file=h5_path)

    # init resamp obj
    resamp = isce3.cuda.image.ResampSlc(grid, slc.getDopplerCentroid(), invalid_value=5+4j)
    resamp.lines_per_tile = 249

    # prepare rasters
    h5_ds = f'//science/LSAR/SLC/swaths/frequencyA/HH'
    raster_ref = f'HDF5:{h5_path}:{h5_ds}'
    input_slc = isce3.io.Raster(raster_ref)

    az_off_raster = isce3.io.Raster(os.path.join(iscetest.data, "offsets/azimuth.off"))
    rg_off_raster = isce3.io.Raster(os.path.join(iscetest.data, "offsets/range.off"))

    output_slc = isce3.io.Raster('warped.slc', rg_off_raster.width, rg_off_raster.length,
                                 rg_off_raster.num_bands, gdal.GDT_CFloat32, 'ENVI')

    # run resamp
    resamp.resamp(input_slc, output_slc, rg_off_raster, az_off_raster)


def test_validate():
    '''
    compare pybind CPU resamp against golden data
    '''
    # Init common variables.
    out_shape = (500, 500)
    trimmed_slice = np.index_exp[20:-20, 20:-20]

    # load generated data and avoid edges
    test_slc = np.fromfile('warped.slc', dtype=np.complex64).reshape(out_shape)[trimmed_slice]

    # load reference data and avoid edges
    ref_slc = np.fromfile(iscetest.data+'warped_envisat.slc', dtype=np.complex64).reshape(out_shape)[trimmed_slice]

    # get normalized error
    abs_error = np.nanmean(np.abs(test_slc - ref_slc))

    # check error
    assert (abs_error < 4.5e-5), f'pybind CPU resamp error {abs_error} > 1e-6'
