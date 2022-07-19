#!/usr/bin/env python3
import os
import numpy as np
from osgeo import gdal, gdal_array
import iscetest
import isce3.ext.isce3 as isce3


def _create_raster(outpath, array, width, length, nbands, dtype,
                   driver_name):
    '''
    create and return ISCE3 raster obj. containing given array
    '''
    driver = gdal.GetDriverByName(driver_name)
    dir_name = os.path.dirname(outpath)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    print('creating test file: ', outpath)
    dset = driver.Create(outpath, width, length, nbands, dtype)
    if dset is None:
        error_message = 'ERROR creating test file: ' + outpath
        raise RuntimeError(error_message)
    dset.GetRasterBand(1).WriteArray(array)
    raster_obj = isce3.io.Raster(outpath)
    return raster_obj

def test_run():
    '''
    run test
    '''

    # set parameters
    width = 10
    length = 10
    nbands = 1
    band = 1
    dtype = gdal.GDT_CFloat32
    memory_mode_list = [
        isce3.core.MemoryModeBlocksY.SingleBlockY,
        isce3.core.MemoryModeBlocksY.MultipleBlocksY] 
    symmetrization_error_threshold = 1e-6
   
    # set file paths
    hv_file = "polsar/symmetrize_hv.bin"
    vh_file = "polsar/symmetrize_vh.bin"
    output_file = "polsar/symmetrize_output.bin"

    # create input arrays
    hv_array = np.zeros((length, width), dtype=np.complex64)
    vh_array = np.zeros((length, width), dtype=np.complex64)
    for i in range(length):
        for j in range(width):
            hv_array[i, j] = i + 1j * j
            vh_array[i, j] = 2 * i + 2j * j 

    # create input raster objects
    hv_raster = _create_raster(hv_file, hv_array, width, length, nbands,
                                    dtype, "ENVI")
    vh_raster = _create_raster(vh_file, vh_array, width, length, nbands,
                                    dtype, "ENVI")

    # iterate over memory modes
    for memory_mode in memory_mode_list:

        # create output raster object
        output_raster = isce3.io.Raster(output_file, width, length, nbands,
                                    dtype, "ENVI")

        # symmetrize cross-pol channels
        isce3.polsar.symmetrize_cross_pol_channels(
            hv_raster, vh_raster, output_raster, memory_mode)
        del output_raster

        # read output raster
        ds = gdal.Open(output_file, gdal.GA_ReadOnly)
        output_array = ds.GetRasterBand(band).ReadAsArray()

        # compute max error
        symmetrization_max_error = np.max(np.abs(output_array[i, j] - 
                                    (hv_array[i, j] + vh_array[i, j]) / 2.0))

        # evaluate max error
        print('PolSAR symmetrization max. error:', symmetrization_max_error)
        assert(symmetrization_max_error  < symmetrization_error_threshold )


if __name__ == "__main__":
    test_run()
