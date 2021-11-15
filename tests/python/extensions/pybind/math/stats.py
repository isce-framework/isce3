#!/usr/bin/env python3
import os
import numpy as np
import numpy.testing as npt
from osgeo import gdal, gdal_array
import iscetest
import pybind_isce3 as isce3


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
    error_threshold = 1e-6
   
    # set file paths
    real_file = "math/stats_real.bin"
    imag_file = "math/stats_imag.bin"
    complex_file = "math/stats_complex.bin"

    # create input arrays
    real_array = np.zeros((length, width), dtype=np.float64)
    imag_array = np.zeros((length, width), dtype=np.float64)
    complex_array = np.zeros((length, width), dtype=np.complex128)

    for i in range(length):
        for j in range(width):
            real_array[i, j] = (np.sin(2. * np.pi * float(i) / length) + 
                                np.cos(4. * np.pi * float(j) / width))
            imag_array[i, j] = 2. * (np.sin(6. * np.pi * float(i) / length) -
                                    np.cos(8. * np.pi * float(j) / width))
            complex_array[i, j] = real_array[i, j] + 1.j * imag_array[i, j]

    # create input raster objects
    real_raster = _create_raster(real_file, real_array, width, length, nbands,
                                 gdal.GDT_Float32, "ENVI")
    imag_raster = _create_raster(imag_file, imag_array, width, length, nbands,
                                 gdal.GDT_Float32, "ENVI")
    complex_raster = _create_raster(complex_file, complex_array, width, length,
                                    nbands, gdal.GDT_CFloat32, "ENVI")

    # compute ISCE3 stats
    stats_real_obj = isce3.math.compute_raster_stats_float32(real_raster)[0]
    stats_imag_obj = isce3.math.compute_raster_stats_float32(imag_raster)[0]
    stats_real_imag_obj = isce3.math.compute_raster_stats_real_imag(
        complex_raster)[0]
    
    atol = 1e-7

    # compare first image ("real") with NumPy
    npt.assert_allclose(stats_real_obj.min, np.min(real_array),
                        atol=atol, err_msg="min values differ (real part)")
    npt.assert_allclose(stats_real_obj.mean, np.mean(real_array),
                        atol=atol, err_msg="mean values differ (real part)")
    npt.assert_allclose(stats_real_obj.max, np.max(real_array),
                        atol=atol, err_msg="max values differ (real part)")
    npt.assert_allclose(stats_real_obj.sample_stddev,
                        np.std(real_array, ddof=1), atol=atol,
                        err_msg="sample stddev values differ (real part)")

    # compare second image ("real") part with NumPy
    npt.assert_allclose(stats_imag_obj.min, np.min(imag_array),
                        atol=atol, err_msg="min values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.mean, np.mean(imag_array),
                        atol=atol, err_msg="mean values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.max, np.max(imag_array),
                        atol=atol, err_msg="max values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.sample_stddev,
                        np.std(imag_array, ddof=1), atol=atol,
                        err_msg="sample stddev values differ (imaginary part)")

    # compare first image ("real") with real part of the complex image
    npt.assert_allclose(stats_real_obj.min, stats_real_imag_obj.min_real,
                        atol=atol, err_msg="min values differ (real part)")
    npt.assert_allclose(stats_real_obj.mean, stats_real_imag_obj.mean_real,
                        atol=atol, err_msg="mean values differ (real part)")
    npt.assert_allclose(stats_real_obj.max, stats_real_imag_obj.max_real,
                        atol=atol, err_msg="max values differ (real part)")
    npt.assert_allclose(stats_real_obj.sample_stddev, 
                        stats_real_imag_obj.sample_stddev_real, atol=atol,
                        err_msg="sample stddev values differ (real part)")

    # compare first image ("imag") with imaginary part of the complex image
    npt.assert_allclose(stats_imag_obj.min, stats_real_imag_obj.min_imag,
                        atol=atol, err_msg="min values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.mean, stats_real_imag_obj.mean_imag,
                        atol=atol, err_msg="mean values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.max, stats_real_imag_obj.max_imag,
                        atol=atol, err_msg="max values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.sample_stddev,
                        stats_real_imag_obj.sample_stddev_imag, atol=atol,
                        err_msg="sample stddev values differ (imaginary part)")


if __name__ == "__main__":
    test_run()
