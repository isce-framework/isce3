#!/usr/bin/env python3
from functools import partial
import os
import numpy as np
import numpy.testing as npt
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
    npt.assert_allclose(stats_real_obj.min, stats_real_imag_obj.real.min,
                        atol=atol, err_msg="min values differ (real part)")
    npt.assert_allclose(stats_real_obj.mean, stats_real_imag_obj.real.mean,
                        atol=atol, err_msg="mean values differ (real part)")
    npt.assert_allclose(stats_real_obj.max, stats_real_imag_obj.real.max,
                        atol=atol, err_msg="max values differ (real part)")
    npt.assert_allclose(stats_real_obj.sample_stddev, 
                        stats_real_imag_obj.real.sample_stddev, atol=atol,
                        err_msg="sample stddev values differ (real part)")

    # compare first image ("imag") with imaginary part of the complex image
    npt.assert_allclose(stats_imag_obj.min, stats_real_imag_obj.imag.min,
                        atol=atol, err_msg="min values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.mean, stats_real_imag_obj.imag.mean,
                        atol=atol, err_msg="mean values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.max, stats_real_imag_obj.imag.max,
                        atol=atol, err_msg="max values differ (imaginary part)")
    npt.assert_allclose(stats_imag_obj.sample_stddev,
                        stats_real_imag_obj.imag.sample_stddev, atol=atol,
                        err_msg="sample stddev values differ (imaginary part)")


def test_array():
    shape = (5, 3)
    np.random.seed(0)
    x = np.random.normal(size=shape).astype("f4")
    y = np.random.normal(size=shape).astype("f4")
    z = (x + 1j * y).astype("c8")

    assert_allclose = partial(npt.assert_allclose, rtol=1e-6)

    # ctor
    s = isce3.math.StatsFloat32(x)
    assert_allclose(s.mean, x.mean())
    assert_allclose(s.min, x.min())
    assert_allclose(s.max, x.max())
    assert_allclose(s.sample_stddev, x.std(ddof=1))

    # update with array
    s2 = isce3.math.StatsFloat32()
    s2.update(x)
    assert_allclose(s2.min, s.min)
    assert_allclose(s2.max, s.max)
    assert_allclose(s2.mean, s.mean)
    assert_allclose(s2.sample_stddev, s.sample_stddev)

    # update with stats
    s3 = isce3.math.StatsFloat32(x)
    s3.update(s)
    assert_allclose(s3.min, s.min)
    assert_allclose(s3.max, s.max)
    assert_allclose(s3.mean, s.mean)
    # stddev differs because 2*n-1 != 2*(n-1)
    npt.assert_(s3.n_valid == 2 * s.n_valid)

    # real/imag
    sri = isce3.math.StatsRealImagFloat32(z)
    sri2 = isce3.math.StatsRealImagFloat32()
    sri2.update(z)
    sri2.update(sri)


def test_strided_array():
    np.random.seed(12345)
    full = np.random.normal(size=(10, 10)).astype("float32")
    even_rows = full[::2, :]
    even_cols = full[:, ::2]

    # Make sure test will trap wrong stride by checking that stdev of views
    # are noticeably different from the stdev of the full array.
    assert not np.isclose(full.std(ddof=1), even_rows.std(ddof=1))
    assert not np.isclose(full.std(ddof=1), even_cols.std(ddof=1))

    s = isce3.math.StatsFloat32(even_rows)
    npt.assert_allclose(s.sample_stddev, even_rows.std(ddof=1))

    s = isce3.math.StatsFloat32(even_cols)
    npt.assert_allclose(s.sample_stddev, even_cols.std(ddof=1))


def test_str():
    stats = isce3.math.StatsFloat32()
    x = np.ones(10).astype("f4")
    stats.update(x)
    expected = "StatsFloat32(n_valid=10, mean=1.0, min=1.0, max=1.0, sample_stddev=0.0)"
    assert str(stats) == expected

    stats_cpx = isce3.math.StatsRealImagFloat32()
    z = x + 1j * x
    stats_cpx.update(z)
    expected_cpx = f"StatsRealImagFloat32(real={expected}, imag={expected})"
    assert str(stats_cpx) == expected_cpx


if __name__ == "__main__":
    test_run()
