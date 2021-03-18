import pybind_isce3 as isce3
import numpy as np
from scipy.signal import convolve2d

from osgeo import gdal
from osgeo import gdal_array


def to_gdal_dataset(outpath, array):
    driver = gdal.GetDriverByName("GTiff")
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    length, width = array.shape
    dset = driver.Create(outpath, xsize=width, ysize=length, bands=1,
                         eType=dtype)
    dset.GetRasterBand(1).WriteArray(array)


def open_raster(filepath):
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    array = ds.GetRasterBand(1).ReadAsArray()

    return array


def test_run_filter2D():
    # Data dimension
    length = 200
    width = 311

    # Kernel dimensions
    kernel_length = 3

    block = 20

    # Create filter kernels
    kernel1d = np.ones([kernel_length, 1], dtype=np.float64) / kernel_length

    # Create real data to filter
    data_real = np.zeros([length, width], dtype=np.float64)
    data_cpx = np.zeros([length, width], dtype=np.complex128)

    for line in range(0, length):
        for col in range(0, width):
            data_real[line, col] = line + col
            data_cpx[line, col] = np.cos(line * col) + 1.0j * np.sin(line * col)

    # Save data
    to_gdal_dataset('data.real', data_real)
    to_gdal_dataset('data.cpx', data_cpx)

    # Filter data
    filt_data_real = isce3.io.Raster('data_real.filt', width, length, 1,
                                     gdal.GDT_Float64, "ENVI")
    filt_data_cpx = isce3.io.Raster('data_cpx.filt', width, length, 1,
                                    gdal.GDT_CFloat64, "ENVI")

    data_raster_real = isce3.io.Raster('data.real')
    data_raster_cpx = isce3.io.Raster('data.cpx')

    isce3.signal.filter2D(filt_data_real, data_raster_real, kernel1d,
                          kernel1d, block)
    isce3.signal.filter2D(filt_data_cpx, data_raster_cpx, kernel1d,
                          kernel1d, block)


def test_validate_filter_real():
    # Create 2D kernel
    kernel_length = 3
    kernel_width = 3
    kernel = np.ones((kernel_length, kernel_width), dtype=np.float64) / (
            kernel_width * kernel_length)

    # Validate filter2D for real data
    data = open_raster('data.real')

    filt_data = open_raster('data_real.filt')

    # Regenerate filtered data
    out = convolve2d(data, kernel, mode='same')

    diff = np.abs(out - filt_data)
    assert diff.max() < 1e-12


def test_validate_filter_complex():
    # Create kernel 2 D
    kernel_width = 3
    kernel_length = 3
    kernel = np.ones((kernel_length, kernel_width), dtype=np.float64) / (
            kernel_width * kernel_length)

    # Open data
    data = open_raster('data.cpx')

    filt_data = open_raster('data_cpx.filt')

    # Regenerate filter data
    out = convolve2d(data, kernel, mode='same')

    diff_pha = np.angle(filt_data * np.conj(out))
    diff_amp = np.abs(filt_data) - np.abs(out)

    assert diff_pha.max() < 1e-12
    assert diff_amp.max() < 1e-12
