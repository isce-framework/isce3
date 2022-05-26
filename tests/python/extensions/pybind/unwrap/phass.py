'''
Unit test for Phass unwrapper
'''

import os

import numpy as np
import numpy.testing as npt
import isce3.ext.isce3 as isce3
from osgeo import gdal, gdal_array

width = 256
length = 1100


def to_gdal_dataset(outpath, array):
    driver = gdal.GetDriverByName("Gtiff")
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    length, width = array.shape
    dset = driver.Create(outpath, xsize=width, ysize=length, bands=1,
                         eType=dtype)
    dset.GetRasterBand(1).WriteArray(array)


def create_datasets():
    # Generate interferogram
    xx = np.linspace(0.0, 50.0, width)
    yy = np.linspace(0.0, 50.0, length)

    x, y = np.meshgrid(xx, yy)
    igram = np.exp(1j * (x + y))
    phase = np.angle(igram)

    to_gdal_dataset('phase.tif', phase)

    # Generate coherence
    corr = np.zeros((length, width), dtype=np.float32)
    corr[100:900, 50:100] = 1.0
    corr[100:900, 150:200] = 1.0
    corr[900:950, 50:200] = 1.0
    corr[1000:1050, 50:200] = 1.0

    to_gdal_dataset('coherence.tif', corr)


def read_raster(infile):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    array = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return array


def test_getter_setter():
    phass = isce3.unwrap.Phass()

    phass.correlation_threshold = 0.5
    npt.assert_equal(phass.correlation_threshold, 0.5)

    phass.good_correlation = 0.6
    npt.assert_equal(phass.good_correlation, 0.6)

    phass.min_pixels_region = 100
    npt.assert_equal(phass.min_pixels_region, 100)


def test_run_phass():
    # Create interferogram and coherence
    create_datasets()

    # Open created datasets as ISCE3 rasters
    phase = isce3.io.Raster('phase.tif')
    corr = isce3.io.Raster('coherence.tif')

    # Generate output rasters
    unwRaster = isce3.io.Raster('unw.f4', phase.width,
                                phase.length, 1, gdal.GDT_Float32, "ENVI")
    labelRaster = isce3.io.Raster('label.u1', phase.width,
                                  phase.length, 1, gdal.GDT_Byte, "ENVI")

    # Configure and run Phass
    phass = isce3.unwrap.Phass()
    phass.unwrap(phase, corr, unwRaster, labelRaster)


def test_check_unwrapped_phase():
    # Read interferogram and connected components
    label = read_raster('label.u1')
    unw = read_raster('unw.f4')

    # Generate reference interferogram
    xx = np.linspace(0.0, 50.0, width)
    yy = np.linspace(0.0, 50.0, length)

    x, y = np.meshgrid(xx, yy)
    ref_unw = x + y

    # Reference to each label differently
    labels = np.unique(label)
    diff = (ref_unw[np.where(label == labels[1])] - ref_unw[102, 52]) - \
           (unw[np.where(label == labels[1])] - unw[102, 52])
    npt.assert_array_less(np.abs(diff).max(), 1e-5)

    diff = (ref_unw[np.where(label == labels[2])] - ref_unw[1002, 52]) - \
           (unw[np.where(label == labels[2])] - unw[1002, 52])
    npt.assert_array_less(np.abs(diff).max(), 1e-5)


def test_check_labels():
    # Open labels
    label = read_raster('label.u1')
    l, w = label.shape

    npt.assert_equal(w, width)
    npt.assert_equal(l, length)

    # Check all pixels within the U
    # have the same label

    npt.assert_equal(np.all(label[100:900, 50:100] == label[100, 50]), True)
    npt.assert_equal(np.all(label[100:900, 150:200] == label[100, 50]), True)
    npt.assert_equal(np.all(label[900:950, 50:200] == label[900, 50]), True)
    npt.assert_equal(np.all(label[1000:1050, 50:200] == label[1000, 50]), True)

    # Check different connected components 
    # have different labels
    npt.assert_raises(AssertionError, npt.assert_array_equal, label[100, 50],
                      label[1000, 50])
    npt.assert_raises(AssertionError, npt.assert_array_equal, label[900, 50],
                      label[1000, 50])


if __name__ == '__main__':
    test_getter_setter()
    test_run_phass()
    test_check_unwrapped_phase()
    test_check_labels()

