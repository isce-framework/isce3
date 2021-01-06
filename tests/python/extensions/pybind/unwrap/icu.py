'''
Unit tests for CPU pybind ICU
'''
import numpy.testing as npt
import numpy as np
import pybind_isce3 as isce3
from osgeo import gdal
from osgeo import gdal_array
import os

width = 256
length = 1100


def test_getter_setter():
    icu = isce3.unwrap.ICU()

    icu.buffer_lines = 1024
    npt.assert_equal(icu.buffer_lines, 1024)

    icu.overlap_lines = 50
    npt.assert_equal(icu.overlap_lines, 50)

    icu.use_phase_grad_neut = True
    npt.assert_equal(icu.use_phase_grad_neut, True)

    icu.use_intensity_neut = True
    npt.assert_equal(icu.use_intensity_neut, True)

    icu.phase_grad_win_size = 3
    npt.assert_equal(icu.phase_grad_win_size, 3)

    icu.neut_phase_grad_thr = 1.5
    npt.assert_equal(icu.neut_phase_grad_thr, 1.5)

    icu.neut_intensity_thr = 4.0
    npt.assert_equal(icu.neut_intensity_thr, 4.0)

    icu.neut_correlation_thr = 0.5
    npt.assert_equal(icu.neut_correlation_thr, 0.5)

    icu.trees_number = 3
    npt.assert_equal(icu.trees_number, 3)

    icu.max_branch_length = 32
    npt.assert_equal(icu.max_branch_length, 32)

    icu.ratio_dxdy = 2.0
    npt.assert_equal(icu.ratio_dxdy, 2.0)

    icu.init_corr_thr = 0.4
    npt.assert_almost_equal(icu.init_corr_thr, 0.4, decimal=8)

    icu.max_corr_thr = 0.8
    npt.assert_almost_equal(icu.max_corr_thr, 0.8, decimal=8)

    icu.corr_incr_thr = 0.2
    npt.assert_almost_equal(icu.corr_incr_thr, 0.2, decimal=8)

    icu.min_cc_area = 0.01
    npt.assert_almost_equal(icu.min_cc_area, 0.01, decimal=8)

    icu.num_bs_lines = 8
    npt.assert_equal(icu.num_bs_lines, 8)

    icu.min_overlap_area = 12
    npt.assert_equal(icu.min_overlap_area, 12)

    icu.phase_var_thr = 3.0
    npt.assert_equal(icu.phase_var_thr, 3.0)


def to_gdal_dataset(outpath, array):
    driver = gdal.GetDriverByName("GTiff")
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    length, width = array.shape
    dset = driver.Create(outpath, xsize=width, ysize=length, bands=1, eType=dtype)
    dset.GetRasterBand(1).WriteArray(array)
    dset = None


def create_datasets():
    # Generate interferogram
    xx = np.linspace(0.0, 50.0, width)
    yy = np.linspace(0.0, 50.0, length)

    x, y = np.meshgrid(xx, yy)
    igram = np.exp(1j * (x + y))

    to_gdal_dataset('igram.int', igram)

    # Generate coherence 
    corr = np.zeros((length, width), dtype=np.float32)
    corr[100:900, 50:100] = 1.0
    corr[100:900, 150:200] = 1.0
    corr[900:950, 50:200] = 1.0
    corr[1000:1050, 50:200] = 1.0

    to_gdal_dataset('coherence.coh', corr)


def read_raster(infile):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    array = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return array


def test_run_icu():
    # Create interferogram & coherence
    create_datasets()

    # Open datasets as ISCE3 rasters
    igram = isce3.io.Raster('igram.int')
    corr = isce3.io.Raster('coherence.coh')

    # Generate output rasters
    unwRaster = isce3.io.Raster('unw.int', igram.width,
                                igram.length, 1, gdal.GDT_Float32, "GTiff")
    cclRaster = isce3.io.Raster('ccl.cc', igram.width,
                                igram.length, 1, gdal.GDT_Byte, "GTiff")

    # Configure and run ICU
    icu = isce3.unwrap.ICU()
    icu.max_corr_thr = 1
    icu.unwrap(unwRaster, cclRaster, igram, corr)


def test_check_unwrapped_phase():
    # Read interferograms and ccl
    ccl = read_raster('ccl.cc')
    unw = read_raster('unw.int')

    # Generate reference interferogram
    xx = np.linspace(0.0, 50.0, width)
    yy = np.linspace(0.0, 50.0, length)

    x, y = np.meshgrid(xx, yy)
    ref_unw = x + y

    # Reference each CC differently
    labels = np.unique(ccl)
    diff = (ref_unw[np.where(ccl == labels[1])] - ref_unw[102, 52]) - \
           (unw[np.where(ccl == labels[1])] - unw[102, 52])
    npt.assert_array_less(diff.max(), 1e-5)

    diff = (ref_unw[np.where(ccl == labels[2])] - ref_unw[1002, 52]) - \
           (unw[np.where(ccl == labels[2])] - unw[1002, 52])
    npt.assert_array_less(diff.max(), 1e-5)


def test_check_connected_components():
    # Open Connected components
    ccl = read_raster('ccl.cc')
    l, w = ccl.shape

    npt.assert_equal(w, width)
    npt.assert_equal(l, length)

    # Check all pixels within the U
    # have the same connected component
    npt.assert_equal(np.all(ccl[100:900, 50:100] == ccl[100, 50]), True)
    npt.assert_equal(np.all(ccl[100:900, 150:200] == ccl[100, 50]), True)
    npt.assert_equal(np.all(ccl[900:950, 50:200] == ccl[900, 50]), True)
    npt.assert_equal(np.all(ccl[1000:1050, 50:200] == ccl[1000, 50]), True)

    # Check that different connected components 
    # have different labels
    npt.assert_raises(AssertionError, npt.assert_array_equal, ccl[100, 50], ccl[1000, 50])
    npt.assert_raises(AssertionError, npt.assert_array_equal, ccl[900, 50], ccl[1000, 50])


if __name__ == '__main__':
    test_getter_setter()
    test_run_icu()
    test_check_unwrapped_phase()
    test_check_connected_components()

    # Remove generated data
    os.remove("igram.int")
    os.remove("coherence.coh")
    os.remove("ccl.cc")
    os.remove("unw.int")
