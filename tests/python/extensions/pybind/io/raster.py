#!/usr/bin/env python3

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import numpy.testing as npt
import pytest
from numpy.typing import DTypeLike
from osgeo import gdal, gdal_array

import isce3
import iscetest


class commonClass:
    def __init__(self):
        self.nc = 100
        self.nl = 200
        self.nbx = 5
        self.nby = 7
        self.lat_file = 'lat.tif'
        self.lon_file = 'lon.vrt'
        self.inc_file = 'inc.bin'
        self.msk_file = 'msk.bin'
        self.vrt_file = 'topo.vrt'


def test_create_geotiff_float():
    # load shared params and clean up if necessary
    cmn = commonClass()
    if os.path.exists(cmn.lat_file):
        os.remove(cmn.lat_file)

    # create raster object
    raster = isce3.io.Raster(path=cmn.lat_file,
            width=cmn.nc, length=cmn.nl, num_bands=1,
            dtype=gdal.GDT_Float32, driver_name='GTiff')

    # check generated raster
    assert( os.path.exists(cmn.lat_file) )
    assert( raster.width == cmn.nc )
    assert( raster.length == cmn.nl )
    assert( raster.num_bands == 1 )
    assert( raster.datatype() == gdal.GDT_Float32 )
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nc)[None,:]

    ds = gdal.Open(cmn.lat_file, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None


def test_create_vrt_double():
    # load shared params and clean up if necessary
    cmn = commonClass()
    if os.path.exists(cmn.lon_file):
        os.remove(cmn.lon_file)

    # create raster object
    raster = isce3.io.Raster(path=cmn.lon_file,
            width=cmn.nc, length=cmn.nl, num_bands=1,
            dtype=gdal.GDT_Float64, driver_name='VRT')

    # check generated raster
    assert( os.path.exists(cmn.lon_file) )
    assert( raster.datatype() == gdal.GDT_Float64 )
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nl)[:,None]

    # open and populate
    ds = gdal.Open(cmn.lon_file, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    arr = ds.GetRasterBand(1).ReadAsArray()
    npt.assert_array_equal(data, arr, err_msg='RW in Update mode')
    ds = None

    # read array
    ds = gdal.Open(cmn.lon_file, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    npt.assert_array_equal(data, arr, err_msg='Readonly mode')


def test_create_2band_envi():
    # load shared params and clean up if necessary
    cmn = commonClass()
    if os.path.exists(cmn.inc_file):
        os.remove(cmn.inc_file)

    # create raster object
    raster = isce3.io.Raster(path=cmn.inc_file,
            width=cmn.nc, length=cmn.nl, num_bands=2,
            dtype=gdal.GDT_Int16, driver_name='ENVI')

    # check generated raster
    assert( os.path.exists(cmn.inc_file) )
    assert( raster.width == cmn.nc )
    assert( raster.length == cmn.nl )
    assert( raster.num_bands == 2 )
    assert( raster.datatype() == gdal.GDT_Int16 )
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nl)[:,None]

    # open and populate
    ds = gdal.Open(cmn.lon_file, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    arr = ds.GetRasterBand(1).ReadAsArray()
    npt.assert_array_equal(data, arr, err_msg='RW in Update mode')
    ds = None

    # read array
    ds = gdal.Open(cmn.lon_file, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    npt.assert_array_equal(data, arr, err_msg='Readonly mode')


def test_create_multiband_vrt():
    # load shared params and clean up if necessary
    cmn = commonClass()

    lat = isce3.io.Raster(cmn.lat_file)
    lon = isce3.io.Raster(cmn.lon_file)
    inc = isce3.io.Raster(cmn.inc_file)

    if os.path.exists( cmn.vrt_file):
        os.remove(cmn.vrt_file)

    vrt = isce3.io.Raster(cmn.vrt_file, raster_list=[lat,lon,inc])

    assert( vrt.width == cmn.nc )
    assert( vrt.length == cmn.nl )
    assert( vrt.num_bands == 4 )
    assert( vrt.datatype(1) == gdal.GDT_Float32 )
    assert( vrt.datatype(2) == gdal.GDT_Float64 )
    assert( vrt.datatype(3) == gdal.GDT_Int16 )
    assert( vrt.datatype(4) == gdal.GDT_Int16 )

    vrt = None



def test_createNumpyDataset():
    ny, nx = 200, 100
    data = np.random.randn(ny, nx).astype(np.float32)

    dset = gdal_array.OpenArray(data)
    raster = isce3.io.Raster(np.uintp(dset.this))

    assert( raster.width == nx )
    assert( raster.length == ny )
    assert( raster.datatype() == gdal.GDT_Float32 )

    dset = None
    del raster


def get_gdal_numpy_dtype_pairs() -> list[tuple[int, DTypeLike]]:
    """Get a list of GDAL and NumPy datatype pairs."""

    # Check that the GDAL version satisfies some major/minor version lower bound (may
    # give the wrong answer for pre-releases and development versions).
    def gdal_version_at_least(major: int, minor: int) -> bool:
        gdal_major, gdal_minor, *_ = gdal.__version__.split(".")[:2]
        return (int(gdal_major), int(gdal_minor)) >= (major, minor)

    dtype_pairs = [
        (gdal.GDT_Byte, np.uint8),
        (gdal.GDT_UInt16, np.uint16),
        (gdal.GDT_Int16, np.int16),
        (gdal.GDT_UInt32, np.uint32),
        (gdal.GDT_Int32, np.int32),
        (gdal.GDT_Float32, np.float32),
        (gdal.GDT_Float64, np.float64),
        (gdal.GDT_CFloat32, np.complex64),
        (gdal.GDT_CFloat64, np.complex128),
    ]
    if gdal_version_at_least(major=3, minor=5):
        dtype_pairs += [
            (gdal.GDT_UInt64, np.uint64),
            (gdal.GDT_Int64, np.int64),
        ]
    if gdal_version_at_least(major=3, minor=7):
        dtype_pairs += [(gdal.GDT_Int8, np.int8)]

    return dtype_pairs

@contextmanager
def make_temp_gtiff(
    shape: tuple[int, int],
    dtype: int,
    *,
    num_bands: int = 1,
) -> Generator[isce3.io.Raster, None, None]:
    """
    Create a temporary GeoTiff dataset.

    The dataset will be closed and the file deleted upon exiting the context
    manager.

    Parameters
    ----------
    shape : (int, int)
        The (length, width) dimensions of the raster dataset.
    dtype : int
        The GDAL datatype identifier of each raster in the dataset.
    num_bands : int, optional
        The number of bands in the raster dataset. Defaults to 1.

    Yields
    ------
    isce3.io.Raster
        The raster dataset.
    """
    with tempfile.NamedTemporaryFile(suffix=".tiff") as file_:
        length, width = shape
        raster = isce3.io.Raster(
            path=file_.name,
            width=width,
            length=length,
            num_bands=num_bands,
            dtype=dtype,
            driver_name="GTiff",
        )
        yield raster
        raster.close_dataset()

class TestRasterGetSetItem:
    @pytest.mark.parametrize("gdal_dtype,numpy_dtype", get_gdal_numpy_dtype_pairs())
    def test_dtype(self, gdal_dtype: int, numpy_dtype: DTypeLike):
        with make_temp_gtiff(shape=(1, 1), dtype=gdal_dtype) as raster:
            assert raster[:, :].dtype == numpy_dtype

    @pytest.mark.parametrize("gdal_dtype,numpy_dtype", get_gdal_numpy_dtype_pairs())
    def test_roundtrip(self, gdal_dtype: int, numpy_dtype: DTypeLike):
        with make_temp_gtiff(shape=(4, 5), dtype=gdal_dtype) as raster:
            data = np.arange(12, dtype=numpy_dtype).reshape(3, 4)
            raster[1:, 1:] = data
            npt.assert_array_equal(raster[1:, 1:], data)

    def test_buffer(self):
        with make_temp_gtiff(shape=(4, 5), dtype=gdal.GDT_Float32) as raster:
            data = np.arange(12, dtype=np.float32).reshape(3, 4)
            raster[1:, 1:] = memoryview(data)
            npt.assert_array_equal(raster[1:, 1:], data)

    @pytest.mark.parametrize("endian", ["<", ">"])
    def test_nonnative_endian(self, endian: str):
        with make_temp_gtiff(shape=(4, 5), dtype=gdal.GDT_Float32) as raster:
            data = np.arange(12, dtype=f"{endian}f4").reshape(3, 4)
            raster[1:, 1:] = data
            npt.assert_array_equal(raster[1:, 1:], data)

    def test_f_contiguous(self):
        with make_temp_gtiff(shape=(4, 5), dtype=gdal.GDT_Float32) as raster:
            data = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ], order="F")
            raster[1:, 1:] = data
            npt.assert_array_equal(raster[1:, 1:], data)

    def test_multiband(self):
        with make_temp_gtiff(
            shape=(2, 2),
            dtype=gdal.GDT_Float32,
            num_bands=2,
        ) as raster:
            regex = "only single-band raster datasets are supported, got numBands=2$"
            with pytest.raises(ValueError, match=regex):
                raster[:, :]

    def test_strided_slice(self):
        with make_temp_gtiff(shape=(4, 5), dtype=gdal.GDT_Float32) as raster:
            regex = "only contiguous slices are supported, got step=2$"
            with pytest.raises(ValueError, match=regex):
                raster[::2, 1::2]

    def test_bad_slice_dims(self):
        with make_temp_gtiff(shape=(4, 5), dtype=gdal.GDT_Float64) as raster:
            data = np.arange(20).reshape(5, 4)
            regex = r"could not broadcast array from shape \(5, 4\) into shape \(4, 5\)"
            with pytest.raises(ValueError, match=regex):
                raster[:, :] = data

    def test_bad_dtype(self):
        with make_temp_gtiff(shape=(4, 5), dtype=gdal.GDT_Int32) as raster:
            data = np.full((4, 5), fill_value="asdf")
            with pytest.raises(Exception):
                raster[:, :] = data


class TestRasterGetSetBlock:
    def test_roundtrip(self):
        with make_temp_gtiff(
            shape=(4, 5),
            dtype=gdal.GDT_Float32,
            num_bands=3,
        ) as raster:
            input = np.arange(12, dtype=np.float32).reshape(3, 4)
            indices = np.s_[1:, 1:]
            raster.set_block(indices, input, band=2)
            output = raster.get_block(indices, band=2)
            npt.assert_array_equal(output, input)

    @pytest.mark.parametrize("band", [0, 4])
    def test_bad_band(self, band: int):
        with make_temp_gtiff(
            shape=(3, 3),
            dtype=gdal.GDT_Float32,
            num_bands=3,
        ) as raster:
            indices = np.s_[:, :]
            regex = f"band {band} is out of bounds for a raster dataset with 3 bands$"
            with pytest.raises(IndexError, match=regex):
                raster.get_block(indices, band=band)


def test_raster_shape():
    shape = (4, 5)
    with make_temp_gtiff(shape=shape, dtype=gdal.GDT_Float64) as raster:
        assert raster.shape == shape


@pytest.mark.parametrize("gdal_dtype,numpy_dtype", get_gdal_numpy_dtype_pairs())
def test_raster_dtype(gdal_dtype: int, numpy_dtype: DTypeLike):
    with make_temp_gtiff(shape=(4, 5), dtype=gdal_dtype) as raster:
        assert raster.datatype() == gdal_dtype
        assert raster.dtype == numpy_dtype


def test_grid_coords():
    raster_file = os.path.join(iscetest.data, "winnipeg_dem.tif")
    raster = isce3.io.Raster(raster_file)

    # Check that these attributes match what we get from `gdalinfo`.
    assert np.isclose(raster.x0, -97.8507)
    assert np.isclose(raster.y0, 49.5642)
    assert np.isclose(raster.dx, 0.00027760989011)
    assert np.isclose(raster.dy, -0.00027765625)


if __name__ == "__main__":
    test_create_geotiff_float()
    test_create_vrt_double()
    test_create_2band_envi()
    test_create_multiband_vrt()
    test_createNumpyDataset()

# end of file
