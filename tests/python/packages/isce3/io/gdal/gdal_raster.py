import os
import pytest
import numpy as np
from dataclasses import dataclass
from osgeo import gdal
from pathlib import Path
from tempfile import TemporaryDirectory
from journal.ext.journal import ApplicationError

from isce3.io.gdal.gdal_raster import GDALRaster


def test_1_band_creation():
    with TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "1_band.gdal"
        obj = GDALRaster.create_dataset_file(
            filepath=filepath,
            dtype=np.float64,
            shape=(100, 100),
            num_bands=1,
        )

        dataset: gdal.Dataset = gdal.Open(os.fspath(filepath), gdal.GA_ReadOnly)
        num_bands = dataset.RasterCount

        # The output raster should have only one band.
        assert num_bands == 1
        # The create_dataset_file function should have returned a GDALRaster object
        # and not a list of them.
        assert isinstance(obj, GDALRaster)


@pytest.mark.parametrize("num_bands", [0, -1, -99999])
def test_lt_1_band_creation(num_bands):
    with TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "negative_bands.gdal"

        # The class should fail with an ApplicationError if the number of bands given
        # is less than 1.
        with pytest.raises(ApplicationError):
            GDALRaster.create_dataset_file(
                filepath=filepath,
                dtype=np.float64,
                shape=(100, 100),
                num_bands=num_bands,
            )


@pytest.mark.parametrize("num_bands", [2, 3, 10])
def test_multi_band_creation(num_bands):
    with TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / f"{num_bands}_bands.gdal"
        obj = GDALRaster.create_dataset_file(
            filepath=filepath,
            dtype=np.float64,
            shape=(100, 100),
            num_bands=num_bands,
        )

        dataset: gdal.Dataset = gdal.Open(os.fspath(filepath), gdal.GA_ReadOnly)
        dataset_bands = dataset.RasterCount

        # The dataset file should have the correct number of bands.
        assert num_bands == dataset_bands

        # The returned object should be a list of GDALRasters when num_bands > 1.
        assert isinstance(obj, list)
        assert all(map(lambda item: isinstance(item, GDALRaster), obj))

        # The length of the list should be the same as num_bands.
        assert len(obj) == num_bands


@pytest.mark.parametrize("shape", [(100, 100), (100, 200), (200, 100)])
def test_shape(shape):
    length, width = shape
    with TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / f"shape_{length}_{width}.gdal"
        raster = GDALRaster.create_dataset_file(
            filepath=filepath,
            dtype=np.float64,
            shape=shape,
            num_bands=1,
        )

        dataset: gdal.Dataset = gdal.Open(os.fspath(filepath), gdal.GA_ReadOnly)

        raster_length: int = dataset.RasterYSize
        raster_width: int = dataset.RasterXSize
        
        # The length/width of the raster as it is known to gdal should be equal to
        # those of the given shape.
        assert length == raster_length
        assert width == raster_width

        # The length/width of the data returned from 
        accessed_data: np.ndarray = raster[:, :]
        data_length, data_width = accessed_data.shape

        assert length == data_length
        assert width == data_width


@pytest.mark.parametrize("step_size", [2, 3, 999, -1, -999])
def test_stride_errors(step_size):
    shape = (200, 100)

    with TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / f"array_stride_err.gdal"
        raster = GDALRaster.create_dataset_file(
            filepath=filepath,
            dtype=np.float64,
            shape=shape,
            num_bands=1,
        )
        dummy_array = np.empty(shape, dtype=np.float64)

        # ApplicationErrors should be raised when strided slicing is used in either
        # dimension to access or write to a raster.
        with pytest.raises(ApplicationError):
            nope = raster[::step_size, :]
        with pytest.raises(ApplicationError):
            nope = raster[:, ::step_size]
        with pytest.raises(ApplicationError):
            raster[::step_size, :] = dummy_array
        with pytest.raises(ApplicationError):
            raster[:, ::step_size] = dummy_array


def test_create_dataset_file_dtype_character_codes():
    """
    Test the GDALRaster.create_dataset_file for acceptance of character codes as a dtype
    """
    with TemporaryDirectory() as tempdir:
        raster = GDALRaster.create_dataset_file(
            filepath=Path(tempdir) / "dtype_test_f4.gdal",
            dtype="f4",
            shape=(100, 100),
            num_bands=1,
        )

        assert raster.dtype == np.float32


def test_create_dataset_file_dtype_classes():
    """
    Test the GDALRaster.create_dataset_file for acceptance of class instances that have
    a .dtype attribute as a dtype parameter
    """
    @dataclass
    class a:
        dtype = np.complex64

    with TemporaryDirectory() as tempdir:
        raster = GDALRaster.create_dataset_file(
            filepath=Path(tempdir) / "dtype_test_a.gdal",
            dtype=a(),
            shape=(100, 100),
            num_bands=1,
        )

        assert raster.dtype == np.complex64

        raster_2 = GDALRaster.create_dataset_file(
            filepath=Path(tempdir) / "dtype_test_raster.gdal",
            dtype=raster,
            shape=(100, 100),
            num_bands=1,
        )

        assert raster_2.dtype == np.complex64


def test_create_dataset_file_dtype_character_codes():
    """
    Test the GDALRaster.create_dataset_file for acceptance of np.dtype objects as a
    dtype parameter.
    """
    with TemporaryDirectory() as tempdir:
        raster = GDALRaster.create_dataset_file(
            filepath=Path(tempdir) / "dtype_test_float64.gdal",
            dtype=np.float64,
            shape=(100, 100),
            num_bands=1,
        )

        assert raster.dtype == np.float64
