from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import journal
import numpy as np
from journal.Error import Error
from numpy.typing import DTypeLike
from osgeo import gdal

from isce3.io.dataset import DatasetReader, DatasetWriter

gdal.UseExceptions()

GDALRasterT = TypeVar("GDALRasterT", bound="GDALRaster")


def get_gdal_dtype(type: DTypeLike) -> int:
    """Returns the GDAL data type associated with a given NumPy dtype."""
    np_dtype = np.dtype(type)

    if np_dtype == np.complex128:
        return gdal.GDT_CFloat64
    if np_dtype == np.complex64:
        return gdal.GDT_CFloat32
    if np_dtype == np.int64:
        return gdal.GDT_Int64
    if np_dtype == np.int32:
        return gdal.GDT_Int32
    if np_dtype == np.int16:
        return gdal.GDT_Int16
    if np_dtype == np.uint64:
        return gdal.GDT_UInt64
    if np_dtype == np.uint32:
        return gdal.GDT_UInt32
    if np_dtype == np.uint16:
        return gdal.GDT_UInt16
    if np_dtype == np.float64:
        return gdal.GDT_Float64
    if np_dtype == np.float32:
        return gdal.GDT_Float32
    if np_dtype == np.ubyte:
        return gdal.GDT_Byte
    
    raise ValueError(f"Type {type} does not correspond to a GDAL data type.")


def get_numpy_dtype_from_gdal(type: int) -> np.dtype:
    """Returns the NumPy dtype associated with a given GDAL data type."""
    if type == gdal.GDT_CFloat64:
        return np.complex128
    if type == gdal.GDT_CFloat32:
        return np.complex64
    if type == gdal.GDT_Int64:
        return np.int64
    if type == gdal.GDT_Int32:
        return np.int32
    if type == gdal.GDT_Int16:
        return np.int16
    if type == gdal.GDT_UInt64:
        return np.uint64
    if type == gdal.GDT_UInt32:
        return np.uint32
    if type == gdal.GDT_UInt16:
        return np.uint16
    if type == gdal.GDT_Float64:
        return np.float64
    if type == gdal.GDT_Float32:
        return np.float32
    if type == gdal.GDT_Byte:
        return np.ubyte
    
    raise ValueError(f"GDAL type {type} does not correspond to a NumPy data type.")


@dataclass(frozen=True)
class GDALRaster(DatasetReader, DatasetWriter):
    """
    A GDAL raster accessible by array indexing.

    Attributes
    ----------
    filepath : pathlib.Path
        The path to the GDAL raster to be read.
    band : int
        The band of the raster (1-based).
    shape : tuple[int, int]
        The shape of the raster.
    dtype : np.dtype
        The data type of the raster, in NumPy dtype format.
    driver_name : str
        The short name of the raster's GDAL driver.
    """

    filepath: Path
    band: int
    shape: tuple[int, int]
    dtype: np.dtype
    driver_name: str

    def __init__(
        self,
        filepath: os.PathLike,
        band: int,
    ) -> GDALRasterT:
        """
        Intakes information about an existing GDAL raster and returns an associated
        GDALRaster object.

        Parameters
        ----------
        filepath : os.PathLike
            The path to the GDAL dataset to be read.
        band : int
            The band of the raster (1-based).
        """
        error_channel: Error = journal.error("GDALRaster.__init__")

        dataset: gdal.Dataset = gdal.Open(os.fspath(filepath), gdal.GA_ReadOnly)
        cols: int = dataset.RasterXSize
        rows: int = dataset.RasterYSize
        my_shape = (rows, cols)

        num_bands = dataset.RasterCount
        if band > num_bands:
            err_str = (
                f"GDALRaster initialized with band {band} - file {filepath} has "
                f"only {num_bands} bands."
            )
            error_channel.log(err_str)
            raise ValueError(err_str)
        
        driver: gdal.Driver = dataset.GetDriver()
        driver_name = driver.ShortName

        band_obj: gdal.Band = dataset.GetRasterBand(band)
        gdal_dtype: int = band_obj.DataType
        dtype = get_numpy_dtype_from_gdal(gdal_dtype)

        my_path = Path(filepath)

        object.__setattr__(self, "filepath", my_path)
        object.__setattr__(self, "shape", my_shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "driver_name", driver_name)

    @classmethod
    def create_dataset_file(
        cls: type[GDALRasterT],
        filepath: os.PathLike,
        dtype: DTypeLike,
        shape: tuple[int, int],
        *,
        num_bands: int = 1,
        driver_name: str = "ENVI",
    ) -> GDALRasterT | list[GDALRasterT]:
        """
        A factory method that intakes information about a GDAL dataset,
        initializes a file to contain it, and returns an associated GDALRaster object
        or list of GDALRaster objects.

        Parameters
        ----------
        filepath : path-like
            The file path to be created.
        dtype : data-type
            The data type of the new dataset.
        shape : tuple[int, int]
            The shape of the new dataset in (length, width) order.
        num_bands : int, optional
            The number of bands in the dataset. Must be a positive number. If 1, this
            function will return a single GDALRaster object, otherwise it will return a
            list. Defaults to 1.
        driver_name : str, optional
            The name of the GDAL driver. Defaults to "ENVI".

        Returns
        -------
        GDALRaster or list[GDALRaster]
            A GDALRaster object for each band in the newly-created file. Returns
            a list only if num_bands is greater than 1.
        """
        error_channel: Error = journal.error("GDALRaster.create_dataset_file")

        if num_bands < 1:
            err_str = f"num_bands given as {num_bands}; must be a positive integer."
            error_channel.log(err_str)
            raise ValueError(err_str)

        length, width = shape
        driver: gdal.Driver = gdal.GetDriverByName(driver_name)
        driver.Create(
            os.fspath(filepath),
            xsize=width,
            ysize=length,
            bands=num_bands,
            eType=get_gdal_dtype(dtype),
        )

        if num_bands == 1:
            return cls(filepath=Path(filepath), band=1)

        rasters: list[GDALRasterT] = [
            cls(filepath=Path(filepath), band=band) for band in range(1, num_bands + 1)
        ]

        return rasters

    @property
    def ndim(self) -> int:
        """int : Number of array dimensions."""
        return 2
    
    @property
    def length(self) -> int:
        length, _ = self.shape
        return length

    @property
    def width(self) -> int:
        _, width = self.shape
        return width

    def __array__(self) -> np.ndarray:
        return self[:, :]

    def __getitem__(self, key: tuple[slice, slice], /) -> np.ndarray:
        error_channel: Error = journal.error("GDALRaster.__getitem__")
        if len(key) != 2:
            err_str = (
                f"Key given with {len(key)} slice(s). GDALRaster only supports "
                "two-dimensional keys."
            )
            error_channel.log(err_str)
            raise ValueError(err_str)

        slice_y, slice_x = key

        start_y, stop_y, step_y = slice_y.indices(self.length)
        start_x, stop_x, step_x = slice_x.indices(self.width)
        length = stop_y - start_y
        width = stop_x - start_x

        if (step_y != 1) or (step_x != 1):
            err_str = "GDALRaster does not support strided slicing."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Benchmarking shows no detectable change in runtime if datasets are opened
        # repeatedly instead of being kept open. Since there is no option in GDAL
        # to close datasets, this method is simpler and therefore preferred.
        dataset: gdal.Dataset = gdal.Open(os.fspath(self.filepath), gdal.GA_ReadOnly)
        band_obj: gdal.Band = dataset.GetRasterBand(self.band)
        return band_obj.ReadAsArray(
            xoff=start_x, yoff=start_y, win_xsize=width, win_ysize=length
        ).astype(self.dtype, copy=False)

    def __setitem__(self, key: tuple[slice, slice], value: np.ndarray, /) -> None:
        error_channel: Error = journal.error("GDALRaster.__setitem__")
        if len(key) != 2:
            err_str = (
                f"Key given with {len(key)} slice(s). GDALRaster only supports "
                "two-dimensional keys."
            )
            error_channel.log(err_str)
            raise ValueError(err_str)
        
        if value.dtype != self.dtype:
            err_str = (
                f"GDAL raster with dtype {self.dtype}: Attempted to write data with "
                f"dtype {value.dtype}."
            )
            error_channel.log(err_str)
            raise TypeError(err_str)
        
        slice_y, slice_x = key

        start_y, stop_y, step_y = slice_y.indices(self.length)
        start_x, stop_x, step_x = slice_x.indices(self.width)

        if (step_y != 1) or (step_x != 1):
            err_str = "GDALRaster does not support strided slicing."
            error_channel.log(err_str)
            raise ValueError(err_str)

        slice_shape = (stop_y - start_y, stop_x - start_x)
        if not slice_shape == value.shape:
            raise ValueError(
                f"could not broadcast input array from shape {value.shape} into shape "
                f"{slice_shape}"
            )

        # Benchmarking shows no detectable change in runtime if datasets are opened
        # repeatedly instead of being kept open. Since there is no option in GDAL
        # to close datasets, this method is simpler and therefore preferred.
        dataset: gdal.Dataset = gdal.Open(os.fspath(self.filepath), gdal.GA_Update)
        band_obj: gdal.Band = dataset.GetRasterBand(self.band)
        band_obj.WriteArray(array=value, xoff=start_x, yoff=start_y)

        # Forces the GDAL dataset to write to disk, although this is not guaranteed
        # (see https://gdal.org/api/python_gotchas.html#certain-objects-contain-a-destroy-method-but-you-should-never-use-it).
        dataset.FlushCache()
