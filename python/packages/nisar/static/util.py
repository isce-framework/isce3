from __future__ import annotations

import itertools
import os
import shutil
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp, mkstemp

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from osgeo import gdal, gdal_array

import isce3

gdal.UseExceptions()


def truncate_datetime_to_integer_seconds(t: isce3.core.DateTime) -> isce3.core.DateTime:
    """
    Truncate a datetime object to integer seconds.

    Parameters
    ----------
    t : isce3.core.DateTime
        The input datetime object.

    Returns
    -------
    isce3.core.DateTime
        A copy of the input datetime with fractional seconds component set to zero.
    """
    return isce3.core.DateTime(
        year=t.year,
        month=t.month,
        day=t.day,
        hour=t.hour,
        minute=t.minute,
        second=t.second,
        frac=0.0,
    )


def isoformat_integer_seconds(t: datetime) -> str:
    """
    Serialize the input datetime object to a string in ISO 8601 format.

    Only datetimes with integer seconds precision are supported.

    Parameters
    ----------
    t : datetime.datetime
        The input datetime object. Must not have a sub-second component.

    Returns
    -------
    str
        A string representation of the input datetime in ISO 8601 format.

    Raises
    ------
    ValueError
        If the input datetime object has a nonzero microsecond component.
    """
    if t.microsecond != 0:
        raise ValueError(
            "the input datetime must have integer seconds precision with no sub-second"
            f" component, got {t}"
        )
    return t.isoformat()[:19]


def look_side_to_str(look_side: isce3.core.LookSide | str) -> str:
    """
    Serialize the input look side to a string.

    Convert `look_side` to a string in the format expected by NISAR HDF5 products
    ('Left' or 'Right').

    Parameters
    ----------
    look_side : isce3.core.LookSide
        The input look side.

    Returns
    -------
    str
        A string representation of the input look side.
    """
    look_side = isce3.core.normalize_look_side(look_side)

    if look_side == isce3.core.LookSide.Left:
        return "Left"
    if look_side == isce3.core.LookSide.Right:
        return "Right"

    # Should be unreachable.
    assert False, f"unexpected look_side {look_side}"


def parse_processing_type_code(code: str) -> str:
    """
    Infer the processing type from the mnemonic used in the run configuration file.

    Parameters
    ----------
    code : {'PR', 'OD'}
        The mnemonic using by the Static Layers run configuration file to describe the
        processing type.

    Returns
    -------
    str
        'Nominal' if the input `code` was 'PR'. 'Custom if the input code was 'OD'.
    """
    if code == "PR":
        return "Nominal"
    if code == "OD":
        return "Custom"

    raise ValueError(f"{code=}, must be one of {{'PR', 'OD'}}")


def get_reference_ellipsoid(raster: isce3.io.Raster) -> isce3.core.Ellipsoid:
    """
    Get the reference ellipsoid associated with the CRS of the input raster.

    Parameters
    ----------
    raster : isce3.io.Raster
        The input raster.

    Returns
    -------
    isce3.core.Ellipsoid
        The reference ellipsoid associated with the coordinate reference system (CRS) of
        the input raster.
    """
    epsg = raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    return proj.ellipsoid


@contextmanager
def scratch_directory(
    dir_: str | os.PathLike | None = None, *, delete: bool = True
) -> Generator[Path, None, None]:
    """
    Context manager that creates a (possibly temporary) file system directory.

    If `dir_` is a path-like object, a directory will be created at the specified
    file system path if it did not already exist. Otherwise, if `dir_` is None, a
    temporary directory will instead be created as though by ``tempfile.mkdtemp()``.

    If a directory was created this way, it may be automatically removed from the file
    system upon exiting the context manager, depending on the `delete` argument. If the
    directory already existed, it will not be removed.

    Parameters
    ----------
    dir_ : path-like or None, optional
        Scratch directory path. If None, a temporary directory will be created. Defaults
        to None.
    delete : bool, optional
        If True, the directory and its contents are recursively removed from the
        file system upon exiting the context manager. This parameter is ignored if the
        specified path was an existing directory. Defaults to True.

    Yields
    ------
    pathlib.Path
        Scratch directory path. If `delete` was True, the directory will be removed from
        the file system upon exiting the context manager scope.
    """
    if dir_ is None:
        scratchdir = Path(mkdtemp())
    else:
        scratchdir = Path(dir_)

        # If the directory already existed, don't delete it upon exiting the context
        # manager. Otherwise, create the directory.
        if scratchdir.exists():
            delete = False
        else:
            scratchdir.mkdir(parents=True)

    try:
        yield scratchdir
    finally:
        if delete:
            shutil.rmtree(scratchdir)


def make_scratch_file(
    *,
    dir_: os.PathLike | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
) -> Path:
    """
    Create a uniquely-named file.

    The file is readable and writeable only by the creating user's ID. The user is
    responsible for deleting the file when done with it.

    Parameters
    ----------
    dir_ : path-like or None, optional
        Directory in which to create the file. If None, a platform-specific default
        temporary directory will be used. Otherwise, it must be the file system path to
        an existing directory. Defaults to None.
    prefix : str or None, optional
        An optional prefix to begin the file name with. Defaults to no prefix.
    suffix : str or None, optional
        An optional suffix to end the file name with. Defaults to no suffix.

    Returns
    -------
    pathlib.Path
        The file path.
    """
    if dir_ is not None:
        dir_ = os.fsdecode(dir_)
    file, filename = mkstemp(dir=dir_, prefix=prefix, suffix=suffix)
    os.close(file)
    return Path(filename)


def create_single_band_gtiff(
    path: os.PathLike | str,
    shape: tuple[int, int],
    dtype: DTypeLike,
) -> isce3.io.Raster:
    """
    Create a single-band GeoTIFF raster dataset.

    Parameters
    ----------
    path : path-like
        The file path at which to create the raster file.
    shape : (int, int)
        The raster shape as a tuple of (length, width).
    dtype : data-type
        The datatype of the raster data. Must be a datatype that's supported by GDAL.

    Returns
    -------
    isce3.io.Raster
        The raster dataset.
    """
    path = os.fsdecode(path)
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(dtype))
    length, width = shape
    return isce3.io.Raster(
        path=path,
        width=width,
        length=length,
        num_bands=1,
        dtype=dtype,
        driver_name="GTiff",
    )


def make_scratch_gtiff(
    shape: tuple[int, int],
    dtype: DTypeLike,
    *,
    dir_: os.PathLike | str | None = None,
    prefix: str | None = None,
) -> isce3.io.Raster:
    """
    Create a uniquely-named single-band GeoTIFF raster dataset.

    Parameters
    ----------
    shape : (int, int)
        The raster shape as a tuple of (length, width).
    dtype : data-type
        The datatype of the raster data. Must be a datatype that's supported by GDAL.
    dir_ : path-like or None, optional
        Directory in which to create the file. If None, a platform-specific default
        temporary directory will be used. Otherwise, it must be the file system path to
        an existing directory. Defaults to None.
    prefix : str or None, optional
        An optional prefix to begin the file name with. Defaults to no prefix.
    """
    path = make_scratch_file(dir_=dir_, prefix=prefix, suffix=".tif")
    return create_single_band_gtiff(path, shape, dtype)


def get_raster_dataset_metadata_item(
    raster_file: os.PathLike | str, name: str, *, default: str
) -> str:
    """
    Extract the contents of a metadata attribute from a geospatial raster dataset.

    Parameters
    ----------
    raster_file : path-like or str
        The file path or name of the input raster dataset.
    name : str
        The name of the metadata item to access.
    default : str
        Default value to return if the metadata item does not exist.

    Returns
    -------
    str
        The contents of the metadata item, or `default` if the metadata item was not
        found.
    """
    dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
    item = dataset.GetMetadataItem(name)
    return default if (item is None) else item


def ceil_divide(n: ArrayLike, d: ArrayLike) -> np.generic:
    """
    Return the smallest integer greater than or equal to the quotient of the inputs.

    Computes integer division of dividend `n` by divisor `d`, rounding up instead of
    truncating.

    Parameters
    ----------
    n : array_like
        The numerator.
    d : array_like
        The denominator.

    Returns
    -------
    q : numpy.generic
        The quotient, rounded up to the next integer.
    """
    return np.ceil(np.divide(n, d)).astype(np.int_, copy=False)


def block_iterator(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> Iterator[tuple[slice, ...]]:
    """
    Iterate over chunks of an N-dimensional array.

    Returns an iterator over regularly-sized non-overlapping blocks of a
    multidimensional array. Each block is represented by an index expression (i.e. a
    tuple of `slice` objects) that can be used to access the corresponding block of data
    from the array. The full set of blocks spans the entire array.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array to be partitioned into blocks. Each dimension must be
        positive-valued.
    chunks : tuple of int
        The shape of a typical block. The last block along each axis may be smaller.
        Each chunk dimension must be positive-valued.

    Yields
    ------
    tuple of slice
        A tuple of slices that can be used to access the corresponding block of data
        from an array.
    """
    if len(chunks) != len(shape):
        errmsg = (
            "size mismatch: shape and chunks must have the same number of elements,"
            f" instead got len(shape) != len(chunks) ({len(shape)} !="
            f" {len(chunks)})"
        )
        raise ValueError(errmsg)

    if not all(n > 0 for n in shape):
        errmsg = f"shape elements must all be > 0, instead got {shape}"
        raise ValueError(errmsg)
    if any(n <= 0 for n in chunks):
        errmsg = f"chunk elements must all be > 0, instead got {chunks}"
        raise ValueError(errmsg)

    # Number of blocks along each array axis.
    nblocks = ceil_divide(shape, chunks)

    # Iterate over blocks.
    for block_ind in itertools.product(*[range(n) for n in nblocks]):
        # Get the lower & upper index bounds for the current block.
        start = np.multiply(block_ind, chunks)
        stop = np.minimum(start + chunks, shape)

        # Yield a tuple of slice objects.
        yield tuple(itertools.starmap(slice, zip(start, stop)))


def unary_transform_blockwise(
    transform: Callable[[np.ndarray], np.ndarray],
    src: isce3.io.DatasetReader,
    dst: isce3.io.DatasetWriter,
    *,
    chunks: tuple[int, int] = (512, 512),
) -> None:
    """
    Transform the contents of a dataset by applying a unary function block-by-block.

    Parameters
    ----------
    transform : callable
        The function to apply to each block from the input dataset. It should accept a
        `numpy.ndarray` as a single positional argument and return a `numpy.ndarray` of
        the same shape.
    src : isce3.io.DatasetReader
        The input dataset to transform.
    dst : isce3.io.DatasetWriter
        The output dataset to write the result to.
    chunks : (int, int), optional
        The shape of a typical block. The last block along each axis may be smaller.
        Each chunk dimension must be positive-valued. Defaults to (512, 512).
    """
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch: {src.shape=} must be equal to {dst.shape=}")

    for subblock in block_iterator(src.shape, chunks):
        dst[subblock] = transform(src[subblock])


def copy_blockwise(
    src: isce3.io.DatasetReader,
    dst: isce3.io.DatasetWriter,
    *,
    chunks: tuple[int, int] = (512, 512),
) -> None:
    """
    Copy the contents of `src` to `dst` block-by-block.

    Parameters
    ----------
    src : isce3.io.DatasetReader
        The input dataset to read from.
    dst : isce3.io.DatasetWriter
        The output dataset to write to.
    chunks : (int, int), optional
        The shape of a typical block. The last block along each axis may be smaller.
        Each chunk dimension must be positive-valued. Defaults to (512, 512).
    """
    unary_transform_blockwise(lambda x: x, src, dst, chunks=chunks)
