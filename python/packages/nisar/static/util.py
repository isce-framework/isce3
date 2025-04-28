from __future__ import annotations

import itertools
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from collections.abc import Callable, Generator, Iterable, Iterator

import numpy as np
from osgeo import gdal_array
from numpy.typing import DTypeLike, ArrayLike

import isce3


def get_reference_ellipsoid(raster: isce3.io.Raster) -> isce3.core.Ellipsoid:
    """ """
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
            delete = False  # FIXME: remove this
        else:
            scratchdir.mkdir(parents=True)

    yield scratchdir

    if delete:
        shutil.rmtree(scratchdir)


def make_scratch_file(
    *,
    dir_: os.PathLike | str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
) -> Path:
    """ """
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
    """ """
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
    prefix: str | None = None
) -> isce3.io.Raster:
# ) -> GDALRaster:
    """ """
    path = make_scratch_file(dir_=dir_, prefix=prefix, suffix=".tif")
    return create_single_band_gtiff(path, shape, dtype)


def as_tuple_of_int(ints: int | Iterable[int]) -> tuple[int, ...]:
    """
    Convert the input to a tuple of ints.

    Parameters
    ----------
    ints : int or iterable of int
        One or more integers.

    Returns
    -------
    out : tuple of int
        Tuple containing the input(s).
    """
    try:
        return (int(ints),)
    except TypeError:
        return tuple(int(i) for i in ints)


def ceil_divide(n: ArrayLike, d: ArrayLike) -> np.ndarray:
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
    q : numpy.ndarray
        The quotient, rounded up to the next integer.
    """
    n = np.asanyarray(n)
    d = np.asanyarray(d)
    return (n + d - np.sign(d)) // d


@dataclass(frozen=True)
class BlockIterator(Iterable[tuple[slice, ...]]):
    """
    An iterable over chunks of an N-dimensional array.

    `BlockIterator` represents a partitioning of a multidimensional array into
    regularly-sized non-overlapping blocks. Each block is represented by an index
    expression (i.e. a tuple of `slice` objects) that can be used to access the
    corresponding block of data from the partitioned array. The full set of blocks spans
    the entire array.

    Iterating over a `BlockIterator` object yields each block in unspecified order.

    Attributes
    ----------
    shape : tuple of int
        The shape of the array to be partitioned into blocks.
    chunks : tuple of int
        The shape of a typical block. The last block along each axis may be smaller.
    """

    shape: tuple[int, ...]
    chunks: tuple[int, ...]

    def __init__(self, shape: int | Iterable[int], chunks: int | Iterable[int]):
        """
        Construct a new `BlockIterator` object.

        Parameters
        ----------
        shape : int or iterable of int
            The shape of the array to be partitioned into blocks. Each dimension must be
            > 0.
        chunks : int or iterable of int
            The shape of a typical block. Must be the same length as `shape`. Each chunk
            dimension must be > 0.
        """
        # Normalize `shape` and `chunks` into tuples of ints.
        shape = as_tuple_of_int(shape)
        chunks = as_tuple_of_int(chunks)

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

        # XXX Workaround for `frozen=True`.
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "chunks", chunks)

    def __iter__(self) -> Iterator[tuple[slice, ...]]:
        """
        Iterate over blocks in unspecified order.

        Yields
        ------
        block : tuple of slice
            A tuple of slices that can be used to access the corresponding block of data
            from an array.
        """
        # Number of blocks along each array axis.
        nblocks = ceil_divide(self.shape, self.chunks)

        # Iterate over blocks.
        for block_ind in itertools.product(*[range(n) for n in nblocks]):
            # Get the lower & upper index bounds for the current block.
            start = np.multiply(block_ind, self.chunks)
            stop = np.minimum(start + self.chunks, self.shape)

            # Yield a tuple of slice objects.
            yield tuple(itertools.starmap(slice, zip(start, stop)))


def copy_blockwise(
    src: isce3.io.DatasetReader,
    dst: isce3.io.DatasetWriter,
    *,
    chunks: tuple[int, int] = (512, 512),
) -> None:
    """ """
    if src.shape != dst.shape:
        raise ValueError  # FIXME

    for subblock in BlockIterator(src.shape, chunks):
        dst[subblock] = src[subblock]


def unary_transform_blockwise(
    transform: Callable[[np.ndarray], np.ndarray],
    src: isce3.io.DatasetReader,
    dst: isce3.io.DatasetWriter,
    *,
    chunks: tuple[int, int] = (512, 512),
) -> None:
    """ """
    if src.shape != dst.shape:
        raise ValueError  # FIXME

    for subblock in BlockIterator(src.shape, chunks):
        dst[subblock] = transform(src[subblock])


def binary_transform_blockwise(
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
    src1: isce3.io.DatasetReader,
    src2: isce3.io.DatasetReader,
    dst: isce3.io.DatasetWriter,
    *,
    chunks: tuple[int, int] = (512, 512),
) -> None:
    """ """
    shape = src1.shape
    if src2.shape != shape:
        raise ValueError  # FIXME
    if dst.shape != shape:
        raise ValueError  # FIXME

    for subblock in BlockIterator(shape, chunks):
        dst[subblock] = transform(src1[subblock], src2[subblock])
