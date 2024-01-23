from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class DatasetReader(Protocol):
    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: tuple[slice, ...], /) -> np.ndarray:
        """
        Reads a slice of the dataset and returns the list of arrays associated with it.

        Parameters
        ----------
        slice : tuple[slice, ...]
            The index expression describing the slice.

        Returns
        -------
        np.ndarray
            The data within that slice on the dataset.
        """
        ...


@runtime_checkable
class DatasetWriter(Protocol):
    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __setitem__(self, key: tuple[slice, ...], value: np.ndarray, /) -> None:
        """
        Writes the data at value to the key slice on the dataset.

        Parameters
        ----------
        slice : tuple[slice, ...]
            The index expression describing the slice.
        value : np.ndarray
            The data to write.
        """
        ...
