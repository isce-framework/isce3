from __future__ import annotations

import journal
import numpy as np

from isce3.ext.isce3.image.flatten import (
    _get_flattening_phase_at_coords,
    _flatten_at_coords,
)
from isce3.product import RadarGridParameters


def flatten_at_coords(
    data_block: np.ndarray,
    range_indices: np.ndarray,
    radar_grid_out: RadarGridParameters,
    radar_grid_in: RadarGridParameters,
    out_rg_first_pixel: int,
) -> np.ndarray:
    """
    Re-flatten a grid of SLC data from its' original grid parameters into a new set
    of grid parameters.

    Parameters
    ----------
    data_block : np.ndarray of complex64
        The SLC data to flatten
    range_indices : np.ndarray of float64
        range index of each coordinate pixel in the data block in the coordinate
        system of the alternate radar grid
    radar_grid_out : isce3.product.RadarGridParameters
        radar grid parameters of the alternate grid
    radar_grid_in : isce3.product.RadarGridParameters
        radar grid parameters of the original grid
    out_rg_first_pixel : int
        range index of the first sample of the alternate grid

    Returns
    -------
    np.ndarray of np.complex64
        The flattened SLC block. If `out` was given, this will be the same array as
        the `out` array.
    """
    error_channel = journal.error("flatten.flatten_at_coords")

    if data_block.shape != range_indices.shape:
        err_log = (
            f"Data block shape {data_block.shape} and range indices block shape "
            f"{range_indices.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    out_array = data_block.copy()

    # Ensure that all of the input data blocks meet the requirements of the
    # _flatten_at_coords pybind (correct dtype, with flags C_CONTIGUOUS)
    # These function calls will return conforming copies of the data blocks if they
    # are not already conforming.
    out_array = np.require(out_array, dtype=np.complex64, requirements=["C"])
    range_indices = np.require(range_indices, dtype=np.float64, requirements=["C"])

    _flatten_at_coords(
        data_block=out_array,
        range_indices=range_indices,
        radar_grid_out=radar_grid_out,
        radar_grid_in=radar_grid_in,
        out_rg_first_pixel=out_rg_first_pixel,
    )
    
    return out_array


def get_flattening_phase_at_coords(
    range_indices: np.ndarray,
    radar_grid_out: RadarGridParameters,
    radar_grid_in: RadarGridParameters,
    out_rg_first_pixel: int,
) -> np.ndarray:
    """
    Acquire the phase necessary to flatten an SLC at each given index of a radar scene.

    Parameters
    ----------
    range_indices : np.ndarray of float64
        range index of each coordinate pixel in the data block in the coordinate
        system of the alternate radar grid
    radar_grid_out : isce3.product.RadarGridParameters
        radar grid parameters of the alternate grid
    radar_grid_in : isce3.product.RadarGridParameters
        radar grid parameters of the original grid
    out_rg_first_pixel : int
        range index of the first sample of the alternate grid

    Returns
    -------
    np.ndarray of np.complex64
        The flattened SLC block. If `out` was given, this will be the same array as
        the `out` array.
    """
    out_array = np.full(
        range_indices.shape,
        fill_value=np.nan + 1.0j * np.nan,
        dtype=np.complex64,
    )

    # Ensure that all of the input data blocks meet the requirements of the
    # _get_flattening_phase_at_coords pybind (correct dtype, with flags C_CONTIGUOUS)
    # These function calls will return conforming copies of the data blocks if they
    # are not already conforming.
    range_indices = np.require(range_indices, dtype=np.float64, requirements=["C"])

    _get_flattening_phase_at_coords(
        data_block=out_array,
        range_indices=range_indices,
        radar_grid_out=radar_grid_out,
        radar_grid_in=radar_grid_in,
        out_rg_first_pixel=out_rg_first_pixel,
    )
    
    return out_array
