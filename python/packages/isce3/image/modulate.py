from __future__ import annotations

import journal
import numpy as np

from isce3.core import LUT2d
from isce3.core.poly2d import Poly2d
from isce3.ext.isce3.image.modulate import (
    _get_modulation_phase,
    _get_modulation_phase_at_coords,
    _modulate,
    _modulate_at_coords,
)
from isce3.product import RadarGridParameters


def modulate(
    slc_data_block: np.ndarray[np.complex64],
    carrier_phase: LUT2d | Poly2d,
    radar_grid: RadarGridParameters,
    conjugate: bool = False,
    fill_value: np.complex64 = np.nan + 1.j * np.nan,
) -> np.ndarray[np.complex64]:
    """
    Evaluate and modulate or demodulate the phase carrier onto the given SLC data block.

    Parameters
    ----------
    slc_data_block : np.ndarray of np.complex64
        The block of SLC data to modulate.
    carrier_phase : LUT2d or Poly2d
        Carrier phase of the SLC data, in radian, as a function of azimuth and range.
        This phase will be modulated to or demodulated from the image.
    radar_grid : RadarGridParameters
        Parameters for the given radar grid corresponding to `slc_data_block`.
    conjugate : bool, optional
        If True, modulate the conjugate of the phase. Defaults to False.
    fill_value: complex
        The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined
        here as any pixels that cannot be evaluated by the carrier_phase function.
        Poly2d functions do not have out-of-bounds pixels, but LUT2d functions may.
        Defaults to NaN + j*NaN.

    Returns
    -------
    np.ndarray of np.complex64
        The modulated SLC block.
    """
    out_arr = slc_data_block.copy()
    out_arr = np.require(out_arr, dtype=np.complex64, requirements=["C", "W"])

    _modulate(
        slc_data_block=out_arr,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_arr


def modulate_at_coords(
    slc_data_block: np.ndarray[np.complex64],
    carrier_phase: LUT2d | Poly2d,
    radar_grid: RadarGridParameters,
    azimuth_indices: np.ndarray[np.float64],
    range_indices: np.ndarray[np.float64],
    conjugate: bool = False,
    fill_value: np.complex64 = np.nan + 1.j * np.nan,
) -> np.ndarray[np.complex64]:
    """
    Evaluate and modulate or demodulate the phase carrier onto the given SLC data block
    at the given indices.

    Parameters
    ----------
    slc_data_block : np.ndarray of np.complex64
        The block of SLC data to modulate.
    carrier_phase : LUT2d or Poly2d
        Carrier phase of the SLC data, in radian, as a function of azimuth and range.
        This phase will be modulated to or demodulated from the image.
    radar_grid : RadarGridParameters
        Parameters for the given radar grid corresponding to `slc_data_block`.
    azimuth_indices : np.ndarray of np.float64
        Azimuth index of each output coordinate pixel in the given radar coordinate
        system. Must be the same shape as phase_data_block.
    range_indices : np.ndarray of np.float64
        Range index of each output coordinate pixel in the given radar coordinate
        system. Must be the same shape as phase_data_block.
    conjugate : bool, optional
        If True, modulate the conjugate of the phase. Defaults to False.
    fill_value: complex
        The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined
        here as any pixels that cannot be evaluated by the carrier_phase function.
        Poly2d functions do not have out-of-bounds pixels, but LUT2d functions may.
        Defaults to NaN + j*NaN.

    Returns
    -------
    np.ndarray of np.complex64
        The modulated SLC block.
    """
    error_channel = journal.error("modulate.modulate_at_coords")

    if azimuth_indices.shape != range_indices.shape:
        err_log = (
            f"Azimuth indices block shape {azimuth_indices.shape} and range indices "
            f"block shape {range_indices.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    if azimuth_indices.shape != slc_data_block.shape:
        err_log = (
            f"Indices block shapes {azimuth_indices.shape} and SLC data block shape "
            f"{slc_data_block.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    out_arr = slc_data_block.copy()
    out_arr = np.require(out_arr, dtype=np.complex64, requirements=["C", "W"])

    # Ensure that all of the index data blocks meet the requirements of the
    # _modulate_at_coords pybind (correct dtype, with flag C_CONTIGUOUS)
    # These function calls will return conforming copies of the data blocks if they
    # are not already conforming.
    range_indices = np.require(range_indices, dtype=np.float64, requirements=["C"])
    azimuth_indices = np.require(azimuth_indices, dtype=np.float64, requirements=["C"])

    _modulate_at_coords(
        slc_data_block=out_arr,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        azimuth_indices=azimuth_indices,
        range_indices=range_indices,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_arr


def get_modulation_phase(
    carrier_phase: LUT2d | Poly2d,
    radar_grid: RadarGridParameters,
    conjugate: bool = False,
    fill_value: np.complex64 = np.nan + 1.j * np.nan,
) -> np.ndarray[np.complex64]:
    """
    Acquire the phase of the given carrier of a radar scene.

    Parameters
    ----------
    carrier_phase : LUT2d or Poly2d
        Carrier phase, in radian, as a function of azimuth and range.
    radar_grid : RadarGridParameters
        Parameters for the given radar grid corresponding to the output block.
    conjugate : bool, optional
        If True, get the conjugate of the phase. Defaults to False.
    fill_value: complex
        The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined
        here as any pixels that cannot be evaluated by the carrier_phase function.
        Poly2d functions do not have out-of-bounds pixels, but LUT2d functions may.
        Defaults to NaN + j*NaN.

    Returns
    -------
    np.ndarray of np.complex64
        The carrier phase, in the form of complex unit vectors.
    """
    out_arr = np.full(
        (radar_grid.length, radar_grid.width),
        fill_value=fill_value,
        dtype=np.complex64,
    )

    _get_modulation_phase(
        out=out_arr,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_arr


def get_modulation_phase_at_coords(
    carrier_phase: LUT2d | Poly2d,
    radar_grid: RadarGridParameters,
    azimuth_indices: np.ndarray[np.float64],
    range_indices: np.ndarray[np.float64],
    conjugate: bool = False,
    fill_value: np.complex64 = np.nan + 1.j * np.nan,
) -> np.ndarray[np.complex64]:
    """
    Acquire the phase of the given carrier at each given index of a radar scene.

    Parameters
    ----------
    carrier_phase : LUT2d or Poly2d
        Carrier phase, in radian, as a function of azimuth and range.
    radar_grid : RadarGridParameters
        Parameters for the radar grid corresponding to the output block.
    azimuth_indices : np.ndarray of np.float64
        Azimuth index of each output coordinate pixel in the given radar coordinate
        system. Must be the same shape as phase_data_block.
    range_indices : np.ndarray of np.float64
        Range index of each output coordinate pixel in the given radar coordinate
        system. Must be the same shape as phase_data_block.
    conjugate : bool, optional
        If True, get the conjugate of the phase. Defaults to False.
    fill_value: complex
        The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined
        here as any pixels that cannot be evaluated by the carrier_phase function.
        Poly2d functions do not have out-of-bounds pixels, but LUT2d functions may.
        Defaults to NaN + j*NaN.

    Returns
    -------
    np.ndarray of np.complex64
        The carrier phase, in the form of complex unit vectors.
    """
    error_channel = journal.error("modulate.get_modulation_phase_at_coords")

    if azimuth_indices.shape != range_indices.shape:
        err_log = (
            f"Azimuth indices block shape {azimuth_indices.shape} and range indices "
            f"block shape {range_indices.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    out_arr = np.full(azimuth_indices.shape, fill_value=fill_value, dtype=np.complex64)

    # Ensure that all of the index data blocks meet the requirements of the
    # _get_modulation_phase_at_coords pybind (correct dtype, with flag C_CONTIGUOUS)
    # These function calls will return conforming copies of the data blocks if they
    # are not already conforming.
    range_indices = np.require(range_indices, dtype=np.float64, requirements=["C"])
    azimuth_indices = np.require(azimuth_indices, dtype=np.float64, requirements=["C"])

    _get_modulation_phase_at_coords(
        out=out_arr,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        azimuth_indices=azimuth_indices,
        range_indices=range_indices,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_arr
