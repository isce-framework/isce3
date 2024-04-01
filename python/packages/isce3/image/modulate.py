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
    out: np.ndarray[np.complex64] | None = None,
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
        If True, modulate the conjugate of the phase, by default False
    fill_value: complex
        The value to fill out-of-bounds pixels with. Defaults to NaN + j*NaN.
    out : np.ndarray of np.complex64 | None, optional
        The array to output data to, or None. If given, must be the same size as
        slc_data_block. Any contents of this array will he overwritten, by default None

    Returns
    -------
    np.ndarray of np.complex64
        The modulated SLC block. If `out` was given, this will be the same array as
        the `out` array.
    """
    out_array = out if out is not None else np.full(
        (radar_grid.length, radar_grid.width),
        fill_value=np.nan + 1.0j * np.nan,
        dtype=np.complex64,
    )
    np.copyto(out_array, slc_data_block)

    _modulate(
        slc_data_block=out_array,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_array


def modulate_at_coords(
    slc_data_block: np.ndarray[np.complex64],
    carrier_phase: LUT2d | Poly2d,
    radar_grid: RadarGridParameters,
    azimuth_indices: np.ndarray[np.float64],
    range_indices: np.ndarray[np.float64],
    conjugate: bool = False,
    fill_value: np.complex64 = np.nan + 1.j * np.nan,
    out: np.ndarray[np.complex64] | None = None,
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
        If True, modulate the conjugate of the phase, by default False
    fill_value: complex
        The value to fill out-of-bounds pixels with. Defaults to NaN + j*NaN.
    out : np.ndarray of np.complex64 | None, optional
        The array to output data to, or None. If given, must be the same size as
        slc_data_block. Any contents of this array will he overwritten, by default None

    Returns
    -------
    np.ndarray of np.complex64
        The modulated SLC block. If `out` was given, this will be the same array as
        the `out` array.
    """
    error_channel = journal.error("modulate.modulate_at_coords")
    
    out_array = out if out is not None else np.full(
        (radar_grid.length, radar_grid.width),
        fill_value=np.nan + 1.0j * np.nan,
        dtype=np.complex64,
    )
    np.copyto(out_array, slc_data_block)

    if out_array.shape != azimuth_indices.shape:
        err_log = (
            f"Output block shape {out_array.shape} and azimuth indices block shape "
            f"{azimuth_indices.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    if out_array.shape != range_indices.shape:
        err_log = (
            f"Output block shape {out_array.shape} and range indices block shape "
            f"{azimuth_indices.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    _modulate_at_coords(
        slc_data_block=out_array,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        azimuth_indices=azimuth_indices,
        range_indices=range_indices,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_array


def get_modulation_phase(
    carrier_phase: LUT2d | Poly2d,
    radar_grid: RadarGridParameters,
    conjugate: bool = False,
    fill_value: np.complex64 = np.nan + 1.j * np.nan,
    out: np.ndarray[np.complex64] | None = None,
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
        If True, get the conjugate of the phase, by default False
    fill_value: complex
        The value to fill out-of-bounds pixels with. Defaults to NaN + j*NaN.
    out : np.ndarray[np.complex64] | None, optional
        The output phase array to modify. Anything in this array will be overwritten.
        Defaults to None

    Returns
    -------
    np.ndarray of np.complex64
        The carrier phase, in the form of complex unit vectors. If `out` was given, this
        will be the same array as the `out` array.
    """
    out_array = out if out is not None else np.full(
        (radar_grid.length, radar_grid.width),
        fill_value=np.nan + 1.0j * np.nan,
        dtype=np.complex64,
    )

    _get_modulation_phase(
        out=out_array,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_array


def get_modulation_phase_at_coords(
    carrier_phase: LUT2d | Poly2d,
    radar_grid: RadarGridParameters,
    azimuth_indices: np.ndarray[np.float64],
    range_indices: np.ndarray[np.float64],
    conjugate: bool = False,
    fill_value: np.complex64 = np.nan + 1.j * np.nan,
    out: np.ndarray[np.complex64] | None = None,
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
        If True, get the conjugate of the phase, by default False
    fill_value: complex
        The value to fill out-of-bounds pixels with. Defaults to NaN + j*NaN.
    out : np.ndarray[np.complex64] | None, optional
        The output phase array to modify. Anything in this array will be overwritten.
        Defaults to None

    Returns
    -------
    np.ndarray of np.complex64
        The carrier phase, in the form of complex unit vectors. If `out` was given, this
        will be the same array as the `out` array.
    """
    error_channel = journal.error("modulate.get_modulation_phase_at_coords")
    out_array = out if out is not None else np.full(
        azimuth_indices.shape,
        fill_value=np.nan + 1.0j * np.nan,
        dtype=np.complex64,
    )

    if out_array.shape != azimuth_indices.shape:
        err_log = (
            f"Output block shape {out_array.shape} and azimuth indices block shape "
            f"{azimuth_indices.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    if out_array.shape != range_indices.shape:
        err_log = (
            f"Output block shape {out_array.shape} and range indices block shape "
            f"{azimuth_indices.shape} are unequal."
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    _get_modulation_phase_at_coords(
        out=out_array,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        azimuth_indices=azimuth_indices,
        range_indices=range_indices,
        conjugate=conjugate,
        fill_value=fill_value,
    )
    
    return out_array
