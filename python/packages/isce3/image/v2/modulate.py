from __future__ import annotations

import journal
import numpy as np

from isce3.core import LUT2d
from isce3.ext.isce3.image.v2 import (
    _get_modulation_phase,
    _get_modulation_phase_at_coords,
    _modulate,
    _modulate_at_coords,
)
from isce3.product import RadarGridParameters


def modulate(
    slc_data_block: np.ndarray[np.complex64],
    carrier_phase: LUT2d,
    radar_grid: RadarGridParameters,
    input_azimuth_first_line: int,
    input_range_first_pixel: int,
    conjugate: bool = False,
    out: np.ndarray[np.complex64] | None = None,
) -> np.ndarray[np.complex64]:
    out_array = out if out is not None else np.copy(slc_data_block)

    _modulate(
        slc_data_block=out_array,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        input_azimuth_first_line=input_azimuth_first_line,
        input_range_first_pixel=input_range_first_pixel,
        conjugate=conjugate,
    )
    
    return out_array


def modulate_at_coords(
    slc_data_block: np.ndarray[np.complex64],
    carrier_phase: LUT2d,
    radar_grid: RadarGridParameters,
    azimuth_indices: np.ndarray[np.float64],
    range_indices: np.ndarray[np.float64],
    conjugate: bool = False,
    out: np.ndarray[np.complex64] | None = None,
) -> np.ndarray[np.complex64]:
    error_channel = journal.error("modulate.modulate_at_coords")
    out_array = out if out is not None else np.copy(slc_data_block)

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
    )
    
    return out_array


def get_modulation_phase(
    carrier_phase: LUT2d,
    radar_grid: RadarGridParameters,
    input_azimuth_first_line: int,
    input_range_first_pixel: int,
    conjugate: bool = False,
    out: np.ndarray[np.complex64] | None = None,
) -> np.ndarray[np.complex64]:
    out_array = out if out is not None else np.full(
        (radar_grid.length, radar_grid.width),
        fill_value=np.nan + 1.0j * np.nan,
        dtype=np.complex64,
    )

    _get_modulation_phase(
        out=out_array,
        carrier_phase=carrier_phase,
        radar_grid=radar_grid,
        input_azimuth_first_line=input_azimuth_first_line,
        input_range_first_pixel=input_range_first_pixel,
        conjugate=conjugate,
    )
    
    return out_array


def get_modulation_phase_at_coords(
    carrier_phase: LUT2d,
    radar_grid: RadarGridParameters,
    azimuth_indices: np.ndarray[np.float64],
    range_indices: np.ndarray[np.float64],
    conjugate: bool = False,
    out: np.ndarray[np.complex64] | None = None,
) -> np.ndarray[np.complex64]:
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
    )
    
    return out_array
