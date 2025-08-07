from __future__ import annotations

import numpy as np

from isce3.core import LUT2d
from isce3.ext.isce3.image.v2 import _resample_to_coords
from isce3.image.v2.resample_slc import resample_slc_blocks
from isce3.product import RadarGridParameters

def block_resample(
    output_slc: np.ndarray,
    input_slc: np.ndarray,
    az_offsets: np.ndarray,
    rg_offsets: np.ndarray,
    input_radar_grid: RadarGridParameters,
    doppler: LUT2d,
    fill_value: np.complex64,
) -> None:
    az_length, rg_width = output_slc.shape

    resample_slc_blocks(
        output_resampled_slcs=[output_slc],
        input_slcs=[input_slc],
        az_offsets_dataset=az_offsets,
        rg_offsets_dataset=rg_offsets,
        input_radar_grid=input_radar_grid,
        doppler=doppler,
        fill_value=fill_value,
        block_size_rg=rg_width // 2,
        block_size_az=az_length // 2,
        quiet=False,
    )


def pybind_resample(
    output_slc: np.ndarray,
    input_slc: np.ndarray,
    az_offsets: np.ndarray,
    rg_offsets: np.ndarray,
    input_radar_grid: RadarGridParameters,
    doppler: LUT2d,
    fill_value: np.complex64,
) -> None:
    rows, cols = np.indices(output_slc.shape)
    azimuth_indices = np.array(rows + az_offsets, dtype=np.float64)
    range_indices = np.array(cols + rg_offsets, dtype=np.float64)

    _resample_to_coords(
        output_slc,
        input_slc,
        range_indices,
        azimuth_indices,
        input_radar_grid,
        doppler,
        fill_value,
    )