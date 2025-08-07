from __future__ import annotations

import numpy as np

from isce3.core import LUT2d
from isce3.ext.isce3.cuda.image.v2 import _gpu_resample_to_coords
from isce3.product import RadarGridParameters


def gpu_pybind_resample(
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

    _gpu_resample_to_coords(
        output_slc,
        input_slc,
        range_indices,
        azimuth_indices,
        input_radar_grid,
        doppler,
        fill_value,
    )