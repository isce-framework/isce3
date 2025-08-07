from __future__ import annotations

import journal
import numpy as np
from isce3.core import LUT2d
from isce3.ext.isce3.cuda.image.v2 import _gpu_resample_to_coords
from isce3.product import RadarGridParameters


def gpu_resample_to_coords(
    input_data_block: np.ndarray,
    range_input_indices: np.ndarray,
    azimuth_input_indices: np.ndarray,
    input_radar_grid: RadarGridParameters,
    native_doppler: LUT2d,
    fill_value: np.complex64 = (np.nan + 1.0j * np.nan),
) -> np.ndarray:
    """
    Interpolate input SLC block into the index values of the output block.

    Parameters
    ----------
    input_data_block : numpy.ndarray (complex64)
        Input SLC basebanded in range direction.
    range_input_indices : numpy.ndarray (float64)
        The range (radar-coordinates x) index of the output pixels in the input
        grid.
    azimuth_input_indices : numpy.ndarray (float64)
        The azimuth (radar-coordinates y) index of the output pixels in the input
        grid.
    input_radar_grid : isce3.product.RadarGridParameters
        Radar grid parameters of the input SLC data.
    native_doppler : isce3.core.LUT2d
        2D LUT describing the native doppler of the input SLC image, in Hz.
    fill_value : complex
        The value to fill out-of-bounds pixels with. Defaults to NaN + j*NaN.

    Returns
    -------
    resampled_block : numpy.ndarray (complex64)
        The resampled data.
    """
    error_channel = journal.error("resample_slc.gpu_resample_to_coords")

    # First, check that the indices arrays have equal shapes.
    if azimuth_input_indices.shape != range_input_indices.shape:
        err_log = (
            "Azimuth indices block and range indices block must "
            "be the same shape. Shapes: "
            f"Azimuth indices: {azimuth_input_indices.shape} "
            f"Range indices: {range_input_indices.shape}"
        )
        error_channel.log(err_log)
        raise ValueError(err_log)

    # Ensure that all of the input data blocks meet the requirements of the
    # _gpu_resample_to_coords pybind (correct dtype, with flags C_CONTIGUOUS and
    # WRITABLE) These function calls will return conforming copies of the data blocks if
    # they are not already conforming.
    input_data_block = np.require(
        input_data_block, dtype=np.complex64, requirements=["C"]
    )
    range_input_indices = np.require(
        range_input_indices, dtype=np.float64, requirements=["C"]
    )
    azimuth_input_indices = np.require(
        azimuth_input_indices, dtype=np.float64, requirements=["C"]
    )
    
    # The shape of the output block is the same as that of the indices shapes.
    output_block = np.full(
        range_input_indices.shape, fill_value=fill_value, dtype=np.complex64
    )

    _gpu_resample_to_coords(
        output_data_block=output_block,
        input_data_block=input_data_block,
        range_input_indices=range_input_indices,
        azimuth_input_indices=azimuth_input_indices,
        in_radar_grid=input_radar_grid,
        native_doppler=native_doppler,
        fill_value=fill_value,
    )

    return output_block
