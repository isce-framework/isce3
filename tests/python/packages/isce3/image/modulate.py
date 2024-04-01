"""Test resample_slc ver. 2"""
from __future__ import annotations

import pytest
import numpy as np

from isce3.image.modulate import (
    get_modulation_phase,
    get_modulation_phase_at_coords,
    modulate,
    modulate_at_coords,
)
from isce3.core import DateTime, LUT2d
from isce3.product import RadarGridParameters

from .resample_slc_utils import (
    generate_doppler_ramp_complex,
    validate_test_results,
)


def generate_carrier_lut(
    grid_params: RadarGridParameters, frequency: float
) -> LUT2d:
    """
    Create a constant carrier LUT.

    Parameters
    ----------
    grid_params : RadarGridParameters
        The radar grid parameters to evaluate azimuth and range with.
    frequency : float
        The frequency of the LUT to be created.

    Returns
    -------
    LUT2d
        The generated LUT.
    """
    # Trivially, for zero doppler just return an LUT2d that always evaluates to 0.
    if frequency == 0:
        return LUT2d()

    array_length = grid_params.length
    array_width = grid_params.width
    az_indices = np.arange(array_length)
    rg_indices = np.arange(array_width)

    az_time = grid_params.sensing_start + az_indices / grid_params.prf
    rg_dist = grid_params.starting_range + rg_indices / grid_params.range_pixel_spacing

    # Evaluate the integral 
    lut_array = np.full(
        shape=(array_length, array_width),
        fill_value=2 * np.pi * frequency,
        dtype=np.float64,
    ) * az_time[:, np.newaxis]

    lut = LUT2d(
        rg_dist,
        az_time,
        lut_array,
    )

    return lut


@pytest.mark.parametrize(
    "function",
    [
        "get_modulation_phase",
        "get_modulation_phase_at_coords",
        "modulate",
        "modulate_at_coords"
    ]
)
class TestModulate:
    """Tests for the image.modulate code."""

    @pytest.mark.parametrize("frequency", [0.1, 0.25, 0.5])
    @pytest.mark.parametrize("conjugate", [True, False])
    def test_modulation_constant_frequency(
        self, frequency: float, function: str, conjugate: bool
    ) -> tuple[np.ndarray[np.complex64], np.ndarray[np.complex64]]:
        """
        Tests the Resample SLC V2 on a randomly distributed target signal.

        This test is designed to assess if Resample SLC works for a reasonably educated
        attempt at a realistic input signal. Fail cases caught by this test include all
        those tested for in the sinusoidal test, but using a more rigorous and
        complicated secondary signal.

        This test also tests the zero-doppler case as well as a set of several doppler
        frequencies as provided by the frequency fixture.

        Fixtures
        ----------
        frequency : float
            A doppler frequency for this test, in Hz.
        """
        # The size of the generated image.
        az_length = 100
        rg_width = 100
        out_shape = (az_length, rg_width)

        # Set up a dummy radar grid
        radar_grid: RadarGridParameters = RadarGridParameters(
            sensing_start=1,
            wavelength=1,
            prf=1,
            starting_range=1,
            range_pixel_spacing=1,
            look_side="right",
            length=az_length,
            width=rg_width,
            ref_epoch=DateTime(),
        )

        # Generate a doppler centroid that has a ramp
        lut: LUT2d = generate_carrier_lut(
            grid_params=radar_grid,
            frequency=frequency,
        )

        # This will hold the output of the function
        signal = np.full(
            (az_length, rg_width),
            fill_value=1. + 0.j,
            dtype=np.complex64,
        )

        # This will hold the actual expected value
        doppler_ramp_complex: np.ndarray[np.complex64]

        # Perform the interpolation.
        if function in ["get_modulation_phase", "modulate"]:
            
            # Create the expected output signal
            doppler_ramp_complex = signal * generate_doppler_ramp_complex(
                grid_params=radar_grid,
                az_indices=np.arange(az_length),
                doppler_frequency=frequency,
            )[:, np.newaxis]

            # Run the selected function
            if function == "get_modulation_phase":
                signal = get_modulation_phase(
                    carrier_phase=lut,
                    radar_grid=radar_grid,
                    conjugate=conjugate,
                )

            elif function == "modulate":
                signal = modulate(
                    slc_data_block=signal,
                    carrier_phase=lut,
                    radar_grid=radar_grid,
                    conjugate=conjugate,
                )

        elif function in ["get_modulation_phase_at_coords", "modulate_at_coords"]:
            # Set the offsets at random positions with a range of -1.5 to 1.5 with a
            # flat probability distribution. This ensures that the difference in phase
            # and potential edge effects near the ends of an image are detectable.
            mag_offset = 3
            az_offsets = np.random.random(out_shape) * mag_offset - mag_offset / 2
            rg_offsets = np.random.random(out_shape) * mag_offset - mag_offset / 2

            # Add the offsets to these indices to get the indices to evaluate at.
            rows, cols = np.indices(out_shape)
            azimuth_indices = np.array(az_offsets + rows, dtype=np.float64)
            range_indices = np.array(rg_offsets + cols, dtype=np.float64)

            # Create the expected output signal
            doppler_ramp_complex = generate_doppler_ramp_complex(
                grid_params=radar_grid,
                az_indices=azimuth_indices,
                doppler_frequency=frequency,
            )

            # Run the selected function
            if function == "get_modulation_phase_at_coords":
                signal = get_modulation_phase_at_coords(
                    carrier_phase=lut,
                    radar_grid=radar_grid,
                    azimuth_indices=azimuth_indices,
                    range_indices=range_indices,
                    conjugate=conjugate,
                )
            elif function == "modulate_at_coords":
                signal = modulate_at_coords(
                    slc_data_block=signal,
                    carrier_phase=lut,
                    radar_grid=radar_grid,
                    azimuth_indices=azimuth_indices,
                    range_indices=range_indices,
                    conjugate=conjugate,
                )
        if conjugate:
            doppler_ramp_complex = np.conjugate(doppler_ramp_complex)

        try:
            # Validate the generated data against the true data.
            validate_test_results(
                test_arr=signal,
                true_arr=doppler_ramp_complex,
                correlation_min=0.99999,
                phase_stdev_max=1e-6,
                nan_percent_max=3,
                buffer_size_az=2,
                buffer_size_rg=2,
                az_offset=0,
                rg_offset=0,
            )
        except AssertionError as err:
            err.add_note(function)
            err.add_note(f"frequency: {frequency}")
            raise err
