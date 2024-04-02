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


def generate_carrier_ramp_complex(
    grid_params: RadarGridParameters,
    az_indices: np.ndarray,
    rg_indices: np.ndarray,
    az_frequency: float,
    rg_frequency: float,
) -> np.ndarray:
    """
    Evaluate the carrier phase ramp at the given set of indices.

    Parameters
    ----------
    grid_params : RadarGridParameters
        The radar grid parameters object for the doppler ramp.
    az_indices : np.ndarray
        The indices to evaluate the carrier at.
    rg_indices : np.ndarray
        The indices to evaluate the carrier at.
    az_frequency : float
        The azimuth carrier frequency.
    rg_frequency : float
        The range carrier frequency.

    Returns
    -------
    np.ndarray
        An array of the carrier phase ramp evaluated at each index passed in.
    """
    # Trivially, for zero-doppler, just return an array of 1+0j
    if az_frequency == 0 or rg_frequency == 0:
        return np.ones(az_indices.shape, dtype=np.complex64)

    # Get the absolute azimuth time at each index.
    azimuth = grid_params.sensing_start + az_indices / grid_params.prf
    range = grid_params.starting_range + rg_indices * grid_params.range_pixel_spacing

    # Evaluate and return the ramp.
    az_ramp = np.exp(1.0j * 2 * np.pi * az_frequency * azimuth)
    rg_ramp = np.exp(1.0j * 2 * np.pi * rg_frequency * range)

    return az_ramp * rg_ramp


def generate_carrier_lut(
    grid_params: RadarGridParameters, az_frequency: float, rg_frequency: float
) -> LUT2d:
    """
    Create a constant carrier LUT.

    Parameters
    ----------
    grid_params : RadarGridParameters
        The radar grid parameters to evaluate azimuth and range with.
    az_frequency : float
        The azimuth carrier frequency of the LUT to be created.
    rg_frequency : float
        The range carrier frequency of the LUT to be created.

    Returns
    -------
    LUT2d
        The generated LUT.
    """

    # Get the dimensions and indices of the array
    array_length = grid_params.length
    array_width = grid_params.width
    az_indices = np.arange(array_length)
    rg_indices = np.arange(array_width)

    # Get the azimuth times and range distances
    azimuth = grid_params.sensing_start + az_indices / grid_params.prf
    range = grid_params.starting_range + rg_indices * grid_params.range_pixel_spacing

    # Get the LUT array by evaluating the array at each index
    az_array = np.full(
        shape=(array_length, array_width),
        fill_value=2 * np.pi * az_frequency,
        dtype=np.float64,
    ) * azimuth[:, np.newaxis]
    rg_array = np.full(
        shape=(array_length, array_width),
        fill_value=2 * np.pi * rg_frequency,
        dtype=np.float64,
    ) * range[np.newaxis, :]
    lut_array = az_array + rg_array

    # Generate the LUT
    return LUT2d(range, azimuth, lut_array)


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

    @pytest.mark.parametrize("az_freq", [0.1, 0.25, 0.5])
    @pytest.mark.parametrize("rg_freq", [0.1, 0.25, 0.5])
    @pytest.mark.parametrize("conjugate", [True, False])
    def test_modulation_constant_frequency(
        self,
        az_freq: float,
        rg_freq: float,
        function: str,
        conjugate: bool,
    ) -> tuple[np.ndarray[np.complex64], np.ndarray[np.complex64]]:
        """
        Tests the four carrier phase acquisition/modulation functions.

        Fixtures
        --------
        az_freq : float
            The azimuth carrier frequency for this test, in Hz.
        rg_freq : float
            The azimuth carrier frequency for this test, in Hz.
        function : str
            The name of the function to test.
        conjugate : bool
            If True, conjugate the output signal; else do not.
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
            az_frequency=az_freq,
            rg_frequency=rg_freq,
        )

        # All of the functions call for a radar grid, carrier phase, and conjugate bool.
        # Begin putting together a set of keyword arguments, since the function
        # signatures are all very similar.
        signal = np.full(
            (az_length, rg_width),
            fill_value=1. + 0.j,
            dtype=np.complex64,
        )

        carrier_ramp_complex: np.ndarray[np.complex64]

        kwargs = {
            "radar_grid": radar_grid,
            "carrier_phase": lut,
            "conjugate": conjugate
        }

        # Perform the interpolation.
        if function in ["get_modulation_phase", "modulate"]:
            az_indices, rg_indices = np.indices((az_length, rg_width), dtype=np.float64)
            
            # Create the expected output signal. For this test, a simple phase ramp in
            # complex phasor format is the output.
            carrier_ramp_complex = generate_carrier_ramp_complex(
                grid_params=radar_grid,
                az_indices=az_indices,
                rg_indices=rg_indices,
                az_frequency=az_freq,
                rg_frequency=rg_freq,
            )

            if function == "get_modulation_phase":
                signal = get_modulation_phase(**kwargs)
            elif function == "modulate":
                # A dummy SLC of 1 + 0j to modulate - this will give the carrier
                # phase.
                input_slc = np.full(
                    (az_length, rg_width),
                    fill_value=1. + 0.j,
                    dtype=np.complex64,
                )
                kwargs["slc_data_block"] = input_slc
                signal = modulate(**kwargs)

        elif function in ["get_modulation_phase_at_coords", "modulate_at_coords"]:
            # Set the offsets at random positions with a range of -1.5 to 1.5 with a
            # flat probability distribution. This ensures that the difference in phase
            # and potential edge effects near the ends of an image are detectable.
            mag_offset = 3
            az_offsets = np.random.random(out_shape) * mag_offset - mag_offset/2
            rg_offsets = np.random.random(out_shape) * mag_offset - mag_offset/2

            # Add the offsets to these indices to get the indices in the ground truth grid.
            rows, cols = np.indices(out_shape)
            azimuth_indices = np.array(az_offsets + rows, dtype=np.float64)
            range_indices = np.array(rg_offsets + cols, dtype=np.float64)

            # Create the expected output signal. For this test, a simple phase ramp in
            # complex phasor format is the output.
            carrier_ramp_complex = generate_carrier_ramp_complex(
                grid_params=radar_grid,
                az_indices=azimuth_indices,
                rg_indices=range_indices,
                az_frequency=az_freq,
                rg_frequency=rg_freq,
            )

            kwargs["azimuth_indices"] = azimuth_indices
            kwargs["range_indices"] = range_indices

            if function == "get_modulation_phase_at_coords":
                signal = get_modulation_phase_at_coords(**kwargs)
            elif function == "modulate_at_coords":
                # A dummy SLC of 1 + 0j to modulate - this will give the carrier
                # phase.
                input_slc = np.full(
                    (az_length, rg_width),
                    fill_value=1. + 0.j,
                    dtype=np.complex64,
                )
                kwargs["slc_data_block"] = input_slc
                signal = modulate_at_coords(**kwargs)

        # If conjugate is True, the ground-truth array is currently the conjugate
        # of the output array (we hope) and must be conjugated for validation.
        if conjugate:
            carrier_ramp_complex = np.conjugate(carrier_ramp_complex)
        try:
            # Validate the generated data against the true data.
            # The correlation is expected to be very high and the standard deviation
            # very low. The percentage of NaN values will be variable depending on
            # offset distance for the "at_coords" functions and zero for the others.
            validate_test_results(
                test_arr=signal,
                true_arr=carrier_ramp_complex,
                correlation_min=0.99999,
                phase_stdev_max=1e-6,
                nan_percent_max=3,
                buffer_size_az=2,
                buffer_size_rg=2,
                az_offset=0,
                rg_offset=0,
            )
        except AssertionError as err:
            # If an error is caught in the validation function, add some clarifying
            # notes for readability.
            err.add_note(f"function: {function}")
            err.add_note(f"az_freq: {az_freq}")
            err.add_note(f"rg_freq: {rg_freq}")
            err.add_note(f"conjugate: {conjugate}")
            raise err
