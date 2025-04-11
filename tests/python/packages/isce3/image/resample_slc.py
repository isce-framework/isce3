"""Test resample_slc ver. 2"""
from __future__ import annotations

import pytest
import numpy as np

from isce3.ext.isce3.image.v2 import _resample_to_coords
from isce3.image.v2.resample_slc import resample_slc_blocks
from isce3.core import DateTime, LUT2d
from isce3.product import RadarGridParameters


from .resample_slc_utils import (
    generate_doppler_lut,
    generate_doppler_ramp_complex,
    generate_signal_sinusoidal,
    nan_complex,
    SignalGenerator,
    validate_test_results,
)


@pytest.mark.parametrize("test_pybind", [True, False])
class TestResampleSLCV2:
    """Tests for the Resample SLC Ver. 2 code."""

    def test_interpolate_sinusoidal(self, test_pybind: bool):
        """
        Tests the Resample SLC V2 on a simple sinusoidal signal.

        This test is designed to assess if Resample SLC works at all. Fail cases caught
        by this test include: Large numbers of NaN values in the output image, loss of
        correlation between the ground truth and output image, and large differences
        between the output image and the ground truth, as well as failure of the
        `resample_slc_blocks` function to complete operation.
        """
        # Scaling factor: Adjusts the period of the sinusoidal functions in both
        # azimuth and range. A large scaling factor corresponds to a short period.
        # Aliasing may become a problem for scaling factors outside of the range
        # (-pi, pi).
        # Unit: Radians per sample
        scaling_factor = 2 / np.pi

        # Offsets: Adjust the distance in azimuth and range at which the pixels are
        # offset.
        az_offset = 5.5
        rg_offset = 5.5

        # Azimuth length and range width: Adjust the dimensions of the produced image.
        az_length = 2048
        rg_width = 2048
        out_shape = (az_length, rg_width)

        # Secondary length and width: Adjust the dimensions of the secondary image from
        # which interpolation is performed.
        sec_shape = out_shape
        sec_length, sec_width = out_shape

        # The rows and columns are just the given indices in azimuth and range on the
        # secondary grid.
        rows, cols = np.indices(sec_shape)
        secondary = generate_signal_sinusoidal(
            rows=rows,
            cols=cols,
            scaling_factor=scaling_factor,
        )

        # Add the offsets to these indices to get the indices in the ground truth grid.
        azimuth_indices = np.array(rows + az_offset, dtype=np.float64)
        range_indices = np.array(cols + rg_offset, dtype=np.float64)

        # Use the above offset indices to get the ground truth signal.
        true_coregistered_secondary = generate_signal_sinusoidal(
            rows=azimuth_indices,
            cols=range_indices,
            scaling_factor=scaling_factor,
        )

        # Generate an output array and radar grid parameters for the resampling.
        resamp_coregistered_secondary = np.empty(out_shape, dtype=np.complex64)

        # Only used in doppler, which for this test is 0. A set of 1's is suitable for
        # this object.
        sec_grid: RadarGridParameters = RadarGridParameters(
            sensing_start=1,
            wavelength=1,
            prf=1,
            starting_range=1,
            range_pixel_spacing=1,
            look_side="right",
            length=sec_length,
            width=sec_width,
            ref_epoch=DateTime(),
        )

        # Generate a doppler centroid that always returns 0 Hz.
        doppler = LUT2d()

        # The value to fill out-of-bounds pixels with.
        fill_value = nan_complex()

        # Perform the interpolation.
        if test_pybind:
            _resample_to_coords(
                resamp_coregistered_secondary,
                secondary,
                range_indices,
                azimuth_indices,
                sec_grid,
                doppler,
                fill_value,
            )
        else:
            resample_slc_blocks(
                output_resampled_slcs=[resamp_coregistered_secondary],
                input_slcs=[secondary],
                az_offsets_dataset=np.full(out_shape, az_offset, dtype=np.float64),
                rg_offsets_dataset=np.full(out_shape, rg_offset, dtype=np.float64),
                input_radar_grid=sec_grid,
                doppler=doppler,
                fill_value=fill_value,
                block_size_rg=rg_width // 4,
                block_size_az=az_length // 4,
                quiet=False,
            )

        try:
            # Validate the interpolated data against the true data.
            validate_test_results(
                test_arr=resamp_coregistered_secondary,
                true_arr=true_coregistered_secondary,
                correlation_min=0.998,
                phase_stdev_max=0.001,
                nan_percent_max=0.9,
                buffer_size_az=5,
                buffer_size_rg=5,
                az_offset=0,
                rg_offset=0,
            )
        except AssertionError as err:
            if test_pybind:
                error_note = "C++ _resample_to_coords pybind"
            else:
                error_note = "Python resample_slc_blocks"
            err.add_note(error_note)
            raise err

    @pytest.mark.parametrize("doppler_frequency", [0, 0.5])
    def test_interpolate_distributed_target_doppler(
        self, doppler_frequency: float, test_pybind: bool
    ):
        """
        Tests the Resample SLC V2 on a randomly distributed target signal.

        This test is designed to assess if Resample SLC works for a reasonably educated
        attempt at a realistic input signal. Fail cases caught by this test include all
        those tested for in the sinusoidal test, but using a more rigorous and
        complicated secondary signal.

        This test also tests the zero-doppler case as well as a set of several doppler
        frequencies as provided by the doppler_frequency fixture.

        Fixtures
        ----------
        doppler_frequency : float
            A doppler frequency for this test, in Hz.
        """
        # The bandwidth of the point signals in both azimuth and range.
        bandwidth = 0.8
        # The size of the generated image.
        az_length = 100
        rg_width = 100
        out_shape = (az_length, rg_width)

        sec_grid: RadarGridParameters = RadarGridParameters(
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

        # Generate the secondary signal with the given inputs.
        test_signal = SignalGenerator(
            az_length=az_length,
            rg_width=rg_width,
            az_bandwidth=bandwidth,
            rg_bandwidth=bandwidth,
            pad_size=1,
            seed=0,
            is_complex=True,
        )
        secondary = test_signal.generate_signal()

        # Generate a doppler centroid that has a ramp
        doppler = generate_doppler_lut(
            grid_params=sec_grid,
            doppler_frequency=doppler_frequency,
        )

        # Ramp the secondary SLC.
        doppler_ramp_complex = generate_doppler_ramp_complex(
            grid_params=sec_grid,
            az_indices=np.arange(az_length),
            doppler_frequency=doppler_frequency,
        )
        secondary *= doppler_ramp_complex[:, np.newaxis]

        # Set the outputs at random positions with a range of -0.5 to 0.5 with a flat
        # probability distribution.
        az_offsets = np.random.random(out_shape) - 0.5
        rg_offsets = np.random.random(out_shape) - 0.5

        # Add the offsets to these indices to get the indices in the ground truth grid.
        rows, cols = np.indices(out_shape)
        azimuth_indices = np.array(rows + az_offsets, dtype=np.float64)
        range_indices = np.array(cols + rg_offsets, dtype=np.float64)

        # Generate the ground-truth coregistered secondary signal with these offsets.
        true_coregistered_secondary = test_signal.generate_signal(
            az_offset=az_offsets,
            rg_offset=rg_offsets,
        )
        coregistered_doppler_ramp = generate_doppler_ramp_complex(
            grid_params=sec_grid,
            az_indices=azimuth_indices,
            doppler_frequency=doppler_frequency,
        )

        # Ramp the true coregistered secondary SLC.
        true_coregistered_secondary *= coregistered_doppler_ramp

        # Create the empty coregistered secondary array.
        resamp_coregistered_secondary = np.empty(shape=out_shape, dtype=np.complex64)

        # The value to fill out-of-bounds pixels with.
        fill_value = nan_complex()

        # Perform the interpolation.
        if test_pybind:
            _resample_to_coords(
                resamp_coregistered_secondary,
                secondary,
                range_indices,
                azimuth_indices,
                sec_grid,
                doppler,
                fill_value,
            )
        else:
            resample_slc_blocks(
                output_resampled_slcs=[resamp_coregistered_secondary],
                input_slcs=[secondary],
                az_offsets_dataset=az_offsets,
                rg_offsets_dataset=rg_offsets,
                input_radar_grid=sec_grid,
                doppler=doppler,
                fill_value=fill_value,
                block_size_rg=rg_width // 2,
                block_size_az=az_length // 2,
                quiet=False,
            )

        try:
            # Validate the interpolated data against the true data.
            validate_test_results(
                test_arr=resamp_coregistered_secondary,
                true_arr=true_coregistered_secondary,
                correlation_min=0.997,
                phase_stdev_max=10.0,
                nan_percent_max=16.0,
                buffer_size_az=5,
                buffer_size_rg=5,
                az_offset=0,
                rg_offset=0,
            )
        except AssertionError as err:
            if test_pybind:
                error_note = "C++ _resample_to_coords pybind"
            else:
                error_note = "Python resample_slc_blocks"
            err.add_note(error_note)
            err.add_note(f"doppler_frequency: {doppler_frequency}")
            raise err
