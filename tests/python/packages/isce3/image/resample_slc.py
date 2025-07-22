"""Test resample_slc ver. 2"""
from __future__ import annotations

import pytest

from .resamp_funcs import block_resample, pybind_resample
from ..resample_slc_utils import (
    distributed_target_resample_test,
    ResampFunc,
    sinusoidal_resample_test,
)


@pytest.mark.parametrize(
    "resamp_func,error_name",
    [
        (pybind_resample, "C++ _resample_to_coords pybind"),
        (block_resample, "Python resample_slc_blocks"),
    ]
)
class TestResampleSLCV2:
    """Tests for the Resample SLC Ver. 2 code."""

    def test_interpolate_sinusoidal(self, resamp_func: ResampFunc, error_name: str):
        """
        Tests the Resample SLC V2 on a simple sinusoidal signal.

        This test is designed to assess if Resample SLC works at all. Fail cases caught
        by this test include: Large numbers of NaN values in the output image, loss of
        correlation between the ground truth and output image, and large differences
        between the output image and the ground truth, as well as failure of the
        `resample_slc_blocks` function to complete operation.
        """
        sinusoidal_resample_test(resamp_func=resamp_func, error_name=error_name)

    @pytest.mark.parametrize("doppler_frequency", [0, 0.5])
    def test_interpolate_distributed_target_doppler(
        self, doppler_frequency: float, resamp_func: ResampFunc, error_name: str,
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
        distributed_target_resample_test(
            doppler_frequency=doppler_frequency,
            resamp_func=resamp_func,
            error_name=error_name,
        )
