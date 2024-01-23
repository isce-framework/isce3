"""Utils for resample_slc tests"""
from __future__ import annotations

import numpy as np

from isce3.core import LUT2d
from isce3.product import RadarGridParameters


def correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute the similarity of two (possibly complex) signals."""

    def cdot(x, y):
        return np.nansum(x * y.conj())

    return np.abs(cdot(a, b) / np.sqrt(cdot(a, a) * cdot(b, b)))


def phase_std_dev(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute the standard deviation of the angles between two signals."""
    dx = np.angle(a * b.conj())
    return np.nanstd(dx)


def nan_complex() -> np.complex64:
    """Return a complex value NaN + Nan*j."""
    return np.nan + 1.0j * np.nan


def validate_test_results(
    test_arr: np.ndarray,
    true_arr: np.ndarray,
    correlation_min: float,
    phase_stdev_max: float,
    nan_percent_max: float,
    buffer_size_az: int = 5,
    buffer_size_rg: int = 5,
    az_offset: int = 0,
    rg_offset: int = 0,
):
    """
    Perform a set of tests to determine if a test result satisfies tolerances.

    Parameters
    ----------
    test_arr : np.ndarray of complex values.
        An array of generated data, to be checked.
    true_arr : np.ndarray of complex values.
        An array of ground truth data, to check against.
    correlation_min : float
        The minimum tolerable correlation between `test_arr` and `true_arr`.
    nan_percent_max : float
        The maximum tolerable standard deviation between the phases `test_arr` and
        `true_arr`.
    nan_percent_tolerance : float
        The maximum tolerable percentage of NaN values in the whole unbuffered
        `test_arr` array.
    buffer_size_az : int, optional
        The number of pixels to omit from the start and end of the arrays in the azimuth
        direction. Defaults to 5.
    buffer_size_rg : int, optional
        The number of pixels to omit from the start and end of the arrays in the range
        direction. Defaults to 5.
    az_offset : int, optional
        The number of pixels in azimuth to offset the test array by. Defaults to 0.
    rg_offset : int, optional
        The number of pixels in range to offset the test array by. Defaults to 0.
    """
    # Check that the arrays have the same dimensions.
    az_length, rg_width = test_arr.shape
    true_az_length, true_rg_width = true_arr.shape
    assert az_length == true_az_length
    assert rg_width == true_rg_width

    # The buffers denote how far to collar in the region of the two images to check for
    # statistics. This prevents the correlation function from finding artificially low
    # values due to the small field of NaNs at the edges of images that appear due to
    # processing.
    az_buffer = buffer_size_az + int(max(az_offset, 0))
    rg_buffer = buffer_size_rg + int(max(rg_offset, 0))
    # buffered_test = test_arr[az_slice, rg_slice]
    # buffered_true = true_arr[az_slice, rg_slice]
    buffered_test = test_arr[
        az_buffer : az_length - az_buffer, rg_buffer : rg_width - rg_buffer
    ]
    buffered_true = true_arr[
        az_buffer : az_length - az_buffer, rg_buffer : rg_width - rg_buffer
    ]

    # Determine the correlation between the buffered testing and ground truth
    # images.
    corr = correlation(
        buffered_test,
        buffered_true,
    )

    # Determine the standard deviation between the buffered testing and ground
    # truth images.
    phase_stdev = phase_std_dev(
        buffered_test,
        buffered_true,
    )
    phase_stdev_degrees = np.rad2deg(phase_stdev)

    # Get the number of NaN values in the whole testing image as % of all pixels.
    nans = np.count_nonzero(np.isnan(test_arr))
    nan_percent = nans / test_arr.size * 100

    fail = False
    fail_strings = []
    if not corr > correlation_min:
        fail = True
        fail_strings.append(
            f"Correlation {corr} of resample output is less than minimum acceptable "
            f"value {correlation_min}"
        )
    if not phase_stdev_degrees < phase_stdev_max:
        fail = True
        fail_strings.append(
            f"Phase standard deviation between ground-truth and resample output "
            f"{phase_stdev_degrees} is greater than maximum acceptable value "
            f"{phase_stdev_max}"
        )
    if not nan_percent < nan_percent_max:
        fail = True
        fail_strings.append(
            f"percentage of NaNs in resample output: {nan_percent}% is greater than "
            f"maximum acceptable value: {nan_percent_max}%"
        )

    if fail:
        raise AssertionError("; ".join(fail_strings))


def generate_signal_sinusoidal(
    rows: np.ndarray, cols: np.ndarray, scaling_factor: float
) -> np.ndarray:
    """Calculate a sinusoidal complex signal at each given position, scaled.

    Parameters
    ----------
    rows : np.ndarray of float values
        The row indices at each point to calculate the signal from.
    cols : np.ndarray of float values
        The column indices at each point to calculate the signal from.
    scaling_factor : float
        The degree of scaling, directly proportional to the signal frequency.

    Returns
    -------
    np.ndarray of complex values
        The generated signal.
    """
    # Scale the row and column indices to ajust the frequency of the signal.
    scaled_rows = rows * scaling_factor
    scaled_cols = cols * scaling_factor
    # The contribution of the row indices to the signal is
    # sin(row) + i * cos(row)
    row_cpx = np.sin(scaled_rows) + np.cos(scaled_rows) * 1.0j
    # The contribution of the column indices to the signal is
    # cos(column) + i * sin(column)
    col_cpx = np.cos(scaled_cols) + np.sin(scaled_cols) * 1.0j

    return np.array(row_cpx + col_cpx, dtype=np.complex64)


class SignalGenerator:
    """A band-limited signal generator for testing interpolators."""

    def __init__(
        self,
        az_length: int = 1024,
        rg_width: int = 1024,
        az_bandwidth: float = 0.8,
        rg_bandwidth: float = 0.8,
        targets_per_bin: int = 4,
        pad_size: float = 0.0,
        seed: int | None = None,
        is_complex: bool = False,
    ):
        """
        A band-limited signal generator for testing interpolators.

        Parameters
        ----------
        az_length : int, optional
            The length of the swath in azimuth pulses. Defaults to 1024.
        rg_width : int, optional
            The width of the swath in range indices. Defaults to 1024.
        az_bandwidth : float, optional
            The bandwidth of the sinc signals, relative to the sampling rate, in the
            azimuth direction. Defaults to 0.8.
        rg_bandwidth : float, optional
            The bandwidth of the sinc signals, relative to the sampling rate, in the
            range direction. Defaults to 0.8.
        targets_per_bin : int, optional
            The number of targets to generate per bin. Defaults to 4.
        pad_size : float, optional
            The extent around the generated frame at which targets can be generated.
        seed : int or None, optional
            The random seed or None. If none, the seed will be whatever numpy.random
            defaults to. Defaults to None.
        is_complex : bool, optional
            Whether or not to generate numbers in the complex domain. Defaults to False.
        """
        # Set random number generator to deterministic state if requested.
        if seed:
            np.random.seed(seed)

        self.az_length = az_length
        self.rg_width = rg_width
        self.az_bandwidth = az_bandwidth
        self.rg_bandwidth = rg_bandwidth
        self.is_complex = is_complex
        self.pad_size = pad_size

        # Number of targets to simulate.  Default to four per bin.
        self.num_targets = int(
            targets_per_bin * (az_length + 2 * pad_size) * (rg_width + 2 * pad_size)
        )

        self.generate_targets()

    def generate_targets(self):
        """Generate targets that will appear as sinc functions within the signal."""
        az_length = self.az_length
        rg_width = self.rg_width
        num_targets = self.num_targets
        pad_size = self.pad_size

        # Targets randomly fall within the padded breadth of the signal.
        az_breadth = az_length + 2 * pad_size
        rg_breadth = rg_width + 2 * pad_size
        self.targets_az_locs = np.random.random(num_targets) * az_breadth - pad_size
        self.targets_rg_locs = np.random.random(num_targets) * rg_breadth - pad_size

        # Generate random amplitudes and phases.
        self.dtype = np.complex64 if self.is_complex else np.float64
        self.target_values = np.zeros(self.num_targets, dtype=self.dtype)

        self.target_values += np.random.normal(size=self.num_targets)
        if self.is_complex:
            self.target_values += 1j * np.random.normal(size=self.num_targets)
        self.target_values *= 1.0 * az_length / num_targets

    def generate_signal(
        self,
        az_offset: float | np.ndarray = 0.0,
        rg_offset: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """
        Realize the test signal.

        Parameters
        -------
        az_offset : float or np.ndarray
            The azimuth offset(s) for each index on the grid.
        rg_offset : float or np.ndarray
            The range offset(s) for each index on the grid.

        Returns
        -------
        signal: np.ndarray
            The generated signal.
        """
        if isinstance(az_offset, np.ndarray):
            assert az_offset.shape == (self.az_length, self.rg_width)
        if isinstance(rg_offset, np.ndarray):
            assert rg_offset.shape == (self.az_length, self.rg_width)

        num_targets = self.num_targets

        # Create arrays with the range and azimuth positions at each pixel.
        az_range = np.arange(self.az_length, dtype=float)
        rg_range = np.arange(self.rg_width, dtype=float)
        rg_indices, az_indices = np.meshgrid(az_range, rg_range)
        az_indices += az_offset
        rg_indices += rg_offset

        # Signal real if real weights, complex if complex weights.
        signal = np.zeros((self.az_length, self.rg_width), dtype=self.dtype)

        # Add each target's effect into the signal.
        for target_index in range(num_targets):
            # Get the value at this target.
            weight = self.target_values[target_index]
            # Get the (az, rg) position of the target.
            targ_az_loc = self.targets_az_locs[target_index]
            targ_rg_loc = self.targets_rg_locs[target_index]
            # The relative positions are the difference in each dimension between the
            # shifted indices and the target location.
            rel_az_positions = az_indices - targ_az_loc
            rel_rg_positions = rg_indices - targ_rg_loc

            # Calculate the values of the weighted sinc function for this target at the
            # given distances, and add them to the signal.
            az_sinc = np.sinc(self.az_bandwidth * rel_az_positions)
            rg_sinc = np.sinc(self.rg_bandwidth * rel_rg_positions)

            target_sinc = az_sinc * rg_sinc * weight
            signal += target_sinc

        return signal


def generate_doppler_ramp_complex(
    grid_params: RadarGridParameters,
    az_indices: np.ndarray,
    doppler_frequency: float,
) -> np.ndarray:
    """
    Evaluate the doppler phase ramp at the given set of indices.

    Parameters
    ----------
    grid_params : RadarGridParameters
        The radar grid parameters object for the doppler ramp.
    az_indices : np.ndarray
        The azimuth indices to evaluate doppler at.
    doppler_frequency : float
        The doppler frequency.

    Returns
    -------
    np.ndarray
        An array of the doppler phase ramp evaluated at each azimuth index passed in.
    """
    # Trivially, for zero-doppler, just return an array of 1+0j
    if doppler_frequency == 0:
        return np.ones(az_indices.shape, dtype=np.complex64)

    # Get the absolute azimuth time at each index.
    az_times = grid_params.sensing_start + az_indices / grid_params.prf
    # Evaluate and return the ramp.
    doppler_ramp = np.exp(1.0j * 2 * np.pi * doppler_frequency * az_times)

    return doppler_ramp


def generate_doppler_lut(
    grid_params: RadarGridParameters, doppler_frequency: float
) -> LUT2d:
    """
    Create a constant doppler LUT.

    Parameters
    ----------
    grid_params : RadarGridParameters
        The radar grid parameters to evaluate azimuth and range with.
    doppler_frequency : float
        The frequency of the to be created.

    Returns
    -------
    LUT2d
        The generated LUT.
    """
    # Trivially, for zero doppler just return an LUT2d that always evaluates to 0.
    if doppler_frequency == 0:
        return LUT2d()

    array_length = grid_params.length
    array_width = grid_params.width
    az_indices = np.arange(array_length)
    rg_indices = np.arange(array_width)

    az_time = grid_params.sensing_start + az_indices / grid_params.prf
    rg_dist = grid_params.starting_range + rg_indices / grid_params.range_pixel_spacing

    lut_array = np.full(
        shape=(array_length, array_width),
        fill_value=doppler_frequency,
        dtype=np.float64,
    )

    doppler_lut = LUT2d(
        rg_dist,
        az_time,
        lut_array,
    )

    return doppler_lut
