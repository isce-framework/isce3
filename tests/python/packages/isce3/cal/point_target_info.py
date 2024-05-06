from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import isce3
import iscetest
import numpy as np
import numpy.testing as npt
import pickle
import typing
from isce3.cal import point_target_info as pt
from numpy.fft import fftfreq, fftshift, ifft
from numpy.typing import ArrayLike
from pytest import mark

from isce3.cal.point_target_info import analyze_point_target

class RectNotch:
    def __init__(self, freq: float, bandwidth: float):
        """
        Define a rectangular notch function at frequency `f` with width `b`
        where both are normalized by the sample rate.
        """
        self.freq = freq
        self.bandwidth = bandwidth

    def apply(self, freqs: np.ndarray, values: np.ndarray):
        "Set `values` to zero where `freqs` are within the notch."
        low = self.freq - self.bandwidth / 2
        hi = self.freq + self.bandwidth / 2
        mask = (low <= freqs) & (freqs <= hi)
        values[mask] = 0.0


def kaiser_irf(bandwidth=1.0, window_parameter=0.0, qpe=0.0, notches=[], n=256, oversamp_ratio=32):
    # Generate spectrum
    nfft = n * oversamp_ratio
    f = fftfreq(nfft, d=1 / oversamp_ratio)
    X = np.zeros(nfft, complex)
    mask = abs(f) <= bandwidth / 2
    X[mask] = np.i0(window_parameter * np.sqrt(1 - (2 * f[mask] / bandwidth) ** 2)) / np.i0(window_parameter)
    X[mask] *= np.exp(1j * qpe / (bandwidth / 2) ** 2 * f[mask] ** 2)
    for notch in notches:
        notch.apply(f, X)
    # Transform
    x = ifft(X)
    x /= np.max(np.abs(x))
    t = fftfreq(nfft, d=1 / n)

    return fftshift(t), fftshift(x)


def coswin_irf(x, window_parameter, bandwidth):
    c = isce3.core.speed_of_light
    delta_x = c/(2*bandwidth)
    q = x / delta_x

    y = np.sinc(q) + (1-window_parameter)/(1+window_parameter)/np.pi * np.sin(np.pi*q) * q / (1 - q**2)
    
    return y


def test_kaiser_win():
    """
    Test null search with 4 given patterns
    """

    # Search parameters
    left_null_idx_benchmark = [4064, 4044, 4028, 4065]
    right_null_idx_benchmark = [4128, 4148, 4164, 4127]

    kaiser_pslr_bench = [-13.27, -30.10, -20.13, -10.82]
    kaiser_islr_bench = [-10.11, -28.97, -17.21, -5.61]
    pslr_max_err = 0.1
    islr_max_err = 0.1

    predict_null = False
    num_sidelobes = 10
    fs_bw_ratio = 32
    window_type = 'kaiser'

    # window window_parameter = Kaiser window coefficient
    cases = {
        "A": dict(), #ideal rectangular win
        "B": dict(window_parameter=4.0),
        "C": dict(window_parameter=2.0, qpe=np.radians(90)),
        "D": dict(notches=[RectNotch(0.23, 0.15)]),
    }

    null_left_idx_list = []
    null_right_idx_list = []
    pslr_cases = []
    islr_cases = []

    for label, kw in cases.items():
        t, z = kaiser_irf(oversamp_ratio=fs_bw_ratio, **kw)

        window_parameter = kw.get("window_parameter", 0.0)

        z_pwr_db = 20 * np.log10(np.abs(z))
        main_peak_idx = np.argmax(z_pwr_db)

        #Test null search algorithm
        null_left_idx, null_right_idx = pt.search_first_null_pair(
            z_pwr_db, main_peak_idx
        )

        null_left_idx_list.append(null_left_idx)
        null_right_idx_list.append(null_right_idx)

        #Test ISLR and PSLR of raised cosine windowed patterns using Default Fs / BW
        islr_db, pslr_db = pt.compute_islr_pslr(
            z,
            fs_bw_ratio,
            num_sidelobes,
            predict_null,
            window_type,
            window_parameter
            )
        pslr_cases.append(pslr_db)
        islr_cases.append(islr_db)

    pslr_err = np.abs(np.array(pslr_cases) - kaiser_pslr_bench)
    islr_err = np.abs(np.array(islr_cases) - kaiser_islr_bench)

    npt.assert_array_equal(
        null_left_idx_list,
        left_null_idx_benchmark,
        "Mainlobe left Null(s) do not match with their bench mark(s)",
    )

    npt.assert_array_equal(
        null_right_idx_list,
        right_null_idx_benchmark,
        "Mainlobe right Null(s) do not match with their bench mark(s)",
    )

    npt.assert_array_less(
        pslr_err, 
        pslr_max_err, 
        "PSLR of Kaiser window(s) do not match with their bench mark(s)",
    )

    npt.assert_array_less(
        islr_err, 
        islr_max_err, 
        "ISLR of Kaiser window(s) do not match with their bench mark(s)",
    )

def test_cosine_win():
    cosine_pslr_bench = [-13.26, -40.06, -21.21]
    cosine_islr_bench = [-10.12, -33.10, -16.53]
    pslr_max_err = 0.1
    islr_max_err = 0.1

    predict_null = False
    num_sidelobes = 10
    fs_bw_ratio = 32
    bandwidth = 20e6
    window_type = 'cosine'

   # Setup cases
    num_lobes_cos = 50
    c = isce3.core.speed_of_light
    delta_x = c / (2*bandwidth)
    x = np.linspace(-num_lobes_cos*delta_x, num_lobes_cos*delta_x, int(2*num_lobes_cos*fs_bw_ratio))

    cases = {
        "A-Ideal": dict(x=x, window_parameter=1, bandwidth=bandwidth),
        "B": dict(x=x, window_parameter=0.1, bandwidth=bandwidth),
        "C": dict(x=x, window_parameter=0.5, bandwidth=bandwidth),
    }

    pslr_cases = []
    islr_cases = []

    for label, kw in cases.items():
        z = coswin_irf(**kw)

        window_parameter = kw.get("window_parameter")
        if window_parameter is None:
            window_parameter = 0.0

        #Test ISLR and PSLR of Kaiser Windowed patterns using Default Fs / BW
        islr_db, pslr_db = pt.compute_islr_pslr(
            z,
            fs_bw_ratio,
            num_sidelobes,
            predict_null,
            window_type,
            window_parameter
            )
        pslr_cases.append(pslr_db)
        islr_cases.append(islr_db)

    pslr_err = np.abs(np.array(pslr_cases) - cosine_pslr_bench)
    islr_err = np.abs(np.array(islr_cases) - cosine_islr_bench)

    npt.assert_array_less(
        pslr_err, 
        pslr_max_err, 
        "PSLR of Raised Cosine window(s) do not match with their bench mark(s)",
    )

    npt.assert_array_less(
        islr_err, 
        islr_max_err, 
        "ISLR of Raised Cosine window(s) do not match with their bench mark(s)",
    )


def test_search_null_pair():
    # This data has a pair of equal values at one of the sidelobe peaks.
    fn = Path(iscetest.data) / "search_first_null.pkl"
    with open(fn, "rb") as f:
        d = pickle.load(f)
    ileft, iright = pt.search_first_null_pair(d["matched_output"],
                                              d["mainlobe_peak_idx"])
    # obtained correct values by plotting the data
    true_left, true_right = 949, 1073
    npt.assert_equal(ileft, true_left)
    npt.assert_equal(iright, true_right)

    # Stick another equal value right next to the main peak.
    x = d["matched_output"][:]
    ipeak = d["mainlobe_peak_idx"]
    x[ipeak + 1] = x[ipeak]
    ileft, iright = pt.search_first_null_pair(x, ipeak)
    npt.assert_equal(ileft, true_left)
    npt.assert_equal(iright, true_right)

    # Try again where now we have two equal values at a null.
    x[true_right - 1] = x[true_right]
    ileft, iright = pt.search_first_null_pair(x, ipeak)
    npt.assert_equal(ileft, true_left)
    npt.assert_equal(iright, true_right)


def sinc2d(x: ArrayLike, y: ArrayLike, dx: float, dy: float) -> np.ndarray:
    """
    Generate a two-dimensional array calculated by the sinc of x multiplied by the sinc
    of y, in the complex domain.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    return np.array(np.sinc(x / dx) * np.sinc(y / dy), dtype=np.complex64)


def complex_noise(shape: tuple[int, ...], scale: float) -> np.ndarray:
    """
    Generate normally distributed noise in the complex domain.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the output array.
    scale : float
        The standard deviation of the noise in the real and imaginary components.

    Returns
    -------
    np.ndarray
        The generated noise.
    """
    generator = np.random.default_rng(seed=0)
    noise = (
        generator.normal(size=shape, scale=scale) +
        1.j * generator.normal(size=shape, scale=scale)
    )
    return noise


def check_tol(value: float, lo: float, hi: float) -> str | None:
    try:
        assert lo < value < hi
    except AssertionError as err:
        return str(err)
    return None


def append_if_given(coll: list, value: typing.Any | None, prefix: str = "") -> None:
    if value is None:
        return
    if not isinstance(value, str) and prefix != "":
        raise ValueError(
            "'value' given as non-string object, cannot be prefixed with string "
            f'"{prefix}".'
        )
    coll += [prefix + ": " + value]


@mark.parametrize(
    "spacing,sinc_scale,heading_deg",
    [
        ([1., 1.], [2., 2.], 0),
        ([1., 1.], [2., 2.], 90),
        ([1., 1.], [2., 2.], 180),
        ([1., 1.], [3., 3.], -45),
        ([1., 1.], [2., 2.], -30),
        ([3., 9.], [6., 18.], 0),
        ([3., 9.], [18., 18.], 90),
        ([3., 9.], [18., 18.], 180),
        ([3., 9.], [9., 27.], -45),
        ([3., 9.], [27., 9.], -30),
    ],
)
@mark.parametrize("offsets", [[0, 0], [5, -5], [5, 5]])
def test_simulated_geocoded_cr(
    spacing : Sequence[float, float],
    sinc_scale: Sequence[float, float],
    offsets : Sequence[float, float],
    heading_deg : float,
    snr : float = 40.0,
    dim : int = 300,
    nov : int = 32,
):
    """
    Test analysis of a simulated geocoded point target.

    Parameters
    ----------
    spacing : Sequence[float, float]
        The pixel spacing of the image.
    offsets : Sequence[float, float]
        The offset of the main lobe peak relative to the center of the
        image, in the same units as the spacing.
    sinc_scale : Sequence[float, float]
        The resolution between the center of the sinc and the first null of the range,
        azimuth axes of the image, in the same units as the spacing.
    heading_deg : float
        The heading, in degrees east of north.
    snr : float, optional
        The signal to noise ratio of the generated data. Defaults to 40.0 dB
    dim : int, optional
        The dimension, determines the shape of the square input image. The chip size is
        1/4 of this. Defaults to 300
    nov : int, optional
        The oversampling factor. Defaults to 32
    """
    chipsize = dim // 4
    if chipsize % 2 == 1:
        chipsize += 1

    e_spacing, n_spacing = spacing
    rg_scale, az_scale = sinc_scale
    x_offset, y_offset = offsets

    # Generate the base image - a simulated radar-encoded point target.
    x = e_spacing * np.arange(-dim//2, dim//2)
    y = n_spacing * np.arange(-dim//2, dim//2)

    # `heading` is the clockwise rotation angle to apply, in radians.
    heading = np.deg2rad(heading_deg)
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)

    X, Y = np.meshgrid(x - x_offset, y - y_offset, indexing="xy")
    X_rotated = cos_heading * X - sin_heading * Y
    Y_rotated = sin_heading * X + cos_heading * Y

    magnitude = 10.0 ** (snr / 20.0)
    image = sinc2d(x=X_rotated, y=Y_rotated, dx=rg_scale, dy=az_scale) * magnitude

    # Add simulated thermal noise to the base image.
    # Set the noise power to 1 (standard deviation of real & imag components should be 1/sqrt(2)).
    noise = complex_noise(shape=image.shape, scale=1.0 / np.sqrt(2))
    image += noise

    # Get "expected position" of the point target at the center of the image.
    # This is expected to be different than the actual position due to the offset,
    # which this test will ensure is correct.
    targ_rows, targ_columns = image.shape

    ret_dict, _ = analyze_point_target(
        slc=image,
        i=targ_rows / 2,
        j=targ_columns / 2,
        nov=nov,
        cuts=True,
        chipsize=chipsize,
        geo_heading=heading,
        pixel_spacing=(n_spacing, e_spacing)
    )

    err_msgs: list[str] = []

    # Acquire and check azimuth and range ISLR, PSLR
    az_islr = ret_dict["azimuth"]["ISLR"]
    append_if_given(err_msgs, check_tol(az_islr, -20, -9), "Azimuth ISLR")
    az_pslr = ret_dict["azimuth"]["PSLR"]
    append_if_given(err_msgs, check_tol(az_pslr, -20, -9), "Azimuth PSLR")

    rg_islr = ret_dict["range"]["ISLR"]
    append_if_given(err_msgs, check_tol(rg_islr, -20, -9), "Range ISLR")
    rg_pslr = ret_dict["range"]["PSLR"]
    append_if_given(err_msgs, check_tol(rg_pslr, -20, -9), "Range PSLR")

    expected_x_position = x_offset / e_spacing
    x_error = expected_x_position - ret_dict["x"]["offset"]
    append_if_given(err_msgs, check_tol(x_error, -1, 1), "X Offset Error")
    expected_y_position = y_offset / n_spacing
    y_error = expected_y_position - ret_dict["y"]["offset"]
    append_if_given(err_msgs, check_tol(y_error, -1, 1), "Y Offset Error")

    # Scale values are known in units of meters to the first null in the main lobe,
    # but the function outputs the 3dB resolution in units of pixels. Convert our known
    # values to pixels using the heading and spacing, then get multiply by 0.886 to get
    # expected 3dB resolution in pixels and calculate a percent error, then check it.
    az_res_out = ret_dict["azimuth"]["resolution"]
    az_res_n = az_scale * np.cos(heading) / n_spacing
    az_res_e = az_scale * np.sin(heading) / e_spacing
    az_res_px = np.sqrt(az_res_n**2 + az_res_e**2)
    az_res_3db = az_res_px * 0.886
    az_res_err = np.abs(az_res_3db - az_res_out) / az_res_3db * 100
    append_if_given(err_msgs, check_tol(az_res_err, 0, 3.5), "Az Resolution %Error")

    rg_res_out = ret_dict["range"]["resolution"]
    rg_res_n = rg_scale * -np.sin(heading) / n_spacing
    rg_res_e = rg_scale * np.cos(heading) / e_spacing
    rg_res_px = np.sqrt(rg_res_n**2 + rg_res_e**2)
    rg_res_3db = rg_res_px * 0.886
    rg_res_err = np.abs(rg_res_3db - rg_res_out) / rg_res_3db * 100
    append_if_given(err_msgs, check_tol(rg_res_err, 0, 3.5), "Rg Resolution %Error")

    if len(err_msgs) > 0:
        raise AssertionError("\n\t" + "\n\t".join(err_msgs))


if __name__ == "__main__":
    test_kaiser_win()
    test_cosine_win()
