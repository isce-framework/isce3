import iscetest
import numpy as np
import numpy.testing as npt
import isce3
from isce3.signal import point_target_info as pt
from numpy.fft import fftfreq, fftshift, fft, ifft

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
        null_left_idx, null_right_idx = pt.search_first_null(
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

if __name__ == "__main__":
    test_kaiser_win()
    test_cosine_win()
