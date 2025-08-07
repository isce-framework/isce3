#!/usr/bin/env python3
import iscetest
from isce3.focus import form_linear_chirp
from isce3.signal import corr_doppler_est, sign_doppler_est

import numpy as np
import numpy.testing as npt


# List of functions used in generating simulated data
def rcoswin(size, ped=1.0):
    """Raised-cosine symmetric window function.

    Parameters
    ----------
    size : int
        size of the window
    ped : float, default=1.0
        pedestal, a value within [0. 1.0]

    Returns
    -------
    np.ndarray(float)

    Raises
    ------
    AssertionError
        For bad inputs

    """
    assert 0 <= ped and ped <= 1, 'Pedestal value shall be within [0, 1]!'
    assert size > 0, 'Size must be a positive number!'
    return (1 + ped) / 2. - (1 - ped) / 2. * np.cos(2.0 * np.pi / (size - 1) *
                                                    np.arange(0, size))


def form_weighted_chirp(bandwidth, duration, prf, pedestal_win):
    """Form a weighted complex baseband chirp.

    Parameters
    ----------
    bandwidth : float
        chirp bandwidth in (Hz)
    duration : float
        chirp duration in (sec)
    prf : float
        PRF in (Hz)
    pedestal_win : float
        Pedestal of a raised cosine window

    Returns
    -------
    np.ndarray(complex)
        Complex windowed baseband chirp

    """
    chirp_rate = bandwidth / duration  # (Hz/sec)
    chirp = np.asarray(form_linear_chirp(chirp_rate, duration, prf, 0.0))
    # apply weighting
    chirp *= rcoswin(len(chirp), ped=pedestal_win)
    return chirp


class TestDopplerEstFunc:
    # List of parameters for generating noisy azimuth chirp signal

    # azimuth sampling rate (Hz)
    prf = 2000.
    # bandwidth of azimuth chirp(Hz)
    bandwidth = 1500.0
    # duration of azimuth chirp (sec)
    duration = 2.5
    # signal to noise ratio (dB)
    snr = 8.0
    # number of range bins
    num_rgb = 8
    # pedestal of raised cosine window function
    pedestal_win = 0.4
    # seed number for random generator
    seed_rnd = 10

    # list of desired doppler centroids to be tested
    doppler_list = [-550., -215, -50, 0, 50, 215, 550]  # (Hz)

    # absolute doppler tolerance in (Hz) per requirement for validating
    # Doppler estimators output against list of Dopplers
    atol_dop = 15.0

    # generating noise-free baseband chirp as well as noise signal
    # used in testing all methods

    # form a noise-free weighted complex baseband chirp
    chirp = form_weighted_chirp(bandwidth, duration, prf, pedestal_win)

    # generate complex Gaussian zero-mean random noise,
    # one independent set per range bin used for all dopplers
    std_iq_noise = 1./np.sqrt(2) * 10 ** (-snr / 20)
    rnd_gen = np.random.RandomState(seed=seed_rnd)
    noise = std_iq_noise * (rnd_gen.randn(chirp.size, num_rgb) +
                            1j * rnd_gen.randn(chirp.size, num_rgb))

    def _validate_doppler_est(self, method: str):
        """Validate estimated doppler for a method"""
        # form a generic doppler estimator function covering both methods        
        if method == 'CDE':
            dop_func = corr_doppler_est
        else: # 'SDE'
            def dop_func(echo, prf, lag=1, axis=None):
                return sign_doppler_est(echo, prf, lag, axis), 1            
            
        # loop over list of doppler centroid to generate noisy pass-band signal
        for doppler in self.doppler_list:
            # create a pass-band chirp from baseband one per doppler
            chirp_dop = self.chirp * np.exp(1j * 2 * np.pi * doppler / self.prf
                                            * np.arange(self.chirp.size))
            # create a noisy complex signal , one set per range bin
            sig = np.repeat(chirp_dop.reshape((chirp_dop.size, 1)),
                            self.num_rgb, axis=1) + self.noise
            # estimate doppler and check its value
            dop_est, corr_coef = dop_func(sig, self.prf)
            npt.assert_allclose(dop_est, doppler, atol=self.atol_dop,
                                err_msg='Large error for doppler '
                                f'{doppler:.2f} (Hz) in method "{method}"')
            npt.assert_equal((corr_coef >= 0 and corr_coef <= 1), True,
                             err_msg = 'Correlation coeff is out of range'
                             f' for method {method}')
            print(f'Correlation coef for method {method} & Doppler '
                  f'{doppler:.1f} (Hz) -> {corr_coef:.3f}')

    def test_corr_doppler_est(self):
        self._validate_doppler_est('CDE')

    def test_sign_doppler_est(self):
        self._validate_doppler_est('SDE')
