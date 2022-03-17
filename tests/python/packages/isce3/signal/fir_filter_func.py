#!/usr/bin/env python3
import iscetest
from isce3.signal import cheby_equi_ripple_filter

import numpy as np
import numpy.testing as npt
import bisect


def est_centroid_spectrum(pow_spec, freq):
    """Estimate the centroid of the power spectrum"""
    return (pow_spec * freq).sum() / pow_spec.sum()


def est_equivalent_bandwidth(pow_spec, df):
    """Estimate rectangle equivalent bandwidth of spectrum"""
    return pow_spec.sum() * df / pow_spec.max()


def test_cheby_equi_ripple_filter():
    # number of fft for power spectrum calculation
    nfft = 2048

    # relative tolerance for pass-band ripple (dB), stop-band attenuation (dB)
    # and bandwidth (MHz)
    rtol_pass = 0.5
    rtol_stop = 0.1
    rtol_bw = 0.02

    # desired filter spec
    samprate = 1.6  # (MHz)
    bandwidth = 1.0  # (MHz)
    rolloff = 1.2  # roll off or oversampling factor (-)
    ripple = 0.2  # pass band ripple (dB)
    stopatt = 35.0  # min stop-band attenutation (dB)
    centerfreq = 0.1  # (MHz)

    # generate filter coeffs
    coefs = cheby_equi_ripple_filter(samprate, bandwidth, rolloff, ripple,
                                     stopatt, centerfreq)

    # get the power spectrum of the filter in linear scale
    pow_spec = abs(np.fft.fftshift(np.fft.fft(coefs, nfft)))**2
    # frequency vector within [-samprate/2., samprate/2.[
    # freq resolution defines absolute tolerance in ceter freq,
    # and roll-off factor estimation!
    df = samprate / nfft
    freq = -0.5 * samprate + df * np.arange(nfft)

    # check the center freq
    freq_cent_est = est_centroid_spectrum(pow_spec, freq)
    print(f'Estimated frequency centroid -> {freq_cent_est:.3f} (MHz)')
    npt.assert_allclose(freq_cent_est, centerfreq, atol=df,
                        err_msg='Wrong center frequency!')

    # check bandwidth
    rect_bw_est = est_equivalent_bandwidth(pow_spec, df)
    print(f'Estimated bandwidth -> {rect_bw_est:.3f} (MHz)')
    npt.assert_allclose(rect_bw_est, bandwidth, rtol=rtol_bw,
                        err_msg='Wrong bandwidth!')

    # get the expected [low, high[ index wihtin pass-band region
    frq_pass_low = -0.5 * bandwidth + centerfreq
    frq_pass_high = 0.5 * bandwidth + centerfreq
    idx_pass_low = bisect.bisect_left(freq, frq_pass_low)
    idx_pass_high = bisect.bisect_right(freq, frq_pass_high)
    slice_pass = slice(idx_pass_low, idx_pass_high)

    # make sure the peak occurs wihtin expected [low,high[ of passband
    idx_max = pow_spec.argmax()
    npt.assert_equal(idx_pass_low <= idx_max and idx_max <= idx_pass_high,
                     True, err_msg='The peak gain occurs outside expected \
pass-band region')
    # get peak-to-peak ripple within pass band to check pass-band ripple
    max_val_pass = pow_spec[idx_max]
    min_val_pass = pow_spec[slice_pass].min()
    est_ripple = 5.0 * np.log10(max_val_pass / min_val_pass)
    print(f'Estimated ripple within passband -> {est_ripple:.2f} (dB)')
    npt.assert_allclose(est_ripple, ripple, rtol=rtol_pass,
                        err_msg='Wrong pass-band ripple')

    # get expected start {left, right} index for edge of stop band region
    frq_stop_left = -0.5 * rolloff * bandwidth + centerfreq
    frq_stop_right = 0.5 * rolloff * bandwidth + centerfreq
    idx_stop_left = bisect.bisect_left(freq, frq_stop_left)
    idx_stop_right = bisect.bisect_right(freq, frq_stop_right)
    slice_stop_left = slice(0, idx_stop_left)
    slice_stop_right = slice(idx_stop_right, nfft)
    # check min stop-band attenuation within stop band on each side
    min_att_left = abs(10 * np.log10(pow_spec[slice_stop_left].max()))
    min_att_right = abs(10 * np.log10(pow_spec[slice_stop_right].max()))
    print(f'Estimated min stop-band attenuation on the left side -> \
{min_att_left:.2f} (dB)')
    print(f'Estimated min stop-band attenuation on the right side -> \
{min_att_right:.2f} (dB)')
    npt.assert_allclose(min_att_left, stopatt, rtol=rtol_stop,
                        err_msg='Wrong stop-band attenuation on the left')
    npt.assert_allclose(min_att_right, stopatt, rtol=rtol_stop,
                        err_msg='Wrong stop-band attenuation on the right')
