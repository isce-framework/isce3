#!/usr/bin/env python3
import iscetest
from isce3.signal import (cheby_equi_ripple_filter,
    design_shaped_lowpass_filter, design_shaped_bandpass_filter,
    build_multi_rate_fir_filter)
import numpy as np
import numpy.testing as npt
import bisect
import scipy.signal as sig
import pytest


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


def test_build_multi_rate_fir_filter():
    # desired filter spec of multi-rate filter
    samprate = 240e6  # (Hz)
    bandwidth = 80e6  # (Hz)
    over_samp_fact = 1.2  # roll off or oversampling factor (-)
    ripple = 0.1  # pass band ripple (dB)
    stopatt = 40.0  # min stop-band attenutation (dB)
    centerfreq = 1257.5e6  # (Hz)

    # expected filter parameters and output rate
    up_fact = 2
    down_fact = 5
    rate_out = 96e6
    num_taps = 111
    group_del = 55

    flt = build_multi_rate_fir_filter(
        samprate, bandwidth, over_samp_fact, centerfreq,
        ripple=ripple, stopatt=stopatt
        )
    npt.assert_equal(
        flt.numtaps, num_taps, err_msg='Wrong filter length.'
        )
    npt.assert_equal(
        flt.upfact, up_fact, err_msg='Wrong up-sampling factor.'
        )
    npt.assert_equal(
        flt.downfact, down_fact, err_msg='Wrong down-sampling factor.'
        )
    npt.assert_equal(
        flt.groupdelsamp, group_del, err_msg='Wrong group delay samples.'
        )
    npt.assert_allclose(
        flt.rateout, rate_out, rtol=0, atol=1e-8,
        err_msg='Wrong output sampling rate.'
        )


def db2amp(x):
    return 10.0**(x / 20.0)


def transition_region(bandwidth, transition_width):
    "bounds of transition region straddle cutoff at width/2"
    a = bandwidth * (1 - transition_width / 2) / 2
    b = bandwidth * (1 + transition_width / 2) / 2
    return a, b


@pytest.mark.parametrize("width,tw,att,odd", [
    (77/96, 0.20, 40, True),
    (40/96, 0.20, 40, True),
    (20/96, 0.20, 40, True),
    ( 5/96, 0.20, 40, True),
    (40/96, 0.20, 40, False),
    (40/96, 0.10, 50, False),
])
def test_lowpass(width, tw, att, odd):
    # design standard low pass filter
    h = design_shaped_lowpass_filter(width, stopatt=att, transition_width=tw,
                                     force_odd_len=odd)
    if odd:
        npt.assert_(len(h) % 2 == 1)
    # compute frequency response
    f, H = sig.freqz(h, fs=1.0)
    # bounds of transition region
    a, b = transition_region(width, tw)
    # should have unit gain in passband with ripple < -att
    atol = db2amp(-att)
    mask_pass = abs(f) < a
    npt.assert_allclose(np.abs(H[mask_pass]), 1.0, atol=atol)
    # should have less than -att gain in stopband
    mask_stop = abs(f) > b
    npt.assert_allclose(np.abs(H[mask_stop]), 0.0, atol=atol)
    # should have half amplitude at passband edges
    npt.assert_allclose(np.interp(width/2, f, abs(H)), 0.5, atol=atol)


def coswin(t, eta):
    a = (1 + eta) / 2
    b = (1 - eta) / 2
    return a + b * np.cos(2 * np.pi * t)


def kaiser(t, beta):
    return np.i0(beta * np.sqrt(1 - 4*t**2)) / np.i0(beta)


@pytest.mark.parametrize("width,window,window_fun", [
    (40/96, ("cosine", 1.00), coswin),  # boxcar
    (40/96, ("cosine", 0.70), coswin),  # NISAR performance model
    (20/96, ("cosine", 0.70), coswin),
    (40/96, ("cosine", 0.08), coswin),  # Hamming
    (40/96, ("cosine", 0.00), coswin),  # Hann
    (40/96, ("kaiser", 0.00), kaiser),  # boxcar
    (40/96, ("kaiser", 1.60), kaiser),  # NISAR RSLC product spec
    (20/96, ("kaiser", 1.60), kaiser),
    (40/96, ("kaiser", 4.50), kaiser),
])
def test_lowpass_shape(width, window, window_fun):
    tw = 0.2
    odd = True
    att = 40.0
    atol = db2amp(-att)
    # design filter with shaped passband
    h = design_shaped_lowpass_filter(width, stopatt=att, transition_width=tw,
                                     force_odd_len=odd, window=window)
    f, H = sig.freqz(h, fs=1.0)
    a, b = transition_region(width, tw)
    # should have shape of window in passband
    mask_pass = abs(f) < a
    name, shape = window
    expected = window_fun(f[mask_pass] / width, shape)
    npt.assert_allclose(np.abs(H[mask_pass]), expected, atol=atol)
    # should have less than -att gain in stopband
    mask_stop = abs(f) > b
    npt.assert_allclose(np.abs(H[mask_stop]), 0.0, atol=atol)


def wrapped_distance(x, fs=1.0):
    xm = x % fs
    return np.where(xm > fs/2, fs - xm, xm)


@pytest.mark.parametrize("width,fc", [
    (40/96, (1239   - 1257.5) / 96),    # intersect L40 and L80
    (20/96, (1229   - 1257.5) / 96),    # intersect L20 and L80
    ( 5/96, (1221.5 - 1257.5) / 96),    # intersect L05 and L80
    (20/48, (1229   - 1239  ) / 48),    # intersect L20 and L40
    ( 5/48, (1221.5 - 1239  ) / 48),    # intersect L05 and L40
    ( 5/24, (1221.5 - 1229  ) / 24),    # intersect L05 and L20
    ( 5/96, (1293.5 - 1257.5) / 96),    # intersect AUX and L80
])
def test_bandpass(width, fc):
    tw = 0.2
    att = 40
    odd = True
    atol = db2amp(-att)
    # design bandpass filter with unit passband
    h = design_shaped_bandpass_filter(width, fc, stopatt=att,
                                      transition_width=tw, force_odd_len=odd)
    # compute frequency response over entire unit circle
    f, H = sig.freqz(h, fs=1.0, whole=True)
    a, b = transition_region(width, tw)
    # should have unit gain in passband with ripple < -att
    mask_pass = wrapped_distance(f - fc) < a
    npt.assert_allclose(np.abs(H[mask_pass]), 1.0, atol=atol)
    # should have less than -att gain in stopband
    mask_stop = wrapped_distance(f - fc) > b
    npt.assert_allclose(np.abs(H[mask_stop]), 0.0, atol=atol)
