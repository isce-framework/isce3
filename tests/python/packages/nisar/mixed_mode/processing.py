from isce3.focus import RangeComp, form_linear_chirp
from nisar.mixed_mode import Band, get_common_band_filter
import numpy as np
import numpy.testing as npt


L20 = Band(1229e6, 20e6)
L40 = Band(1239e6, 40e6)
fs = 48e6
T = 20e-6
K = L40.width / T
plot = False


def test_filter():
    # filter design runs
    h, shift = get_common_band_filter(L40, L20, fs)
    # length must be odd for integer delay
    npt.assert_(len(h) % 2 == 1)
    # lower chunk of 40 MHz must be shifted up to make it baseband.
    npt.assert_(shift > 0)


def test_rangecomp_delay():
    """Make sure we understand how to bookkeep the starting range when using
    a common-band filter for mixed-mode processing.

    Assume that "starting range" is understood to mean the delay since the
    *start* of the transmit event.
    """
    nraw = 1024
    rawdata = np.zeros(nraw, dtype="c8")
    rawdata[0] = 1

    nchirp = 255
    chirp = np.zeros(nchirp, dtype="c8")
    chirp[0] = 1

    # Common-band filter design will always have delay (n-1)/2
    cb_filters = [
        np.array([1.0]),
        np.array([0, 1, 0.0]),
        np.array([0, 0, 1, 0, 0.0])
    ]

    rc_modes = [RangeComp.Mode.Full, RangeComp.Mode.Same, RangeComp.Mode.Valid]
    for mode in rc_modes:
        for cb_filter in cb_filters:
            cb_chirp = np.convolve(cb_filter, chirp, mode="full")
            rc = RangeComp(cb_chirp, nraw, mode=mode)
            rcdata = np.zeros(rc.output_size, "c8")
            rc.rangecompress(rcdata, rawdata)
            i = rc.first_valid_sample - (len(cb_filter) - 1) // 2
            # NOTE negative for Mode.Valid is correct but not testable this way
            if i >= 0:
                npt.assert_allclose(rcdata[i], 1.0, atol=1.2e-7, err_msg=
                    f"Incorrect delay for mode={mode} filter={cb_filter}")


def test_rangecomp():
    # simulate raw data
    zsim = np.zeros(2048, dtype="c8")
    zsim[0] = 1.0
    chirp = np.array(form_linear_chirp(K, T, fs), "c8")
    zraw = np.convolve(chirp, zsim, mode="full")
    traw = np.arange(len(zraw))

    # rangecomp at full res
    rc = RangeComp(chirp, len(zraw))
    zrc = np.zeros(rc.output_size, "c8")
    trc = np.arange(len(zrc)) - rc.first_valid_sample
    rc.rangecompress(zrc, zraw)

    # rangecomp common band
    # apply filter to chirp so that rangecomp does both
    h, shift = get_common_band_filter(L40, L20, fs)
    chirp_cb = np.convolve(h, chirp, mode="full")

    rc_cb = RangeComp(chirp_cb, len(zraw))
    zrc_cb = np.zeros(rc_cb.output_size, "c8")

    # RangeComp.first_valid_sample is no longer correct because it bookkeeps
    # full filter length as delay, but CB filter delay is only half.
    trc_cb = np.arange(len(zrc_cb)) - (len(chirp) - 1) - (len(h) - 1) // 2

    # Compress and shift to baseband using offset calculated by CB design.
    rc_cb.rangecompress(zrc_cb, zraw)
    zrc_cb *= np.exp(1j * shift * trc_cb)

    if plot:
        import matplotlib.pyplot as p
        p.plot(trc, abs(zrc))
        p.plot(trc_cb, abs(zrc_cb))
        p.show()

    # Target should show up in expected location.
    i = np.argmax(np.abs(zrc))
    j = np.argmax(np.abs(zrc_cb))
    npt.assert_almost_equal(trc[i], 0.0)
    npt.assert_almost_equal(trc_cb[j], 0.0)

    # Amplitude should be half given how things have been normalized.
    npt.assert_allclose(np.abs(zrc_cb[j]), 0.5 * np.abs(zrc[i]), rtol=0.02)

    # Phase should be same.
    # FIR filter has linear phase, so random error is due to FFT precision.
    # see https://en.wikipedia.org/wiki/Fast_Fourier_transform#Accuracy
    # Note tan(x)≈x for small x.
    atol = np.log2(rc_cb.fft_size) * np.finfo(np.float32).eps
    npt.assert_allclose(np.angle(zrc_cb[j]), np.angle(zrc[i]), atol=atol, rtol=0)

    # Image should be baseband.
    carrier = np.angle(np.sum(zrc_cb[1:] * zrc_cb[:-1].conj()))
    npt.assert_allclose(carrier, 0.0, atol=0.01, rtol=0)
