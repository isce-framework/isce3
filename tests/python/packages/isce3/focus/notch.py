from isce3.focus import Notch, FrequencyDomain, form_linear_chirp, RangeComp
import numpy.testing as npt

def test_notch():
    # Conversion between str and enum
    assert FrequencyDomain("baseband") == FrequencyDomain.BASEBAND == "baseband"
    assert FrequencyDomain("radio_frequency")

    carrier = 1257e6
    fs = 96e6
    notch_freq_rf = 1260e6
    notch_freq_bb = notch_freq_rf - carrier
    notch_bandwidth = 0.5e6

    # ctor
    notch1 = Notch(notch_freq_bb, notch_bandwidth, "baseband")
    notch2 = Notch(notch_freq_rf, notch_bandwidth, "radio_frequency")

    # Normalization
    assert notch1.normalized(fs).frequency == notch_freq_bb / fs
    assert notch1.normalized(fs).bandwidth == notch_bandwidth / fs

    # Equivalence of notches after normalization
    assert notch1.normalized(fs) == notch2.normalized(fs, carrier)


def test_rangecomp():
    # Generate a representative chirp.
    duration = 40e-6
    bandwidth = 40e6
    fs = 48e6
    chirp = form_linear_chirp(bandwidth / duration, duration, fs)

    # Use it to construct a range compression object and get its spectrum.
    rc = RangeComp(chirp, 8 * len(chirp))
    chirp_spectrum_before = rc.get_chirp_spectrum()

    # Check DC level is nonzero.  FFTW order, DC should be in first bin.
    assert abs(chirp_spectrum_before[0]) > 0.0

    # Apply DC notch and check it got zeroed-out.
    notch = Notch(0, 1e6, "baseband").normalized(fs)
    rc.apply_notch(notch.frequency, notch.bandwidth)
    chirp_spectrum_after = rc.get_chirp_spectrum()
    assert abs(chirp_spectrum_after[0]) == 0.0
