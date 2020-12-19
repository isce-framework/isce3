import numpy as np
import numpy.testing as npt
import pybind_isce3 as isce3


def test_crossmultiply():

    nrows = 100
    ncols = 200
    crsmulObj = isce3.signal.CrossMultiply(nrows, ncols)

    assert crsmulObj.nrows == nrows
    assert crsmulObj.ncols == ncols
    assert crsmulObj.upsample_factor == 2
    assert crsmulObj.fftsize >= ncols

    x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    phase1 = np.sin(10.0 * np.pi * y / ncols)
    phase2 = np.sin(8.0 * np.pi * y / ncols)
    slc1 = np.exp(1j * phase1)
    slc2 = np.exp(1j * phase2)
    expected_ifgram = np.exp(1j * (phase1 - phase2))
    ifgram = np.zeros((nrows, ncols), dtype=np.complex64)
    crsmulObj.crossmultiply(ifgram, slc1, slc2)
    diff = np.max(np.abs(np.angle(ifgram * np.conjugate(expected_ifgram))))
    npt.assert_allclose(diff, 0.0, atol=1e-6, rtol=0.0)


def test_crossmultiply_flatten():

    nrows = 100
    ncols = 50
    upsample = 1
    crsmulObj = isce3.signal.CrossMultiply(nrows, ncols, upsample)

    wvl = 0.23
    range_spacing = 7.0
    ref_starting_range = 800000.0
    sec_starting_range = 800020.0

    x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    ref_rng = ref_starting_range + x * range_spacing
    sec_rng = sec_starting_range + (x + 1) * range_spacing
    range_offset = (ref_rng - sec_rng) / range_spacing
    delay = 0.1 * np.sin(np.pi * x / ncols)

    ref_phase = 4.0 * np.pi * ref_rng / wvl
    ref_slc = np.exp(1j * ref_phase)

    sec_phase = 4.0 * np.pi * (sec_rng + delay) / wvl
    sec_slc = np.exp(1j * sec_phase)

    expected_ifgram = np.exp(1j * (-4.0 * np.pi * delay / wvl))

    ifgram = np.zeros((nrows, ncols), dtype=np.complex64)
    crsmulObj.crossmultiply(ifgram, ref_slc, sec_slc)

    isce3.signal.flatten(ifgram, range_offset, range_spacing, wvl)

    diff = np.max(np.abs(np.angle(ifgram * np.conjugate(expected_ifgram))))
    npt.assert_allclose(diff, 0.0, atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    test_crossmultiply()
    test_crossmultiply_flatten()
