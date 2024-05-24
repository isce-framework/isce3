import iscetest
import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from numpy.random import randn, randint, uniform
import numpy.testing as npt
import pytest
from isce3.signal.rfi_freq_null import run_freq_notch

def gaussian_data_gen(num_pulses, num_rng_samples, pwr_db):
    """Generate normally distributed signals of desired input power.

    Parameters
    ----------
    num_pulses: int
        number of pulses in the output
    num_range_samples: int
        number of range samples in the output
    sig_pwr_db: float
        desired signal power of the output in dB

    Returns
    -------
    random_data: 2D array of complex64
        Normally distributed random complex output [num_pulses x num_range_samples]
    """

    mag = 10 ** (pwr_db / 20)
    random_data = mag / np.sqrt(2) * (
        randn(num_pulses, num_rng_samples).astype(np.float32)
        + 1j * randn(num_pulses, num_rng_samples).astype(np.float32)
    )

    return random_data

def rfi_nb_gen(
    raw_data,
    freq_rfi_nb,
    fs,
    rfi_nb_pulse_idx,
    pwr_rfi_nb_db,
):
    """Generate complex sinusoidal tone as narrowband RFI.

    Parameters
    ----------
    raw_data: 2D array of complex64
        raw data input
    freq_rfi_nb: array of float
        narrowband RFI center frequencies
    fs: float
        sampling frequency
    rfi_nb_pulse_idx: array of int
        index of pulses to be contaminated by narrowband RFI.
        These pulses are contaminated by all narrowband RFI emitters.
    pwr_rfi_nb_db: array of float
        narrowband RFI power in dB

    Returns
    -------
    raw_data_nb_rfi: same data type and shape as input raw_data
        raw_data output contaminated by narrowband RFI
    """

    raw_data_nb_rfi = raw_data.copy()
    num_rfi_nb = pwr_rfi_nb_db.size

    # Verify RFI frequency is less than sampling frequency
    if np.any(freq_rfi_nb > fs):
        raise ValueError(
            f"RFI frequency of {freq_rfi_nb} Hz must be less than sampling frequency of {fs} Hz!"
        )

    # Verify dimensions for NB RFI parameters are correct
    assert (
        len(freq_rfi_nb) == num_rfi_nb
    ), "Length of freq_rfi_nb list must be equal to the number of NB RFI"
    assert (
        len(pwr_rfi_nb_db) == num_rfi_nb
    ), "Length of pwr_rfi_nb_db list must be equal to the number of NB RFI"

    num_pulses, num_rng_samples = raw_data.shape

    ts = np.linspace(0, (num_rng_samples - 1) * (1 / fs), num_rng_samples)
    tone_rfi_sum = np.zeros([num_rng_samples], dtype=np.complex64)

    mag_rfi = 10 ** (pwr_rfi_nb_db / 20)

    # Simple NB RFI model with each contaminated line having the same RFI
    for k in range(num_rfi_nb):
        # Apply a uniformly distributed random phase offset
        phase_rand = uniform(-np.pi, np.pi)
        tone_rfi = (
            mag_rfi[k] * np.exp(1j * (2 * np.pi * freq_rfi_nb[k] * ts  + phase_rand ))
        )
        tone_rfi_sum += tone_rfi

    raw_data_nb_rfi[rfi_nb_pulse_idx] += tone_rfi_sum

    return raw_data_nb_rfi

def rfi_wb_gen(
    raw_data,
    rfi_wb_pulse_start,
    rfi_wb_pulse_stop,
    rfi_wb_sample_start,
    rfi_wb_sample_stop,
    pwr_rfi_wb_db,
):
    """Generate white barrage type of RFI as wideband RFI in time domain.

    Parameters
    ----------
    raw_data: 2D array of complex64
        raw data input
    rfi_wb_pulse_start: array of int
        first pulse of each wideband RFI
    rfi_wb_pulse_stop: array of int
        last pulse of each wideband RFI
    rfi_wb_sample_start: array of int
        first range sample of wideband RFI
    rfi_wb_sample_stop: array of int
        last range sample of wideband RFI
    pwr_rfi_wb_db: array of float
        wideband RFI power in dB

    Returns
    -------
    raw_data_wb_rfi: same data type as raw_data input
        raw_data output contaminated by wideband RFI
    """

    raw_data_wb_rfi = raw_data.copy()
    num_wb_rfi = pwr_rfi_wb_db.size

    num_pulses_rfi_wb = rfi_wb_pulse_stop - rfi_wb_pulse_start
    num_samples_rfi_wb = rfi_wb_sample_stop - rfi_wb_sample_start

    # Confirm RFI input parameter lengths matest_caseh with number of WB RFI
    assert (
        len(num_samples_rfi_wb) == num_wb_rfi
    ), "Length of num_samples_rfi_wb list must be equal to the number of WB RFI"
    assert (
        len(pwr_rfi_wb_db) == num_wb_rfi
    ), "Length of pwr_rfi_wb_db list must be equal to the number of WB RFI"

    # WB RFI Power
    mag_rfi_wb = 10 ** (pwr_rfi_wb_db / 20)

    # Generate white random noise from frequency domain
    for idx_rfi in range(num_wb_rfi):
        rfi_az_blk = np.s_[rfi_wb_pulse_start[idx_rfi] : rfi_wb_pulse_stop[idx_rfi]]
        rfi_rng_blk = np.s_[
            rfi_wb_sample_start[idx_rfi] : rfi_wb_sample_stop[idx_rfi]
        ]

        # Generate wideband RFI
        rfi_wb_az = gaussian_data_gen(
            num_pulses_rfi_wb[idx_rfi],
            num_samples_rfi_wb[idx_rfi],
            pwr_rfi_wb_db[idx_rfi],
        )

        raw_data_wb_rfi[rfi_az_blk, rfi_rng_blk] = rfi_wb_az + raw_data[rfi_az_blk, rfi_rng_blk]


    return raw_data_wb_rfi

def test_freq_null():
    """Test RFI detection and mitigation using frequency domain nulling
    algorithms by Franz Meyer, J. Nicoll, and A. Doulgeris,
    “Correction and Characterization of Radio Frequency Interference
    Signatures in L-Band Synthetic Aperture Radar Data"
    """

    # Set pseudo random number generator seed
    np.random.seed(0)

    # Raw data parameters
    num_pulses = 2000
    num_rng_samples = 2148

    # Noise Power
    noise_pwr_db = 0

    # Generate raw data and systems noise
    sig_pwr_db = 8

    # signal data is of normally distribution
    sig_data = gaussian_data_gen(num_pulses, num_rng_samples, sig_pwr_db)

    # Noise is normally distributed
    noise_data = gaussian_data_gen(num_pulses, num_rng_samples, noise_pwr_db)

    # Signal + Noise
    raw_data = sig_data + noise_data

    # Narrowband RFI parameters
    num_nb_rfi = 10

    # Interference to Noise Ratio in dB
    inr_nb_low = 15
    inr_nb_high = 18
    pwr_nb_rfi_db = randint(inr_nb_low + noise_pwr_db, inr_nb_high + noise_pwr_db, num_nb_rfi)

    # Sampling Frequency
    freq_fs_factor = 20
    fs_nb = freq_fs_factor * 1e6

    # Narrowband RFI tone frequencies
    num_rfi_nb_pulses = num_pulses // 50
    freq_nb_tone = 1e6 * randint(1, freq_fs_factor // 2, num_nb_rfi)
    rfi_nb_pulse_idx = randint(50, num_pulses, num_rfi_nb_pulses)

    # Wideband RFI parameters
    num_wb_rfi = 10
    inr_wb = 20
    pwr_rfi_wb_db = inr_wb * np.ones(num_wb_rfi)

    # WB RFI contaminated pulses
    min_rfi_wb_pulses = 10
    max_rfi_wb_pulses = 12
    num_wb_rfi_pulses = randint(min_rfi_wb_pulses, max_rfi_wb_pulses, num_wb_rfi)
    rfi_wb_pulse_start = randint(
        50, num_pulses - 2 * num_wb_rfi_pulses.max(), num_wb_rfi
    )
    rfi_wb_pulse_stop = rfi_wb_pulse_start + num_wb_rfi_pulses

    # Randomly set number of range samples contaminated by RFI
    rfi_wb_sample_start = 128
    rfi_wb_sample_end = 256

    num_wb_rfi_samples = randint(rfi_wb_sample_start, rfi_wb_sample_end, num_wb_rfi)

    # Randomly select starting range sample contaminated by RFI
    rfi_wb_sample_start = randint(
        1, num_rng_samples - 2 * num_wb_rfi_samples, num_wb_rfi
    )
    rfi_wb_sample_stop = rfi_wb_sample_start + num_wb_rfi_samples

    # Generate Narrowband RFI
    raw_data_nb = rfi_nb_gen(
        raw_data, freq_nb_tone, fs_nb, rfi_nb_pulse_idx, pwr_nb_rfi_db
    )

    # Generate Wideband RFI
    raw_data_rfi = rfi_wb_gen(
        raw_data_nb,
        rfi_wb_pulse_start,
        rfi_wb_pulse_stop,
        rfi_wb_sample_start,
        rfi_wb_sample_stop,
        pwr_rfi_wb_db,
    )

    # Frequency Domain Nulling Parameters
    az_winsize = 25
    rng_winsize = 22
    num_rng_blks = 3
    num_pulses_az_blk = 600
    trim_frac = 0.01
    pvalue_threshold = 0.005
    cdf_threshold = 0.68
    nb_detect = True
    wb_detect = True
    mitigate_enable = True
    raw_data_mitigated = np.zeros(raw_data.shape, dtype=raw_data.dtype)

    # Run Frequency Domain Notch Filtering
    rfi_likelihood = run_freq_notch(
        raw_data_rfi,
        num_pulses_az_blk,
        num_rng_blks=num_rng_blks,
        az_winsize=az_winsize,
        rng_winsize=rng_winsize,
        trim_frac=trim_frac,
        pvalue_threshold=pvalue_threshold,
        cdf_threshold=cdf_threshold,
        nb_detect=nb_detect,
        wb_detect=wb_detect,
        mitigate_enable=mitigate_enable,
        raw_data_mitigated=raw_data_mitigated,
    )

    # Compare maximum pulse power before and after RFI mitigation
    rfi_residue = 1  # Residual RFI power in dB
    raw_data_pulse_pwr_db = 10 * np.log10(np.var(raw_data, axis=1))
    raw_miti_pulse_pwr_db = 10 * np.log10(np.var(raw_data_mitigated, axis=1))

    max_raw_pulse_pwr_db = raw_data_pulse_pwr_db.max()
    max_miti_pulse_pwr_db = raw_miti_pulse_pwr_db.max()

    npt.assert_allclose(
        max_raw_pulse_pwr_db,
        max_miti_pulse_pwr_db,
        atol=rfi_residue,
    )
