import iscetest
import numpy as np
from numpy.random import randn, randint, uniform
import numpy.testing as npt
import pytest
from isce3.signal.rfi_process_evd import run_slow_time_evd
from isce3.signal.rfi_detection_evd import ThresholdParams


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


@pytest.mark.parametrize(
    "cpi_len, num_rng_blks, num_cpi_tb, max_deg_freedom, num_max_trim, num_min_trim, max_num_rfi_ev, mitigate_enable, test_case",
    [  
        (32, 1, 20, 16, 1, 1, 2, True, 'mitigate'),  # No range blks
        (32, 8, 20, 16, 1, 1, 2, True, 'mitigate'),  # 8 range blocks
        (32, 1, 20, 16, 0, 0, 2, True, 'no-op'),  # No-op: no rng blks
        (32, 8, 20, 16, 0, 0, 2, True, 'no-op'),  # No-op: 8 blocks
        (32, 1, 20, 16, 0, 0, 2, False,'no-op'),  # No-op: detection only
    ],
)
def test_slow_time_evd(
    cpi_len,
    num_rng_blks,
    num_cpi_tb,
    max_deg_freedom,
    num_max_trim,
    num_min_trim,
    max_num_rfi_ev,
    mitigate_enable,
    test_case,
):
    """Verify slow-time EVD with five test cases:
    1: cpi_len=32, no range blocking, 20 cpi/thresh blk, max_deg_freedom=16,
       num_max_trim=1, num_min_trim=1, max_num_rfi_ev=2, mitigate_enable=True, test_case=mitigate
    2: cpi_len=32, 8 range blocks, 20 cpi/thresh blk, max_deg_freedom=16,
       num_max_trim=1, num_min_trim=1, max_num_rfi_ev=2, mitigate_enable=True, test_case=mitigate
    3: cpi_len=32, no range blocking, 20 cpi/thresh blk, max_deg_freedom=16,
       num_max_trim=1, num_min_trim=1, max_num_rfi_ev=2, mitigate_enable=True, test_case=no-op
    4: cpi_len=32, 8 range blocks, 20 cpi/thresh blk, max_deg_freedom=16,
       num_max_trim=1, num_min_trim=1, max_num_rfi_ev=2, mitigate_enable=True, test_case=no-op
    5: cpi_len=32, no range blocking, 20 cpi/thresh blk, max_deg_freedom=16, 
       num_max_trim=1, num_min_trim=1, max_num_rfi_ev=2, mitigate_enable=False, 
       test_case=no-op, detection only

    """
    """Test slow-time EVD with and without range blocking."""

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

    # If test case is mitigate, add narrowband and wideband RFI
    if test_case == 'mitigate':
        # Narrowband RFI parameters
        num_nb_rfi = 10

        # Interference to Noise Ratio
        inr_nb_low = 15
        inr_nb_high = 18
        pwr_nb_rfi_db = randint(inr_nb_low, inr_nb_high, num_nb_rfi)

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

        num_wb_rfi_sampless = randint(rfi_wb_sample_start, rfi_wb_sample_end, num_wb_rfi)

        # Randomly select starting range sample contaminated by RFI
        rfi_wb_sample_start = randint(
            1, num_rng_samples - 2 * num_wb_rfi_sampless, num_wb_rfi
        )
        rfi_wb_sample_stop = rfi_wb_sample_start + num_wb_rfi_sampless

        # Generate Narrowband RFI
        raw_data_nb = rfi_nb_gen(
            raw_data, freq_nb_tone, fs_nb, rfi_nb_pulse_idx, pwr_nb_rfi_db
        )

        # Generate Wideband RFI: Y = S + N + RFI_NB + RFI_WB
        raw_data_rfi = rfi_wb_gen(
            raw_data_nb,
            rfi_wb_pulse_start,
            rfi_wb_pulse_stop,
            rfi_wb_sample_start,
            rfi_wb_sample_stop,
            pwr_rfi_wb_db,
        )

        raw_data_mitigated = np.zeros(raw_data.shape, dtype=raw_data.dtype)
    elif test_case == 'no-op':
        raw_data_mitigated = raw_data.copy()
        raw_data_rfi = raw_data

    # Perform Slow-Time EVD Detection and Mitigation
    # Compute Eigenvalues and Eigenvectors

    threshold_params = ThresholdParams([2, 20], [5, 2])

    rfi_likelihood = run_slow_time_evd(
        raw_data_rfi,
        cpi_len,
        max_deg_freedom,
        num_max_trim=num_max_trim,
        num_min_trim=num_min_trim,
        max_num_rfi_ev=max_num_rfi_ev,
        num_rng_blks=num_rng_blks,
        threshold_params=threshold_params,
        num_cpi_tb=num_cpi_tb,
        mitigate_enable=mitigate_enable,
        raw_data_mitigated=raw_data_mitigated,
    )

    # test_case 1 mitigate: Verify mitigation results: compare max pulse power
    # test_case 2 no-op: Verify raw data without RFI does not get altered by ST-EVD.

    # Compute raw data pulse power before and after RFI mitigation
    if test_case == 'mitigate':
        rfi_residue = 1  # Residual RFI power in dB
        raw_data_pulse_pwr_db = 10 * np.log10(np.var(raw_data, axis=1))
        raw_miti_pulse_pwr_db = 10 * np.log10(np.var(raw_data_mitigated, axis=1))

        max_raw_pulse_pwr_db = raw_data_pulse_pwr_db.max()
        max_miti_pulse_pwr_db = raw_miti_pulse_pwr_db.max()

        npt.assert_allclose(
            max_raw_pulse_pwr_db,
            max_miti_pulse_pwr_db,
            rtol=0.0,
            atol=rfi_residue,
        )
    elif test_case == 'no-op':  # test_case 2: compare raw data against mitigated data
        npt.assert_equal(
            raw_data_rfi,
            raw_data_mitigated,
            "Raw data and output data of No-op test case are not equal.",
        )
