import iscetest
import numpy as np
from numpy.random import randn, randint
import numpy.testing as npt
import pytest
from isce3.signal.compute_evd_cpi import compute_evd
from isce3.signal.rfi_detection_evd import rfi_detect_evd
from isce3.signal.rfi_mitigation_evd import rfi_mitigate_evd


def raw_data_gen(
    nb_pwr_db, wb_pwr_db, noise_pwr_db, fc, fs, num_pulses, num_rng_samples
):
    raw_data = np.zeros((num_pulses, num_rng_samples), dtype=np.complex64)

    # Set Random number seed
    np.random.seed(0)

    # Narrowband RFI
    nb_mag = 10 ** (nb_pwr_db / 20)
    ts = 1 / fs
    t = np.linspace(0, (num_rng_samples - 1) * ts, num_rng_samples)
    nb_pulse = nb_mag * np.exp(1j * 2 * np.pi * fc * t)
    nb_data = np.tile(nb_pulse, (num_pulses, 1))

    # Only some of the pulses are corrupted by NB RFI
    # Null NB RFI in some of the pulses defined by nb_null_pulse_idx
    num_sig_pulses = num_pulses // 3
    nb_null_pulse_idx = np.random.choice(
        range(num_pulses), size=int(num_sig_pulses), replace=False
    )
    nb_data[nb_null_pulse_idx] = 0.000001 * (
        randn(num_sig_pulses, num_rng_samples).astype(np.float32)
        + 1j * randn(num_sig_pulses, num_rng_samples).astype(np.float32)
    )

    # Broadband RFI Data
    wb_mag = 10 ** (wb_pwr_db / 20) / np.sqrt(2)
    wb_pulse = wb_mag * (
        randn(num_rng_samples).astype(np.float32)
        + 1j * randn(num_rng_samples).astype(np.float32)
    )

    wb_data = np.tile(wb_pulse, (num_pulses, 1))

    # Null WB RFI in some of the pulses
    wb_null_pulse_idx = np.random.choice(
        range(num_pulses), size=int(num_sig_pulses), replace=False
    )
    wb_data[wb_null_pulse_idx] = 0.000001 * (
        randn(num_sig_pulses, num_rng_samples).astype(np.float32)
        + 1j * randn(num_sig_pulses, num_rng_samples).astype(np.float32)
    )

    # Add Random Noise
    noise_mag = 10 ** (noise_pwr_db / 20) / np.sqrt(2)
    noise_data = noise_mag * (
        randn(num_pulses, num_rng_samples).astype(np.float32)
        + 1j * randn(num_pulses, num_rng_samples).astype(np.float32)
    )

    # Decorrelate RFI between pulse to pulse
    for idx_pulse in range(num_pulses):
        wb_data[idx_pulse] = np.roll(wb_data[idx_pulse], randint(16))
        nb_data[idx_pulse] = np.roll(nb_data[idx_pulse], randint(6))
        raw_data[idx_pulse] = wb_data[idx_pulse] + nb_data[idx_pulse]

    #Remove the possibilty of any RFI in the last partial block
    num_pulses_remaining = 10
    raw_data[-num_pulses_remaining:] =  0.000001 * (
        randn(num_pulses_remaining, num_rng_samples).astype(np.float32) + 1j * randn(num_pulses_remaining, num_rng_samples).astype(np.float32)
    )

    #Add random noise to raw data
    raw_data += noise_data

    return raw_data


@pytest.mark.parametrize(
    "cpi_len, num_rng_blks, detect_threshold, rfi_ev_idx_stop",
    [
        (32, 1, 1, 16),     #No range blocking
        (64, 6, 0.75, 32),  #Range blocking: 6 range blocks
        (32, 1, 20, 16),    #No-op with 20 dB/idx threshold
        (64, 6, 20, 10),    #No-op with 20 dB/idx threshold
    ],
)
def test_rfi_az_evd(cpi_len, num_rng_blks, detect_threshold, rfi_ev_idx_stop):
    """Verify slow-time EVD with four test cases:
    1: cpi_len = 32, no range blocking, threshold = 1 dB / EV idx, stop_idx = 16
    2: cpi_len = 64, 6 range blocks, threshold = 0.75 dB / EV idx, stop_idx = 32
    3: No-op: cpi_len = 32, no range blocking, threshold = 20 dB / EV idx, stop_idx = 16
    4: No-op: cpi_len = 64, 6 range blocks, threshold = 20 dB / EV idx, stop_idx = 10
    """
    # Raw data parameters
    nb_pwr_db = 20
    wb_pwr_db = 16
    noise_pwr_db = 10
    fc = 1e6
    fs = 2 * fc

    num_pulses = 648
    num_rng_samples = 1200

    # Generate raw data with RFI
    raw_data = raw_data_gen(
        nb_pwr_db, wb_pwr_db, noise_pwr_db, fc, fs, num_pulses, num_rng_samples
    )

    # Compute Eigenvalues and Eigenvectors
    (
        eig_val_sort_array,
        eig_vec_sort_array,
    ) = compute_evd(raw_data, cpi_len, num_rng_blks)

    rfi_cpi_flag_array = rfi_detect_evd(
        eig_val_sort_array, rfi_ev_idx_stop, detect_threshold
    )

    # Test EVD mitigation
    data_mitigated = np.zeros(raw_data.shape, dtype=np.complex64)

    rfi_mitigate_evd(
        raw_data,
        eig_vec_sort_array,
        rfi_cpi_flag_array,
        cpi_len,
        data_mitigated,
    )

    # Verify Mitigation results
    rfi_residual = 0.5 
    if detect_threshold < 3:  #Mitigated
        max_pulse_pwr_db_exp = noise_pwr_db + rfi_residual

        # Maximum distributed target power leve is set at 10 dB
        # Given some residual RFI, maximum pulse power of mitigated raw data
        # should be less than 10.5 dB

        data_mitigated_pulse_pwr_db = 10*np.log10(np.var(data_mitigated, axis=1))
        max_data_mitigated_pulse_pwr_db = np.max(data_mitigated_pulse_pwr_db)

        npt.assert_allclose(
            max_data_mitigated_pulse_pwr_db,
            max_pulse_pwr_db_exp,
            rtol = 0.0,
            atol = 0.5,
        )
    # With relaxed threshold, input raw data should be exactly the same as output
    # data
    elif detect_threshold > 10:  #No-op: no mitigation
        npt.assert_equal(
            raw_data,
            data_mitigated,
            'Raw data and output no-op data are not equal.',
        )



