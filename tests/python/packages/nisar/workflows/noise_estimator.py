import iscetest
import numpy as np
import numpy.testing as npt
import os
from nisar.workflows.noise_estimator import extract_cal_lines, noise_est_avg, noise_est_evd

def get_test_file():
    raw_data_file = os.path.join(iscetest.data, "bf", "REE_L0B_ECHO_DATA_NOISE_EST.h5")

    return raw_data_file

def read_cal_lines():
    freq_group = 'A'
    pols = 'HH'
    raw_data_file = get_test_file()

    raw_cal_lines = extract_cal_lines(raw_data_file, freq_group, pols)

    return raw_cal_lines

def test_noise_avg_estimate_entire_rng_line():
    noise_est_bench = 41.82
    noise_est_margin = 0.1
    raw_cal_lines = read_cal_lines()
    noise_pwr_beam = noise_est_avg(raw_cal_lines)
    
    noise_est_error = noise_pwr_beam - noise_est_bench

    npt.assert_array_less(
        noise_est_error,
        noise_est_margin,
        "Noise power estimate error is larger than error margin",
    )

    return noise_pwr_beam

def test_noise_avg_estimate_single_beam():
    noise_est_bench = 35.15
    noise_est_margin = 0.5
    rng_start = [2000]
    rng_stop = [4000]
    raw_cal_lines = read_cal_lines()
    noise_pwr_beam = noise_est_avg(raw_cal_lines, rng_start, rng_stop)
    
    noise_est_error = noise_pwr_beam - noise_est_bench

    npt.assert_array_less(
        noise_est_error,
        noise_est_margin,
        "Noise power estimate error is larger than error margin",
    )

    return noise_pwr_beam

def test_noise_avg_estimate_multiple_beams():
    noise_est_bench = 35.15
    noise_est_margin = 0.5
    rng_start = [2000, 4000, 8000, 10000, 14000, 20000, 25000]
    rng_stop = [4000, 6000, 10000, 14000, 20000, 25000, 28000]
    raw_cal_lines = read_cal_lines()
    noise_pwr_beam = noise_est_avg(raw_cal_lines, rng_start, rng_stop)
    
    noise_est_error = noise_pwr_beam - noise_est_bench

    npt.assert_array_less(
        noise_est_error,
        noise_est_margin,
        "Noise power estimate error is larger than error margin",
    )

    return noise_pwr_beam

def test_noise_evd_estimate_entire_rng_line():
    noise_est_bench = 35.15
    noise_est_margin = 0.5
    cpi = 2
    raw_cal_lines = read_cal_lines()
    noise_pwr_beam = noise_est_evd(raw_cal_lines, cpi)
    
    noise_est_error = noise_pwr_beam - noise_est_bench

    npt.assert_array_less(
        noise_est_error,
        noise_est_margin,
        "Noise power estimate error is larger than error margin",
    )

    return noise_pwr_beam

def test_noise_evd_estimate_single_beam():
    noise_est_bench = 35.15
    noise_est_margin = 0.5
    cpi = 2
    rng_start = [2000]
    rng_stop = [4000]
    raw_cal_lines = read_cal_lines()
    noise_pwr_beam = noise_est_evd(raw_cal_lines, cpi, rng_start, rng_stop)
    
    noise_est_error = noise_pwr_beam - noise_est_bench

    npt.assert_array_less(
        noise_est_error,
        noise_est_margin,
        "Noise power estimate error is larger than error margin",
    )

    return noise_pwr_beam

def test_noise_evd_estimate_multiple_beams():
    noise_est_bench = 35.15
    noise_est_margin = 0.5
    cpi = 8
    rng_start = [2000, 4000, 8000, 10000, 14000, 20000, 25000]
    rng_stop = [4000, 6000, 10000, 14000, 20000, 25000, 28000]
    raw_cal_lines = read_cal_lines()
    noise_pwr_beam = noise_est_evd(raw_cal_lines, cpi, rng_start, rng_stop)
    
    noise_est_error = noise_pwr_beam - noise_est_bench

    npt.assert_array_less(
        noise_est_error,
        noise_est_margin,
        "Noise power estimate error is larger than error margin",
    )

    return noise_pwr_beam

