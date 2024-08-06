import os
import argparse

import pytest

from nisar.workflows.noise_estimator import run_noise_estimator
import iscetest


@pytest.mark.parametrize(
    "algorithm,num_rng_block,cpi,perc_invalid_rngblk,"
    "plot,no_diff,no_median_ev,exclude_first_last,diff_method",
    [
        ('MVE', None, 3, 5, False, False, False, True, 'mean'),
        ('MVE', None, 3, 5, False, False, False, False, 'diff'),
        ('MVE', 50, 3, 20, True, False, False, False, 'single'),
        ('MEE', None, 4, 5, False, True, True, True, 'single'),
        ('MEE', 50, None, 20, True, False, False, False, 'single'),
    ]
)
def test_noise_estimator(algorithm, num_rng_block,
                         cpi, perc_invalid_rngblk,
                         plot, no_diff, no_median_ev,
                         exclude_first_last, diff_method):
    # sub directory for all test files under "isce3/tests/data"
    sub_dir = 'bf'
    # sample single-pol single-band L0B file (noise-only)
    l0b_name = 'REE_L0B_ECHO_DATA_NOISE_EST.h5'

    # l0b filepath
    l0b_file = os.path.join(iscetest.data, sub_dir, l0b_name)

    # form the input args
    args = argparse.Namespace(
        l0b_file=l0b_file,
        algorithm=algorithm,
        num_rng_block=num_rng_block,
        cpi=cpi,
        pct_invalid_rngblk=perc_invalid_rngblk,
        plot=plot,
        output_path='.',
        json_file='noise_power_est_info.json',
        diff_quad=False,
        no_diff=no_diff,
        no_median_ev=no_median_ev,
        exclude_first_last=exclude_first_last,
        diff_method=diff_method
    )
    run_noise_estimator(args)
