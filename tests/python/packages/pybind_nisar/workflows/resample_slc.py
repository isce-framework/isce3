import argparse
import os

import numpy as np

from pybind_nisar.workflows import resample_slc
from pybind_nisar.workflows.resample_slc_runconfig import ResampleSlcRunConfig

import iscetest


def test_resample_slc_run():
    '''
    Run resample slc
    '''

    # load yaml file
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')

    # Load text and substitute directory path
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data).\
                replace('@TEST_OUTPUT@', 'rifg.h5').\
                replace('@TEST_PRODUCT_TYPES@', 'RIFG')

    # create CLI input namespace
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    runconfig = ResampleSlcRunConfig(args, 'coarse')
    runconfig.geocode_common_arg_load()

    resample_slc.run(runconfig.cfg, 'coarse')


def test_resample_slc_validate():
    '''
    Validate resample_slc output VS golden dataset
    '''

    scratch_path = '.'
    ref_slc = np.fromfile(os.path.join(iscetest.data, 'warped_winnipeg.slc'),
                          dtype=np.complex64).reshape(250, 250)[20:-20, 20:-20]
    test_slc = np.fromfile(os.path.join(scratch_path, 'coarse_resample_slc', 'freqA',
                                        'HH', 'coregistered_secondary.slc'), dtype=np.complex64).reshape(250, 250)[
               20:-20, 20:-20]

    # Quantify error
    error = np.nanmean(np.abs(ref_slc - test_slc))

    # Check error magnitude
    assert (error < 1.0e-6), f'CPU resample_slc error {error} > 1e-6'
