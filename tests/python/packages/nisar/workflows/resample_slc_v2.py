import argparse
import os

import numpy as np

from nisar.workflows import resample_slc_v2
from nisar.workflows.resample_slc_runconfig import ResampleSlcRunConfig

import iscetest


def test_resample_slc_v2_run():
    '''
    Run resample slc
    '''

    # load yaml file
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')

    # Load text and substitute directory path
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data).\
                replace('@TEST_OUTPUT@', 'rifg.h5').\
                replace('@TEST_PRODUCT_TYPES@', 'RIFG').\
                replace('@TEST_RDR2GEO_FLAGS@', 'True')

    # create CLI input namespace
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    runconfig = ResampleSlcRunConfig(args, 'coarse')
    runconfig.geocode_common_arg_load()

    runconfig.cfg["product_path_group"]["scratch_path"] = "resamp_slc_v2"

    resample_slc_v2.run(runconfig.cfg, 'coarse')


def test_resample_slc_v2_validate():
    '''
    Validate resample_slc output VS golden dataset
    '''

    scratch_path = 'resamp_slc_v2'
    ref_slc = np.fromfile(os.path.join(iscetest.data, 'warped_winnipeg.slc'),
                          dtype=np.complex64).reshape(250, 250)[20:-20, 20:-20]
    test_slc = np.fromfile(os.path.join(scratch_path, 'coarse_resample_slc', 'freqA',
                                        'HH', 'coregistered_secondary.slc'), dtype=np.complex64).reshape(250, 250)[
               20:-20, 20:-20]

    # Quantify error
    error = np.nanmean(np.abs(ref_slc - test_slc))

    # Check error magnitude
    assert error < 5.0e-5
