import argparse
import os

import numpy as np

import isce3
from nisar.workflows import resample_slc
from nisar.workflows.resample_slc_runconfig import ResampleSlcRunConfig

import iscetest


# Original test output to be renamed based on processing type (CPU, GPU).
test_output= os.path.join('.', 'coarse_resample_slc', 'freqA', 'HH',
                          'coregistered_secondary.slc')


def test_resample_slc_run():
    '''
    Run resample slc
    '''
    test_yaml_path = os.path.join(iscetest.data, 'insar_test.yaml')

    # Iterate over processing type.
    for pu in ['cpu', 'gpu']:
        # Skip GPU geocode insar if cuda not included
        if pu == 'gpu' and not hasattr(isce3, 'cuda'):
            continue

        # Load text and substitute directory path
        with open(test_yaml_path) as fh_test_yaml:
            test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data).\
                    replace('@TEST_OUTPUT@', 'rifg.h5').\
                    replace('@TEST_PRODUCT_TYPES@', 'RIFG').\
                    replace('@TEST_RDR2GEO_FLAGS@', 'True')
            if pu == 'gpu':
                test_yaml = test_yaml.replace('gpu_enabled: False',
                                              'gpu_enabled: True')

        # create CLI input namespace
        args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

        # Initialize runconfig object
        runconfig = ResampleSlcRunConfig(args, 'coarse')
        runconfig.geocode_common_arg_load()

        resample_slc.run(runconfig.cfg, 'coarse')

        # Mark output file according CPU or GPU processing.
        os.rename(test_output, test_output + f".{pu}")


def test_resample_slc_validate():
    '''
    Validate resample_slc output VS golden dataset
    '''
    # Init common variables.
    out_shape = (250, 250)
    trimmed_slice = np.index_exp[20:-20, 20:-20]

    # Iterate over processing type.
    for pu in ['cpu', 'gpu']:
        # Skip GPU geocode insar if cuda not included
        if pu == 'gpu' and not hasattr(isce3, 'cuda'):
            continue

        ref_slc = np.fromfile(os.path.join(iscetest.data, 'warped_winnipeg.slc'),
                              dtype=np.complex64).reshape(out_shape)[trimmed_slice]
        test_slc = np.fromfile(test_output + f'.{pu}',
                               dtype=np.complex64).reshape(out_shape)[trimmed_slice]

        # Compute error
        error = np.nanmean(np.abs(test_slc - ref_slc))

        # Check error magnitude
        assert (error < 4.5e-5), f'{pu} resample_slc error {error} > 1e-6'
