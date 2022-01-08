import argparse
import os

import h5py
import numpy as np
import numpy.testing as npt

from nisar.workflows import crossmul, h5_prep
from nisar.workflows.crossmul_runconfig import CrossmulRunConfig

import iscetest


def test_crossmul_run():
    '''
    Check if crossmul runs without crashing.
    '''
    # load yaml file
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data).\
                replace('@TEST_OUTPUT@', 'rifg.h5').\
                replace('@TEST_PRODUCT_TYPES@', 'RIFG').\
                replace('@TEST_RDR2GEO_FLAGS@', 'True')

    # Create CLI input namespace with yaml text instead of filepath
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    runconfig = CrossmulRunConfig(args)
    runconfig.geocode_common_arg_load()

    h5_prep.run(runconfig.cfg)

    crossmul.run(runconfig.cfg)


def test_crossmul_validate():
    '''
    Validate products generated by crossmul workflow.
    '''
    scratch_path = '.'

    group_path = '/science/LSAR/RIFG/swaths/frequencyA/interferogram/HH'
    with h5py.File(os.path.join(scratch_path, 'rifg.h5'), 'r') as h_rifg:

        # check generated interferogram has 0 phase
        igram = h_rifg[f'{group_path}/wrappedInterferogram'][()]
        npt.assert_allclose(np.abs(np.angle(igram)), 0, atol=1e-6)

        # check resulting coherence sufficiently close to 1
        coherence = h_rifg[f'{group_path}/coherenceMagnitude'][()]
        npt.assert_allclose(coherence, 1.0, atol=1e-6)


if __name__ == "__main__":
    test_crossmul_run()
    test_crossmul_validate()
