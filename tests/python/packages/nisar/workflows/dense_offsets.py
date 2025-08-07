import argparse
import os

import numpy as np
import numpy.testing as npt
from nisar.workflows import dense_offsets
from nisar.workflows.dense_offsets_runconfig import DenseOffsetsRunConfig

import iscetest

def test_dense_offsets_run():
    '''
    Run dense offsets estimation
    '''

    # Load yaml
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')

    # Load text and substitute directory path
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_OUTPUT@', 'rifg.h5'). \
            replace('@TEST_PRODUCT_TYPES@', 'RIFG'). \
            replace('@TEST_RDR2GEO_FLAGS@', 'True'). \
            replace('gpu_enabled: False', 'gpu_enabled: True')

    # Create CLI input namespace with yaml test instead of filepath
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    runconfig = DenseOffsetsRunConfig(args)
    runconfig.geocode_common_arg_load()

    # run dense offsets
    dense_offsets.run(runconfig.cfg)


def check_errors(infile, layers, tol):
    data = np.memmap(infile, shape=(18, 18, layers), dtype=np.float32)

    for k in range(layers):
        data_layer = np.array(data[:, :, k])
        npt.assert_allclose(data_layer, 0, atol=tol)


def test_dense_offsets_validate():
    '''
    Validate dense offsets outputs
    '''

    scratch_path = '.'
    # Tolerance for dense_offsets is set as 1/32, where 32 is
    # the correlation_surface_oversampling_factor.
    # This threshold set the "accuracy/granularity" of the offsets
    # we can estimate
    fnames = ['dense_offsets', 'gross_offset', 'covariance']
    layers = [2, 2, 3]
    tols = [0.03125, 1e-6, 1e-6]

    for fname, layer, tol in zip(fnames, layers, tols):
        in_file = os.path.join(scratch_path, 'dense_offsets', 'freqA', 'HH',
                               fname)
        check_errors(in_file, layer, tol)


