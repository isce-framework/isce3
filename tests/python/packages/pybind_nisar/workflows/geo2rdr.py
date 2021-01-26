import argparse
import os

import numpy as np

from pybind_nisar.workflows import geo2rdr
from pybind_nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig

import iscetest


def test_geo2rdr_run():
    '''
    Check if geo2rdr runs
    '''
    # load yaml file
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data).\
                replace('@TEST_OUTPUT@', 'rifg.h5').\
                replace('@TEST_PRODUCT_TYPES@', 'RIFG')

    # Create CLI input namespace with yaml text instead of filepath
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    runconfig = Geo2rdrRunConfig(args)
    runconfig.geocode_common_arg_load()
    geo2rdr.run(runconfig.cfg)


def check_error(f_test, dtype, tol, test_type):
    '''
    Calculate residual (validate geo2rdr outputs)
    '''

    # Retrieve data

    test = np.fromfile(f_test, dtype=dtype).astype('float64')
    test = np.ma.masked_array(test, mask=np.abs(test) > 999.0)

    # Calculate average error
    error = np.nansum(test * test)

    # Check the error
    fname = os.path.basename(f_test)
    assert (error < tol), f'NISAR Python {test_type} geo2rdr fail at {fname}: {error} >= {tol}'


def test_geo2rdr_validate():
    '''
    Validate geo2rdr outputs
    '''

    # Get files to compare
    fnames = ['azimuth.off', 'range.off']
    dtypes = [np.float32, np.float32]
    tols = [1.0e-9, 1.0e-9]

    # Check errors
    scratch_path = '.'
    for fname, dtype, tol in zip(fnames, dtypes, tols):
        output_file = os.path.join(scratch_path, 'geo2rdr', 'freqA', fname)
        check_error(output_file, dtype, tol, 'CPU')
