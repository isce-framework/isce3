import argparse
import os

import numpy as np
from osgeo import gdal

from nisar.workflows import rdr2geo
from nisar.workflows.rdr2geo_runconfig import Rdr2geoRunConfig

import iscetest


def test_rdr2geo_run():
    '''
    run rdr2geo
    '''
    # load yaml
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')
    # load text then substitute test directory paths since data dir is read only
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data).\
                replace('@TEST_OUTPUT@', 'rifg.h5').\
                replace('@TEST_PRODUCT_TYPES@', 'RIFG').\
                replace('@TEST_RDR2GEO_FLAGS@', 'True')

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    runconfig = Rdr2geoRunConfig(args)
    runconfig.geocode_common_arg_load()

    rdr2geo.run(runconfig.cfg)


def check_error(f_test, f_ref, dtype, tol, test_type):
    '''
    calculate error for file in vrt
    '''
    # retrieve test output
    ds = gdal.Open(f_test, gdal.GA_ReadOnly)
    test = ds.GetRasterBand(1).ReadAsArray()

    # retrieve reference data
    ref = np.fromfile(f_ref, dtype=dtype)

    # calculate average error
    diff = np.abs(test.flatten() - ref)
    diff = diff[diff < 5]
    error = np.mean(diff)

    # error check
    fname = os.path.basename(f_test)
    assert (error < tol), f'NISAR Python {test_type} rdr2geo fail at {fname}: {error} >= {tol}'


def test_rdr2geo_validate():
    '''
    validate rdr2geo outputs
    '''
    # vrt constituent files to compare
    ref_fnames = ['x.rdr', 'y.rdr', 'z.rdr', 'inc.rdr', 'hdg.rdr',
                  'localInc.rdr', 'localPsi.rdr']
    test_fnames = ['x.rdr', 'y.rdr', 'z.rdr', 'incidence.rdr', 'heading.rdr',
              'localIncidence.rdr', 'localPsi.rdr']
    # dtypes of vrt constituent files
    dtypes = [np.float64, np.float64, np.float64, np.float32, np.float32, np.float32, np.float32]
    # tolerances per vrt constituent file taken from C++ tests
    tols = [1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02]

    # check errors with scratch set to cwd
    scratch_path = '.'
    for ref_fname, test_fname, dtype, tol in zip(ref_fnames, test_fnames, dtypes, tols):
        output_file = os.path.join(scratch_path, 'rdr2geo', 'freqA', test_fname)
        ref_file = os.path.join(iscetest.data, 'topo_winnipeg', ref_fname)
        check_error(output_file, ref_file, dtype, tol, 'CPU')
