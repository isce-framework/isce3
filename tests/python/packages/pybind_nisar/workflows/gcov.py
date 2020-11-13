#!/usr/bin/env python3
import argparse
import os

import pybind_isce3 as isce3
from pybind_nisar.workflows import gcov, h5_prep
from pybind_nisar.workflows.gcov_runconfig import GCOVRunConfig

import iscetest

geocode_modes = {'interp':isce3.geocode.GeocodeOutputMode.INTERP,
        'area':isce3.geocode.GeocodeOutputMode.AREA_PROJECTION}
input_axis = ['x', 'y']

def test_run():
    '''
    run gcov with same rasters and DEM as geocodeSlc test
    '''
    test_yaml = os.path.join(iscetest.data, 'geocode/test_gcov.yaml')

    # load text then substitude test directory paths since data dir is read only
    with open(test_yaml) as fh_test_yaml:
        test_yaml = ''.join(
                [line.replace('ISCETEST', iscetest.data) for line in fh_test_yaml])

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    runconfig = GCOVRunConfig(args)
    runconfig.geocode_common_arg_load()

    # geocode same rasters as C++/pybind geocodeCov
    for axis in input_axis:
        #  iterate thru geocode modes
        for key, value in geocode_modes.items():
            runconfig.cfg['ProductPathGroup']['SASOutputFile'] = f'{axis}_{key}.h5'

            # prepare output hdf5
            h5_prep.run(runconfig.cfg, 'GCOV')

            # geocode test raster
            gcov.run(runconfig.cfg)


if __name__ == '__main__':
    test_run()
