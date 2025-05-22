#!/usr/bin/env python3
import argparse
import os

from nisar.workflows import defaults, gslc_array, h5_prep
from nisar.workflows.gslc_runconfig import GSLCRunConfig

import iscetest

def test_run():
    '''
    run gslc with same rasters and DEM as geocodeSlc test
    '''
    test_yaml = os.path.join(iscetest.data, 'geocodeslc/test_gslc.yaml')

    # load text then substitute test directory paths since data dir is read only
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read(). \
            replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_BLOCK_SZ_X@', '133'). \
            replace('@TEST_BLOCK_SZ_Y@', '275')

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    runconfig = GSLCRunConfig(args)
    runconfig.geocode_common_arg_load()

    # geocode same 2 rasters as C++/pybind geocodeSlc
    for xy in ['x', 'y']:
        # adjust runconfig to match just created raster
        runconfig.cfg['product_path_group']['sas_output_file'] = f'{xy}_out.h5'

        # prepare output hdf5
        h5_prep.run(runconfig.cfg)

        # geocode test raster
        gslc_array.run(runconfig.cfg)


if __name__ == '__main__':
    test_run()
