#!/usr/bin/env python3
import argparse
import os

from nisar.workflows import defaults, gslc
from nisar.workflows.gslc_runconfig import GSLCRunConfig
from nisar.products.writers import GslcWriter

import iscetest

def test_run():
    '''
    run gslc with same rasters and DEM as geocodeSlc test
    '''
    test_yaml = os.path.join(iscetest.data, 'geocodeslc/test_gslc.yaml')

    # load text then substitude test directory paths since data dir is read only
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read(). \
            replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_BLOCK_SZ_X@', '133'). \
            replace('@TEST_BLOCK_SZ_Y@', '1000')

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    runconfig = GSLCRunConfig(args)
    runconfig.geocode_common_arg_load()

    # geocode same 2 rasters as C++/pybind geocodeSlc
    for xy in ['x', 'y']:
        # adjust runconfig to match just created raster
        sas_output_file = f'{xy}_out.h5'
        runconfig.cfg['product_path_group']['sas_output_file'] = \
            sas_output_file

        partial_granule_id = \
            ('NISAR_L2_PR_GSLC_105_091_D_006_{MODE}_{POLE}_A'
                '_{StartDateTime}_{EndDateTime}_D00344_P_P_J_001.h5')
        expected_granule_id = \
            ('NISAR_L2_PR_GSLC_105_091_D_006_1600_SHNA_A'
                '_20030226T175530_20030226T175531_D00344_P_P_J_001.h5')
        runconfig.cfg['primary_executable']['partial_granule_id'] = \
            partial_granule_id

        if os.path.isfile(sas_output_file):
            os.remove(sas_output_file)

        # geocode test raster
        gslc.run(runconfig.cfg)

        with GslcWriter(runconfig=runconfig) as gslc_obj:
            gslc_obj.populate_metadata()
            assert gslc_obj.granule_id == expected_granule_id


if __name__ == '__main__':
    test_run()
