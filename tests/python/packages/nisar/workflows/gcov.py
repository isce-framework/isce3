#!/usr/bin/env python3
import argparse
import os

import isce3.ext.isce3 as isce3
from nisar.workflows import gcov
from nisar.workflows.gcov_runconfig import GCOVRunConfig
from nisar.products.writers import GcovWriter

import iscetest

geocode_modes = {'interp': isce3.geocode.GeocodeOutputMode.INTERP,
                 'area': isce3.geocode.GeocodeOutputMode.AREA_PROJECTION}
input_axis = ['x', 'y']


def test_run():
    '''
    run gcov with same rasters and DEM as geocodeSlc test
    '''
    test_yaml = os.path.join(iscetest.data, 'geocode/test_gcov.yaml')

    # load text then substitude test directory paths since data dir is read
    # only
    with open(test_yaml) as fh_test_yaml:
        test_yaml = ''.join(
            [line.replace('@ISCETEST@', iscetest.data)
             for line in fh_test_yaml])

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    runconfig = GCOVRunConfig(args)
    runconfig.geocode_common_arg_load()

    # geocode same rasters as C++/pybind geocodeCov
    for axis in input_axis:
        #  iterate thru geocode modes
        for key, value in geocode_modes.items():
            sas_output_file = f'{axis}_{key}.h5'
            runconfig.cfg['product_path_group']['sas_output_file'] = \
                sas_output_file
            partial_granule_id = \
                ('NISAR_L2_PR_GCOV_105_091_D_006_{MODE}_{POLE}_A'
                 '_{StartDateTime}_{EndDateTime}_D00344_P_P_J_001.h5')
            expected_granule_id = \
                ('NISAR_L2_PR_GCOV_105_091_D_006_2000_SHNA_A'
                 '_20120717T143647_20120717T144244_D00344_P_P_J_001.h5')
            runconfig.cfg['primary_executable']['partial_granule_id'] = \
                partial_granule_id

            if os.path.isfile(sas_output_file):
                os.remove(sas_output_file)

            # geocode test raster
            gcov.run(runconfig.cfg)

            with GcovWriter(runconfig=runconfig) as gcov_obj:
                gcov_obj.populate_metadata()
                assert gcov_obj.granule_id == expected_granule_id


if __name__ == '__main__':
    test_run()
