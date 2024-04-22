#!/usr/bin/env python3
import argparse
import os

import isce3.ext.isce3 as isce3
from nisar.workflows import gcov
from nisar.workflows.gcov_runconfig import GCOVRunConfig
from nisar.products.writers import GcovWriter
from nisar.products.readers import open_product
import numpy as np

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

                doppler_centroid_lut_path = (
                    '/science/LSAR/GCOV/metadata/sourceData/'
                    'processingInformation/parameters/frequencyA/'
                    'dopplerCentroid')

                # verify that Doppler Centroid LUT in radar coordinates
                # is saved into the GCOV product
                assert doppler_centroid_lut_path in gcov_obj.output_hdf5_obj

            gcov_product = open_product(sas_output_file)
            gcov_doppler_centroid_lut = gcov_product.getDopplerCentroid()
            assert isinstance(gcov_doppler_centroid_lut, isce3.core.LUT2d)

            # The GCOV Doppler Centroid LUT in radar coordiantes must match
            # the RSLC Doppler Centroid LUT
            rslc_product = open_product(f'{iscetest.data}/winnipeg.h5')
            rslc_doppler_centroid_lut = rslc_product.getDopplerCentroid()

            assert np.array_equal(gcov_doppler_centroid_lut.data,
                                    rslc_doppler_centroid_lut.data)

            lut_attributes_to_check_list = ['length', 'width',
                                            'y_spacing', 'x_spacing',
                                            'y_start', 'x_start']

            for attr in lut_attributes_to_check_list:
                assert (gcov_doppler_centroid_lut.__getattribute__(attr) ==
                        rslc_doppler_centroid_lut.__getattribute__(attr))


if __name__ == '__main__':
    test_run()
