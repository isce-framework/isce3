#!/usr/bin/env python3
import argparse
import os

import numpy as np
import isce3.ext.isce3 as isce3
from nisar.workflows import gslc
from nisar.workflows.gslc_runconfig import GSLCRunConfig
from nisar.products.writers import GslcWriter
from nisar.products.readers import open_product
from osgeo import gdal

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

            doppler_centroid_lut_path = (
                '/science/LSAR/GSLC/metadata/sourceData/'
                'processingInformation/parameters/frequencyA/'
                'dopplerCentroid')

            # verify that Doppler Centroid LUT in radar coordinates
            # is saved into the GSLC product
            assert doppler_centroid_lut_path in gslc_obj.output_hdf5_obj

        # assert that the metadata cubes geogrid is larger than the
        # GSLC images by a margin
        hh_ref = (f'NETCDF:{sas_output_file}://science/LSAR/GSLC/'
                  'grids/frequencyA/HH')
        hh_xmin, hh_xmax, hh_ymin, hh_ymax, _, _ = get_raster_geogrid(hh_ref)

        metadata_cubes_ref = (
            f'NETCDF:{sas_output_file}://science/LSAR/GSLC'
            '/metadata/radarGrid/incidenceAngle')
        cubes_xmin, cubes_xmax, cubes_ymin, cubes_ymax, cubes_dx, \
            cubes_dy = get_raster_geogrid(metadata_cubes_ref)

        # we should have a margin of at least 5 metadata cubes pixels
        margin_x = 5 * cubes_dx
        margin_y = 5 * abs(cubes_dy)

        # hh_xmin needs to start after cubes_xmin
        assert (hh_xmin - cubes_xmin > margin_x)

        # hh_xmax needs to end before cubes_xmax
        assert (cubes_xmax - hh_xmax > margin_x)

        # hh_ymin needs to start after cubes_ymin
        assert (hh_ymin - cubes_ymin > margin_y)

        # hh_ymax needs to end before cubes_ymax
        assert (cubes_ymax - hh_ymax > margin_y)

        gslc_product = open_product(sas_output_file)
        gslc_doppler_centroid_lut = gslc_product.getDopplerCentroid()
        assert isinstance(gslc_doppler_centroid_lut, isce3.core.LUT2d)

        # The GSLC Doppler Centroid LUT in radar coordiantes must match
        # the RSLC Doppler Centroid LUT
        rslc_product = open_product(f'{iscetest.data}/envisat.h5')
        rslc_doppler_centroid_lut = rslc_product.getDopplerCentroid()

        assert np.array_equal(gslc_doppler_centroid_lut.data,
                              rslc_doppler_centroid_lut.data)

        lut_attributes_to_check_list = ['length', 'width',
                                        'y_spacing', 'x_spacing',
                                        'y_start', 'x_start']

        for attr in lut_attributes_to_check_list:
            assert (gslc_doppler_centroid_lut.__getattribute__(attr) ==
                    rslc_doppler_centroid_lut.__getattribute__(attr))


def get_raster_geogrid(dataset_reference):
    gdal_ds = gdal.Open(dataset_reference, gdal.GA_ReadOnly)
    geotransform = gdal_ds.GetGeoTransform()
    length = gdal_ds.RasterYSize
    width = gdal_ds.RasterXSize

    dx = geotransform[1]
    dy = geotransform[5]
    xmin = geotransform[0]
    xmax = geotransform[0] + width * dx
    ymax = geotransform[3]
    ymin = geotransform[3] + length * dy

    return xmin, xmax, ymin, ymax, dx, dy


if __name__ == '__main__':
    test_run()
