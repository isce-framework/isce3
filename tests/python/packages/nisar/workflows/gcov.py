#!/usr/bin/env python3
import argparse
import os
import h5py

import numpy as np
import numpy.testing as npt
from osgeo import gdal
from pathlib import Path

import isce3.ext.isce3 as isce3
from nisar.workflows import gcov
from nisar.workflows.gcov_runconfig import GCOVRunConfig
from nisar.products.writers import GcovWriter
from nisar.products.readers import open_product

import iscetest

geocode_modes = {'interp': isce3.geocode.GeocodeOutputMode.INTERP,
                 'area': isce3.geocode.GeocodeOutputMode.AREA_PROJECTION}


def test_run_winnipeg():
    '''
    Test the GCOV using the UAVSAR Winipeg dataset
    '''

    # load text then substitute test directory paths
    test_yaml_file = Path(iscetest.data) / 'geocode/test_gcov.yaml'
    test_yaml = test_yaml_file.read_text().replace('@ISCETEST@', iscetest.data)

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    runconfig = GCOVRunConfig(args)
    runconfig.geocode_common_arg_load()

    #  iterate thru geocode modes
    for key in geocode_modes.keys():
        sas_output_file = f'gcov_winnipeg_{key}.h5'
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

        # We need to remove existing products because we are bypassing
        # the part of the GCOV workflow that is responsible for this.
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

        # assert that the metadata cubes geogrid is larger than the
        # GCOV images by a margin
        hh_ref = (f'NETCDF:{sas_output_file}://science/LSAR/GCOV/'
                  'grids/frequencyA/HHHH')
        hh_xmin, hh_xmax, hh_ymin, hh_ymax, _, _ = get_raster_geogrid(
            hh_ref)

        metadata_cubes_ref = (
            f'NETCDF:{sas_output_file}://science/LSAR/GCOV'
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

        output_h5_obj = h5py.File(sas_output_file, 'r')

        zero_doppler_time_dataset = \
            output_h5_obj['//science/LSAR/GCOV/metadata/sourceData/'
                          'processingInformation/parameters/'
                          'zeroDopplerTime']

        assert 'units' in zero_doppler_time_dataset.attrs.keys()

        assert zero_doppler_time_dataset.attrs[
            'units'].decode().startswith('seconds since ')

        assert zero_doppler_time_dataset.attrs['description'].decode() == \
            ('Vector of zero Doppler azimuth times, measured relative to'
             ' a UTC epoch, corresponding to source data processing'
             ' information records')


def test_run_envisat():
    '''
    Test GCOV workflow using the Envisat dataset.
    '''

    # load text then substitute test directory paths
    test_yaml_file = Path(iscetest.data) / 'geocode/test_gcov_envisat.yaml'
    test_yaml = test_yaml_file.read_text().replace('@ISCETEST@', iscetest.data)

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    runconfig = GCOVRunConfig(args)
    runconfig.geocode_common_arg_load()

    #  iterate thru geocode modes
    for key in geocode_modes.keys():

        # dictionary that will hold the HH image with (key: True)
        # and without (key: False) noise correction
        hh_noise_correction = {}

        for apply_noise_correction in [False, True]:

            apply_noise_correction_str = \
                str(apply_noise_correction).lower()
            sas_output_file = (f'gcov_envisat_{key}_noise'
                               f'_correction_{apply_noise_correction_str}.h5')
            runconfig.cfg['product_path_group']['sas_output_file'] = \
                sas_output_file

            runconfig.cfg['processing']['noise_correction'][
                'apply_correction'] = apply_noise_correction

            # We need to remove existing products because we are bypassing
            # the part of the GCOV workflow that is responsible for this.
            if os.path.isfile(sas_output_file):
                os.remove(sas_output_file)

            # geocode test raster
            gcov.run(runconfig.cfg)

            with GcovWriter(runconfig=runconfig) as gcov_obj:
                gcov_obj.populate_metadata()

            with h5py.File(sas_output_file, 'r') as output_h5_obj:
                hh_noise_correction[apply_noise_correction] = \
                    output_h5_obj[
                        '//science/LSAR/GCOV/grids/frequencyA/HHHH'][()]

        rslc_product = open_product(os.path.join(iscetest.data,
                                                 'envisat.h5'))
        radar_grid = rslc_product.getRadarGrid(frequency='A')
        noise_product = \
            rslc_product.getResampledNoiseEquivalentBackscatter(
                sensing_times=radar_grid.sensing_times,
                slant_ranges=radar_grid.slant_ranges,
                frequency='A')

        diff_noise = np.nanmean(hh_noise_correction[False] -
                                hh_noise_correction[True])
        mean_noise_power_from_rslc = np.nanmean(noise_product.power_linear)

        print('Estimated noise power from GCOV:', diff_noise)
        print('Noise power from RSLC metadata:',
              noise_product.power_linear)

        err_msg = ('Estimated noise power from the GCOV images'
                   f' ({diff_noise}) does not match the noise power'
                   ' in the RSLC metadata')
        npt.assert_allclose(diff_noise, mean_noise_power_from_rslc,
                            err_msg=err_msg, rtol=0.1)


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
    test_run_winnipeg()
    test_run_envisat()
