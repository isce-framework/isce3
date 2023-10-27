import argparse
import os

import h5py
import isce3
import iscetest
import numpy as np
from nisar.workflows import geocode_insar, prepare_insar_hdf5
from nisar.products.insar.product_paths import RUNWGroupsPaths, GUNWGroupsPaths
from nisar.workflows.geocode_insar_runconfig import GeocodeInsarRunConfig
from osgeo import gdal


def test_geocode_run():
    '''
    Test if phase geocoding runs smoothly
    '''
    test_yaml_path = os.path.join(iscetest.data, 'insar_test.yaml')
    for pu in ['cpu', 'gpu']:
        # Skip GPU geocode insar if cuda not included
        if pu == 'gpu' and not hasattr(isce3, 'cuda'):
            continue

        with open(test_yaml_path) as fh_test_yaml:
            test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data). \
                replace('@TEST_OUTPUT@', f'{pu}_gunw.h5'). \
                replace('@TEST_PRODUCT_TYPES@', 'GUNW'). \
                replace('@TEST_RDR2GEO_FLAGS@', 'True')
            if pu == 'gpu':
                test_yaml = test_yaml.replace('gpu_enabled: False', 'gpu_enabled: True')

        # Create CLI input name space with yaml instead of filepath
        args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

        # Initialize runconfig object
        runconfig = GeocodeInsarRunConfig(args)
        runconfig.geocode_common_arg_load()

        # save geogrid for later validation
        geogrid = runconfig.cfg['processing']['geocode']['geogrids']['A']
        geo_tx = np.array([geogrid.start_x, geogrid.spacing_x, 0, geogrid.start_y, 0, geogrid.spacing_y])
        geo_tx.tofile('gunw_geogrid.txt', sep=',')

        # prepare HDF5 outputs
        out_paths = prepare_insar_hdf5.run(runconfig.cfg)

        # insert rdr2geo outputs into RUNW HDF5
        rdr2geo_dict = {'x': 'unwrappedPhase', 'y': 'coherenceMagnitude'}

        # Create RUNW obj to avoid hard-coded path to RUNW datasets
        runw_product_path = f'{RUNWGroupsPaths().SwathsPath}/frequencyA/interferogram/HH'
        az_looks = runconfig.cfg['processing']['crossmul']['azimuth_looks']
        rg_looks = runconfig.cfg['processing']['crossmul']['range_looks']
        with h5py.File(out_paths['RUNW'], 'a', libver='latest', swmr=True) as h_runw:
            for axis, ds_name in rdr2geo_dict.items():
                # extract rdr2geo raster to array
                f_rdr2geo = f'rdr2geo/freqA/{axis}.rdr'
                ds = gdal.Open(f_rdr2geo, gdal.GA_ReadOnly)
                arr = ds.GetRasterBand(1).ReadAsArray()
                ds = None

                # decimate array (take center pixel of mlook grid) and save to HDF5
                sz_x, sz_y = arr.shape
                arr_mlook = arr[az_looks//2:sz_x-az_looks//2:az_looks, az_looks//2:sz_y-az_looks//2:az_looks]
                h_runw[f'{runw_product_path}/{ds_name}'][:, :] = arr_mlook

        # disable unused runw dataset
        runconfig.cfg['processing']['geocode']['gunw_datasets']['connected_components'] = False

        # run geocodeing of runw
        geocode_insar.run(runconfig.cfg, out_paths['RUNW'], out_paths['GUNW'])


def test_geocode_validate():
    '''
    validate generated geocode data
    '''
    # load geotransform
    geo_trans = np.fromfile('gunw_geogrid.txt', sep=',')

    # prepare the check gunw hdf5
    scratch_path = '.'
    for pu in ['cpu', 'gpu']:
        # Skip GPU geocode insar if cuda not included
        if pu == 'gpu' and not hasattr(isce3, 'cuda'):
            continue

        # Create GUNW obj to avoid hard-coded paths to GUNW datasets
        path_gunw = os.path.join(scratch_path, f'{pu}_gunw.h5')
        product_path = f'{GUNWGroupsPaths().GridsPath}/frequencyA/unwrappedInterferogram/HH'
        with h5py.File(path_gunw, 'r', libver='latest', swmr=True) as h:
            # iterate over axis
            rdr2geo_dict = {'x': 'unwrappedPhase', 'y': 'coherenceMagnitude'}
            for axis, ds_name in rdr2geo_dict.items():
                # get dataset as masked array
                ds_path = f'{product_path}/{ds_name}'
                geo_arr = h[ds_path][()]
                geo_arr = np.ma.masked_array(geo_arr, mask=np.isnan(geo_arr))

                # get transform and meshgrids once for common geogrid
                if ds_name == 'unwrappedPhase':
                    x0 = geo_trans[0] + geo_trans[1] / 2.0
                    dx = geo_trans[1]
                    y0 = geo_trans[3] + geo_trans[5] / 2.0
                    dy = geo_trans[5]

                    lines, pixels = geo_arr.shape
                    meshx, meshy = np.meshgrid(np.arange(pixels), np.arange(lines))
                    grid_lon = x0 + meshx * dx
                    grid_lat = y0 + meshy * dy

                # set calculate error and RMSE error thresholds per axis
                if ds_name == 'unwrappedPhase':
                    err = geo_arr - grid_lon
                    rmse_err_threshold = 0.5 * dx
                else:
                    err = geo_arr - grid_lat
                    rmse_err_threshold = 0.5 * abs(dy)

                # check max error
                max_err = np.nanmax(err)
                # 1e-4 is ~11m
                assert(max_err < 1e-4), f'{axis}-axis max error fail on {pu}'

                # check RMSE
                rmse = np.sqrt(np.sum(err**2) / np.count_nonzero(~geo_arr.mask))
                assert(rmse < rmse_err_threshold), f'{axis}-axis RMSE fail on {pu}'


if __name__ == "__main__":
    test_geocode_run()
