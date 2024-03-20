import argparse
import os

import h5py
import iscetest
import pytest
import types
import numpy as np

from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.products.insar.product_paths import CommonPaths
from nisar.products.readers import SLC
from nisar.workflows import baseline
import isce3


@pytest.fixture(scope='session')
def unit_test_params():
    '''
    test parameters shared by all baseline tests
    '''
    # load h5 for doppler and orbit
    params = types.SimpleNamespace()
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_OUTPUT@', 'RIFG.h5'). \
            replace('@TEST_PRODUCT_TYPES@', 'RIFG'). \
            replace('@TEST_RDR2GEO_FLAGS@', 'True')
    # Create CLI input namespace with yaml text instead of filepath
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)
    # Initialize runconfig object
    insar_runcfg = InsarRunConfig(args)
    insar_runcfg.geocode_common_arg_load()
    insar_runcfg.yaml_check()
    cfg = insar_runcfg.cfg
    refslc_path = cfg["input_file_group"]["reference_rslc_file"]
    scratch_path = cfg['product_path_group']['scratch_path']
    params.baseline_dir_path = f'{scratch_path}/baseline'
    os.makedirs(params.baseline_dir_path, exist_ok=True)
    params.ref_slc = SLC(hdf5file=refslc_path)
    radar_grid = params.ref_slc.getRadarGrid()
    params.ellipsoid = isce3.core.Ellipsoid()
    params.ref_orbit = params.ref_slc.getOrbit()
    params.ref_radargrid = radar_grid
    # native-doppler
    params.ref_doppler = params.ref_slc.getDopplerCentroid(frequency='A')
    params.ref_doppler.bounds_error = False
    params.geo2rdr_parameters = {'threshold': 1.0e-8,
                                 'maxiter': 50,
                                 'delta_range': 1.0e-8}
    params.coord_x = -97.71044127296169
    params.coord_y = 49.4759022631287
    params.coord_z = 240
    params.epsg = 4326
    params.range_start = radar_grid.starting_range
    params.range_end = params.range_start + \
        radar_grid.width * radar_grid.range_pixel_spacing

    return params


def test_compute_baseline(unit_test_params):
    '''
    Check if compute_baseline runs without crashing and returns zero-baseline
    '''

    grid_x = np.ones([2, 2]) * unit_test_params.coord_x
    grid_y = np.ones([2, 2]) * unit_test_params.coord_y
    grid_z = np.ones([2, 2]) * unit_test_params.coord_z

    coord_set = np.zeros([3, 2, 2])
    coord_set[0, :, :] = grid_x
    coord_set[1, :, :] = grid_y
    coord_set[2, :, :] = grid_z

    ref_rngs = np.ones([2, 2]) * 13774.94775418
    ref_azts = np.ones([2, 2]) * 172802.73308736

    parb, perb = baseline.compute_baseline(
        ref_rngs,
        ref_azts,
        ref_rngs,
        ref_azts,
        coord_set,
        unit_test_params.ref_orbit,
        unit_test_params.ref_orbit,
        unit_test_params.ellipsoid,
        unit_test_params.epsg)

    assert np.nanmean(parb) < 1e-5
    assert np.nanmean(perb) < 1e-5

def test_add_baseline(unit_test_params):
    '''
    test the add_baseline without crushing and wrong values
    '''
    # Instantiate common path object to avoid hard-code paths to datasets
    prod_obj = CommonPaths()
    common_path = prod_obj.RootPath

    output_paths = dict({'RIFG': 'RIFG.h5', 'RUNW': 'RUNW.h5', 'GUNW': 'GUNW.h5'})
    for dst in ['RIFG', 'GUNW']:
        with h5py.File(output_paths[dst], 'a') as h5_src:

            product_path = f'{common_path}/{dst}'
            if dst in ['RIFG']:
                grid_path = f'{product_path}/metadata/geolocationGrid'
                metadata_path_dict = {
                    "heights": f"{grid_path}/heightAboveEllipsoid",
                    "azimuthTime": f"{grid_path}/zeroDopplerTime",
                    "slantRange": f"{grid_path}/slantRange",
                    "coordX": f"{grid_path}/coordinateX",
                    "coordY": f"{grid_path}/coordinateY",
                    "perpendicularBaseline": f"{grid_path}/perpendicularBaseline",
                    "parallelBaseline": f"{grid_path}/parallelBaseline",
                    "epsg": f"{grid_path}/epsg",
                    "range_start": unit_test_params.range_start,
                    "range_end": unit_test_params.range_end,
                    }

                # winnipeg data does not have coordinate X and Y in metadata cube
                # we need to create to compute the baseline
                grid_x = np.ones([1, 2, 2]) * unit_test_params.coord_x
                grid_y = np.ones([1, 2, 2]) * unit_test_params.coord_y

                if metadata_path_dict['coordX'] in h5_src:
                    del h5_src[metadata_path_dict['coordX']]
                h5_src.create_dataset(metadata_path_dict['coordX'],
                                    dtype=np.float32,
                                    shape=[1,2,2], data=grid_x)

                if metadata_path_dict['coordY'] in h5_src:
                    del h5_src[metadata_path_dict['coordY']]
                h5_src.create_dataset(metadata_path_dict['coordY'],
                                    dtype=np.float32,
                                    shape=[1,2,2], data=grid_y)
                h5_src[metadata_path_dict['coordY']][:] = grid_y

                if metadata_path_dict['heights'] in h5_src:
                    del h5_src[metadata_path_dict['heights']]
                h5_src.create_dataset(metadata_path_dict['heights'],
                                        dtype=np.float32,
                                        shape=[1],
                                        data=unit_test_params.coord_z)

                if metadata_path_dict['epsg'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['epsg'],
                                        dtype=np.int64,
                                        data=4326)
                # replace the metadata to have only two elements.
                if metadata_path_dict['slantRange'] in h5_src:
                    del h5_src[metadata_path_dict['slantRange']]
                h5_src.create_dataset(metadata_path_dict['slantRange'],
                                        dtype=np.float64,
                                        shape=[2],
                                        data=[13150.0574, 13649.71149664])

                if metadata_path_dict['azimuthTime'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['azimuthTime'],
                                        dtype=np.float64,
                                        shape=[2])
                if h5_src[metadata_path_dict['azimuthTime']].shape[0] == 0:
                    del h5_src[metadata_path_dict['azimuthTime']]
                    h5_src.create_dataset(metadata_path_dict['azimuthTime'],
                                        dtype=np.float64,
                                        shape=[2])
                if h5_src[metadata_path_dict['slantRange']].shape[0] == 0:
                    del h5_src[metadata_path_dict['slantRange']]
                    h5_src.create_dataset(metadata_path_dict['slantRange'],
                                        dtype=np.float64,
                                        shape=[2])

                output_paths_rifg = {"RIFG": output_paths["RIFG"],
                                     "RUNW": output_paths["RUNW"]}

                baseline.add_baseline(output_paths_rifg,
                                unit_test_params.ref_orbit,
                                unit_test_params.ref_orbit,
                                unit_test_params.ref_radargrid,
                                unit_test_params.ref_radargrid,
                                unit_test_params.ref_doppler,
                                unit_test_params.ref_doppler,
                                metadata_path_dict,
                                unit_test_params.geo2rdr_parameters,
                                use_gpu=False,
                                baseline_dir_path=unit_test_params.baseline_dir_path,
                                baseline_mode='top_bottom')

                validate_baseline(output_paths_rifg["RIFG"],
                    perp_path=metadata_path_dict["perpendicularBaseline"],
                    par_path=metadata_path_dict["parallelBaseline"])
                validate_baseline(output_paths_rifg["RUNW"],
                    perp_path=metadata_path_dict["perpendicularBaseline"].replace('RIFG', 'RUNW'),
                    par_path=metadata_path_dict["parallelBaseline"].replace('RIFG', 'RUNW'))

            elif dst in ['GUNW']:
                product_path = f'{common_path}/{dst}'
                grid_path = f'{product_path}/metadata/geolocationGrid'
                grid_path = grid_path.replace('geolocation', 'radar')
                cube_ref_dataset = f'{grid_path}/slantRange'

                metadata_path_dict = {
                    "heights": f"{grid_path}/heightAboveEllipsoid",
                    "azimuthTime": f"{grid_path}/referenceZeroDopplerAzimuthTime",
                    "slantRange": f"{grid_path}/referenceSlantRange",
                    "coordX": f"{grid_path}/xCoordinates",
                    "coordY": f"{grid_path}/yCoordinates",
                    "perpendicularBaseline": f"{grid_path}/perpendicularBaseline",
                    "parallelBaseline": f"{grid_path}/parallelBaseline",
                    "projection": f"{grid_path}/projection",
                    "range_start": unit_test_params.range_start,
                    "range_end": unit_test_params.range_end,
                    }

                x_array = np.ones(2) * unit_test_params.coord_x
                y_array = np.ones(2) * unit_test_params.coord_y
                if metadata_path_dict['coordX'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['coordX'],
                                        dtype=np.float32,
                                        shape=[2],
                                        data=x_array)
                if metadata_path_dict['coordY'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['coordY'],
                                        dtype=np.float32,
                                        shape=[2],
                                        data=y_array)
                if metadata_path_dict['heights'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['heights'],
                                        dtype=np.float32,
                                        shape=[1],
                                        data=unit_test_params.coord_z)
                if metadata_path_dict['slantRange'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['slantRange'],
                                        dtype=np.float32,
                                        shape=[1, 2, 2])
                output_paths_gunw = {"GUNW": output_paths["GUNW"]}

                baseline.add_baseline(output_paths_gunw,
                                unit_test_params.ref_orbit,
                                unit_test_params.ref_orbit,
                                unit_test_params.ref_radargrid,
                                unit_test_params.ref_radargrid,
                                unit_test_params.ref_doppler,
                                unit_test_params.ref_doppler,
                                metadata_path_dict,
                                unit_test_params.geo2rdr_parameters,
                                use_gpu=False,
                                baseline_dir_path=unit_test_params.baseline_dir_path,
                                baseline_mode='top_bottom')

                validate_baseline(output_paths_gunw["GUNW"],
                    perp_path=metadata_path_dict["perpendicularBaseline"],
                    par_path=metadata_path_dict["parallelBaseline"])

def validate_baseline(output_path, perp_path, par_path):
    '''
    validate the baselines if they are all zeros.
    '''

    with h5py.File(output_path) as src_h5:
        perp_base = np.array(src_h5[perp_path])
        par_base = np.array(src_h5[par_path])

        assert np.nanmean(perp_base) < 1e-5
        assert np.nanmean(par_base) < 1e-5

# if __name__ == '__main__':
#     # test_compute_baseline()
#     test_add_baseline()
