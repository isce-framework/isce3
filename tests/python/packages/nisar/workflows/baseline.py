import argparse
import os

import h5py
# from isce3.signal.filter_data import filter_data
import iscetest
import numpy as np
from osgeo import gdal

# from nisar.workflows import filter_interferogram, h5_prep
from nisar.workflows import h5_prep, insar
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.products.readers import SLC
from nisar.workflows import baseline
import isce3

def test_compute_baseline():
    '''
    Check if compute_baseline runs without crashing and returns zero-baseline
    '''

    # Load yaml file
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
    refslc_path = cfg["input_file_group"]["reference_rslc_file_path"]

    ref_slc = SLC(hdf5file=refslc_path)
    ellipsoid = isce3.core.Ellipsoid()
    ref_orbit = ref_slc.getOrbit()

    ref_radargrid = ref_slc.getRadarGrid()

    # native-doppler
    ref_doppler = ref_slc.getDopplerCentroid(frequency='A')
    ref_doppler.bounds_error = False

    proj = isce3.core.make_projection(4326)
    geo2rdr_parameters = {'threshold': 1.0e-8,
                          'maxiter': 50,
                          'delta_range': 1.0e-8}

    target_proj = np.array([-97.71044127296169, 49.4759022631287, 240])
    target_llh = proj.inverse(target_proj)

    parb, perb = baseline.compute_baseline(target_llh,
                            ref_orbit,
                            ref_orbit,
                            ref_doppler,
                            ref_doppler,
                            ref_radargrid,
                            ref_radargrid,
                            ellipsoid,
                            geo2rdr_parameters)
    assert parb < 1e-5
    assert perb < 1e-5

def test_add_baseline():
    '''
    test the add_baseline without crushing and wrong values
    '''

    # Load yaml file
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
    refslc_path = cfg["input_file_group"]["reference_rslc_file_path"]

    ref_slc = SLC(hdf5file=refslc_path)
    ellipsoid = isce3.core.Ellipsoid()
    ref_orbit = ref_slc.getOrbit()

    ref_radargrid = ref_slc.getRadarGrid()

    # native-doppler
    ref_doppler = ref_slc.getDopplerCentroid(frequency='A')
    ref_doppler.bounds_error = False

    proj = isce3.core.make_projection(4326)
    geo2rdr_parameters = {'threshold': 1.0e-8,
                          'maxiter': 50,
                          'delta_range': 1.0e-8}

    output_paths = dict({'RIFG': 'RIFG.h5', 'RUNW': 'RUNW.h5', 'GUNW': 'GUNW.h5'})
    for dst in ['RIFG', 'GUNW']:
        with h5py.File(output_paths[dst], 'a', libver='latest', swmr=True) as h5_src:
            common_path = 'science/LSAR'
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
                    }
                # winnipeg data does not have coordinate X and Y in metadata cube
                # we need to create to compute the baseline
                if metadata_path_dict['coordX'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['coordX'],
                                        dtype=np.float32,
                                        shape=[1,1,1], data=-97.71044127296169)
                    h5_src.create_dataset(metadata_path_dict['coordY'],
                                        dtype=np.float32,
                                        shape=[1,1,1], data= 49.4759022631287)
                    h5_src[metadata_path_dict['coordY']][:] = 49.4759022631287
                    h5_src.create_dataset(metadata_path_dict['heights'],
                                        dtype=np.float32,
                                        shape=[1])
                if metadata_path_dict['epsg'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['epsg'],
                                        dtype=np.int,
                                        data=4326)
                if metadata_path_dict['slantRange'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['slantRange'],
                                        dtype=np.int,
                                        shape=[1])
                if metadata_path_dict['azimuthTime'] not in h5_src:
                    h5_src.create_dataset(metadata_path_dict['azimuthTime'],
                                        dtype=np.int,
                                        shape=[1])
                if h5_src[metadata_path_dict['azimuthTime']].shape[0] == 0:
                    del h5_src[metadata_path_dict['azimuthTime']]
                    h5_src.create_dataset(metadata_path_dict['azimuthTime'],
                                        dtype=np.int,
                                        shape=[1])
                if h5_src[metadata_path_dict['slantRange']].shape[0] == 0:
                    del h5_src[metadata_path_dict['slantRange']]
                    h5_src.create_dataset(metadata_path_dict['slantRange'],
                                        dtype=np.int,
                                        shape=[1])

                target_proj = np.array([-97.71044127296169, 49.4759022631287, 240])
                output_paths_rifg = {"RIFG": output_paths["RIFG"], "RUNW": output_paths["RUNW"]}

                baseline.add_baseline(output_paths_rifg,
                                ref_orbit,
                                ref_orbit,
                                ref_radargrid,
                                ref_radargrid,
                                ref_doppler,
                                ref_doppler,
                                ellipsoid,
                                metadata_path_dict,
                                geo2rdr_parameters)
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
                heights = f"{grid_path}/heightAboveEllipsoid"
                coordX = f"{grid_path}/xCoordinates"
                coordY = f"{grid_path}/yCoordinates"
                cube_ref_dataset = f'{grid_path}/slantRange'

                metadata_path_dict = {
                    "heights": f"{grid_path}/heightAboveEllipsoid",
                    "azimuthTime": f"{grid_path}/zeroDopplerAzimuthTime",
                    "slantRange": f"{grid_path}/slantRange",
                    "coordX": f"{grid_path}/xCoordinates",
                    "coordY": f"{grid_path}/yCoordinates",
                    "perpendicularBaseline": f"{grid_path}/perpendicularBaseline",
                    "parallelBaseline": f"{grid_path}/parallelBaseline",
                    "epsg": f"{grid_path}/epsg",
                    }
                if coordX not in h5_src:
                    h5_src.create_dataset(coordX,
                                        dtype=np.float32,
                                        shape=[1])
                    h5_src.create_dataset(coordY,
                                        dtype=np.float32,
                                        shape=[1])
                    h5_src.create_dataset(heights,
                                        dtype=np.float32,
                                        shape=[1])
                    h5_src.create_dataset(cube_ref_dataset,
                                        dtype=np.float32,
                                        shape=[1])
                output_paths_gunw = {"GUNW": output_paths["GUNW"]}

                baseline.add_baseline(output_paths_gunw,
                                ref_orbit,
                                ref_orbit,
                                ref_radargrid,
                                ref_radargrid,
                                ref_doppler,
                                ref_doppler,
                                ellipsoid,
                                metadata_path_dict,
                                geo2rdr_parameters)
                validate_baseline(output_paths_gunw["GUNW"],
                    perp_path=metadata_path_dict["perpendicularBaseline"],
                    par_path=metadata_path_dict["parallelBaseline"])

def validate_baseline(output_path, perp_path, par_path):
    '''
    validate the baselines if they are all zeros.
    '''

    with h5py.File(output_path) as src_h5:
        perp_base = src_h5[perp_path]
        par_base = src_h5[par_path]

        assert perp_base < 1e-5
        assert par_base < 1e-5

if __name__ == '__main__':
    test_compute_baseline()
    test_add_baseline()
