'''
collection of functions for processing and validating args
'''

import argparse
import os

import h5py
import journal
import numpy as np
import osr
from ruamel.yaml import YAML
import yamale

import pybind_isce3 as isce
from pybind_nisar.workflows import geogrid


def deep_update(original, update):
    '''
    update default runconfig key with user supplied dict
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for key, val in update.items():
        if isinstance(val, dict):
            original[key] = deep_update(original.get(key, {}), val)
        else:
            original[key] = val

    # return updated original
    return original


def dir():
    """
    convenience function that returns file directory
    """
    return os.path.dirname(os.path.realpath(__file__))


def load(workflow_name):
    '''
    load user provided yaml into run configuration dictionary
    '''
    error_channel = journal.error('runconfig.load')

    # process args from terminal
    args = process_args()

    # assign default config and yamale schema
    # assume defaults have already been yamale validated
    if workflow_name == 'GCOV':
        default_cfg = f'{dir()}/defaults/gcov.yaml'
        schema = yamale.make_schema(f'{dir()}/schemas/gcov.yaml', parser='ruamel')
    elif workflow_name == 'GSLC':
        default_cfg = f'{dir()}/defaults/gslc.yaml'
        schema = yamale.make_schema(f'{dir()}/schemas/gslc.yaml', parser='ruamel')
    elif workflow_name == 'INSAR':
        default_cfg = f'{dir()}/defaults/insar.yaml'
        schema = yamale.make_schema(f'{dir()}/schemas/insar.yaml', parser='ruamel')
    else:
        # quit on unrecognized workflow_name
        err_str = f'Unsupported geocode workflow: {workflow_name}'
        error_channel.log(err_srt)
        raise ValueError(err_str)

    # validate yaml from terminal
    try:
        data = yamale.make_data(args.run_config_path, parser='ruamel')
    except yamale.YamaleError as e:
        err_str = f'Yamale unable to load {workflow_name} runconfig yaml {args.run_config_path} for validation.'
        raise yamale.YamaleError(err_str) from e
    try:
        yamale.validate(schema, data)
    except yamale.YamaleError as e:
        raise yamale.YamaleError(f'Validation fail for {workflow_name} runconfig yaml {args.run_config_path}.')

    # overwrite default with user provided values
    cfg = load_yaml(args.run_config_path, default_cfg)

    prep_paths(cfg, workflow_name)
    prep_frequency_and_polarizations(cfg)
    prep_geocode_cfg(cfg)

    if workflow_name == 'GCOV':
        prep_gcov(cfg)

    return cfg


def load_yaml(yaml, default):
    """
    Load default runconfig, override with user input, and convert to dict
    Leading namespaces can be stripped off down the line
    """
    parser = YAML(typ='safe')
    cfg = parser.load(open(default, 'r'))
    # remove top
    cfg = cfg['runconfig']['groups']
    with open(yaml) as f_yaml:
        user = parser.load(f_yaml)
        # remove top levels of dict
        user = user['runconfig']['groups']

    # copy user suppiled config into default config
    deep_update(cfg, user)

    # remove default frequency(s) if not chosen by user
    default_freqs = cfg['processing']['input_subset']['list_of_frequencies']
    user_freqs = user['processing']['input_subset']['list_of_frequencies'].keys()
    discards = [freq for freq in default_freqs if freq not in user_freqs]
    for discard in discards:
        del default_freqs[discard]

    return cfg


def prep_gcov(cfg):
    '''
    Check gcov specific config parameters
    and replace all enum strings with isce3 enum values.
    '''
    error_channel = journal.error('runconfig.prep_gcov')

    geocode_dict = cfg['processing']['geocode']

    if geocode_dict['abs_rad_cal'] is None:
        geocode_dict['abs_rad_cal'] = 1.0

    if geocode_dict['memory_mode'] == 'single_block':
        geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.SINGLE_BLOCK
    elif geocode_dict['memory_mode'] == 'geogrid':
        geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID
    elif geocode_dict['memory_mode'] == 'geogrid_radargrid':
        geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID_AND_RADARGRID
    else:
        geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.AUTO

    if geocode_dict['algorithm_type'] == 'interp':
        geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.INTERP
    elif geocode_dict['algorithm_type'] == 'area_projection':
        geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.AREA_PROJECTION
    elif geocode_dict['algorithm_type'] == 'area_projection_gamma_naught':
        geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.AREA_PROJECTION_GAMMA_NAUGHT
    else:
        err_str = f'Unsupported geocode algorithm: {geocode_dict["algorithm_type"]}'
        error_channel.log(err_str)
        raise ValueError(err_str)

    rtc_dict = cfg['processing']['rtc']

    # only 2 RTC algorithms supported: david-small (default) & area-projection
    if rtc_dict['algorithm_type'] == "area_projection":
        rtc_dict['algorithm_type'] = isce.geometry.RtcAlgorithm.RTC_AREA_PROJECTION
    else:
        rtc_dict['algorithm_type'] = isce.geometry.RtcAlgorithm.RTC_DAVID_SMALL

    if rtc_dict['input_terrain_radiometry'] == "sigma0":
        rtc_dict['input_terrain_radiometry'] = isce.geometry.RtcInputRadiometry.SIGMA_NAUGHT_ELLIPSOID
    else:
        rtc_dict['input_terrain_radiometry'] = isce.geometry.RtcInputRadiometry.BETA_NAUGHT

    rtc_min_value_db = rtc_dict['rtc_min_value_db']
    if rtc_dict['rtc_min_value_db'] is None:
        rtc_dict['rtc_min_value_db'] = np.nan


def _get_polarizations(h5_path, freq):
    '''
    return list of polarizations from h5 for a given frequency
    '''
    with h5py.File(h5_path, 'r') as h5_obj:

        # get polarization list dataset
        ds_path = f'science/LSAR/SLC/swaths/frequency{freq}/listOfPolarizations'
        if ds_path not in h5_obj:
            return None
        pols = h5_obj[ds_path][()]

    # convert from numpy.bytes_ to str
    pols = [pol.decode('UTF-8') for pol in pols]

    return pols


def _get_epsg_ellipsoid(epsg):
    '''
    make ellipsoid from EPSG
    '''
    sr = osr.SpatialReference()
    res = sr.ImportFromEPSG(epsg)
    if res != 0:
        raise RuntimeError(f'osr could not import EPSG {epsg}')
    a = sr.GetSemiMajor()
    b = sr.GetSemiMinor()
    # calculate eccentricity
    e2 = 1 - (b/a)**2

    # create ellipsoid
    ellipsoid = isce.core.Ellipsoid(a, e2)
    return ellipsoid


def prep_paths(cfg, workflow_name):
    '''
    make sure input paths is valid
    '''
    error_channel = journal.info('runconfig.check_common')

    # check input file value
    input_path = cfg['InputFileGroup']['InputFilePath']

    if isinstance(input_path, list):
        n_inputs = len(input_path)
        if workflow_name in ['GCOV', 'GSLC']:
            if n_inputs != 1:
                err_str = f'{n_inputs} inputs provided. Only one input file is required.'
                error_channel.log(err_str)
                raise ValueError(err_str)
        elif workflow_name == 'INSAR':
            if n_inputs == 2:
                secondary_path = input_path[1]
                if not os.path.isfile(secondary_path):
                    raise ValueError(f'Secondary RSLC not found {secondary_path} ')
                cfg['InputFileGroup']['SecondaryFilePath'] = secondary_path
            else:
                err_str = f"{n_inputs} provided. 2 input files are required."
                error_channel.log(err_str)
                raise ValueError(err_str)
        else:
            err_str = f'{workflow_name} unsupported'
            error_channel.log(err_str)
            raise ValueError(err_str)

        input_path = input_path[0]
        if not os.path.isfile(input_path):
            err_str = f'Reference SLC not found {input_path}'
            error_channel.log(err_str)
            raise ValueError(err_str)

    if type(input_path) != str:
        err_str = 'String type not provided for path to YAML.'
        error_channel.log(err_str)
        raise ValueError(err_str)

    if not os.path.isfile(input_path):
        err_str = f"{input_path} input not found."
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)

    cfg['InputFileGroup']['InputFilePath'] = input_path

    # ensure validity of DEM inputs
    dem_path = cfg['DynamicAncillaryFileGroup']['DEMFile']
    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"{dem_path} not valid")

    # create output dir (absolute or relative) if it does exist. do nothing if nothing
    output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']
    output_dir = os.path.dirname(output_hdf5)
    if output_dir and not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError:
            raise OSError(f"Unable to create {output_dir}")


def prep_geocode_cfg(cfg):
    '''
    check geocode config and initialize as needed
    '''
    geocode_dict = cfg['processing']['geocode']

    # check for user provided EPSG and grab from DEM if none provided
    if geocode_dict['outputEPSG'] is None:
        geocode_dict['outputEPSG'] = isce.io.Raster(cfg['DynamicAncillaryFileGroup']['DEMFile']).get_epsg()

    # create ellipsoid from ESPG
    ellipsoid = _get_epsg_ellipsoid(geocode_dict['outputEPSG'])
    cfg['processing']['geocode']['ellipsoid'] = ellipsoid

    # make geogrids for each frequency
    geogrids = {}

    # for each frequency check source RF polarization values and make geogrids
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    for freq in freq_pols.keys():
        # build geogrids only if pols not None
        geogrids[freq] = geogrid.create(cfg, freq)

    # place geogrids in cfg for later processing
    cfg['processing']['geocode']['geogrids'] = geogrids


def prep_frequency_and_polarizations(cfg):
    '''
    check frequency and polarizations and fix as needed
    '''
    error_channel = journal.error('runconfig.prep_frequency_and_polarizations')
    input_path = cfg['InputFileGroup']['InputFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    for freq in freq_pols.keys():
        # first check polarizations from source hdf5
        rslc_pols = _get_polarizations(input_path, freq)
        # use all RSLC polarizations if None provided
        if freq_pols[freq] is None:
            freq_pols[freq] = rslc_pols
        # use polarizations provided by user
        else:
            # check if user provided polarizations match RSLC ones
            for usr_pol in freq_pols[freq]:
                if usr_pol not in rslc_pols:
                    err_str = f"{usr_pol} invalid; not found in source polarization"
                    error_channel.log(err_str)
                    raise ValueError(err_str)


def process_args():
    '''
    process args from terminal
    '''
    error_channel = journal.error('runconfig.process_args')

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('run_config_path', type=str, help='Path to run config file')

    args = parser.parse_args()

    # make a journal device that is attached to a file
    journal.debug.journal.device = "journal.file"
    journal.debug.journal.device.log = args.run_config_path + ".log"

    if not os.path.isfile(args.run_config_path):
        err_str = f"{args.run_config_path} not valid"
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)

    return args
