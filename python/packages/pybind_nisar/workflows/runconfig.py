'''
base class for processing and validating args
'''

import os

import journal
from ruamel.yaml import YAML
import yamale
import numpy as np

import pybind_isce3 as isce
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import geogrid
import pybind_nisar.workflows.helpers as helpers


class RunConfig:
    def __init__(self, args, workflow_name=''):
        # argparse namespace
        self.args = args
        # workflow name in lower case
        self.workflow_name = workflow_name
        self.cfg = {}
        self.user = {}

    def load_yaml_to_dict(self):
        """
        Load default runconfig, override with user input, and convert to dict
        Leading namespaces can be stripped off down the line
        """
        # assign default config and yamale schema
        # assume defaults have already been yamale validated
        try:
            default_cfg = f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/{self.workflow_name}.yaml'
            schema = yamale.make_schema(f'{helpers.WORKFLOW_SCRIPTS_DIR}/schemas/{self.workflow_name}.yaml',
                                        parser='ruamel')
        except:
            err_str = f'workflow {self.workflow_name} does not have a schema.'
            raise ValueError(err_str)

        # set run config type
        run_config_is_txt = False
        # if newlines then run_config is YAML string (primarily for unit test)
        if self.args.run_config_path is not None:
            if '\n' in self.args.run_config_path:
                run_config_is_txt = True

        # validate yaml file taken from command line
        try:
            if run_config_is_txt:
                data = yamale.make_data(content=self.args.run_config_path, parser='ruamel')
            else:
                data = yamale.make_data(self.args.run_config_path, parser='ruamel')
        except yamale.YamaleError as e:
            err_str = f'Yamale unable to load {self.workflow_name} runconfig yaml {self.args.run_config_path} for validation.'
            raise yamale.YamaleError(err_str) from e
        try:
            yamale.validate(schema, data)
        except yamale.YamaleError as e:
            err_str = f'Validation fail for {self.workflow_name} runconfig yaml {self.args.run_config_path}.'
            raise yamale.YamaleError(err_str) from e

        # load default config
        parser = YAML(typ='safe')
        with open(default_cfg, 'r') as f:
            self.cfg = parser.load(f)

        # load user config based on input type
        if run_config_is_txt:
            self.user = parser.load(self.args.run_config_path)
        else:
            with open(self.args.run_config_path) as f_yaml:
                self.user = parser.load(f_yaml)

        # copy user suppiled config into default config
        helpers.deep_update(self.cfg, self.user)

    def load_geocode_yaml_to_dict(self):
        '''
        Modify config dict for geocoded related workflows.
        '''
        self.load_yaml_to_dict()

        # remove top 2 levels of dict to reduce boiler plate
        self.cfg = self.cfg['runconfig']['groups']
        self.user = self.user['runconfig']['groups']

        # update logging destination
        if self.args.log_file and 'logging' in self.cfg:
            log_path = self.cfg['logging']['path']
            helpers.check_log_dir_writable(log_path)
            # check logging write mode. default to 'a'/append if no mode specified.
            if 'write_mode' in self.cfg['logging']:
                write_mode = self.cfg['logging']['write_mode']
            else:
                write_mode = 'a'
            journal.debug.journal.device.log = open(log_path, write_mode)

        # remove default frequency(s) if not chosen by user
        default_freqs = self.cfg['processing']['input_subset']['list_of_frequencies']
        user_freqs = self.user['processing']['input_subset']['list_of_frequencies'].keys()
        discards = [freq for freq in default_freqs if freq not in user_freqs]
        for discard in discards:
            del default_freqs[discard]

    def prep_paths(self):
        '''
        Prepare input and output paths
        '''
        error_channel = journal.error('RunConfig.load')

        # check input file value
        input_path = self.cfg['InputFileGroup']['InputFilePath']

        # check input HDF5(s) in cfg
        if isinstance(input_path, list):
            n_inputs = len(input_path)
            if self.workflow_name in ['gcov', 'gslc']:
                if n_inputs != 1:
                    err_str = f'{n_inputs} inputs provided. Only one input file is allowed.'
                    error_channel.log(err_str)
                    raise ValueError(err_str)
            elif self.workflow_name == 'insar':
                if n_inputs == 2:
                    secondary_path = input_path[1]
                    if not os.path.isfile(secondary_path):
                        err_str = f'{secondary_path} secondary RSLC not found.'
                        error_channel.log(err_str)
                        raise ValueError(err_str)
                    self.cfg['InputFileGroup']['SecondaryFilePath'] = secondary_path
                else:
                    err_str = f"{n_inputs} provided. 2 input files are required for INSAR workflow."
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)
            else:
                err_str = f'{self.workflow_name} unsupported'
                error_channel.log(err_str)
                raise ValueError(err_str)

            input_path = input_path[0]
            if not os.path.isfile(input_path):
                err_str = f'Reference SLC not found {input_path}'
                error_channel.log(err_str)
                raise ValueError(err_str)

        if not isinstance(input_path, str):
            err_str = 'String type not provided for path to YAML.'
            error_channel.log(err_str)
            raise ValueError(err_str)

        if not os.path.isfile(input_path):
            err_str = f"{input_path} input not found."
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        self.cfg['InputFileGroup']['InputFilePath'] = input_path

        # ensure validity of DEM inputs
        helpers.check_dem(self.cfg['DynamicAncillaryFileGroup']['DEMFile'])

        # check if each product type has an output
        output_hdf5 = self.cfg['ProductPathGroup']['SASOutputFile']
        output_dir = os.path.dirname(output_hdf5)

        helpers.check_write_dir(output_dir)
        helpers.check_write_dir(self.cfg['ProductPathGroup']['ScratchPath'])

    def prep_frequency_and_polarizations(self):
        '''
        check frequency and polarizations and fix as needed
        '''
        error_channel = journal.error('RunConfig.prep_frequency_and_polarizations')
        input_path = self.cfg['InputFileGroup']['InputFilePath']
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']

        slc = SLC(hdf5file=input_path)

        for freq in freq_pols.keys():
            if freq not in slc.frequencies:
                err_str = f"Frequency {freq} invalid; not found in source frequencies."
                error_channel.log(err_str)
                raise ValueError(err_str)

            # first check polarizations from source hdf5
            rslc_pols = slc.polarizations[freq]
            # use all RSLC polarizations if None provided
            if freq_pols[freq] is None:
                freq_pols[freq] = rslc_pols
                continue

            # use polarizations provided by user
            # check if user provided polarizations match RSLC ones
            for usr_pol in freq_pols[freq]:
                if usr_pol not in rslc_pols:
                    err_str = f"{usr_pol} invalid; not found in source polarizations."
                    error_channel.log(err_str)
                    raise ValueError(err_str)

    def prep_geocode_cfg(self):
        '''
        check geocode config and initialize as needed
        '''
        geocode_dict = self.cfg['processing']['geocode']

        # check for user provided EPSG and grab from DEM if none provided
        if geocode_dict['outputEPSG'] is None:
            geocode_dict['outputEPSG'] = isce.io.Raster(self.cfg['DynamicAncillaryFileGroup']['DEMFile']).get_epsg()

        # make geogrids for each frequency
        geogrids = {}

        # for each frequency check source RF polarization values and make geogrids
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        for freq in freq_pols.keys():
            # build geogrids only if pols not None
            geogrids[freq] = geogrid.create(self.cfg, freq)

        # place geogrids in cfg for later processing
        self.cfg['processing']['geocode']['geogrids'] = geogrids

    def prep_cubes_geocode_cfg(self):
        '''
        check cubes geocode config and initialize as needed

        radar_grid_cubes is an optional group. If not provided,
        the geocode group should be used, but with different X and Y
        spacing defaults
        '''
        geocode_dict = self.cfg['processing']['geocode']

        # check for user provided EPSG and grab geocode group EPSG if not provided

        if self.cfg['processing']['radar_grid_cubes']['outputEPSG'] is None: 
            cubes_epsg = geocode_dict['outputEPSG']
        else:
            cubes_epsg = self.cfg['processing']['radar_grid_cubes']['outputEPSG']

        self.cfg['processing']['radar_grid_cubes']['outputEPSG'] = cubes_epsg

        if not self.cfg['processing']['radar_grid_cubes']['heights']:
            self.cfg['processing']['radar_grid_cubes']['heights'] = \
                list(np.arange(-1000, 9001, 500))

        if cubes_epsg == 4326:
            # lat/lon
            default_cube_geogrid_spacing_x = 0.005
            default_cube_geogrid_spacing_y = -0.005
        else:
            # meters
            default_cube_geogrid_spacing_x = 500
            default_cube_geogrid_spacing_y = -500

        radar_grid_cubes_dict = self.cfg['processing']['radar_grid_cubes']
        self.cfg['processing']['radar_grid_cubes']['outputEPSG'] = cubes_epsg

        # build geogrid
        frequency_ref = 'A'
        frequency_group = None
        cubes_geogrid = geogrid.create(
            self.cfg, frequency_group = frequency_group, 
            frequency = frequency_ref,
            geocode_dict = radar_grid_cubes_dict,
            default_spacing_x = default_cube_geogrid_spacing_x,
            default_spacing_y = default_cube_geogrid_spacing_y)

        # place geogrid in cfg for later processing
        self.cfg['processing']['radar_grid_cubes']['geogrid'] = cubes_geogrid

    def geocode_common_arg_load(self):
        '''
        Workflows needing geocoding
        '''
        self.prep_paths()
        self.prep_frequency_and_polarizations()
        self.prep_geocode_cfg()
        self.prep_cubes_geocode_cfg()
