import os

import journal
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


class FilterInterferogramRunConfig(RunConfig):
    def __init__(self, args):
        # All InSAR workflow steps share a common InSAR schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check filter_interferogram inputs from YAML file
        '''
        error_channel = journal.error('FilterInterferogramRunconfig.yaml_check')

        # Extract frequency and polarizations to process
        freq_pols = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

        # interferogram_path is required for stand-alone usage of filter_interferogram.py
        interferogram_path = self.cfg['processing']['filter_interferogram'][
            'interferogram_path']
        if interferogram_path is None:
            err_str = f'{interferogram_path} in filter_interferogram required for stand-alone usage of filter_interferogram.py'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if interferogram_path is a file
        if not os.path.isfile(interferogram_path):
            err_str = f"{interferogram_path} is invalid; needs to be a file"
            error_channel.log(err_str)
            raise ValueError(err_str)
        else:
            # Check that required polarization/frequencies are in interferogram_path
            helpers.check_hdf5_freq_pols(interferogram_path, freq_pols)

        # If mask/weight is assigned check if it exists
        mask_options = self.cfg['processing']['filter_interferogram']['mask']
        if 'general' in mask_options and mask_options['general'] is not None:
            if not os.path.isfile(mask_options['general']):
                err_str = f"The mask file {mask_options['general']} is not a file"
                error_channel.log(err_str)
                raise ValueError(err_str)
        else:
            # Otherwise check that mask for individual freq/pols are correctly assigned
            for freq, pol_list in freq_pols.items():
                if freq in mask_options:
                    for pol in pol_list:
                        if pol in mask_options[freq]:
                            mask_file = mask_options[freq][pol]
                            if mask_file is not None and not os.path.isfile(
                                    mask_file):
                                err_str = f"{mask_file} is invalid; needs to be a file"
                                error_channel.log(err_str)
                                raise ValueError(err_str)

        # Check filter_type and if not allocated, create a default cfg dictionary
        # filter_type will be present at runtime because is allocated in share/nisar/defaults
        filter_type = self.cfg['processing']['filter_interferogram']['filter_type']
        if filter_type != 'no_filter' and filter_type not in \
                self.cfg['processing']['filter_interferogram']:
            self.cfg['processing']['filter_interferogram'][filter_type] = {}

        # Based on filter_type, check if related dictionary and/or parameters
        # are assigned. Note, if filter_type='boxcar', the filter dictionary
        # is filled by share/nisar/defaults
        if filter_type == 'gaussian':
            if 'gaussian' not in self.cfg['processing']['filter_interferogram']:
                self.cfg['processing']['filter_interferogram'][
                    'gaussian'] = {}
            gaussian_options = self.cfg['processing']['filter_interferogram'][
                'gaussian']
            if 'sigma_range' not in gaussian_options:
                gaussian_options['sigma_range'] = 1
            if 'sigma_azimuth' not in gaussian_options:
                gaussian_options['sigma_azimuth'] = 1
            if 'filter_size_range' not in gaussian_options:
                gaussian_options['filter_size_range'] = 9
            if 'filter_size_azimuth' not in gaussian_options:
                gaussian_options['filter_size_azimuth'] = 9
