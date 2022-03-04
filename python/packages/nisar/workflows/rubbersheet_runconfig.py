import os

import journal
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


class RubbersheetRunConfig(RunConfig):

    def __init__(self, args):
        # All InSAR submodules share a common "insar" schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check rubbersheet specifics from YAML
        '''

        error_channel = journal.error('RubbersheetRunConfig.yaml_check')
        scratch_path = self.cfg['product_path_group']['scratch_path']

        # Check if dense_offset is enabled, if not, and rubbersheet is enabled
        # throw an error and do not run rubbersheet
        if not self.cfg['processing']['dense_offsets']['enabled'] and \
                self.cfg['processing']['rubbersheet']['enabled']:
            err_str = 'Dense_offsets must be enabled to run rubbersheet'
            error_channel.log(err_str)
            raise RuntimeError(err_str)

        # Check if offset filtering options are properly allocated. The schema
        # will throw an error if the filter type is not correct.
        filter_type = self.cfg['processing']['rubbersheet']['offsets_filter']

        if filter_type == 'median':
            if 'filter_size_range' not in self.cfg['processing']['rubbersheet']['median']:
                self.cfg['processing']['rubbersheet']['median'][
                    'filter_size_range'] = 5
            if 'filter_size_azimuth' not in \
                    self.cfg['processing']['rubbersheet']['median']:
                self.cfg['processing']['rubbersheet']['median'][
                    'filter_size_azimuth'] = 5
        elif filter_type == 'gaussian':
            if 'sigma_range' not in self.cfg['processing']['rubbersheet']['gaussian']:
                self.cfg['processing']['rubbersheet']['gaussian']['sigma_range'] = 1
            if 'sigma_azimuth' not in self.cfg['processing']['rubbersheet']['gaussian']:
                self.cfg['processing']['rubbersheet']['gaussian']['sigma_azimuth'] = 1

        # If dense_offsets_path is None, assume that we run rubbersheet
        # as part of insar.py. In this case, dense_offsets_path comes from
        # the previous step (dense_offsets) via scratch_path
        if self.cfg['processing']['rubbersheet']['dense_offsets_path'] is None:
            self.cfg['processing']['rubbersheet'][
                'dense_offsets_path'] = scratch_path

        # If geo2rdr_offsets path is None, assume it is scratch_path (see above)
        if self.cfg['processing']['rubbersheet']['geo2rdr_offsets_path'] is None:
            self.cfg['processing']['rubbersheet'][
                'geo2rdr_offsets_path'] = scratch_path

        dense_offsets_path = self.cfg['processing']['rubbersheet'][
            'dense_offsets_path']
        geo2rdr_offsets_path = self.cfg['processing']['rubbersheet'][
            'geo2rdr_offsets_path']
        freq_pols = self.cfg['processing']['input_subset'][
            'list_of_frequencies']
        frequencies = freq_pols.keys()

        # Check if dense_offsets_path is a directory.
        # If yes, check it has dense offsets estimated for
        # the required frequencies and polarizations
        if os.path.isdir(dense_offsets_path):
            helpers.check_mode_directory_tree(dense_offsets_path,
                                              'dense_offsets',
                                              frequencies, freq_pols)
        else:
            # If not a directory, throw an error
            err_str = f"{dense_offsets_path} is invalid; needs to be a directory"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if geo2rdr offset path has appropriate structure
        if os.path.isdir(geo2rdr_offsets_path):
            helpers.check_mode_directory_tree(geo2rdr_offsets_path,
                                              'geo2rdr', frequencies)
        else:
            # If not a directory, throw an error
            err_str = f"{geo2rdr_offsets_path} is invalid; needs to be a directory"
            error_channel.log(err_str)
            raise ValueError(err_str)
