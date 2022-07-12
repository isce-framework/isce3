import os

import journal
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


class OffsetsProductRunConfig(RunConfig):

    def __init__(self, args):
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check offsets product specifics from YAML file
        '''
        error_channel = journal.error('OffsetsProductRunConfig.yaml_check')
        scratch_path = self.cfg['product_path_group']['scratch_path']

        # If coregistered_slc_path is None, assume that we run the offsets
        # products workflow as a part of insar.py. In this case,
        # coregistered_slc_path comes from the previous step via scratch_path
        if self.cfg['processing']['offsets_product'][
            'coregistered_slc_path'] is None:
            self.cfg['processing']['offsets_product'][
                'coregistered_slc_path'] = scratch_path

        # Check if coregistered_slc_path is a path or a directory
        coregistered_slc_path = self.cfg['processing']['offsets_product'][
            'coregistered_slc_path']
        if not os.path.exists(coregistered_slc_path):
            err_str = f"{coregistered_slc_path} invalid; must be a file or a directory"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if geometry-coregistered rasters exists in directory (or HDF5)
        freq_pols = self.cfg['processing']['input_subset'][
            'list_of_frequencies']
        frequencies = freq_pols.keys()
        if os.path.isdir(coregistered_slc_path):
            helpers.check_mode_directory_tree(coregistered_slc_path,
                                              'coarse_resample_slc',
                                              frequencies, freq_pols)
        else:
            helpers.check_hdf5_freq_pols(coregistered_slc_path, freq_pols)

        # Check that at least one layer is specified. If not, throw exception
        offs_params = self.cfg['processing']['offsets_product']
        layer_keys = [key for key in offs_params.keys() if
                      key.startswith('layer')]

        # If no layer is found, throw an exception
        if not layer_keys:
            err_str = "No offset layer specified; at least one layer is required"
            error_channel.log(err_str)
            raise ValueError(err_str)
