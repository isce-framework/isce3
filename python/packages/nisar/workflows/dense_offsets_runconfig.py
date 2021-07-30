import os

import journal
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


class DenseOffsetsRunConfig(RunConfig):

    def __init__(self, args):
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check dense offset specifics from YAML file
        '''

        error_channel = journal.error('DenseOffsetsRunConfig.yaml_check')
        scratch_path = self.cfg['ProductPathGroup']['ScratchPath']

        # If coregistered_slc_path is None, assume that we run dense_offsets
        # as part of insar.py. In this case, coregistered_slc_path comes
        # from the previous processing step via scratch_path
        if self.cfg['processing']['dense_offsets']['coregistered_slc_path'] is None:
            self.cfg['processing']['dense_offsets']['coregistered_slc_path'] = scratch_path

        # Check if coregistered_slc_path is a path or directory
        coregistered_slc_path = self.cfg['processing']['dense_offsets']['coregistered_slc_path']
        if not os.path.exists(coregistered_slc_path):
            err_str = f"{coregistered_slc_path} invalid; must be a file or directory"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if geometry-coregistered rasters
        # exists in directory or HDF5 file
        freq_pols = self.cfg['processing']['input_subset'][
            'list_of_frequencies']
        frequencies = freq_pols.keys()

        if os.path.isdir(coregistered_slc_path):
            helpers.check_mode_directory_tree(coregistered_slc_path,
                                              'coarse_resample_slc',
                                              frequencies, freq_pols)
        else:
            helpers.check_hdf5_freq_pols(coregistered_slc_path, freq_pols)
