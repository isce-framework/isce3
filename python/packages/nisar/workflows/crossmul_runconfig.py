import os

import journal
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


class CrossmulRunConfig(RunConfig):
    def __init__(self, args, resample_type='coarse'):
        # all insar submodules share a common `insar` schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check(resample_type)

    def yaml_check(self, resample_type):
        '''
        Check crossmul specifics from YAML
        '''
        error_channel = journal.error('CrossmulRunConfig.yaml_check')

        scratch_path = self.cfg['product_path_group']['scratch_path']
        crossmul_cfg = self.cfg['processing']['crossmul']
        # if coregistered_slc_path not provided, use scratch_path as source for coregistered SLCs
        if 'coregistered_slc_path' not in crossmul_cfg:
            crossmul_cfg['coregistered_slc_path'] = scratch_path

        # check whether coregistered_slc_path is a directory or file
        coregistered_slc_path = crossmul_cfg['coregistered_slc_path']
        if not os.path.isdir(coregistered_slc_path) \
                and not os.path.isfile(coregistered_slc_path):
            err_str = f"{coregistered_slc_path} is invalid; needs to be a file or directory."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # check if required coregistered frequency/polarization rasters exist in directory or HDF5 file
        # Distinguish between coarse and fine resample_slc directories
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        frequencies = freq_pols.keys()
        if os.path.isdir(coregistered_slc_path):
            if resample_type not in ['coarse', 'fine']:
                err_str = f"{resample_type} not a valid resample slc type"
                error_channel.log(err_str)
                raise ValueError(err_str)
            helpers.check_mode_directory_tree(coregistered_slc_path,
                                              f'{resample_type}_resample_slc',
                                              frequencies, freq_pols)
        else:
            helpers.check_hdf5_freq_pols(coregistered_slc_path, freq_pols)

        # flatten is bool False disables flattening in crossmul
        # flatten is bool True runs flatten and sets data directory to scratch
        # flatten is str assumes value is path to data directory
        # Data directory contains range offset rasters
        # The following directory tree is required:
        # flatten
        # └── geo2rdr
        #     └── freq(A,B)
        #         └── range.off
        # flatten defaults to bool True
        # flatten_path defaults to scratch_path
        flatten = crossmul_cfg['flatten']
        if flatten:
            if 'flatten_path' in crossmul_cfg:
                flatten_path = crossmul_cfg['flatten_path']
            else:
                flatten_path = scratch_path

            # check if required frequency range offsets exist
            helpers.check_mode_directory_tree(flatten_path, 'geo2rdr',
                                              frequencies)

            crossmul_cfg['flatten_path'] = flatten_path
