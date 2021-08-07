import os

import journal
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


class UnwrapRunConfig(RunConfig):
    def __init__(self, args):
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check phase_unwrap specifics from YAML
        '''
        error_channel = journal.error('CrossmulRunConfig.yaml_check')

        # Check if crossmul_path is provided (needed for stand-alone unwrapping)
        if self.cfg['processing']['phase_unwrap']['crossmul_path'] is None:
            err_str = "'crossmul_path' file path under `phase_unwrap' required for standalone execution with YAML"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Allocate, if not present, cfg related to user-unwrapper choice
        algorithm = self.cfg['processing']['phase_unwrap']['algorithm']
        if algorithm not in self.cfg['processing']['phase_unwrap']:
            self.cfg['processing']['phase_unwrap'][algorithm] = {}

        # Check if crossmul path is a directory or a file
        crossmul_path = self.cfg['processing']['phase_unwrap']['crossmul_path']
        if not os.path.isfile(crossmul_path):
            err_str = f"{crossmul_path} is invalid; needs to be a file"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if required polarizations/frequency are in crossmul_path file
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        helpers.check_hdf5_freq_pols(crossmul_path, freq_pols)
