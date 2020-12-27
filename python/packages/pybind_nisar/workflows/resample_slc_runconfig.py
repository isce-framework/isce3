from collections import defaultdict
import os

import journal

from pybind_nisar.workflows.runconfig import RunConfig
import pybind_nisar.workflows.helpers as helpers


class ResampleSlcRunConfig(RunConfig):
    def __init__(self, args):
        # InSAR submodules have a common InSAR schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is None:
            self.cli_arg_load()
        else:
            super().load_geocode_yaml_to_dict()
            super().geocode_common_arg_load()
            self.yaml_check()

    def cli_arg_load(self):
        ''' 
        Load user-provided command line args into minimal cfg dictionary
        '''

        error_channel = journal.error('ResampleSlcRunConfig.cli_arg_load')

        self.cfg = helpers.autovivified_dict

        # Valid input h5?
        if os.path.isfile(self.args.input_h5):
            self.cfg['InputFileGroup']['SecondaryFilePath'] = self.args.input_h5
        else:
            err_str = f"{self.args.input_h5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        # Valid lines_per_tile?
        if isinstance(self.args.lines_per_tile, int):
            self.cfg['processing']['resample']['lines_per_tile'] = self.args.lines_per_tile
        else:
            err_str = f"{self.args.lines_per_tile} not a valid number"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid input for frequencies and polarizations?
        for k, vals in self.args.freq_pols.items():
            # Check if frequency is valid
            if k not in ['A', 'B']:
                err_str = f"frequency {k} not valid"
                error_channel.log(err_str)
                raise ValueError(err_str)
            # Check if polarizations values are valid
            for v in vals:
                if v not in ['HH', 'VV', 'HV', 'VH']:
                    err_str = f"polarization {v} not valid"
                    error_channel.log(err_str)
                    raise ValueError(err_str)

        self.cfg['processing']['input_subset']['list_of_frequencies'] = self.args.freq_pols

        # Check scratch directory
        helpers.check_write_dir(self.args.scratch)
        self.cfg['ProductPathGroup']['ScratchPath'] = self.args.scratch

        # Check if the offset directory is provided
        if self.args.off_dir is None:
            off_dir = self.args.off_dir
        else:
            # Full InSAR workflow expect geometric offsets in scratch
            off_dir = self.args.scratch

        # Check offsets directory structure
        frequencies = self.args.freq_pols.keys()
        helpers.check_mode_directory_tree(off_dir, 'geo2rdr', frequencies)

        self.cfg['processing']['resample']['offset_dir'] = off_dir

    def yaml_check(self):
        '''
        Check resample specifics from YAML.
        '''
        # Use scratch as offset_dir if none given in YAML
        if 'offset_dir' not in self.cfg['processing']['resample']:
            self.cfg['processing']['resample']['offset_dir'] = self.cfg['ProductPathGroup']['ScratchPath']

        # Check offsets directory structure
        off_dir = self.cfg['processing']['resample']['offset_dir']
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        frequencies = freq_pols.keys()
        helpers.check_mode_directory_tree(off_dir, 'geo2rdr', frequencies)
