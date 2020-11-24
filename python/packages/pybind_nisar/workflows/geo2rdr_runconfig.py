import os
from collections import defaultdict

import journal

from pybind_nisar.workflows.runconfig import RunConfig
import pybind_nisar.workflows.helpers as helpers


class Geo2rdrRunConfig(RunConfig):
    def __init__(self, args):
        # InSAR submodules share a common "InSAR" schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is None:
            self.cli_arg_load()
        else:
            super().load_geocode_yaml_to_dict()
            super().geocode_common_arg_load()
            self.yaml_check()

    def cli_arg_load(self):
        """
        Load user-provided command line args into mininal cfg dict
        """
        error_channel = journal.error('Geo2rdrRunConfig.cli_arg_load')

        self.cfg = defaultdict(helpers.autovivified_dict)

        # Valid input h5?
        if os.path.isfile(self.args.input_h5):
            self.cfg['InputFileGroup']['SecondaryFilePath'] = self.args.input_h5
        else:
            err_str = f"{self.args.input_h5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        # Valid geo2rdr threshold?
        if 1e-9 < self.args.threshold < 1e-3:
            self.cfg['processing']['geo2rdr']['threshold'] = self.args.threshold
        else:
            err_str = f"{self.args.threshold} not a valid threshold for geo2rdr"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid number of iterations?
        if isinstance(self.args.max_iter, int):
            self.cfg['processing']['geo2rdr']['maxiter'] = self.args.max_iter
        else:
            err_str = f"{self.args.max_iter} not a valid number of iterations for geo2rdr"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check dem validity
        helpers.check_dem(self.args.dem)
        self.cfg['DynamicAncillaryFileGroup']['DEMFile'] = self.args.dem

        # Check frequency and polarization prior to dict assignment
        for k, vals in self.args.freq_pols.items():
            # Check validity of frequency key
            if k not in ['A', 'B']:
                err_str = f"frequency {k} not valid"
                error_channel.log(err_str)
                raise ValueError(err_str)
            # Check if polarization values are valid
            for v in vals:
                if v not in ['HH', 'VV', 'HV', 'VH']:
                    err_str = f"polarization {v} not valid"
                    error_channel.log(err_str)
                    raise ValueError(err_str)

        self.cfg['processing']['input_subset']['list_of_frequencies'] = self.args.freq_pols

        # Valid scratch directory?
        helpers.check_write_dir(self.args.scratch)
        self.cfg['ProductPathGroup']['ScratchPath'] = self.args.scratch

        # Was topo path provided
        if self.args.topo is None:
            topo_path = self.args.topo
        else:
            # full insar workflow expects topo result directory in scratch
            topo_path = self.args.scratch

        # Valid topo raster?
        frequencies = self.args.freq_pols.keys()
        helpers.check_mode_directory_tree(topo_path, 'rdr2geo', frequencies)

        # check if topo path provided
        if self.args.topo is not None:
            topo_path = self.args.topo
        else:
            # full insar workflow expects topo result directory in scratch
            topo_path = self.args.scratch

        # Check topo directory structure
        frequencies = self.args.freq_pols.keys()
        helpers.check_mode_directory_tree(topo_path, 'rdr2geo', frequencies)

        self.cfg['processing']['geo2rdr']['topo_path'] = topo_path

    def yaml_check(self):
        '''
        Check geo2rdr specifics from YAML.
        '''
        # Use scratch as topo_path if none given in YAML
        if 'topo_path' not in self.cfg['processing']['geo2rdr']:
            self.cfg['processing']['geo2rdr']['topo_path'] = self.cfg['ProductPathGroup']['ScratchPath']

        # Check topo directory structure
        topo_path = self.cfg['processing']['geo2rdr']['topo_path']
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        frequencies = freq_pols.keys()
        helpers.check_mode_directory_tree(topo_path, 'rdr2geo', frequencies)
