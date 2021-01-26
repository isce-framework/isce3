from collections import defaultdict
import os

import journal

from pybind_nisar.workflows.runconfig import RunConfig
import pybind_nisar.workflows.helpers as helpers


class Rdr2geoRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a commmon `insar` schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is None:
            self.cli_arg_load()
        else:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()

    def cli_arg_load(self):
        """
        Load user provided command line args into minimal cfg dict
        """
        error_channel = journal.error('Rdr2geoRunConfig.cli_arg_load')

        self.cfg = helpers.autovivified_dict

        # input h5 valid?
        if os.path.isfile(self.args.input_h5):
            self.cfg['InputFileGroup']['InputFilePath'] = self.args.input_h5
        else:
            err_str = f"{self.args.input_h5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        # check dem validity. if invalid check_dem raises error.
        helpers.check_dem(self.args.dem)
        self.cfg['DynamicAncillaryFileGroup']['DEMFile'] = self.args.dem

        # check frequency and polarization dict prior to dict assignment
        for k, vals in self.args.freq_pols.items():
            # check if frequency key is valid
            if k not in ['A', 'B']:
                err_str = f"frequency {k} not valid"
                error_channel.log(err_str)
                raise ValueError(err_str)
            # check if polarization values are valid
            for v in vals:
                if v not in ['HH', 'VV', 'HV', 'VH']:
                    err_str = f"polarization {v} not valid"
                    error_channel.log(err_str)
                    raise ValueError(err_str)

        self.cfg['processing']['input_subset']['list_of_frequencies'] = self.args.freq_pols

        helpers.check_write_dir(self.args.scratch)
        self.cfg['ProductPathGroup']['ScratchPath'] = self.args.scratch
