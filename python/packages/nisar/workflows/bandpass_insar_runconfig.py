from collections import defaultdict
import os

import journal

from nisar.workflows.runconfig import RunConfig
import nisar.workflows.helpers as helpers


class BandpassRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a commmon `insar` schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is None:
            self.cli_arg_load()
        else:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def cli_arg_load(self):
        """
        Load user provided command line args into minimal cfg dict
        """
        error_channel = journal.error('BandpassRunConfig.cli_arg_load')

        self.cfg = helpers.autovivified_dict()

        # ref hdf5 valid?
        if not os.path.isfile(self.args.ref_hdf5):
            err_str = f"{self.args.ref_hdf5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        self.cfg['InputFileGroup']['InputFilePath'] = self.args.ref_hdf5

        # sec hdf5 valid?
        if not os.path.isfile(self.args.sec_hdf5):
            err_str = f"{self.args.sec_hdf5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        self.cfg['InputFileGroup']['SecondaryFilePath'] = self.args.sec_hdf5

        if self.args.rows_per_block < 1:
            err_str = f"range look of {self.args.rows_per_block} must be >= 1"
            error_channel.log(err_str)
            raise ValueError(err_str)

        self.cfg['processing']['blocksize'] = self.args.rows_per_block

        # check frequency and polarization dict prior to dict assignment
        freq_pols = self.args.freq_pols
        for k, vals in freq_pols.items():
            # check if frequency key is valid
            if k not in ['A', 'B']:
                err_str = f"frequency {k} not valid"
                error_channel.log(err_str)
                raise ValueError(err_str)
            # check if polarization values are valid
            for val in vals:
                if val not in ['HH', 'VV', 'HV', 'VH']:
                    err_str = f"polarization {val} not valid"
                    error_channel.log(err_str)
                    raise ValueError(err_str)

        self.cfg['processing']['input_subset']['list_of_frequencies'] = freq_pols
        frequencies = freq_pols.keys()

        outdir = os.path.dirname(self.args.output_h5)
        helpers.check_write_dir(outdir)
        self.cfg['ProductPathGroup']['SASOutputFile'] = self.args.output_h5

        # check if secondary HDF5 has all the frequencies and polarizations
        helpers.check_hdf5_freq_pols(self.args.sec_hdf5, freq_pols)

    def yaml_check(self):
        '''
        Check bandpass specifics from YAML
        '''
        error_channel = journal.error('BandpassRunConfig.yaml_check')

        scratch_path = self.cfg['ProductPathGroup']['ScratchPath']
        # if coregistered_slc_path not provided, use ScratchPath as source for coregistered SLCs
        if 'bandpass_slc_path' not in self.cfg['processing']['bandpass']:
            self.cfg['processing']['bandpass']['bandpass_slc_path'] = scratch_path

        # check whether bandpass_slc_path is a directory or file
        bandpass_slc_path = self.cfg['processing']['bandpass']['bandpass_slc_path']
        if not os.path.isdir(bandpass_slc_path) and not os.path.isfile(bandpass_slc_path):
            err_str = f"{bandpass_slc_path} is invalid; needs to be a file or directory."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # check if required coregistered frequency/polarization rasters exist in directory or HDF5 file
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        frequencies = freq_pols.keys()
        if os.path.isdir(bandpass_slc_path):
            helpers.check_mode_directory_tree(bandpass_slc_path, 'bandpass_slc',\
                                              frequencies, freq_pols)
        else:
            helpers.check_hdf5_freq_pols(bandpass_slc_path, freq_pols)
