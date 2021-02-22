from collections import defaultdict
import os

import journal

from pybind_nisar.workflows.runconfig import RunConfig
import pybind_nisar.workflows.helpers as helpers

class CrossmulRunConfig(RunConfig):
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
        error_channel = journal.error('CrossmulRunConfig.cli_arg_load')

        self.cfg = helpers.autovivified_dict()

        # ref hdf5 valid?
        if not os.path.isfile(self.args.ref_hdf5):
            err_str = f"{self.args.ref_hdf5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        self.cfg['InputFileGroup']['InputFilePath'] = self.args.ref_hdf5

        # sec hdf5 valid?
        if os.path.isfile(self.args.sec_hdf5):
            err_str = f"{self.args.sec_hdf5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        self.cfg['InputFileGroup']['SecondaryFilePath'] = self.args.sec_hdf5

        # multilooks valid?
        if self.args.azimuth_looks >= 1:
            err_str = f"azimuth look of {self.args.azimuth_looks} must be >= 1"
            error_channel.log(err_str)
            raise ValueError(err_str)

        self.cfg['processing']['crossmul']['azimuth_looks'] = self.args.azimuth_looks

        if self.args.range_looks >= 1:
            err_str = f"range look of {self.args.range_looks} must be >= 1"
            error_channel.log(err_str)
            raise ValueError(err_str)

        self.cfg['processing']['crossmul']['range_looks'] = self.args.range_looks

        if self.args.oversample < 1:
            err_str = f"range look of {self.args.oversample} must be >= 1"
            error_channel.log(err_str)
            raise ValueError(err_str)

        self.cfg['processing']['crossmul']['oversample'] = self.args.oversample

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

        self.cfg['PrimaryExecutable']['ProductType'] = 'RIFG'

        # check and assign coregistered SLCs
        # if args.sec_raster provided, check if provided path is a HDF5 file or directory
        # if args.sec_raster not provided, use sec_hdf5 as coregistered raster source
        # check required coregistered frequency/polarization rasters exist
        if self.args.sec_raster is not None:
            # sec raster valid file or directory?
            if os.path.isfile(self.args.sec_raster):
                # check if HDF5 has all the frequencies and polarizations
                helpers.check_hdf5_freq_pols(self.args.sec_raster, freq_pols)
            elif os.path.isdir(self.args.sec_raster):
                # check if directory has all the frequencies and polarizations
                helpers.check_mode_directory_tree(self.args.sec_raster,
                                                  'coregistered_secondary', frequencies, freq_pols)
            else:
                err_str = f"{self.args.sec_raster} not a valid file or directory"
                error_channel.log(err_str)
                raise TypeError(err_str)

            # validated secondary raster can now be assigned
            self.cfg['processing']['crossmul']['coregistered_slc_path'] = self.args.sec_raster
        else:
            # check if secondary HDF5 has all the frequencies and polarizations
            helpers.check_hdf5_freq_pols(self.args.sec_hdf5, freq_pols)

            # validated secondary hdf5 assigned to secondary raster
            self.cfg['processing']['crossmul']['coregistered_slc_path'] = self.args.sec_hdf5

        # if flatten path provided check directory tree
        self.cfg['processing']['crossmul']['flatten'] = False
        if self.args.flatten_path is not None:
            helpers.check_mode_directory_tree(self.args.flatten_path, 'geo2rdr', frequencies)
            self.cfg['processing']['crossmul']['flatten'] = self.args.flatten_path

    def yaml_check(self):
        '''
        Check crossmul specifics from YAML
        '''
        error_channel = journal.error('CrossmulRunConfig.yaml_check')

        scratch_path = self.cfg['ProductPathGroup']['ScratchPath']
        # if coregistered_slc_path not provided, use ScratchPath as source for coregistered SLCs
        if 'coregistered_slc_path' not in self.cfg['processing']['crossmul']:
            self.cfg['processing']['crossmul']['coregistered_slc_path'] = scratch_path

        # check whether coregistered_slc_path is a directory or file
        coregistered_slc_path = self.cfg['processing']['crossmul']['coregistered_slc_path']
        if not os.path.isdir(coregistered_slc_path) and not os.path.isfile(coregistered_slc_path):
            err_str = f"{coregistered_slc_path} is invalid; needs to be a file or directory."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # check if required coregistered frequency/polarization rasters exist in directory or HDF5 file
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        frequencies = freq_pols.keys()
        if os.path.isdir(coregistered_slc_path):
            helpers.check_mode_directory_tree(coregistered_slc_path, 'resample_slc',\
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
        flatten = self.cfg['processing']['crossmul']['flatten']
        if flatten:
            # check if flatten is bool and if true as scratch path (str)
            if isinstance(flatten, bool):
                self.cfg['processing']['crossmul']['flatten'] = scratch_path
                flatten = scratch_path
            # check if required frequency range offsets exist
            helpers.check_mode_directory_tree(flatten, 'geo2rdr', frequencies)
        else:
            self.cfg['processing']['crossmul']['flatten'] = None

        # oversample CPU-only capability for now
        if 'oversample' not in self.cfg['processing']['crossmul']:
            self.cfg['processing']['crossmul']['oversample'] = 2
