import os

import journal

import pybind_isce3 as isce
from pybind_nisar.workflows.runconfig import RunConfig
import pybind_nisar.workflows.helpers as helpers


class GeocodeInsarRunConfig(RunConfig):
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
        error_channel = journal.error('GeocodeInsarRunConfig.cli_arg_load')

        # rslc h5 valid?
        if os.path.isfile(self.args.rslc_h5):
            self.cfg['InputFileGroup']['InputFilePath'] = self.args.rslc_h5
        else:
            err_str = f"{self.args.rslc_h5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        # runw h5 valid?
        if not os.path.isfile(self.args.runw_h5):
            err_str = f"{self.args.runw_h5} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        # check dem validity. if invalid check_dem raises error.
        helpers.check_dem(self.args.dem)
        self.cfg['DynamicAncillaryFileGroup']['DEMFile'] = self.args.dem

        # multilooks valid?
        az_looks = self.args.azimuth_looks
        if az_looks >= 1:
            if az_looks > 1 and az_looks % 2 == 0:
                err_str = f"azimuth looks = {az_looks} not an odd integer."
                error_channel.log(err_str)
                raise ValueError(err_str)
            self.cfg['processing']['crossmul']['azimuth_looks'] = az_looks
        else:
            err_str = f"azimuth looks = {az_looks} not >= 1"
            error_channel.log(err_str)
            raise ValueError(err_str)

        rg_looks = self.args.range_looks
        if rg_looks >= 1:
            if rg_looks > 1 and rg_looks % 2 == 0:
                err_str = f"range looks = {rg_looks} not an odd integer."
                error_channel.log(err_str)
                raise ValueError(err_str)
            self.cfg['processing']['crossmul']['range_looks'] = rg_looks
        else:
            err_str = f"range look = {rg_looks} not >= 1"
            error_channel.log(err_str)
            raise ValueError(err_str)

        self.cfg['processing']['geocode']['datasets']['connectedComponents'] = not self.args.no_connected_components
        self.cfg['processing']['geocode']['datasets']['phaseSigmaCoherence'] = not self.args.no_phase_sigma
        self.cfg['processing']['geocode']['datasets']['unwrappedPhase'] = not self.args.no_unwrapped_phase

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

        interp_method = self.args.interp_method
        if interp_method not in ['BILINEAR', 'BICUBIC', 'NEAREST','BIQUINTIC']:
            err_str = f"{interp_method} invalid interpolator. Valid options: 'BILINEAR', 'BICUBIC', 'NEAREST', 'BIQUINTIC'"
            error_channel.log(err_str)
            raise ValueError(err_str)

    def yaml_check(self):
        '''
        Check GUNW specifics from YAML
        '''
        error_channel = journal.error('GeocodeInsarRunConfig.yaml_check')

        if 'runw_path' not in self.cfg['processing']['geocode']:
            err_str = "'runw_path' file path under `geocode' required for standalone execution with YAML"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if runw path is a directory or a file
        runw_path = self.cfg['processing']['geocode']['runw_path']
        if not os.path.isfile(runw_path):
            err_str = f"{runw_path} is invalid; needs to be a file"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if required polarizations/frequency are in provided HDF5 file
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        helpers.check_hdf5_freq_pols(runw_path, freq_pols)

        if 'interp_method' not in self.cfg['processing']['geocode']:
            self.cfg['processing']['geocode']['interp_method'] = 'BILINEAR'

        # create empty dict if geocode_datasets not in geocode
        if 'datasets' not in self.cfg['processing']['geocode']:
            self.cfg['processing']['geocode']['datasets'] = {}

        # default to True for datasets not found
        gunw_datasets = ["connectedComponents", "phaseSigmaCoherence", "unwrappedPhase"]
        for gunw_dataset in gunw_datasets:
            if gunw_dataset not in self.cfg['processing']['geocode']:
                self.cfg['processing']['geocode']['datasets'][gunw_dataset] = True

        if self.cfg['processing']['dem_margin'] is None:
            '''
            Default margin as the length of 50 pixels
            (max of X and Y pixel spacing).
            '''
            dem_file = self.cfg['DynamicAncillaryFileGroup']['DEMFile']
            dem_raster = isce.io.Raster(dem_file)
            dem_margin = 50 * max([dem_raster.dx, dem_raster.dy])
            self.cfg['processing']['dem_margin'] = dem_margin

        # multilooks valid?
        az_looks = self.cfg['processing']['crossmul']['azimuth_looks']
        if az_looks > 1 and az_looks % 2 == 0:
            err_str = f"azimuth looks = {az_looks} not an odd integer."
            error_channel.log(err_str)
            raise ValueError(err_str)

        rg_looks = self.cfg['processing']['crossmul']['range_looks']
        if rg_looks > 1 and rg_looks % 2 == 0:
            err_str = f"range looks = {rg_looks} not an odd integer."
            error_channel.log(err_str)
            raise ValueError(err_str)
