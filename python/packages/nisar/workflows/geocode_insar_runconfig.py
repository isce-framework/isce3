import os

import journal
import pybind_isce3 as isce
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


class GeocodeInsarRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a commmon `insar` schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

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
        gunw_datasets = ["connectedComponents", "coherenceMagnitude", "unwrappedPhase",
                         "alongTrackOffset", "slantRangeOffset", "layoverShadowMask"]
        for gunw_dataset in gunw_datasets:
            if gunw_dataset not in self.cfg['processing']['geocode']['datasets']:
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

        # To geocode the offsets we need the offset field shape and
        # the start pixel in range and azimuth. Note, margin and gross_offsets
        # are allocated as defaults in share/nisar/defaults/insar.yaml
        geocode_azimuth_offset = self.cfg['processing']['geocode']['datasets'][
                'alongTrackOffset']
        geocode_range_offset = self.cfg['processing']['geocode']['datasets'][
                'slantRangeOffset']
        if geocode_azimuth_offset or geocode_range_offset:
            offset_cfg = self.cfg['processing']['dense_offsets']
            margin = max(offset_cfg['margin'],
                         offset_cfg['gross_offset_range'],
                         offset_cfg['gross_offset_azimuth'])
            if offset_cfg['start_pixel_range'] is None:
               offset_cfg['start_pixel_range'] = margin + offset_cfg[
                          'half_search_range']
            if offset_cfg['start_pixel_azimuth'] is None:
               offset_cfg['start_pixel_azimuth'] = margin + offset_cfg[
                          'half_search_azimuth']
