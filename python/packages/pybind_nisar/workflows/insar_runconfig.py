import journal
import numpy as np
from pybind_nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig


class InsarRunConfig(Geo2rdrRunConfig):
    def __init__(self, args):
        super().__init__(args)
        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()

    def yaml_check(self):
        '''
        Check submodule paths from YAML
        '''
        scratch_path = self.cfg['ProductPathGroup']['ScratchPath']
        error_channel = journal.error('InsarRunConfig.yaml_check')

        # If dense_offsets is disabled and rubbersheet is enabled
        # throw an exception and do not run the workflow
        if not self.cfg['processing']['dense_offsets']['enabled'] and \
                self.cfg['processing']['rubbersheet']['enabled']:
            err_str = "Dense_offsets must be enabled to run rubbersheet"
            error_channel.log(err_str)
            raise RuntimeError(err_str)

        # for each submodule check if user path for input data assigned
        # if not assigned, assume it'll be in scratch
        if 'topo_path' not in self.cfg['processing']['geo2rdr']:
            self.cfg['processing']['geo2rdr']['topo_path'] = scratch_path

        if 'offset_dir' not in self.cfg['processing']['resample']:
            self.cfg['processing']['resample']['offset_dir'] = scratch_path

        if self.cfg['processing']['dense_offsets']['coregistered_slc_path'] is None:
            self.cfg['processing']['dense_offsets'][
                'coregistered_slc_path'] = scratch_path

        # When running insar.py dense_offsets_path and geo2rdr_offsets_path
        # come from previous step through scratch_path
        if self.cfg['processing']['rubbersheet']['dense_offsets_path'] is None:
            self.cfg['processing']['rubbersheet'][
                'dense_offsets_path'] = scratch_path

        if self.cfg['processing']['rubbersheet']['geo2rdr_offsets_path'] is None:
            self.cfg['processing']['rubbersheet'][
                'geo2rdr_offsets_path'] = scratch_path

        if 'coregistered_slc_path' not in self.cfg['processing']['crossmul']:
            self.cfg['processing']['crossmul'][
                'coregistered_slc_path'] = scratch_path

        flatten = self.cfg['processing']['crossmul']['flatten']
        if flatten:
            if isinstance(flatten, bool):
                self.cfg['processing']['crossmul']['flatten'] = scratch_path
        else:
            self.cfg['processing']['crossmul']['flatten'] = None

        # Create default unwrap cfg dict depending on unwrapping algorithm
        algorithm = self.cfg['processing']['phase_unwrap']['algorithm']
        if algorithm not in self.cfg['processing']['phase_unwrap']:
            self.cfg['processing']['phase_unwrap'][algorithm]={}

        if 'interp_method' not in self.cfg['processing']['geocode']:
            self.cfg['processing']['geocode']['interp_method'] = 'BILINEAR'

        # create empty dict if geocode_datasets not in geocode
        if 'datasets' not in self.cfg['processing']['geocode']:
            self.cfg['processing']['geocode']['datasets'] = {}

        # default to True for datasets not found
        gunw_datasets = ["connectedComponents", "coherenceMagnitude",
                         "unwrappedPhase", "alongTrackOffset", "slantRangeOffset",
                         'layoverShadowMask']

        for gunw_dataset in gunw_datasets:
            if gunw_dataset not in self.cfg['processing']['geocode']['datasets']:
                self.cfg['processing']['geocode']['datasets'][
                    gunw_dataset] = True

        # To geocode the offsets we need the offset field shape and
        # the start pixel in range and azimuth. Note, margin and gross_offsets
        # are allocated as defaults in share/nisar/defaults/insar.yaml
        geocode_azimuth_offset = self.cfg['processing'][
            'geocode']['datasets']['alongTrackOffset']
        geocode_range_offset = self.cfg['processing'][
            'geocode']['datasets']['slantRangeOffset']
        if geocode_azimuth_offset or geocode_range_offset:
            offset_cfg = self.cfg['processing']['dense_offsets']
            margin = max(offset_cfg['margin'], offset_cfg['gross_offset_range'],
                         offset_cfg['gross_offset_azimuth'])
            if offset_cfg['start_pixel_range'] is None:
                offset_cfg['start_pixel_range'] = margin + offset_cfg[
                    'half_search_range']
            if offset_cfg['start_pixel_azimuth'] is None:
                offset_cfg['start_pixel_azimuth'] = margin + offset_cfg[
                    'half_search_azimuth']
