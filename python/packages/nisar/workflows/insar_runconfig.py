from nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig
import journal
import os
import warnings

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

        scratch_path = self.cfg['product_path_group']['scratch_path']
        error_channel = journal.error('InsarRunConfig.yaml_check')
        warning_channel = journal.warning('InsarRunConfig.yaml_check')

        # Extract frequencies and polarizations to process
        freq_pols = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

        # If dense_offsets is disabled and rubbersheet is enabled
        # throw an exception and do not run the workflow
        if not self.cfg['processing']['dense_offsets']['enabled'] and \
                self.cfg['processing']['rubbersheet']['enabled']:
            err_str = "Dense_offsets must be enabled to run rubbersheet"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # If rubbersheet is disabled but fine_resampling is enabled,
        # throw an exception and stop insar.py execution
        if not self.cfg['processing']['rubbersheet']['enabled'] and \
                self.cfg['processing']['fine_resample']['enabled']:
            err_str = "Rubbersheet must be enabled to run fine SLC resampling"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if rdr2geo flags enabled for topo X, Y, and Z rasters
        for xyz in 'xyz':
            # Get write flag for x, y, or z
            write_flag = f'write_{xyz}'

            # Check if it's not enabled (required for InSAR processing)
            if not self.cfg['processing']['rdr2geo'][write_flag]:
                # Raise and log warning
                warning_str = f'{write_flag} incorrectly disabled for rdr2geo; it will be enabled'
                warning_channel.log(warning_str)
                warning.warn(warning_str)

                # Set write flag True
                self.cfg['processing']['rdr2geo'][write_flag] = True

        # for each submodule check if user path for input data assigned
        # if not assigned, assume it'll be in scratch
        if 'topo_path' not in self.cfg['processing']['geo2rdr']:
            self.cfg['processing']['geo2rdr']['topo_path'] = scratch_path

        if self.cfg['processing']['coarse_resample']['offsets_dir'] is None:
            self.cfg['processing']['coarse_resample']['offsets_dir'] = scratch_path

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

        if self.cfg['processing']['fine_resample']['offsets_dir'] is None:
            self.cfg['processing']['fine_resample']['offsets_dir'] = scratch_path

        if 'coregistered_slc_path' not in self.cfg['processing']['crossmul']:
            self.cfg['processing']['crossmul'][
                'coregistered_slc_path'] = scratch_path

        flatten = self.cfg['processing']['crossmul']['flatten']
        if flatten:
            if isinstance(flatten, bool):
                self.cfg['processing']['crossmul']['flatten'] = scratch_path
        else:
            self.cfg['processing']['crossmul']['flatten'] = None

        # Check dictionary for interferogram filtering
        mask_options = self.cfg['processing']['filter_interferogram']['mask']

        # If general mask is provided, check its existence
        if 'general' in mask_options and mask_options['general'] is not None:
            if not os.path.isfile(mask_options['general']):
                err_str = f"The mask file {mask_options['general']} is not a file"
                error_channel.log(err_str)
                raise ValueError(err_str)
        else:
            # Otherwise check that mask for individual freq/pols are correctly assigned
            for freq, pol_list in freq_pols.items():
                if freq in mask_options:
                   for pol in pol_list:
                       if pol in mask_options[freq]:
                          mask_file = mask_options[freq][pol]
                          if mask_file is not None and not os.path.isfile(mask_file):
                             err_str = f"{mask_file} is invalid; needs to be a file"
                             error_channel.log(err_str)
                             raise ValueError(err_str)

        # Check filter_type and if not allocated, create a default cfg dictionary
        # filter_type will be present at runtime because is allocated in share/nisar/defaults
        filter_type = self.cfg['processing']['filter_interferogram']['filter_type']
        if filter_type != 'no_filter' and filter_type not in \
                self.cfg['processing']['filter_interferogram']:
            self.cfg['processing']['filter_interferogram'][filter_type] = {}

        # Based on filter_type, check if related dictionary and/or parameters
        # are assigned. Note, if filter_type='boxcar', the filter dictionary
        # is filled by share/nisar/defaults
        if filter_type == 'gaussian':
            if 'gaussian' not in self.cfg['processing']['filter_interferogram']:
                self.cfg['processing']['filter_interferogram'][
                    'gaussian'] = {}
            gaussian_options = self.cfg['processing']['filter_interferogram'][
                'gaussian']
            if 'sigma_range' not in gaussian_options:
                gaussian_options['sigma_range'] = 1
            if 'sigma_azimuth' not in gaussian_options:
                gaussian_options['sigma_azimuth'] = 1
            if 'filter_size_range' not in gaussian_options:
                gaussian_options['filter_size_range'] = 9
            if 'filter_size_azimuth' not in gaussian_options:
                gaussian_options['filter_size_azimuth'] = 9

        # set to empty dict and default unwrap values will be used
        # if phase_unwrap fields not in yaml
        if 'phase_unwrap' not in self.cfg['processing']:
            self.cfg['processing']['phase_unwrap'] = {}

        # if phase_unwrap fields not in yaml
        if self.cfg['processing']['phase_unwrap'] is None:
            self.cfg['processing']['phase_unwrap'] = {}

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

        # Check if layover shadow output enabled
        if not self.cfg['processing']['rdr2geo']['write_layover_shadow']:
            # Raise and log warning
            warning_str = 'layover_shadow incorrectly disabled for rdr2geo; it will be enabled'
            warning_channel.log(warning_str)
            warning.warn(warning_str)

            # Set write flag True
            self.cfg['processing']['rdr2geo']['write_layover_shadow'] = True

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
