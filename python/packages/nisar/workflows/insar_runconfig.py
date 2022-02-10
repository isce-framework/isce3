from nisar.products.readers import SLC
from nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig
import nisar.workflows.helpers as helpers
import journal
import os
import h5py

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
        info_channel = journal.info("InsarRunConfig.yaml_check")

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

        # Create defaults for ionosphere phase correction
        iono_cfg = self.cfg['processing']['ionosphere_phase_correction']
        # If ionosphere phase correction is enabled, check defaults
        if iono_cfg['enabled']:
            # Extract split-spectrum dictionary
            split_cfg = iono_cfg['split_range_spectrum']
            # Extract main range bandwidth from reference SLC
            ref_slc = SLC(hdf5file=self.cfg['InputFileGroup']['InputFilePath'])
            rg_main_bandwidth = ref_slc.getSwathMetadata(
                'A').processed_range_bandwidth

            # Depending on how the user has selected "spectral_diversity" check if
            # "low_bandwidth" and "high_bandwidth" are assigned. Otherwise, use default
            if split_cfg['spectral_diversity'] == 'split_main_band':
                # If "low_bandwidth" or 'high_bandwidth" is not allocated, split the main range bandwidth
                # into two 1/3 sub-bands.
                if split_cfg['low_band_bandwidth'] is None:
                    split_cfg['low_band_bandwidth'] = rg_main_bandwidth / 3.0
                    info_str = "low band width for low sub-bands are not given;"\
                        "It is automatically set by 1/3 of range bandwidth of freqeuncyA""
                    info_channel.log(info_str)

                if split_cfg['high_band_bandwidth'] is None:
                    split_cfg['high_band_bandwidth'] = rg_main_bandwidth / 3.0
                    info_str = "high band width for high sub-band are not given;"\
                        "It is automatically set by 1/3 of range bandwidth of freqeuncyA""
                    info_channel.log(info_str)

            if split_cfg['spectral_diversity'] == 'main_side_band':
                print(freq_pols.keys())
                if 'B' not in freq_pols.keys():
                    err_str = "polarizations for frequency B are not given;"\
                        "frequency B is required for main-side-band method."
                    error_channel.log(err_str)
                    raise ValueError(err_str)

                # Extract side-band range bandwidth
                rg_side_bandwidth = ref_slc.getSwathMetadata(
                    'B').processed_range_bandwidth

                # Check that main and side-band are at the same polarization. If not, throw an error.
                src_h5 = h5py.File(self.cfg['InputFileGroup']['InputFilePath'],
                                   'r',
                                   libver='latest', swmr=True)
                pol_path = os.path.join(ref_slc.SwathPath, 'frequencyA',
                                        'listOfPolarizations')
                pols_freqA = src_h5[pol_path][()]
                pol_path = os.path.join(ref_slc.SwathPath, 'frequencyB',
                                        'listOfPolarizations')
                pols_freqB = src_h5[pol_path][()]
                src_h5.close()

                if len(set.intersection(set(pols_freqA), set(pols_freqB))) == 0:
                    err_str = "No common polarization between frequency A and B rasters"
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)

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
