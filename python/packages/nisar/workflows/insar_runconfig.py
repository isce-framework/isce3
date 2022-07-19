import warnings

import os
import h5py
import journal
import numpy as np

from nisar.products.readers import SLC
from nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig
import nisar.workflows.helpers as helpers

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

        if self.cfg['processing']['offsets_product'][
            'coregistered_slc_path'] is None:
            self.cfg['processing']['offsets_product'][
                'coregistered_slc_path'] = scratch_path

        # Check a layer of offset exists
        if self.cfg['processing']['offsets_product']['enabled']:
            off_params = self.cfg['processing']['offsets_product']
            layer_keys = [key for key in off_params.keys() if
                          key.startswith('layer')]
            # If no layer of offset is found, throw an exception
            if not layer_keys:
                err_str = "No offset layer specified; at least one layer is required"
                error_channel.log(err_str)
                raise ValueError(err_str)

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

        iono_cfg = self.cfg['processing']['ionosphere_phase_correction']

        # If ionosphere phase correction is enabled, check defaults
        if iono_cfg['enabled']:
            # Extract split-spectrum dictionary
            split_cfg = iono_cfg['split_range_spectrum']
            iono_freq_pol = iono_cfg['list_of_frequencies']
            iono_method = iono_cfg['spectral_diversity']

            # Extract main range bandwidth from reference SLC
            ref_slc_path = self.cfg['input_file_group']['reference_rslc_file_path']
            sec_slc_path = self.cfg['input_file_group']['secondary_rslc_file_path']

            ref_slc = SLC(hdf5file=ref_slc_path)
            sec_slc = SLC(hdf5file=sec_slc_path)

            # extract the polarizations from reference and secondary hdf5
            with h5py.File(ref_slc_path, 'r', libver='latest',
                swmr=True) as ref_h5, \
                h5py.File(sec_slc_path, 'r', libver='latest',
                swmr=True) as sec_h5:

                ref_pol_path = os.path.join(
                    ref_slc.SwathPath, 'frequencyA', 'listOfPolarizations')
                ref_pols_freqA = list(
                    np.array(ref_h5[ref_pol_path][()], dtype=str))

                sec_pol_path = os.path.join(
                    sec_slc.SwathPath, 'frequencyA', 'listOfPolarizations')
                sec_pols_freqA = list(
                    np.array(sec_h5[sec_pol_path][()], dtype=str))

            rg_main_bandwidth = ref_slc.getSwathMetadata(
                'A').processed_range_bandwidth

            # get common polarzations of freqA from reference and secondary
            common_pol_refsec_freqA = set.intersection(
                set(ref_pols_freqA), set(sec_pols_freqA))

            # If no common polarizations found between reference and secondary,
            # then throw errors.
            if not common_pol_refsec_freqA:
                err_str = "No common polarization between frequency A rasters"
                error_channel.log(err_str)
                raise FileNotFoundError(err_str)

            # If polarizations are given, then check if HDF5 has them.
            # If not, then throw error.
            if iono_freq_pol['A']:
                for iono_pol in iono_freq_pol['A']:
                    if (iono_pol not in ref_pols_freqA) or \
                       (iono_pol not in sec_pols_freqA):
                        err_str = f"polarzations {iono_pol} for ionosphere estimation are requested, but not found"
                        error_channel.log(err_str)
                        raise FileNotFoundError(err_str)

            # If common polarization found, but input polarizations are not given,
            # then assign the common polarization for split_main_band
            if (common_pol_refsec_freqA) and (not iono_freq_pol['A']):
                # Co-polarizations are found, split_main_band will be used for co-pols
                common_copol_ref_sec = [pol for pol in common_pol_refsec_freqA
                    if pol in ['VV', 'HH']]
                iono_freq_pol['A'] = common_copol_ref_sec

                # If common co-pols not found, cross-pol will be alternatively used.
                if not common_copol_ref_sec:
                    iono_freq_pol['A'] = common_pol_refsec_freqA

                warning_str = f"{iono_freq_pol} will be used for {iono_method}"
                warning_channel.log(warning_str)
                self.cfg['processing'][
                    'ionosphere_phase_correction'][
                    'list_of_frequencies'] = iono_freq_pol

            # Depending on how the user has selected "spectral_diversity" check if
            # "low_bandwidth" and "high_bandwidth" are assigned. Otherwise, use default
            if iono_method == 'split_main_band':
                # If "low_bandwidth" or 'high_bandwidth" is not allocated, split the main range bandwidth
                # into two 1/3 sub-bands.
                if split_cfg['low_band_bandwidth'] is None:
                    split_cfg['low_band_bandwidth'] = rg_main_bandwidth / 3.0
                    info_str = "low band width for low sub-bands are not given;"\
                        "It is automatically set by 1/3 of range bandwidth of frequencyA"
                    warning_channel.log(info_str)

                if split_cfg['high_band_bandwidth'] is None:
                    split_cfg['high_band_bandwidth'] = rg_main_bandwidth / 3.0
                    info_str = "high band width for high sub-band are not given;"\
                        "It is automatically set by 1/3 of range bandwidth of frequencyA"
                    warning_channel.log(info_str)

                # If polarzations for frequency B are requested
                # for split_main_band method, then throw error
                if iono_freq_pol['B']:
                    err_str = f"Incorrect polarzations {iono_freq_pol['B']} for frequency B are requested. "\
                        f"{iono_method} should not have polarizations in frequency B."
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)

            # methods that use side band
            if iono_method in ['main_side_band', 'main_diff_main_side_band']:
                # extract the polarizations from reference and secondary hdf5
                with h5py.File(ref_slc_path, 'r', libver='latest',
                    swmr=True) as ref_h5, \
                    h5py.File(sec_slc_path, 'r', libver='latest',
                    swmr=True) as sec_h5:

                    ref_pol_path = os.path.join(
                        ref_slc.SwathPath, 'frequencyB', 'listOfPolarizations')
                    ref_pols_freqB = list(
                        np.array(ref_h5[ref_pol_path][()], dtype=str))

                    sec_pol_path = os.path.join(
                        sec_slc.SwathPath, 'frequencyB', 'listOfPolarizations')
                    sec_pols_freqB = list(
                        np.array(sec_h5[sec_pol_path][()], dtype=str))

                # find common polarizations for freq B between ref and sec HDF5
                common_pol_refsec_freqB = set.intersection(
                    set(ref_pols_freqB), set(sec_pols_freqB))

                # when common polarzations are not found, throw error.
                if not common_pol_refsec_freqB:
                    err_str = "No common polarization between frequencyB rasters"
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)

                common_pol_ref_freq_a_b = set.intersection(
                    set(ref_pols_freqA), set(ref_pols_freqB))
                if not common_pol_ref_freq_a_b:
                    err_str = "No common polarization between frequency A and B rasters"
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)

                # If polarizations are given, then check if HDF5 has them.
                # If not, then throw error.
                if iono_freq_pol['B']:
                    for iono_pol in iono_freq_pol['B']:
                        if (iono_pol not in ref_pols_freqB) or \
                            (iono_pol not in sec_pols_freqB):
                            err_str = f"polarzations {iono_pol} for ionosphere"\
                                "estimation are requested, but not found"
                            error_channel.log(err_str)
                            raise FileNotFoundError(err_str)

                # Co-polarizations are found and input pol for freq B is not given
                else:
                    common_pol_refsec_freq_ab = set.intersection(
                    set(common_pol_refsec_freqB), set(common_pol_ref_freq_a_b))

                    # if pol of freq A is given, this pol is used for freq B.
                    if  iono_freq_pol['A']:
                        common_pol_refsec_freq_ab = set.intersection(
                        set(iono_freq_pol['A']), set(common_pol_refsec_freq_ab))

                    common_copol_ref_sec = [pol for pol in common_pol_refsec_freq_ab
                        if pol in ['VV', 'HH']]

                    if common_copol_ref_sec:
                        info_str = f"{common_copol_ref_sec} will be "\
                            f"used for {iono_method}"
                        warning_channel.log(info_str)
                        self.cfg['processing'][
                            'ionosphere_phase_correction'][
                            'list_of_frequencies']['A'] = common_copol_ref_sec
                        self.cfg['processing'][
                            'ionosphere_phase_correction'][
                            'list_of_frequencies']['A'] = common_copol_ref_sec

                    # If common co-pols not found, cross-pol will be alternatively used.
                    else:
                        info_str = f"{common_pol_refsec_freq_ab} will be used for split_main_band"
                        warning_channel.log(info_str)
                        self.cfg['processing'][
                            'ionosphere_phase_correction'][
                            'list_of_frequencies']['A'] = common_pol_refsec_freq_ab
                        self.cfg['processing'][
                            'ionosphere_phase_correction'][
                            'list_of_frequencies']['B'] = common_pol_refsec_freq_ab

        if 'interp_method' not in self.cfg['processing']['geocode']:
            self.cfg['processing']['geocode']['interp_method'] = 'BILINEAR'

        # create empty dict if geocode_datasets not in geocode
        for datasets in ['gunw_datasets', 'goff_datasets']:
            if datasets not in self.cfg['processing']['geocode']:
                self.cfg['processing']['geocode'][datasets] = {}

        # Initialize GUNW and GOFF names
        gunw_datasets = ['connected_components', 'coherence_magnitude',
                         'unwrapped_phase', 'along_track_offset',
                         'slant_range_offset', 'layover_shadow_mask']
        goff_datasets = ['along_track_offset', 'snr',
                         'along_track_offset_variance',
                         'correlation_surface_peak', 'cross_offset_variance',
                         'slant_range_offset', 'slant_range_offset_variance']
        # insert both geocode datasets in dict keyed on datasets name
        geocode_datasets = {'gunw_datasets': gunw_datasets,
                            'goff_datasets': goff_datasets}
        for dataset_group in geocode_datasets:
            for dataset in geocode_datasets[dataset_group]:
                if dataset not in self.cfg['processing']['geocode'][
                    dataset_group]:
                    self.cfg['processing']['geocode'][dataset_group][
                        dataset] = True

        # Check if layover shadow output enabled
        if not self.cfg['processing']['rdr2geo']['write_layover_shadow']:
            # Raise and log warning
            warning_str = 'layover_shadow incorrectly disabled for rdr2geo; it will be enabled'
            warning_channel.log(warning_str)
            warning.warn(warning_str)

            # Set write flag True
            self.cfg['processing']['rdr2geo']['write_layover_shadow'] = True
