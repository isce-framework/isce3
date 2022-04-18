import journal
import os
import h5py
import numpy as np 

from nisar.products.readers import SLC
from nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig

class InsarIonosphereRunConfig(Geo2rdrRunConfig):
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
        error_channel = journal.error('InsarIonosphereRunConfig.yaml_check')
        info_channel = journal.info('InsarIonosphereRunConfig.yaml_check')

        # Extract frequencies and polarizations to process
        freq_pols = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

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

        if not iono_cfg['enabled']:
            err_str = 'Ionosphere phase correction must be enabled to execute split-spectrum'
            error_channel.log(err_str)
            raise ValueError(err_str)

        if iono_cfg['enabled']:
            # Extract split-spectrum dictionary
            split_cfg = iono_cfg['split_range_spectrum']
            iono_method = iono_cfg['spectral_diversity']
            ref_slc_path = self.cfg['input_file_group']['input_file_path']
            sec_slc_path = self.cfg['input_file_group']['secondary_file_path']
            iono_freq_pol = iono_cfg['list_of_frequencies']

            iono_method_side = ['main_side_band', 'main_diff_ms_band', 'hybrid']
            # Extract main range bandwidth from reference SLC
            ref_slc = SLC(hdf5file=ref_slc_path)
            sec_slc = SLC(hdf5file=sec_slc_path)

            rg_main_bandwidth = ref_slc.getSwathMetadata(
                'A').processed_range_bandwidth
            
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
                # If ionosphere estimation method using frequency B,
                # then extract list of polarization from frequency B
                # If frequency B does not exist, throw error.  
                if iono_method in iono_method_side:
                    try:
                        ref_pol_path = os.path.join(
                            ref_slc.SwathPath, 'frequencyB', 'listOfPolarizations')
                        ref_pols_freqB = list(
                            np.array(ref_h5[ref_pol_path][()], dtype=str))

                        sec_pol_path = os.path.join(
                            sec_slc.SwathPath, 'frequencyB', 'listOfPolarizations')
                        sec_pols_freqB = list(
                            np.array(sec_h5[sec_pol_path][()], dtype=str))
                    except:
                        err_str = f"frequencyB is not found in SLC HDF5. "\
                            f"{iono_method} needs frequency B."
                        error_channel.log(err_str)
                        raise FileNotFoundError(err_str)
                    
                    # Check that main and side-band are at the same polarization. 
                    # If not, throw an error.
                    if len(set.intersection(set(ref_pols_freqA), set(ref_pols_freqB))) == 0:
                        err_str = "No common polarization between frequency A and B rasters"
                        error_channel.log(err_str)
                        raise FileNotFoundError(err_str)

            if iono_method == 'split_main_band':
                # If polarizations for frequency B are requested 
                # for split_main_band method, then throw error
                if iono_freq_pol['B']:
                    err_str = f"Incorrect polarzations {iono_freq_pol['B']} for frequency B are requested. "\
                        f"{iono_method} should not have polarizations in frequency B."
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)

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
                        err_str = f"polarizations {iono_pol} of frequency A "\
                            f"for ionosphere estimation are given, but not found"
                        error_channel.log(err_str)
                        raise FileNotFoundError(err_str)

            # If polarizations are given for iono estimation using side-band, 
            # then check if HDF5 has them. If not, then throw error. 
            if iono_method in iono_method_side:
                if iono_freq_pol['B']:
                    for iono_pol in iono_freq_pol['B']:
                        if (iono_pol not in ref_pols_freqB) or \
                            (iono_pol not in sec_pols_freqB):
                            err_str = f"polarizations {iono_pol} of frequency B "\
                                f"for ionosphere estimation are given, but not found"
                            error_channel.log(err_str)
                            raise FileNotFoundError(err_str)
                
                # list_of_frequencies should have one pol at least to run rdr2geo/geo2rdr
                if 'B' not in freq_pols:
                    err_str = f"polarizations { iono_freq_pol['B']} of frequency B "\
                        f"for ionosphere estimation are requested, "\
                        f"list_of_frequencies in InSAR workflow should have frequency B key."
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)
                
                # If polarizations for frequency A and B are given, 
                # check if given polarzations are identical. 
                if (iono_freq_pol['A']) and (iono_freq_pol['B']):
                    diff_pol = [i for i in iono_freq_pol['B'] if i not in iono_freq_pol['A']]
                    # when requested polarization are not same(ex. freqA : VV, freqB: HH)
                    # ionosphere will be computed from two different polarzations
                    # But only one for each frequency is allowed. 
                    if diff_pol:
                        if (len(iono_freq_pol['A']) != 1) and (len(iono_freq_pol['B']) != 1):
                            err_str = f"different polarizations for frequency A and B are requested "\
                            f"for {iono_method}, but only one polarization is allowed for polarization combination"
                            error_channel.log(err_str)
            
            # If common polarization found, but input polarizations are not given, 
            # then assign the common polarization for split_main_band
            if (common_pol_refsec_freqA) and (not iono_freq_pol['A']):
                # Co-polarizations are found, split_main_band will be used for co-pols
                common_copol_ref_sec = [pol for pol in common_pol_refsec_freqA 
                    if pol in ['VV', 'HH']]
                iono_freq_pol['A'] = common_copol_ref_sec
                iono_freq_pol['B'] = None
                
                # If common co-pols not found, cross-pol will be alternatively used.
                if not common_copol_ref_sec:
                    iono_freq_pol['A'] = common_pol_refsec_freqA
                    iono_freq_pol['B'] = None

                info_str = f"{iono_freq_pol['A']} will be used for split_main_band"
                info_channel.log(info_str)
                self.cfg['processing'][
                    'ionosphere_phase_correction'][
                    'list_of_frequencies'] = iono_freq_pol

            if iono_method in ['main_side_band', 'main_diff_ms_band']:
                # get common polarzations of freqA from reference and secondary
                common_pol_refsec_freqB = set.intersection(
                    set(ref_pols_freqB), set(sec_pols_freqB))

                # If common polarization found, but input polarizations are not given, 
                # then assign the common polarization for split_main_band
                if (common_pol_refsec_freqB) and (not iono_freq_pol['B']):
                    # Co-polarizations are found, split_main_band will be used for co-pols
                    common_copol_ref_sec = [pol for pol in common_pol_refsec_freqB 
                        if pol in ['VV', 'HH']]
                    iono_freq_pol['B'] = common_copol_ref_sec
                    
                    # If common co-pols not found, cross-pol will be alternatively used.
                    if not common_copol_ref_sec:
                        iono_freq_pol['B'] = common_pol_refsec_freqB

                    info_str = f"{iono_freq_pol['B']} will be used for {iono_method}"
                    info_channel.log(info_str)
                    self.cfg['processing'][
                        'ionosphere_phase_correction'][
                        'list_of_frequencies'] = iono_freq_pol