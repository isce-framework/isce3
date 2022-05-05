import os
import numpy as np
import h5py
import journal

from nisar.products.readers import SLC
from nisar.workflows.runconfig import RunConfig
import nisar.workflows.helpers as helpers


class SplitSpectrumRunConfig(RunConfig):
    def __init__(self, args):
        # All InSAR submodules share a common InSAR schema
        super().__init__(args, 'insar')

        # When using split spectrum as stand-alone module
        # check that the runconfig has been properly checked
        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check split-spectrum specifics from YAML file
        '''
        
        info_channel = journal.info('SplitSpectrumRunConfig.yaml_check')
        error_channel = journal.error('SplitSpectrumRunConfig.yaml_check')
        iono_cfg = self.cfg['processing']['ionosphere_phase_correction']
        split_cfg = iono_cfg['split_range_spectrum']
        iono_freq_pol = iono_cfg['list_of_frequencies']
        ref_slc_path = self.cfg['input_file_group']['reference_rslc_file_path']
        sec_slc_path = self.cfg['input_file_group']['secondary_rslc_file_path']
        iono_method = iono_cfg['spectral_diversity']

        # Extract main range bandwidth from reference RSLC
        ref_slc = SLC(hdf5file=ref_slc_path)
        sec_slc = SLC(hdf5file=sec_slc_path)

        rg_main_bandwidth = ref_slc.getSwathMetadata(
            'A').processed_range_bandwidth

        # Check if ionosphere_phase_correction is enabled. Otherwise,
        # throw an error and do not execute split-spectrum
        if not iono_cfg['enabled']:
            err_str = 'Ionosphere phase correction must be enabled to execute split-spectrum'
            error_channel.log(err_str)
            raise ValueError(err_str)

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
        
        # Depending on how the user has selected "spectral_diversity" check if
        # "low_bandwidth" and "high_bandwidth" are assigned. Otherwise, use default
        if iono_method == 'split_main_band':
            # If "low_bandwidth" or 'high_bandwidth" is not allocated, 
            # split the main range bandwidth into two 1/3 sub-bands.
            if split_cfg['low_band_bandwidth'] is None or split_cfg[
                'high_band_bandwidth'] is None:
                split_cfg['low_band_bandwidth'] = rg_main_bandwidth / 3.0
                split_cfg['high_band_bandwidth'] = rg_main_bandwidth / 3.0
                info_str = "band_widths for sub-bands are not given; They will be 1/3 of range bandwidth"
                info_channel.log(info_str)

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
                        err_str = f"polarizations {iono_pol} for ionosphere estimation are given, but not found"
                        error_channel.log(err_str)
                        raise FileNotFoundError(err_str)

            # If polarizations for frequency B are requested 
            # for split_main_band method, then throw error
            if iono_freq_pol['B']:
                err_str = f"Incorrect polarzations {iono_freq_pol['B']} for frequency B are requested. "\
                    f"{iono_method} should not have polarizations in frequency B."
                error_channel.log(err_str)
                raise FileNotFoundError(err_str)
                
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
            