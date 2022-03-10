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
        ref_slc_path = self.cfg['input_file_group']['input_file_path']
        sec_slc_path = self.cfg['input_file_group']['secondary_file_path']
        
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
        if split_cfg['spectral_diversity'] == 'split_main_band':
            # If "low_bandwidth" or 'high_bandwidth" is not allocated, 
            # split the main range bandwidth into two 1/3 sub-bands.
            if split_cfg['low_band_bandwidth'] is None or split_cfg[
                'high_band_bandwidth'] is None:
                split_cfg['low_band_bandwidth'] = rg_main_bandwidth / 3.0
                split_cfg['high_band_bandwidth'] = rg_main_bandwidth / 3.0
                info_str = "band_widths for sub-bands are not given; They will be 1/3 of range bandwidth"
                info_channel.log(info_str)

            # Obtains common polarzations of freqA between reference and secondary
            common_pol_refsec_freqA = set.intersection(set(ref_pols_freqA), set(sec_pols_freqA))

            # If no common polarizations found, then raise errors. 
            if not common_pol_refsec_freqA:
                err_str = "No common polarization between frequency A rasters"
                error_channel.log(err_str)
                raise FileNotFoundError(err_str)

            # If common polarization found, but input polarizations are not given, 
            # then assign the common polarization for split_main_band
            elif (common_pol_refsec_freqA) and (not iono_freq_pol['A']):
                # Co-polarizations are found, split_main_band will be used for co-pols
                common_copol_ref_sec = [pol for pol in common_pol_refsec_freqA 
                    if pol in ['VV', 'HH']]
                info_str = f"{common_copol_ref_sec} will be used for split_main_band"
                info_channel.log(info_str)
                iono_freq_pol = {'A': common_copol_ref_sec}
                self.cfg['processing'][
                    'ionosphere_phase_correction'][
                    'list_of_frequencies'] = {'A': common_copol_ref_sec}
                # If common co-pols not found, cross-pol will be alternatively used.
                if not common_copol_ref_sec:
                    info_str = f"{common_pol_refsec_freqA} will be used for split_main_band"
                    info_channel.log(info_str)
                    iono_freq_pol = {'A': common_pol_refsec_freqA}
                    self.cfg['processing'][
                        'ionosphere_phase_correction'][
                        'list_of_frequencies'] = {'A': common_copol_ref_sec}
            
            # search for the common polarizations in runconfig and HDF5
            ref_intersect_pol = [pol for pol in iono_freq_pol['A'] 
                if pol in ref_pols_freqA]
            sec_intersect_pol = [pol for pol in iono_freq_pol['A'] 
                if pol in ref_pols_freqA]
            # if given polarzations are not found in HDF5, then raise error. 
            if (not ref_intersect_pol) or (not sec_intersect_pol):
                err_str = f"polarzations {iono_freq_pol['A']} are given, but not found"
                error_channel.log(err_str)
                raise FileNotFoundError(err_str)

        if split_cfg['spectral_diversity'] == 'main_side_band':
            # Extract side-band range bandwidth
            rg_side_bandwidth = ref_slc.getSwathMetadata(
                'B').processed_range_bandwidth

            # If "low_bandwidth" and "high_bandwidth" are not assigned, assign main range bandwidth
            # and side-band bandwidths, respectively. If assigned, check that
            # "low_bandwidth" and "high_bandwidth" correspond to main and side range bandwidths
            if split_cfg['low_band_bandwidth'] is None or split_cfg[
                'low_band_bandwidth'] != rg_main_bandwidth:
                split_cfg['low_band_bandwidth'] = rg_main_bandwidth
            if split_cfg['high_band_bandwidth'] is None or split_cfg[
                'high_band_bandwidth'] != rg_side_bandwidth:
                split_cfg['high_band_bandwidth'] = rg_side_bandwidth

            # Check that main and side-band are at the same polarization. If not, throw an error.
            with h5py.File(self.cfg['input_file_group']['input_file_path'],
                                   'r', libver='latest', swmr=True) as src_h5:                                   
                    pol_path = os.path.join(ref_slc.SwathPath, 'frequencyA',
                                            'listOfPolarizations')
                    pols_freqA = src_h5[pol_path][()]
                    pol_path = os.path.join(ref_slc.SwathPath, 'frequencyB',
                                            'listOfPolarizations')
                    pols_freqB = src_h5[pol_path][()]

            if len(set.intersection(set(pols_freqA), set(pols_freqB))) == 0:
                err_str = "No common polarization between frequency A and B rasters"
                error_channel.log(err_str)
                raise FileNotFoundError(err_str)
