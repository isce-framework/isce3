import journal
import os
import h5py
import numpy as np

from nisar.products.readers import SLC
from nisar.workflows.runconfig import RunConfig


def common_ionosphere_cfg_check(cfg):
    """Check ionosphere runconfig for all methods

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters

    Returns
    -------
    ref_pols_freqA: list
        frequency A polarization list for reference SLC
    sec_pols_freqA: list
        frequency A polarization list for secondary SLC

    split_main_band
    rg_main_bandwidth: float
        range bandwidth for reference SLC

    ionosphere methods using sideband
    ref_pols_freqB: list
        frequency B polarization list for reference SLC
    sec_pols_freqB: list
        frequency B polarization list for secondary SLC
    """
    error_channel = journal.error('CommonIonosphere.yaml_check')
    info_channel = journal.info('CommonIonosphere.yaml_check')

    # Extract frequencies and polarizations to process
    freq_pols = cfg['processing']['input_subset'][
            'list_of_frequencies']
    # Create defaults for ionosphere phase correction
    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    # If ionosphere phase correction is enabled, check defaults

    # Extract split-spectrum dictionary
    iono_method = iono_cfg['spectral_diversity']
    ref_slc_path = cfg['input_file_group']['reference_rslc_file_path']
    sec_slc_path = cfg['input_file_group']['secondary_rslc_file_path']
    iono_freq_pol = iono_cfg['list_of_frequencies']

    # hard coded methods using side-band
    iono_method_side = ['main_side_band', 'main_diff_ms_band']

    # if any polarizations and frequencies are not given,
    # default is None for both polarizations.
    if iono_freq_pol == None:
        iono_freq_pol = {'A': None, 'B': None}

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

        # available polarizations in frequency A of reference SLC
        ref_pol_path = f'{ref_slc.SwathPath}/frequencyA/listOfPolarizations'
        ref_pols_freqA = list(
            np.array(ref_h5[ref_pol_path][()], dtype=str))

        # available polarizations in frequency A of secondary SLC
        sec_pol_path = f'{sec_slc.SwathPath}/frequencyA/listOfPolarizations'
        sec_pols_freqA = list(
            np.array(sec_h5[sec_pol_path][()], dtype=str))

        # If ionosphere estimation method using frequency B,
        # then extract list of polarization from frequency B
        # If frequency B does not exist, throw error.
        if iono_method in iono_method_side:
            pol_path = \
                    f"{ref_slc.SwathPath}/frequencyB/listOfPolarizations"

            err_str = f"SLC HDF5 needs frequencyB for {iono_method}"
            if 'frequencyB' not in ref_h5[ref_slc.SwathPath]:
                err_str = "reference" + err_str
                error_channel.log(err_str)
                raise ValueError(err_str)

            # available polarizations in frequency B of reference SLC
            ref_pols_freqB = list(
                np.array(ref_h5[ref_pol_path][()], dtype=str))

            if 'frequencyB' not in sec_h5[ref_slc.SwathPath]:
                err_str = "secondary" + err_str
                error_channel.log(err_str)
                raise ValueError(err_str)

            # available polarizations in frequency B of secondary SLC
            sec_pols_freqB = list(
                np.array(sec_h5[ref_pol_path][()], dtype=str))

            # Check that main and side-band are at the same polarization.
            # If not, throw an error.
            if not set.intersection(set(ref_pols_freqA),
                                    set(ref_pols_freqB)):
                err_str = "No common polarization between "\
                          "frequency A and B rasters"
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

    # get common polarizations of freqA from reference and secondary
    common_pol_refsec_freqA = set.intersection(
        set(ref_pols_freqA), set(sec_pols_freqA))

    # If no common polarizations found between reference and secondary,
    # then throw errors.
    if not common_pol_refsec_freqA:
        err_str = "No common polarization between frequency A rasters"
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)

    if (common_pol_refsec_freqA) and (not iono_freq_pol['A']):
        '''
        If input polarization (frequency A) for ionosphere is not given,
        the polarizations assigned for InSAR workflow are copied.
        However, the polarization of InSAR workflow flow is cross-pol,
        then available co-polarizations are used instead.
        '''
        # common co-poliarzations in reference and secondary SLC
        common_copol_ref_sec = [pol for pol in common_pol_refsec_freqA
            if pol in ['VV', 'HH']]
        common_copol_ref_sec_insar = set.intersection(
            set(common_copol_ref_sec), set(freq_pols['A']))
        if common_copol_ref_sec_insar:
            iono_freq_pol['A'] = list(common_copol_ref_sec_insar)
        else:
            iono_freq_pol['A'] = list(common_copol_ref_sec)

        # If common co-pols not found, cross-pol will be alternatively used.
        if not common_copol_ref_sec:
            iono_freq_pol['A'] = common_pol_refsec_freqA

        # Co-polarizations are found, split_main_band will be used
        # for co-pols
        if iono_method not in iono_method_side:
            info_str = f"{iono_freq_pol['A']} will be used "\
                       f"for {iono_method}."
            info_channel.log(info_str)
            iono_freq_pol['B'] = None

        cfg['processing'][
            'ionosphere_phase_correction'][
            'list_of_frequencies'] = iono_freq_pol

    if iono_method not in iono_method_side:
        return ref_pols_freqA, sec_pols_freqA, rg_main_bandwidth
    else:
        return ref_pols_freqA, sec_pols_freqA, ref_pols_freqB, sec_pols_freqB


def split_main_band_cfg_check(cfg):
    """Check ionosphere runconfig for split_main_band method

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    """

    error_channel = journal.error('SplitMainBandIonosphere.yaml_check')
    info_channel = journal.info('SplitMainBandIonosphere.yaml_check')

    # check common ionosphere options
    ref_pols_freqA, sec_pols_freqA, rg_main_bandwidth = \
        common_ionosphere_cfg_check(cfg)

    # Extract split-spectrum dictionary
    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    split_cfg = iono_cfg['split_range_spectrum']
    iono_method = iono_cfg['spectral_diversity']
    iono_freq_pol = iono_cfg['list_of_frequencies']

    # Extract frequencies and polarizations to process InSAR
    freq_pols = cfg['processing']['input_subset'][
            'list_of_frequencies']

    # If polarizations for frequency B are requested
    # for split_main_band method, then throw error
    if iono_freq_pol['B']:
        err_str = f"Incorrect polarizations {iono_freq_pol['B']} "\
            "for frequency B are requested. "\
            f"{iono_method} should not have polarizations in frequency B."
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)

    # if "split_main_band" is selected,
    # check if "low_bandwidth" and "high_bandwidth" are assigned.
    # If "low_bandwidth" or 'high_bandwidth" is not allocated,
    # split the main range bandwidth into two 1/3 sub-bands.
    if split_cfg['low_band_bandwidth'] is None:
        split_cfg['low_band_bandwidth'] = rg_main_bandwidth / 3.0
        info_str = "low bandwidth for low sub-bands are not given;"\
            "It is automatically set by 1/3 of range bandwidth of frequencyA"
        info_channel.log(info_str)

    if split_cfg['high_band_bandwidth'] is None:
        split_cfg['high_band_bandwidth'] = rg_main_bandwidth / 3.0
        info_str = "high bandwidth for high sub-band are not given;"\
            "It is automatically set by 1/3 of range bandwidth of frequencyA"
        info_channel.log(info_str)


def sideband_cfg_check(cfg):
    """Check ionosphere runconfig for methods using sideband

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    """

    error_channel = journal.error('SidebandIonosphere.yaml_check')
    info_channel = journal.info('SidebandIonosphere.yaml_check')
    # check common ionosphere options
    ref_pols_freqA, sec_pols_freqA, \
    ref_pols_freqB, sec_pols_freqB = \
        common_ionosphere_cfg_check(cfg)

    # get common polarizations of freqB from reference and secondary
    common_pol_refsec_freqB = set.intersection(
        set(ref_pols_freqB), set(sec_pols_freqB))

    # If no common polarizations found between reference and secondary,
    # then throw errors.
    if not common_pol_refsec_freqB:
        err_str = "No common polarization between frequency B rasters"
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)

    # get options for ionosphere estimation method using sideband
    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    iono_method = iono_cfg['spectral_diversity']
    iono_freq_pol = iono_cfg['list_of_frequencies']

    # Extract frequencies and polarizations to process InSAR
    freq_pols = cfg['processing']['input_subset'][
                    'list_of_frequencies']
    # hardcoded methods for ionosphere estimation using side-band
    iono_method_side = ['main_side_band', 'main_diff_ms_band']

    # If polarizations are given for iono estimation using side-band,
    # then check if HDF5 has them. If not, then throw error.
    if iono_freq_pol['B']:
        for iono_pol in iono_freq_pol['B']:
            if (iono_pol not in ref_pols_freqB) or \
               (iono_pol not in sec_pols_freqB):
                err_str = f"polarizations {iono_pol} of frequency B "\
                    f"for ionosphere estimation are given, "\
                    "but not found"
                error_channel.log(err_str)
                raise FileNotFoundError(err_str)

    # If polarizations for frequency A and B are given,
    # check if given polarizations are identical.
    if (iono_freq_pol['A']) and (iono_freq_pol['B']):
        diff_pol = [i for i in iono_freq_pol['B']
                    if i not in iono_freq_pol['A']]
        # when requested polarization are not same
        # (ex. freqA : VV, freqB: HH)
        # ionosphere will be computed from two different polarizations
        # But only one for each frequency is allowed.
        if diff_pol:
            if (len(iono_freq_pol['A']) != 1) and \
               (len(iono_freq_pol['B']) != 1):
                err_str = f"different polarizations for frequency A "\
                    f"and B are requested for {iono_method}, "\
                    "but only one polarization is allowed for "\
                    "polarization combination."
                error_channel.log(err_str)

    if not iono_freq_pol['B']:
        '''
        If input polarizations (frequency B) for ionosphere
           are not given, and
        1) if polarization (frequency B) for InSAR are given,
           then copy them to ionosphere
        2) if polarization (frequency B) for InSAR are not given,
           search the co-polarization
            (i.e. HH or VV)
            If polarizations (frequency A) for InSAR are given,
            2-1) If polarizations (frequency A) for InSAR are co-pol,
                    then copy frequency A to ionosphere
            2-2) If polarizations (frequency A) for InSAR are cross-pol
                    then use the common co-pol between reference and secondary
                    for ionosphere.
        '''
        # case 1
        if 'B' in freq_pols:
            iono_freq_pol['B'] = freq_pols['B']

        if not iono_freq_pol['B']:
            common_copol_refsec_freqB = \
                [pol for pol in common_pol_refsec_freqB
                 if pol in ['VV', 'HH']]
            common_pol_refsec_freqB_insar = set.intersection(
                set(common_copol_refsec_freqB), set(freq_pols['A']))
            # case 2-1
            if common_pol_refsec_freqB_insar:
                iono_freq_pol['B'] = common_pol_refsec_freqB_insar

            # case 2-2
            else:
                common_copol_refsec_freqB = [pol for pol in common_pol_refsec_freqB
                    if pol in ['VV', 'HH']]
                iono_freq_pol['B'] = common_copol_refsec_freqB

    # If numbers of the 'list_of_polarizations' for ionosphere
    # in frequency A and B are different
    # find common polarizations from A and B.
    if len(iono_freq_pol['A']) != len(iono_freq_pol['B']):
        common_pol_freq_ab = set.intersection(
                set(iono_freq_pol['A']),
                set(iono_freq_pol['B']))

        min_num_pol = np.nanmin([len(iono_freq_pol['A']),
                                 len(iono_freq_pol['B'])])

        if common_pol_freq_ab:
            iono_freq_pol['A'] = common_pol_freq_ab
            iono_freq_pol['B'] = common_pol_freq_ab
        else:
            iono_freq_pol['A'] = iono_freq_pol['A'][0:min_num_pol]
            iono_freq_pol['B'] = iono_freq_pol['B'][0:min_num_pol]

    info_str = \
        f"A: {iono_freq_pol['A']}, B {iono_freq_pol['B']} "\
        f"will be used for {iono_method}."
    info_channel.log(info_str)
    cfg['processing'][
        'ionosphere_phase_correction'][
        'list_of_frequencies'] = iono_freq_pol


def ionosphere_cfg_check(cfg):
    """Check ionosphere runconfig

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    """

    error_channel = journal.error('Ionosphere.yaml_check')

    # Extract ionosphere options
    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    iono_method = iono_cfg['spectral_diversity']
    iono_method_side = ['main_side_band', 'main_diff_ms_band']

    if not iono_cfg['enabled']:
        err_str = f'Ionosphere phase correction must be enabled '\
                  f'to execute {iono_method}.'
        error_channel.log(err_str)
        raise ValueError(err_str)

    if iono_method in iono_method_side:
        sideband_cfg_check(cfg)
    else:
        split_main_band_cfg_check(cfg)


class InsarIonosphereRunConfig(RunConfig):
    def __init__(self, args):
        super().__init__(args, 'insar')
        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()

    def yaml_check(self):
        '''
        Check submodule paths from YAML
        '''

        error_channel = journal.error('InsarIonosphereRunConfig.yaml_check')
        info_channel = journal.info('InsarIonosphereRunConfig.yaml_check')

        scratch_path = self.cfg['product_path_group']['scratch_path']

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
            self.cfg['processing']['crossmul']['flatten_path'] = scratch_path

        # Check dictionary for interferogram filtering
        mask_options = self.cfg['processing']['filter_interferogram']['mask']

        # If general mask is provided, check its existence
        mask_general_options = mask_options.get('general', None)
        if mask_general_options is not None:
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
        filter_type = self.cfg['processing']['filter_interferogram'].get('filter_type', 'no_filter')
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

        # check ionosphere runfigs
        ionosphere_cfg_check(self.cfg)