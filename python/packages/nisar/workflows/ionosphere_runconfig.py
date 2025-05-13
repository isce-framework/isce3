import os

import h5py
import journal
import numpy as np

from nisar.products.readers import SLC
from nisar.workflows.runconfig import RunConfig


def _get_rslc_h5_freq_pols(slc_h5_path, freq):
    '''
    Attempt to retrieve for frequencies A and B polarizations rom RSLC HDF5 as
    a list. If a frequency not found, it is represented as an empty list.

    Parameters
    ----------
    slc_h5_path: str
        Path to RSLC HDF5 file
    freq: ['A', 'B']
        Frequency whose polarizations are to be extracted from HDF5

    Returns
    -------
    list
        List of polarizations for given frequency. List is empty if no
        polarizations found.
    '''
    with h5py.File(slc_h5_path, 'r', libver='latest', swmr=True) as h:
        # Load SLC object from HDF5 for swath path
        slc = SLC(hdf5file=slc_h5_path)
        swath_path = slc.SwathPath

        # Check if frequency is in HDF5 swath path
        freq = f'frequency{freq}'
        if freq not in h[swath_path]:
            return []

        # Return polarizations for given frequency as a list
        pol_path = f'{swath_path}/{freq}/listOfPolarizations'
        return list(h[pol_path][()].astype('str'))


def _cfg_freq_pol_check(cfg, freq):
    """Check ionosphere polarizations for given frequency

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    freq: ['A', 'B']
        Frequency whose polarizations are to be checked
    """
    freq = freq.upper()
    error_channel = \
        journal.error('ionosphere_runconfig._cfg_freq_pol_check')

    # available polarizations in frequency A of reference SLC
    ref_slc_path = cfg['input_file_group']['reference_rslc_file']
    h5_ref_pols = _get_rslc_h5_freq_pols(ref_slc_path, freq)

    # available polarizations in frequency A of secondary SLC
    sec_slc_path = cfg['input_file_group']['secondary_rslc_file']
    h5_sec_pols = _get_rslc_h5_freq_pols(sec_slc_path, freq)

    # check if both ref and sec that freq + pols exist
    fails = []
    for refsec, pols_freq in zip(['reference', 'secondary'],
                                 [h5_ref_pols, h5_sec_pols]):
        if not pols_freq:
            fails.append(refsec)

    if fails:
        err_str = f' SLC HDF5(s) missing frequency {freq}'
        err_str = ', '.join(fails) + err_str
        error_channel.log(err_str)
        raise LookupError(err_str)

    # get common polarizations of freq from reference and secondary
    h5_common_pols = set.intersection(set(h5_ref_pols), set(h5_sec_pols))

    # If no common frequency A polarizations found between reference and
    # secondary HDF5s, then raise exception.
    if not h5_common_pols:
        err_str = "No common polarization between reference and secondary"\
            f" frequency {freq} rasters"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # If polarizations are given in iono config, then check if HDF5 has them.
    iono_cfg = cfg['processing']['ionosphere_phase_correction']

    if iono_cfg['list_of_frequencies'] is None:
        iono_cfg_freq_pol = None
    else:
        iono_cfg_freq_pol = iono_cfg['list_of_frequencies'][freq]

    if iono_cfg_freq_pol:
        set_iono_cfg_freq_pol = set(iono_cfg_freq_pol)

        # container to store which errors occurred
        fails = []

        # iterate reference and secondary polarizations from h5 for given
        # frequency
        for refsec, pols_freq in zip(['reference', 'secondary'],
                                     [h5_ref_pols, h5_sec_pols]):
            # check for intersection in polarizations
            if not set_iono_cfg_freq_pol.intersection(set(pols_freq)):
                fails.append(refsec)

        # raise error if any non-intersection found
        if fails:
            err_str = ' SLC HDF5(s) polarizations do not intersect with'\
                ' polarizations in ionosphere configuration'
            err_str = f'Frequency {freq}' + ', '.join(fails) + err_str
            error_channel.log(err_str)
            raise LookupError(err_str)
    # If iono config polarizations not given
    else:
        '''
        If input polarization (frequency A) for ionosphere is not given,
        the polarizations assigned for InSAR workflow are copied.
        However, the polarization of InSAR workflow flow is cross-pol,
        then available co-polarizations are used instead.
        '''
        # common co-poliarzations in reference and secondary SLC
        set_copol = set(('VV', 'HH'))

        # find copols in h5 common pols
        h5_common_copol_ref_sec = h5_common_pols.intersection(set_copol)

        # find common copols in h5 and frequency A iono config
        cfg_freq_pols = cfg['processing']['input_subset'][
            'list_of_frequencies']
        h5_iono_cfg_common_copol_ref_sec = set.intersection(
            set(h5_common_copol_ref_sec), set(cfg_freq_pols['A']))

        # use common copols in h5 and frequency A iono config if any found
        if h5_iono_cfg_common_copol_ref_sec:
            iono_cfg_freq_pol = list(h5_iono_cfg_common_copol_ref_sec)
        # use h5 common copols
        elif h5_common_copol_ref_sec:
            iono_cfg_freq_pol = list(h5_common_copol_ref_sec)
        # use h5 common crosspols
        else:
            iono_cfg_freq_pol = list(h5_common_pols)
    cfg['processing']['ionosphere_phase_correction'][
            'list_of_frequencies'][freq] = iono_cfg_freq_pol


def split_main_band_cfg_check(cfg):
    """Check ionosphere runconfig for split_main_band method

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    """

    error_channel = journal.error(
        'ionosphere_runconfig.split_main_band_cfg_check')
    info_channel = journal.info(
        'ionosphere_runconfig.split_main_band_cfg_check')

    # Extract split-spectrum dictionary
    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    split_cfg = iono_cfg['split_range_spectrum']
    iono_method = iono_cfg['spectral_diversity']
    iono_freq_pol = iono_cfg['list_of_frequencies']
    ref_slc_path = cfg['input_file_group']['reference_rslc_file']

    # Extract main range bandwidth from reference SLC
    ref_slc = SLC(hdf5file=ref_slc_path)

    rg_main_bandwidth = ref_slc.getSwathMetadata(
        'A').processed_range_bandwidth

    # If polarizations for frequency B are requested
    # for split_main_band method, then throw error
    if iono_freq_pol['B']:
        err_str = f"Incorrect polarizations {iono_freq_pol['B']} "\
            "for frequency B are requested. "\
            f"{iono_method} should not have polarizations in frequency B."
        error_channel.log(err_str)
        raise ValueError(err_str)

    # if "split_main_band" is selected,
    # check if "low_bandwidth" and "high_bandwidth" are assigned.
    # If "low_bandwidth" or 'high_bandwidth" is not allocated,
    # split the main range bandwidth into two 1/3 sub-bands.
    if split_cfg['low_band_bandwidth'] is None:
        split_cfg['low_band_bandwidth'] = rg_main_bandwidth / 3.0
        info_str = "low bandwidth for low sub-band is not given;"\
            "It is automatically set by 1/3 of range bandwidth of frequencyA"
        info_channel.log(info_str)

    if split_cfg['high_band_bandwidth'] is None:
        split_cfg['high_band_bandwidth'] = rg_main_bandwidth / 3.0
        info_str = "high bandwidth for high sub-band is not given;"\
            "It is automatically set by 1/3 of range bandwidth of frequencyA"
        info_channel.log(info_str)


def sideband_cfg_check(cfg):
    """Check ionosphere runconfig for methods using sideband

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    """

    error_channel = journal.error('ionosphere_runconfig.sideband_cfg_check')
    info_channel = journal.info('ionosphere_runconfig.sideband_cfg_check')

    # get options for ionosphere estimation method using sideband
    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    iono_method = iono_cfg['spectral_diversity']
    iono_cfg_freq_pol = iono_cfg['list_of_frequencies']

    # Extract frequencies and polarizations to process InSAR
    cfg_freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # Assign config pols to iono config pols if they exit
    if not iono_cfg_freq_pol['B'] and 'B' in cfg_freq_pols:
        iono_cfg_freq_pol['B'] = cfg_freq_pols['B']

    _cfg_freq_pol_check(cfg, 'B')

    '''
    If the numbers of the 'list_of_polarizations' for ionosphere in frequencies
    A and B are different, find common polarizations from A and B.
    If the common polarizations are not found, identify the minimum number of
    polarizations between frequencies A and B and use only the same number of
    polarizations in order specified in iono config.
    '''
    if len(iono_cfg_freq_pol['A']) != len(iono_cfg_freq_pol['B']):
        common_pol_freq_ab = set.intersection(set(iono_cfg_freq_pol['A']),
                                              set(iono_cfg_freq_pol['B']))

        if common_pol_freq_ab:
            iono_cfg_freq_pol['A'] = list(common_pol_freq_ab)
            iono_cfg_freq_pol['B'] = list(common_pol_freq_ab)
        else:
            min_num_pol = np.nanmin([len(iono_cfg_freq_pol['A']),
                                     len(iono_cfg_freq_pol['B'])])
            iono_cfg_freq_pol['A'] = iono_cfg_freq_pol['A'][:min_num_pol]
            iono_cfg_freq_pol['B'] = iono_cfg_freq_pol['B'][:min_num_pol]

    info_str = \
        f"A: {iono_cfg_freq_pol['A']}, B {iono_cfg_freq_pol['B']} "\
        f"will be used for {iono_method}."
    info_channel.log(info_str)
    cfg['processing'][
        'ionosphere_phase_correction'][
        'list_of_frequencies'] = iono_cfg_freq_pol


def ionosphere_cfg_check(cfg):
    """Check ionosphere runconfig

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    """
    error_channel = journal.error('ionosphere_runconfig.ionosphere_cfg_check')

    # Extract ionosphere options
    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    iono_method = iono_cfg['spectral_diversity']
    iono_method_side = ['main_side_band', 'main_diff_ms_band']

    if not iono_cfg['enabled']:
        err_str = f'Ionosphere phase correction must be enabled '\
                  f'to execute {iono_method}.'
        error_channel.log(err_str)
        raise ValueError(err_str)

    iono_cfg = cfg['processing']['ionosphere_phase_correction']
    iono_cfg_freq_pol = iono_cfg['list_of_frequencies']

    # if any polarizations and frequencies are not given,
    # default is None for both polarizations.
    if iono_cfg_freq_pol is None:
        cfg['processing']['ionosphere_phase_correction'][
            'list_of_frequencies'] = {'A': None, 'B': None}

    _cfg_freq_pol_check(cfg, 'A')

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
