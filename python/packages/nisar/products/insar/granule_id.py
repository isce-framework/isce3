from datetime import datetime

import h5py
import journal
import numpy as np
from nisar.products.readers import SLC

# Constants for date and time formats
ZERO_DOPPLER_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
INSAR_FILENAME_TIME_FORMAT = '%Y%m%dT%H%M%S'

# InSAR range bandwidth (BW, units: MHz) mapping (Reference BW, Secondary BW) => InSAR BW in filename
# The last 00 indicates a nonexistent frequencyB (as per InSAR processing baseline)
INSAR_RG_BW_MAPPING = {
    (0, 0): np.nan,
    (0, 5): np.nan,
    (5, 0): np.nan,
    (0, 20): np.nan,
    (20, 0): np.nan,
    (0, 40): np.nan,
    (40, 0): np.nan,
    (0, 77): np.nan,
    (77, 0): np.nan,
    (5, 5): np.nan,
    (5, 20): np.nan,
    (20, 5): np.nan,
    (5, 40): np.nan,
    (40, 5): np.nan,
    (5, 77): np.nan,
    (77, 5): np.nan,
    (20, 20): 2000,
    (20, 40): 2000,
    (20, 77): 2000,
    (40, 40): 4000,
    (40, 20): 2000,
    (40, 77): 4000,
    (77, 77): 7700,
    (77, 20): 2000,
    (77, 40): 4000,
}

# InSAR polarization (Pol) (InSAR Pol content) => InSAR Pol code in filename
INSAR_POL_MAPPING = {
    ('HH'): 'SH',
    ('VV'): 'SV',
    ('HH', 'HV'): 'DH',
    ('VV', 'VH'): 'DV',
    ('LH', 'LV'): 'CL',
    ('RH', 'RV'): 'CR',
    ('HH', 'HV', 'VH', 'VH'): 'QP',
    ('HH', 'VV'): 'QD'
}


def get_slc_polarizations(slc_path, freq='A'):
    '''
    Get the list of polarizations from an SLC product.

    Parameters
    ----------
    slc_path: str
        The path to the SLC product file.
    freq: (str, optional)
        The frequency identifier (default is 'A').

    Returns
    -------
    numpy.ndarray:
        An array containing the list of polarizations.
    '''
    slc = SLC(hdf5file=slc_path)
    return slc.polarizations[freq]


def determine_insar_slant_range_bandwidth(ref_bw, sec_bw):
    '''
    Determine the InSAR slant range bandwidth based on reference and secondary bandwidths.
    If the combination of slant range bandwidths is not supported for InSAR baseline processing
    'np.nan' is returned.

    Parameters
    ----------
    ref_bw: float
        The slant range bandwidth (MHz) of the reference RSLC.
    sec_bw: float
        The slant range bandwidth (MHz) of the secondary RSLC.

    Returns
    -------
    float: The slant range bandwidth (MHz) of the InSAR product.
    '''

    return INSAR_RG_BW_MAPPING.get((ref_bw, sec_bw), np.nan)


def get_slc_range_bandwidth(slc_path, freq='A'):
    '''
    Get the acquired slant range bandwidth (MHz) from an SLC product.

    Parameters
    ----------
    slc_path: str
        The path to the SLC product file.
    freq: (str, optional)
        The frequency identifier (default is 'A').

    Returns
    -------
    float: the range bandwidth (MHz) of the RSLC product
    '''
    slc = SLC(hdf5file=slc_path)
    with h5py.File(slc_path, 'r', libver='latest', swmr=True) as h:
        return h[f'{slc.SwathPath}/frequency{freq}/processedRangeBandwidth'][()].astype('float') // 1e6


def get_slc_start_end_time(slc_path, time_type='start'):
    '''
    Retrieve the zero Doppler start or end time from an SLC product.

    Parameters
    ---------
    slc_path: str
        The path to the SLC product file.
    time_type: (str, optional)
        Type of time to retrieve ('start' or 'end', default:'start').

    Returns
    -------
    str:
        The formatted start or end time string in 'YYYYMMDDTHHMMSS' format.
    '''
    error_journal = journal.error('granule_id.get_slc_start_end_time')
    slc = SLC(hdf5file=slc_path)

    if time_type == 'start':
        extracted_time = slc.identification.zdStartTime
    elif time_type == 'end':
        extracted_time = slc.identification.zdEndTime
    else:
        err_str = f'{time_type} is invalid. Only "start" or "end" are valid types'
        error_journal.log(err_str)
        raise ValueError(err_str)

    # Remove fractional seconds from extracted time
    extracted_time = str(extracted_time).split('.')[0]
    date_obj = datetime.strptime(extracted_time, ZERO_DOPPLER_TIME_FORMAT)
    product_time = date_obj.strftime(INSAR_FILENAME_TIME_FORMAT)

    return product_time


def get_insar_polarization_code(polarizations):
    '''
    Determine the InSAR polarization code based on the
    polarization content of the InSAR product for frequencyA.
    It returns None if no matching code is found (i.e., the
    list of polarization is not in the baseline InSAR processing)

    Parameters
    ----------
    polarizations: list
        FrequencyA list of polarization of the InSAR product

    Returns
    -------
    str:
        InSAR polarization code.
    '''
    for key_polarizations, code in INSAR_POL_MAPPING.items():
        if set(key_polarizations) == set(polarizations):
            return code

    return None

def get_radar_band(slc_path):
    '''
    Get the radar band of the RSLC product given
    the frequency code

    Parameters
    ----------
    slc_path: str
        Path to the SLC product file

    radar_band: str
        1 character to indicate the radar band (e.g., L)
    '''
    slc = SLC(hdf5file=slc_path)

    with h5py.File(slc_path, 'r', libver='latest', swmr=True) as h:
        return h[f'{slc.IdentificationPath}/radarBand'][()]


def get_insar_granule_id(ref_slc_path, sec_slc_path, partial_granule_id,
                         pol_process, freq='A', product_type='RIFG'):
    '''
    Generate an InSAR granule ID based on input parameters.

    Parameters
    ----------
    ref_slc_path: str
        Path to the reference SLC product file.
    sec_slc_path: str
        Path to the secondary SLC product file.
    partial_granule_id: str
        Partial granule ID template.
    pol_process: list
        List of polarizations to process for the InSAR product.
    freq: (str, optional)
        Frequency identifier (default: 'A').
    product_type: (str, optional):
        Product type (default: 'RIFG').

    Returns
    -------
    str:
        InSAR granule ID.

    This function constructs an InSAR granule ID by replacing placeholders in
    the provided partial granule ID with information based on the input parameters,
    including slant range bandwidths, polarization codes, and zero Doppler start/end times.

    Example of InSAR granule ID for RIFG product:
    'NISAR_L1_PR_RIFG_034_080_A_010_2005_DVDV_A_20230619T000803_20230619T000835_20230631T000803_20230631T000835_D00340_P_P_J_001.h5'
    '''
    warning_channel = journal.warning('granule_id.get_insar_granule_id')

    radar_band = get_radar_band(ref_slc_path, freq=freq)
    if radar_band not in ['L', 'S']:
        err_str = f"The radar band {radar_band} is not a supported NISAR radar band" \
                  f"Assigning a dummy value of 'A' "
        warning_channel.log(err_str)

    level = '1' if product_type.startswith('R') else '2'
    band_level = f'{radar_band}{level}'
    ref_rg_bw = get_slc_range_bandwidth(ref_slc_path, freq=freq)
    sec_rg_bw = get_slc_range_bandwidth(sec_slc_path, freq=freq)
    insar_bw_mode = determine_insar_slant_range_bandwidth(ref_rg_bw, sec_rg_bw)

    if np.isnan(insar_bw_mode):
        err_str = f'Reference and secondary slant range bandwidths ' \
                  f'combination does not generate a supported NISAR InSAR mode: ' \
                  f'Reference bandwidth (MHz): {ref_rg_bw}, Secondary bandwidth (MHz): {sec_rg_bw}'
        warning_channel.log(err_str)

    ref_pols = get_slc_polarizations(ref_slc_path, freq=freq)
    sec_pols = get_slc_polarizations(sec_slc_path, freq=freq)
    insar_pols = list(set(ref_pols) & set(sec_pols) & set(pol_process))

    if any(p in insar_pols for p in ['HV', 'VH', 'LH', 'LV', 'RH', 'RV']):
        err_str = f'InSAR polarizations to process contain some unsupported polarization channels ' \
                  f'in data production: {insar_pols}'
        warning_channel.log(err_str)

    insar_pol_mode = get_insar_polarization_code(insar_pols)
    ref_start_time = get_slc_start_end_time(ref_slc_path, time_type='start')
    ref_end_time = get_slc_start_end_time(ref_slc_path, time_type='end')
    sec_start_time = get_slc_start_end_time(sec_slc_path, time_type='start')
    sec_end_time = get_slc_start_end_time(sec_slc_path, time_type='end')

    info_values = [band_level, product_type, insar_bw_mode, insar_pol_mode,
                   ref_start_time, ref_end_time, sec_start_time, sec_end_time]
    placeholders = ['{Level}', '{ProductType}', '{MODE}', '{PO}', '{RefStartDateTime}',
                    '{RefEndDateTime}', '{SecStartDateTime}', '{SecEndDateTime}']

    for value, placeholder in zip(info_values, placeholders):
        partial_granule_id = partial_granule_id.replace(placeholder, str(value))

    return partial_granule_id
