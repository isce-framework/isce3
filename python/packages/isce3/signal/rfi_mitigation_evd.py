"""
Perform RFI mitigation by projecting raw data in the direction of
RFI Eigenvectors, and subsequently remove it from input raw data.
"""
import numpy as np
from isce3.signal.compute_evd_cpi import slice_gen

def rfi_mitigate_evd(
    raw_data: np.ndarray,
    eig_vec_sort,
    rfi_flag,
    raw_data_mitigated=None,
):
    """Radio Frequency Interference (RFI) component of the Raw data in a CPI is derived by
    projecting the raw data in the directions of Eigenvectors of dominant
    Eigenvalues (Principal Components).  RFI mitigation is performed by
    removing RFI component of raw data from raw data. The  number of rows (pulses) 
    in the 'raw_data' input is equal to a single CPI length.

    Parameters
    ------------
    raw_data: array-like complex [cpi_len x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    eig_vec_sort: 2D array of complex, [cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on indices of sorted Eigenvalues
    rfi_flag: 1D array of bool, [cpi_len]
        RFI flag array that marks each Eigenvalue index as either RFI or signal.
        1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
    raw_data_mitigated: array-like complex [num_pulses x num_rng_samples] or None, optional
        output array in which the mitigated data values is placed. It
        must be an array-like object supporting `multidimensional array access
        <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
        The array should have the same shape and dtype as the input raw data array.
        If None (the default), an alias will be created internally and 'raw_data' is modified
        in-place.
    """

    if raw_data_mitigated is None:
        raw_data_mitigated = raw_data
    else:
        if raw_data_mitigated.shape != raw_data.shape:
            raise ValueError(
                "Shape mismatch: output mitigated data array must have the same shape"
                " as the input data"
            )

     # Verify total number of pulses is greater than number of pulses per CPI
    if raw_data.shape[0] != eig_vec_sort.shape[0]:
        raise ValueError(
            "Total number of pulses must be equal to CPI length."
        )

    num_rfi = rfi_flag.sum()

    if num_rfi > 0:  # RFI is detected in CPI
        eig_vec_rfi = eig_vec_sort[:, rfi_flag == 1]

        weight_adaptive = np.matmul(eig_vec_rfi, np.conj(eig_vec_rfi).transpose())

        data_proj_rfi = np.matmul(weight_adaptive, raw_data)
        raw_data_mitigated[:] = raw_data - data_proj_rfi
    else:
        raw_data_mitigated[:] = raw_data


def rfi_mitigate_tb(
    raw_data_tb,
    eig_vec_sort_array,
    rfi_cpi_flag_array,
    raw_data_mitigated_tb=None,
):
    """Wrapper function that performs RFI mitigation of the input raw data of a 
    Threshold Block (TB) one CPI at a time. Number of pulses or rows of raw_data_tb must
    be an integer of CPI length

    Parameters
    ------------
    raw_data_tb: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed
    eig_vec_sort_array: 3D array of complex, [num_cpi x cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on indices of sorted Eigenvalues
    rfi_cpi_flag_array: 2D array of bool, [num_cpi x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as either RFI or signal.
        1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
    raw_data_mitigated_tb: array-like complex [num_pulses x num_rng_samples] or None, optional
        output array in which the mitigated data values is placed. It
        must be an array-like object supporting `multidimensional array access
        <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
        The array should have the same shape and dtype as the input raw data array.
        If None (the default), an alias will be created internally, and 'raw_data' is modified
        in-place.
    """

    num_pulses = raw_data_tb.shape[0]
    cpi_len = eig_vec_sort_array.shape[2]

    if num_pulses % cpi_len != 0:
        raise ValueError("Total number of pulses must be an integer multiple of cpi_len")

    # Create an alias for raw_data_tb if there is no 'raw_data_mitigated_tb' as input
    if raw_data_mitigated_tb is None:
        raw_data_mitigated_tb = raw_data_tb

    for idx_cpi, cpi_slow_time in enumerate(slice_gen(num_pulses, cpi_len)):
        data_cpi = raw_data_tb[cpi_slow_time]

        eig_vec_sort = eig_vec_sort_array[idx_cpi]
        rfi_cpi_flag = rfi_cpi_flag_array[idx_cpi]

        rfi_mitigate_evd(
            data_cpi, eig_vec_sort, rfi_cpi_flag, raw_data_mitigated_tb[cpi_slow_time])
