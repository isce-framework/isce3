"""
Compute Eigenvalues and Eigenvectors of input data one
Coherent Processing Intervals (CPI) at a time
"""
from __future__ import annotations
import numpy as np
from numpy import linalg as la
from collections.abc import Iterator

def slice_gen(total_size: int, batch_size: int, combine_rem: bool=True) -> Iterator[slice]:
    """Generate slices with size defined by batch_size.

    Parameters
    ----------
    total_size: int
        size of data to be manipulated by slice_gen
    batch_size: int
        designated data chunk size in which data is sliced into.
    combine_rem: bool
        Combine the remaining values with the last complete block if 'True'.
        If False, ignore the remaining values
        Default = 'True'

    Yields
    ------
    slice: slice obj
        Iterable slices of data with specified input batch size, bounded by start_idx
        and stop_idx.
    """

    num_complete_blks = total_size // batch_size
    num_total_complete = num_complete_blks * batch_size
    num_rem = total_size - num_total_complete

    if combine_rem and num_rem > 0:
        for start_idx in range(0, num_total_complete - batch_size, batch_size):
            stop_idx = start_idx + batch_size
            yield slice(start_idx, stop_idx)

        last_blk_start = num_total_complete - batch_size
        last_blk_stop = total_size
        yield slice(last_blk_start, last_blk_stop)
    else:
        for start_idx in range(0, num_total_complete, batch_size):
            stop_idx = start_idx + batch_size
            yield slice(start_idx, stop_idx)


def eigen_decomp_sort(cov_matrix):
    """Perform Eigenvaule Decomposition of Sample Covariance Matrix which is assumed 
    to be Hermitian, and sort Eigenvalues in descending order.  Re-arrange column 
    vector Eigenvectors based on the indices of sorted Eigenvalue sequence.
    Input cov_matrix needs to be full-rank to ensure correct derivation
    of Eigenvalues and Eigenvectors.

    Parameters
    ------------
    cov_matrix: 2D array of complex
        Sample Covariance Matrix (SCM) constructed from data within a coherent
        processing interval (CPI) with dimension [number of pulses/CPI x number of pulses/CPI]

    Returns
    --------
    eig_val_sort: 1D array of float, same length as number of rows (or columns) of input matrix
        Eigenvalues sorted in descending order
    eig_vec_sort: 2D array of complex, same shape as input matrix
        column vector Eigenvectors sorted based on index of sorted Eigenvalues
    """

    eig_val, eig_vec = la.eigh(cov_matrix)

    eig_val_sort = eig_val[::-1]
    eig_vec_sort = eig_vec[:, ::-1]

    return eig_val_sort, eig_vec_sort

def compute_evd(
    raw_data: np.ndarray,
):
    """Perform Eigenvalue Decomposition along axis 0.

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed

    Returns
    --------
    eig_val_sort: 1D array of float, same length as the number of rows of input matrix
        Eigenvalues sorted in descending order
    eig_vec_sort: 2D array of complex, same shape as input matrix
        column vector Eigenvectors sorted based on index of sorted Eigenvalues
    """

    num_rng_samples = raw_data.shape[1]

    # The raw_data is not necessarily zero-mean when it is corrupted by RFI.
    # If so, estimated sample covariance matrix cov_cpi should be called 
    # sample correlation  matrix instead.  The reference below demonstrates
    # this concept and notation.

    # F. Zhou, R. Wu, M. Xing, and Z. Bao, “Eigensubspace-Based Filtering With 
    # Application in Narrow-Band Interference Suppression for SAR”, IEEE Geoscience 
    # and Remote Sensing Letters, vol. 4, no. 1, pp. 76,2007.

    cov_cpi = np.matmul(raw_data, np.conj(raw_data).transpose()) / num_rng_samples
    eig_val_sort, eig_vec_sort = eigen_decomp_sort(cov_cpi)

    return eig_val_sort, eig_vec_sort

def compute_evd_tb(
    raw_data: np.ndarray,
    cpi_len=32,
):
    """Divide input raw data equivalent to a threshold block into data blocks 
    or Coherent Processing Intervals (CPI) with resepct to axis=0 and perform 
    Eigenvalue Decomposition for all CPIs.

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed
    cpi_len: int, optional
        Number of slow-time pulses within a CPI, default=32

    Returns
    --------
    eig_val_sort_array: 2D array of float with dimension [num_cpi x cpi_len]
        Eigenvalues of all CPIs sorted in descending order
    eig_vec_sort_array: 3D array of complex with dimension [num_cpi x cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on index of sorted Eigenvalues
    """

    # compute number of CPIs
    num_pulses, num_rng_samples = raw_data.shape
    num_cpi = num_pulses // cpi_len

    # Minimum number of range samples to estimate Sample Correlation Matrix L:
    # L ~ 2 * cpi_len
    # Reference: Space Time Adaptive Processing for Radar, Artech House, pp33
    rng_samples_min = 2 * cpi_len

    # Verify number of range samples in raw data is greater than minimum needed
    # to estimate Sample Covariance Matrix
    if num_rng_samples < rng_samples_min:
        raise ValueError(
            "Minimum number of samples in a range block to estimate Sample Covariance"
            f" Matrix is {rng_samples_min}! Current number of samples per range block"
            f" is {num_rng_samples}!"
        )

    # Verify Total number of pulses is greater than CPI length
    if num_pulses < cpi_len:
        raise ValueError(
            f"Coherent Processing Interval length exceeds total number of pulses {num_pulses}!"
        )

    # Output Eigenvalues and Eigenvectors
    eig_val_sort_array = np.zeros([num_cpi, cpi_len], dtype="f4")
    eig_vec_sort_array = np.zeros((num_cpi, cpi_len, cpi_len), dtype="complex64")

    for idx_cpi, cpi_slow_time in enumerate( slice_gen(num_pulses, cpi_len, combine_rem=False)):
        data_cpi = raw_data[cpi_slow_time]

        eig_val_sort, eig_vec_sort = compute_evd(data_cpi)
        eig_val_sort_array[idx_cpi] = eig_val_sort
        eig_vec_sort_array[idx_cpi] = eig_vec_sort

    return eig_val_sort_array, eig_vec_sort_array
