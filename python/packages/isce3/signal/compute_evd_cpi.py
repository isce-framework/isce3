"""
Compute Eigenvalues and Eigenvectors of input data divided into Azimuth blocks
"""
import numpy as np
from numpy import linalg as la


def eigen_decomp_sort(cov_matrix):
    """Perform Eigenvaule Decomposition of Sample Covariance Matrix, and
       sort Eigenvalues in descending order.  Re-arrange column vector Eigenvectors
       based on the indices of sorted Eigenvalue sequence.
       Input cov_matrix needs to be full-rank to ensure correct derivation
       of Eigenvalues and Eigenvectors.

    Parameters
    ------------
    cov_matrix: 2D array of complex
        Sample Covariance Matrix (SCM) constructed from data within a coherent
        processing interval (CPI) with dimension [number of pulses/CPI x number of pulses/CPI]

    Returns
    --------
    eig_val_sort: 2D array of float, same shape as input matrix
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
    cpi_len,
    num_rng_blks=1,
):
    """Divide input raw data into data blocks in slow-time referred to as
    Coherent Processing Interval (CPI) and perform Eigenvalue Decompostion
    for all CPIs.

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed
    cpi_len: int
        Number of slow-time pulses within a CPI
    num_rng_blks: int
        Number of range bin blocks to be processed for EVD, default=1
        When num_rng_blk=1, all range samples are used to compute EVD

    Returns
    --------
    eig_val_sort_array: 3D array of float with dimension [num_cpi x num_rng_blks x cpi_len]
        Eigenvalues of all CPIs sorted in descending order
    eig_vec_sort_array: 4D array of complex with dimension [num_cpi x num_rng_blks x cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on index of sorted Eigenvalues
    """

    num_pulses, num_rng_samples = raw_data.shape
    num_samples_blk = num_rng_samples // num_rng_blks

    # Minimum number of range samples to estimate Sample Correlation Matrix L:
    # L ~ 2 * cpi_len
    # Reference: Space Time Adaptive Processing for Radar, Artech House, pp33
    rng_samples_blk_min = 2 * cpi_len

    # Verify Total number of pulses is greater than CPI length
    if num_pulses < cpi_len:
        raise ValueError(
            f"Coherent Processing Interval length exceeds total number of pulses {num_pulses}!"
        )

    # Verify number samples in a range block is greater than minimum needed to estimate
    # Sample Correlation Matrix
    if num_samples_blk < rng_samples_blk_min:
        raise ValueError(
            f"""Minimum number of samples in a range block to estimate Sample Correlation Matrix is {rng_samples_blk_min}!
            Current number of samples per range block is {num_samples_blk}!
            """
        )

    num_cpi = num_pulses // cpi_len
    num_pulses_cpi = cpi_len * num_cpi

    # CPI start and stop pulse indices
    cpi_start = np.arange(0, num_pulses_cpi, cpi_len)
    cpi_stop = cpi_start + cpi_len

    # If number of lines in the raw data is a small value
    # Range bin start and stop for each range block

    num_samples_blks_all = num_samples_blk * num_rng_blks

    rng_start = np.arange(0, num_samples_blks_all, num_samples_blk)
    rng_stop = rng_start + num_samples_blk

    # Output eigenvalues and eigenvectors
    eig_val_sort_array = np.zeros([num_cpi, num_rng_blks, cpi_len], dtype="f4")
    eig_vec_sort_array = np.zeros(
        (num_cpi, num_rng_blks, cpi_len, cpi_len), dtype="complex64"
    )

    for idx_cpi in range(num_cpi):
        cpi = np.s_[cpi_start[idx_cpi] : cpi_stop[idx_cpi]]
        data_cpi = raw_data[cpi]

        for idx_blk in range(num_rng_blks):
            blk_cpi = data_cpi[:, rng_start[idx_blk] : rng_stop[idx_blk]]
            cov_blk = np.matmul(blk_cpi, np.conj(blk_cpi).transpose()) / num_samples_blk

            eig_val_sort, eig_vec_sort = eigen_decomp_sort(cov_blk)
            eig_val_sort_array[idx_cpi, idx_blk] = eig_val_sort
            eig_vec_sort_array[idx_cpi, idx_blk] = eig_vec_sort

    return eig_val_sort_array, eig_vec_sort_array
