"""
Compute Eigenvalues and Eigenvectors of input data one
Coherent Processing Intervals (CPI) at a time
"""
from __future__ import annotations
import numpy as np
from numpy import linalg as la
from collections.abc import Iterator
import warnings

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

def compute_evd_tb(
    raw_data: np.ndarray,
    *,
    cpi_len: int=16,
    mask_valid: np.ndarray=None,
    off_diag_overlap_ratio: float=0.25,
    diag_valid_ratio: float=0.20,
    min_ev_valid_idx: int=10,
    rx_dynamic_range_db: float=50.0,
):
    """Divide input raw data equivalent to a threshold block into Coherent
    Processing Intervals (CPI) with respect to axis=0 and perform Eigenvalue
    Decomposition for all CPIs.

    For constant-PRF data, standard EVD is used. For dithered-PRF data, gap-exclusion 
    EVD is used. CPI validity is always checked.

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        Raw data to be processed
    cpi_len: int, optional
        Number of slow-time pulses within a CPI, default=16
    mask_valid : np.ndarray bool, [num_pulses x num_rng_samples], optional
        Valid-sample mask with same shape as raw_data.  If provided, CPI sample
        covariance matrix will be computed differently by excluding the invalid
        data gaps.
    off_diag_overlap_ratio : float, optional, default = 0.25
        Minimum overlap ratio used by gap exclusion covariance estimation
    diag_valid_ratio : float, optional, default = 0.20
        Minimum fraction of valid samples required to compute a diagonal term
        in the sample covariance matrix entry R_ii.
    min_ev_valid_idx: int, optional
        Eigenvalue index used by threshold estimation to estimate the slow-time minimum
        Eigenvalue slope. This parameter is also used to validate that the threshold block
        has enough usable eigenvalues for robust sample covaraince estimation of a CPI.
    rx_dynamic_range_db: float, optional, default = 50 dB
        Radar platform receiver dynamic range. This is applied as a threshold
        to determine if the Eigenvalue under test is meaningfully signficant. If the
        Eigenvalue under test is less than this threshold, it will be viewed as unusable.

    Returns
    --------
    eig_val_sort_array: 2D array of float with dimension [num_cpi x cpi_len]
        Eigenvalues of all CPIs sorted in descending order
    eig_vec_sort_array: 3D array of complex with dimension
        [num_cpi x cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on index of sorted
        Eigenvalues
    tb_is_valid : bool
        False if any CPI in the threshold block lacks sufficient usable Eigenvalues,
        determined by checking whether the Eigenvalue at min_ev_valid_idx is meaningful.
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
            f" Matrix is {rng_samples_min} per Reed-Mallet-Brennan Rule."
            f"Current number of samples per range block is {num_rng_samples}!"
        )

    # Verify Total number of pulses is greater than CPI length
    if num_pulses < cpi_len:
        raise ValueError(
            f"Coherent Processing Interval length of {cpi_len} exceeds total number of pulses {num_pulses}!"
        )

    if min_ev_valid_idx >= cpi_len:
        raise ValueError(
            f"min_ev_valid_idx ({min_ev_valid_idx}) must be less than cpi_len ({cpi_len}). "
            "Since Python uses 0-based indexing, the maximum valid index is cpi_len - 1."
        )

    # Verify TB Mask shape
    if (mask_valid is not None) and (mask_valid.shape != raw_data.shape):
        raise ValueError(f"Valid TB mask shape {mask_valid.shape} != TB data shape {raw_data.shape}")
    
    # Output Eigenvalues and Eigenvectors
    eig_val_sort_array = np.zeros([num_cpi, cpi_len], dtype="f4")
    eig_vec_sort_array = np.zeros((num_cpi, cpi_len, cpi_len), dtype="complex64")

    tb_is_valid = True

    # Compute Eigenvalue and Eigenvector pairs for each CPI
    for idx_cpi, cpi_slow_time in enumerate(
        slice_gen(num_pulses, cpi_len, combine_rem=False)
    ):
        data_cpi = raw_data[cpi_slow_time]
        mask_valid_cpi = None if mask_valid is None else mask_valid[cpi_slow_time]
        eig_val_sort, eig_vec_sort = compute_evd(
            data_cpi,
            mask_valid_cpi=mask_valid_cpi,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )

        # Verify if the eigenvalue of CPI at index defind by min_ev_valid_idx is meaningful
        eig_val_sort_abs = np.maximum(np.abs(eig_val_sort), 1e-30)
        noise_ev_norm_db = 10 * np.log10(eig_val_sort_abs[min_ev_valid_idx] / eig_val_sort_abs[0])

        if -noise_ev_norm_db > rx_dynamic_range_db:
            tb_is_valid = False
            break

        eig_val_sort_array[idx_cpi] = eig_val_sort
        eig_vec_sort_array[idx_cpi] = eig_vec_sort

    return eig_val_sort_array, eig_vec_sort_array, tb_is_valid

def compute_evd(
    raw_data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    off_diag_overlap_ratio: float = 0.25,
    diag_valid_ratio: float = 0.20,
):
    """Perform Eigenvalue Decomposition along axis 0.

    Parameters
    ----------
    raw_data : array-like complex [num_pulses x num_rng_samples]
        Raw data to be processed.
    mask_valid_cpi : (num_pulses, num_rng_samples) bool array, optional
        True indicates valid samples. False indicates invalid samples or gaps.
        If provided, CPI sample covariance matrix will be computed differently
        by excluding the invalid data gaps.
    off_diag_overlap_ratio : float, default=0.25
        Minimum fraction of overlapping valid range samples required to compute
        an off-diagonal covariance term R_ij if mask is provided.
    diag_valid_ratio : float, default=0.20
        Minimum fraction of valid range samples required to compute
        a diagonal covariance term R_ii when mask is provided.

    Returns
    -------
    eig_val_sort : 1D array of float
        Eigenvalues sorted in descending order.
    eig_vec_sort : 2D array of complex
        Column eigenvectors sorted to match eig_val_sort.
    """
    # The raw_data is not necessarily zero-mean when it is corrupted by RFI.
    # If so, estimated sample covariance matrix cov_cpi should be called 
    # sample correlation  matrix instead.  The reference below demonstrates
    # this concept and notation.

    # F. Zhou, R. Wu, M. Xing, and Z. Bao, “Eigensubspace-Based Filtering With 
    # Application in Narrow-Band Interference Suppression for SAR”, IEEE Geoscience 
    # and Remote Sensing Letters, vol. 4, no. 1, pp. 76,2007.

    if mask_valid_cpi is not None:
        mask_valid_cpi = mask_valid_cpi.astype(bool, copy=False)

        if mask_valid_cpi.shape != raw_data.shape:
            raise ValueError(
                f"Valid CPI mask shape {mask_valid_cpi.shape} != CPI data shape {raw_data.shape}"
            )

        cov_cpi = compute_gap_exclusion_cov(
            raw_data,
            mask_valid_cpi=mask_valid_cpi,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )
    else:
        num_rng_samples = raw_data.shape[1]
        cov_cpi = (raw_data @ raw_data.conj().T) / num_rng_samples

    eig_val_sort, eig_vec_sort = eigen_decomp_sort(cov_cpi)

    return eig_val_sort, eig_vec_sort


def compute_gap_exclusion_cov(
    data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    off_diag_overlap_ratio: float = 0.25,
    diag_valid_ratio: float = 0.20,
):
    """
    Compute a gap-excluded slow-time sample covariance matrix.

    Parameters
    ----------
    data: (num_pulses, num_rng_samples) complex array
        Slow-time block: K pulses x M range samples.
        Pulses should be contiguous in slow time for ST-EVD.
    mask_valid_cpi: (num_pulses, num_rng_samples) bool array, optional
        True indicates valid samples. False indicates invalid samples or gaps.
        If None, an all true boolean mask is created. All samples are assumed to be valid.
    off_diag_overlap_ratio: float, default = 0.25
        Minimum fraction of overlapping valid range samples required to compute
        an off-diagonal term in the sample covariance matrix entry R_ij.
    diag_valid_ratio : float, optional, deffault = 0.20
        Minimum fraction of valid samples required to compute a diagonal term in the
        sample covariance matrix entry R_ii.
        
    Returns
    -------
    cov : (num_pulses, num_pulses) complex64
        Gap-excluded sample covariance matrix.

    Notes
    -------
    The number of range samples per purlse must be equal or larger than 2 x number of pulses 
    to be processed in order to have a reliable sample covariance matrix estimate.
    """

    num_pulses, num_rng_samples = data.shape

    if mask_valid_cpi is None:
        mask_valid_cpi = np.ones(data.shape, dtype=bool)

    if mask_valid_cpi.shape != data.shape:
        raise ValueError(f"CPI mask shape {mask_valid_cpi.shape} != CPI data shape {data.shape}")

    if not (0.0 < off_diag_overlap_ratio <= 1.0):
        raise ValueError("off_diag_overlap_ratio must be between 0 and 1.")

    if not (0.0 < diag_valid_ratio <= 1.0):
        raise ValueError("diag_valid_ratio must be between 0 and 1.")

    # Minimum Samples required to compute diagonal and off-diagonal terms of
    # Sample Covariance Matrix
    min_valid_off_diag = max(1, int(np.ceil(off_diag_overlap_ratio * num_rng_samples)))
    min_valid_diag = max(1, int(np.ceil(diag_valid_ratio * num_rng_samples))) 

    # The number of range samples per purlse must be equal or larger than 2 x number of pulses
    # to be processed in order to have a reliable sample covariance matrix estimate.
    # However, sometimes we would like to explore the trade-off betweeen less number of samples
    # to estimate sample covariance matrix and reduction of RFI artifact. Hence only a warning
    # is raised.
    rng_samples_min = 2 * num_pulses

    if min_valid_off_diag < rng_samples_min:
        warnings.warn(f"""
            Minimum number of samples required per pulse to estimate sample covariance matrix
            is {rng_samples_min}. The number of valid overlapping off-diagonal samples is
            {min_valid_off_diag}.
        """)

    if min_valid_diag < rng_samples_min:
        warnings.warn(f"""
            Minimum number of samples required per pulse to estimate sample covariance matrix
            is {rng_samples_min}. The number of valid diagonal samples is {min_valid_off_diag}.
        """)

    # Zero-out invalid samples
    x_valid =  data * mask_valid_cpi

    # Count valid sample overalp count for each element of the
    # sample covariance matrix
    mask_int = mask_valid_cpi.astype(np.int32)
    overlap_counts = mask_int @ mask_int.T  # shape (pulse x pulse)

    # Sum of conjugate products over overlapping valid samples
    # without proper normalization
    cov_sum = x_valid @ x_valid.conj().T

    # Initialize gap-excluded sample covariance matrix
    cov = np.zeros((num_pulses, num_pulses), dtype=np.complex64)

    #Count number of valid in diagonal and off-diagonal for normalization

    # Diagonal terms: Generate Diagonal indices
    diag_idx = np.diag_indices(num_pulses)
 
    # Extract the number of valid samples of diagonal terms
    diag_counts = overlap_counts[diag_idx]

    # Extract the uncorrected diagonal values
    diag_cov_sum = cov_sum[diag_idx]

    # Check if there are enough valid samples
    diag_valid_idx = diag_counts >= min_valid_diag

    diag_vals = np.zeros(num_pulses, dtype=np.complex64)

    # Normalize the diagonal terms
    diag_vals[diag_valid_idx] = diag_cov_sum[diag_valid_idx] / diag_counts[diag_valid_idx]
    cov[diag_idx] = diag_vals

    # Off-diagonal: Verify if there are enough valid samples for off-diagonal terms
    off_diag_valid = overlap_counts >= min_valid_off_diag
    
    # Mask out diagonal terms of the off_diagonal_valid matrix
    np.fill_diagonal(off_diag_valid, False)

    # Normalize the off-diagonal terms
    cov[off_diag_valid] = cov_sum[off_diag_valid] / overlap_counts[off_diag_valid]

    # Ensure Hermitian numerically
    cov = (0.5 * (cov + cov.conj().T)).astype(np.complex64)

    return cov
