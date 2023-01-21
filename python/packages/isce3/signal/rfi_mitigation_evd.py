"""
Perform RFI mitigation by projecting raw data in the direction of
signal Eigenvectors
"""
import numpy as np


def rfi_mitigate_evd(
    raw_data,
    eig_vec_sort_array,
    rfi_cpi_flag_array,
    cpi_len=32,
    raw_data_mitigated=None,
):
    """Radio Frequency Interference (RFI) component of the Raw data is derived by
        projecting the raw data in the directions of Eigenvectors of dominant
        Eigenvalues (Principal Components).  RFI mitigation is performed by
        removing RFI component of raw data from raw data.

     Parameters
     ------------
     raw_data: array-like complex [num_pulses x num_rng_samples]
         raw data to be processed
     eig_vec_sort_array: 4D array of complex with dimension [num_cpi x num_rng_blks x cpi_len x cpi_len]
         Sorted column vector Eigenvectors of all CPIs based on index of sorted Eigenvalues
     rfi_cpi_flag_array: 2D array of integer, [num_cpi x num_rng_blks x cpi_len]
         RFI flag array that marks each Eigenvalue index in a CPI as RFI or desired signal.
         1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
     cpi_len: int
         Number of pulses within a CPI, default = 32
     raw_data_mitigated: array-like complex [num_pulses x num_rng_samples]
         Optional: Eigenvalue Decomposition mitigated raw data

     Returns
     --------
     raw_data_mitigated: complex numpy.ndarray, [num_pulses x num_rng_samples]
         Output array for Eigenvalue Decomposition mitigated raw data
    """

    num_pulses, num_rng_samples = raw_data.shape
    num_rng_blks = eig_vec_sort_array.shape[1]

    num_cpi = num_pulses // cpi_len
    num_pulses_cpi = cpi_len * num_cpi

    if raw_data_mitigated is None:
        raw_data_mitigated = np.zeros(raw_data.shape, dtype=np.complex64)

    cpi_start = np.arange(0, num_pulses_cpi, cpi_len)
    cpi_stop = cpi_start + cpi_len

    # Number of range bins per range block
    num_samples_blk = num_rng_samples // num_rng_blks
    num_samples_blks_all = num_samples_blk * num_rng_blks

    # Range Bin start and end for each of the range blocks
    rng_start = np.arange(0, num_samples_blks_all, num_samples_blk)
    rng_stop = rng_start + num_samples_blk

    # Ignore the last partial CPI if there is any.
    for idx_cpi in range(num_cpi):
        cpi = np.s_[cpi_start[idx_cpi] : cpi_stop[idx_cpi]]
        data_cpi = raw_data[cpi]

        for idx_blk in range(num_rng_blks):
            rng_blk = np.s_[rng_start[idx_blk] : rng_stop[idx_blk]]
            blk_cpi = data_cpi[:, rng_blk]

            eig_vec_sort = eig_vec_sort_array[idx_cpi, idx_blk]

            rfi_cpi_flag = rfi_cpi_flag_array[idx_cpi, idx_blk]
            num_rfi_cpi = rfi_cpi_flag.sum()

            if num_rfi_cpi > 0:
                eig_vec_rfi = eig_vec_sort[:, rfi_cpi_flag == 1]

                weight_adaptive = np.matmul(
                    eig_vec_rfi, np.conj(eig_vec_rfi).transpose()
                )
                data_proj_rfi_cpi = np.matmul(weight_adaptive, blk_cpi)
                data_proj_sig_cpi = blk_cpi - data_proj_rfi_cpi

                raw_data_mitigated[cpi, rng_blk] = data_proj_sig_cpi
            else:
                raw_data_mitigated[cpi, rng_blk] = blk_cpi

        # Fill the remaining few range bins of the CPI with original raw data samples
        if num_rng_samples > num_samples_blks_all:
            raw_data_mitigated[cpi, num_samples_blks_all:] = raw_data[
                cpi, num_samples_blks_all:
            ]

    # Fill the remaining few pulses with original raw data samples
    if num_pulses > num_pulses_cpi:
        raw_data_mitigated[num_pulses_cpi:] = raw_data[num_pulses_cpi:]

    return raw_data_mitigated
