"""
Perform RFI Detection by evalutating Eigenvalue variation (slope)
"""
import numpy as np


def rfi_detect_evd(
    eig_val_sort_array,
    rfi_ev_idx_stop,
    detect_threshold=0.65,
):
    """Perform RFI detection using Eigenvaule Decomposition based on Principal
       Component Algorithm. A detection threshold in dB/Eigenvalue index is parameter.
       The slope (gradient) of sorted Eigenvalues within a CPI is computed.
       Any Eigenvalue slope above this threshold is identified as RFI.
       The last RFI Eigenvalue index separates the RFI eigenvalues from desired signal
       Eigenvalues. The algorithm is run for all Coherent Processing Intervals (CPI).

    Parameters
    ------------
    eig_val_sort_array: array of complex, [num_cpi x num_rng_blks x cpi_len]
        Sorted Eigenvalues in descending order of all CPIs in raw data
    rfi_ev_idx_stop: int
        RFI cut-off eigenvalue index. If RFI is detected beyond this cut-off index,
        it will be ignored to ensure less loss of desired data in EVD mitigation.
    detect_threshold: float
        RFI detection threshold used by EVD detection algorithm in dB/Eigenvalue index


    Returns
    --------
    rfi_cpi_flag_array: 3D array of integer, [num_cpi x num_rng_blks x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as RFI or desired signal.
        1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
    """

    # Ensure detection threshold is a positive value
    if not (detect_threshold > 0):
        raise ValueError("Detection threshold must be a positive value!")

    # Number of pulses, CPI length, and number of range blocks in a CPI
    num_cpi, num_rng_blks, cpi_len = eig_val_sort_array.shape

    # Ensure RFI Eigenvalue stop index is less than CPI length
    if not (rfi_ev_idx_stop < cpi_len):
        raise ValueError(
            "RFI Eigenvalue stop index must be less than number of pulses in a CPI!"
        )

    # RFI flag for each eigenvalue index in all CPIs: RFI=1, signal=0:w
    rfi_cpi_flag_array = np.ones((num_cpi, num_rng_blks, cpi_len), dtype="uint8")

    # Compute Eigenvalue Slope for all CPIs
    eig_val_sort_db_array = 10 * np.log10(np.abs(eig_val_sort_array))
    eig_val_db_slope_array = np.diff(eig_val_sort_db_array, axis=2)

    for idx_cpi in range(num_cpi):
        for idx_blk in range(num_rng_blks):
            eig_val_db_slope = eig_val_db_slope_array[idx_cpi, idx_blk]

            sig_ev_idx_start = 0
            rfi_ev_idx = np.where(
                eig_val_db_slope[:rfi_ev_idx_stop] < -detect_threshold
            )[0]
            if rfi_ev_idx.size:
                sig_ev_idx_start = rfi_ev_idx[-1] + 1

            # Sets signal eigenvalue indices to zero
            rfi_cpi_flag_array[idx_cpi, idx_blk, sig_ev_idx_start:] = 0

    return rfi_cpi_flag_array
