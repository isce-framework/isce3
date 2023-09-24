"""
Perform RFI detection and mitigation of input raw data using Slow-Time Eigenvalue Decomposition
(ST-EVD).
"""
import numpy as np
from isce3.signal.compute_evd_cpi import slice_gen
from isce3.signal.rfi_detection_evd import rfi_detect, ThresholdParams
from isce3.signal.rfi_mitigation_evd import rfi_mitigate_tb

def run_slow_time_evd(
    raw_data: np.ndarray,
    cpi_len,
    max_deg_freedom,
    *,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    num_rng_blks=1,
    threshold_params: ThresholdParams = ThresholdParams(),
    num_cpi_tb=20,
    mitigate_enable=False,
    raw_data_mitigated=None,
):

    """This is the top-level wrapper which takes raw data in whole or part and does the following:
    1. Partition data into smaller blocks defined as Threshold Block (TB)
       Each TB is consisted of M Coherent Processing Intervals (CPI) and N range samples
       Each CPI is consisted of K slow-time pulses.
    2. Derive slow-time RFI detection threshold for each TB.
    3. Mitigate RFI of all CPIs above the detection threshold if mitigation is enabled.

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    cpi_len: int
        Number of slow-time pulses within a CPI
    max_deg_freedom: int
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len to avoid unintended removal of signal data.
    num_max_trim: int, default=0
        Number of large value outliers to be trimmed in slow-time minimum Eigenvalues.
    num_min_trim: int, default=0
        Number of small value outliers to be trimmed in slow-time minimum Eigenvalues
    max_num_rfi_ev: int, default=2
        A detection error (miss) happens when a maximum power RFI emitter contaminates 
        multiple consecutive CPIs, resulting in a flat maximum Eigenvalue slope in slow 
        time. Hence the standard (STD) deviation of multiple dominant EVs across slow time 
        defined by this parameter are compared. The one with the maximum STD is used for RFI
        Eigenvalue first difference computation.
    num_rng_blks: int, default=1
        Number of range bin blocks to be processed for EVD, default=1
        When num_rng_blks=1, all range samples are used to compute EVD
    threshold_params: ThresholdParams object, default=ThresholdParams()
        RFI detection threshold interpolation parameters. The x field defines STD
        ratio between maximum and minimum Eigenvalue slopes (MMES) of the
        slow-time threshold interval. The y field defines the number of sigma (STD)
        from the mean of MMES.
    num_cpi_tb: int, default=20
        Number of slow-time CPIs in a TB
    mitigate_enable: bool, default=False
        Enable mitigation
    raw_data_mitigated: array-like complex [num_pulses x num_rng_samples] or None, optional
        output array in which the mitigated data values is placed. It
        must be an array-like object supporting `multidimensional array access
        <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
        The array should have the same shape and dtype as the input raw data array.
        If None (the default), the input 'raw_data' will be modified in-place.

    Returns
    --------
    rfi_likelihood: float
        Ratio of number of CPIs detected with RFI Eigenvalues over that of total number
        of CPIs.

    Notes
    -----
    If the number of pulses is not an integer multiple of the CPI length,
    any remaining pulses after the last full CPI will be unmitigated.

    References
    ----------
    ..[1] Bo Huang, Heresh Fattahi, Hirad Ghaemi, Brian Hawkins, Geoffrey Gunter,
    "Radio Frequency Interference Detection and Mitigation of NISAR DATA using
    Slow Time Eigenvalue Decomposition", IGARSS 2023.'
    """

    num_pulses, num_rng_samples = raw_data.shape
    num_samples_rng_blk = num_rng_samples // num_rng_blks

    # If the number of pulses is not an integer multiple of TB size, following
    # operations will ensue. If the number of remaining pulses is greater than
    # the CPI length, additional CPI(s) will be constructed, the very last TB will
    # include the additional CPI(s). The rest of the remaining pulses not enough to
    # construct a full CPI will not be processed. If the number of remaining pulses 
    # is less than CPI length, then they will not be processed. In both cases, 
    # at most cpi_len-1 number of pulses will be ignored.

    num_cpi = num_pulses // cpi_len
    num_pulses_proc = cpi_len * num_cpi
    num_pulses_tb = cpi_len * num_cpi_tb

    # Modify raw_data in-place
    if raw_data_mitigated is None:
        raw_data_mitigated = raw_data
    else:
        if raw_data_mitigated.shape != raw_data.shape:
            raise ValueError(
                "Shape mismatch: output mitigated data array must have the same shape"
                " as the input data"
            )

    # Collect total number of CPI range blocks contaminated by RFI
    rfi_cpi_count_sum = 0

    # Verify total number of pulses is equal or greater than number of pulses per TB
    if num_pulses < num_pulses_tb:
        raise ValueError(
            "Total number of pulses must be greater or equal to that of a threshold block."
        )

    # Maximum number of degrees of freedom must be less than cpi_len
    if max_deg_freedom >= cpi_len:
        raise ValueError(
            "Max number of deg. of freedom must be less than number of pulses in a CPI."
        )

    # Run RFI Detection and Mitigation
    for idx_tb, tb_slow_time in enumerate(slice_gen(num_pulses_proc, num_pulses_tb)):
        for idx_rng, tb_fast_time in enumerate(
            slice_gen(num_rng_samples, num_samples_rng_blk)
        ):
            raw_tb_blk = raw_data[tb_slow_time, tb_fast_time]

            (rfi_cpi_flag_tb, evec_sort_tb) = rfi_detect(
                raw_tb_blk,
                cpi_len,
                max_deg_freedom,
                num_max_trim,
                num_min_trim,
                max_num_rfi_ev,
                threshold_params,
            )

            num_rfi_ev_cpi = np.sum(rfi_cpi_flag_tb, axis=1)
            rfi_cpi_count = np.sum(num_rfi_ev_cpi != 0)
            rfi_cpi_count_sum += rfi_cpi_count

            # Run Mitigation:
            if mitigate_enable:
                rfi_mitigate_tb(
                    raw_tb_blk,
                    evec_sort_tb,
                    rfi_cpi_flag_tb,
                    raw_data_mitigated[tb_slow_time, tb_fast_time],
                )

    # Percentage of RFI Eigenvalues based on slow-time min EV slope detection
    rfi_likelihood = rfi_cpi_count_sum / (num_cpi * num_rng_blks)

    # Fill the remaining few pulses with original raw data samples
    if num_pulses > num_pulses_proc:
        raw_data_mitigated[num_pulses_proc:] = raw_data[num_pulses_proc:]

    return rfi_likelihood
