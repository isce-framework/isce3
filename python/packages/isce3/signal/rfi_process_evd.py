"""
Perform RFI detection and mitigation of input raw data using Slow-Time Eigenvalue Decomposition
(ST-EVD) using the original ev_slope method only.
"""
import numpy as np
from isce3.signal.compute_evd_cpi import slice_gen, count_excluded_pulses_per_cpi
from isce3.signal.rfi_detection_evd import rfi_detect, ThresholdParams
from isce3.signal.rfi_mitigation_evd import rfi_mitigate_tb
import warnings
import os

def run_slow_time_evd(
    raw_data: np.ndarray,
    cpi_len,
    max_deg_freedom,
    *,
    num_rfi_buffer=2,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    num_samples_rng_blk=250,
    use_entire_pulse=False,
    threshold_params: ThresholdParams = None,
    num_cpi_per_threshold_block=12,
    off_diag_overlap_ratio=0.20,
    diag_valid_ratio=0.15,
    mitigate_enable=False,
    min_rank_frac=0.70,
    rx_dynamic_range_db=50.0,
    swaths=None,
    threshold_method='ev_slope',
    rfi_check=True,
    max_ev_spread_thresh_db=2.0,
    eff_rank_std_thresh=1.0,
    sig_ev_margin_db=1.0,
    rfi_candidate_tolerance_db=3.0,
    raw_data_mitigated=None,
):

    """This is the top-level wrapper which takes raw data in whole or part and does the following:
    1. Partition data into smaller blocks defined as Threshold Block (TB)
       Each TB is consisted of M Coherent Processing Intervals (CPI) and N range samples
       Each CPI is consisted of K slow-time pulses.
    2. Derive slow-time RFI detection threshold for each TB using ev_slope method.
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
    num_samples_rng_blk: int, default=250
        Number of range samples per range block when data blocking is applied in range direction
        for sample covariance matrix estimation. It is recommended that this parameter is at
        least 5 x cpi_len to avoid discrepancy from true sample covariance matrix. In addition,
        in order to avoid a run-time error for ST-EVD, this parameter needs to be
        at least 2 x cpi_len.
    use_entire_pulse: bool, default=False
        Ignore any value passed for num_samples_rng_blk and instead use all samples
        in the slow-time pulses for detection if this is True.
    threshold_params: ThresholdParams object or None, default=None
        RFI detection threshold interpolation parameters. If None, default
        ThresholdParams() (x=[2.0, 20.0], y=[5.0, 2.0]) is used.
        The x field defines the STD ratio between maximum and
        minimum Eigenvalue slopes (MMES) of the slow-time threshold interval, and
        the y field defines the number of sigma (STD) from the mean of MMES.
    num_cpi_per_threshold_block: int, default=12
        Number of slow-time CPIs in a TB
    off_diag_overlap_ratio : float, optional, default=0.20
        Minimum overlap ratio used by gap exclusion covariance estimation
    diag_valid_ratio : float, optional, default=0.15
        Minimum fraction of valid samples required to compute a diagonal term in the
        sample covariance matrix entry R_ii.
    mitigate_enable: bool, default=False
        Enable mitigation
    min_rank_frac: float, default = 0.7
        This fraction will be used to determine the minimum number of valid Eigenvalues
        required for a CPI. min_ev_valid_idx = int(np.floor(min_rank_frac * cpi_len))
        Must be a value within (0,1]
    rx_dynamic_range_db: float, optional, default = 50 dB
        radar platform receiver dynamic range in dB. This is applied as a threshold
        to determine if the Eigenvalue under test is meaningfully significant. If the
        Eigenvalue under test is less than this threshold, it will be viewed as unusable.
    swaths : np.ndarray [int], optional
        Valid subswath samples, dims = (ns, nt, 2) where ns is the number of
        sub-swaths, nt is the number of pulses, and the trailing dimension is
        the [start, stop) indices of the sub-swath.  It's recommended to supply
        this for modes with dithered PRI, where it will be used to normalize
        the sample covariance matrix.
    threshold_method : str, default='ev_slope'
        Detection method: 'ev_slope' or 'max_ev'.
        'ev_slope' uses Eigenvalue Slope Thresholding (TB-wise scalar threshold).
        'max_ev' uses per-CPI adaptive thresholds with fixed margin.
    num_rfi_buffer : int, default=2
        Number of buffer eigenvalue indices to skip after last possible RFI EV before
        starting clean segment interpolation. Used by both 'max_ev' threshold method
        and RFI candidate selection. The clean segment starts at index
        (max_deg_freedom + num_rfi_buffer - 1).
    sig_ev_margin_db : float, default=1.0
        Aggressive safety margin in dB added to the extrapolated estimate for RFI candidates
        in 'max_ev' method. Only used when threshold_method='max_ev'.
    rfi_candidate_tolerance_db : float, default=3.0
        Tolerance in dB for RFI candidate selection. CPIs where
        (actual_EV0 - predicted_clean_EV0) > this value are flagged as RFI candidates
        and receive thresholds. Used by both threshold methods.
    rfi_check : bool, default=True
        Controls RFI-presence characterization. If False, no check is performed
        and all TBs proceed to RFI detection. Otherwise, a TB is screened for
        RFI-like Eigenvalue traits and skipped if RFI is determined to not be
        present, based on dominant Eigenvalue spread and effective rank variability.
    max_ev_spread_thresh_db : float, default=2.0
        Threshold in dB for the spread (std across CPIs) of the dominant Eigenvalues
        in a TB. If the maximum spread among dominant EVs exceeds this value, RFI
        is determined to be present. Only used when rfi_check=True.
    eff_rank_std_thresh : float, default=1.0
        Threshold for the standard deviation of per-CPI effective rank across a TB.
        If exceeded, indicates RFI presence.
        Only used when rfi_check=True.
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

    # Set default threshold_params if not provided
    if threshold_params is None:
        threshold_params = ThresholdParams()

    num_pulses, num_rng_samples = raw_data.shape

    # Override num_rng_samples_blk if use_entire_pulse is True
    if use_entire_pulse:
        num_samples_rng_blk = num_rng_samples

    # If the number of pulses is not an integer multiple of TB size, following
    # operations will ensue. If the number of remaining pulses is greater than
    # the CPI length, additional CPI(s) will be constructed, the very last TB will
    # include the additional CPI(s). The rest of the remaining pulses not enough to
    # construct a full CPI will not be processed. If the number of remaining pulses
    # is less than CPI length, then they will not be processed. In both cases,
    # at most cpi_len-1 number of pulses will be ignored.

    num_cpi = num_pulses // cpi_len
    num_pulses_proc = cpi_len * num_cpi
    num_pulses_tb = cpi_len * num_cpi_per_threshold_block
    num_tb = num_pulses_proc // num_pulses_tb

    # Figure out how many range slices are there
    rng_slices = list(
        slice_gen(num_rng_samples, num_samples_rng_blk, combine_rem=True)
    )
    num_rng_blks = len(rng_slices)

    # RFI EV Count Map
    rfi_ev_count_map = np.zeros(
        (num_cpi, num_rng_blks),
        dtype=np.int16,
    )

    # Boolean Detection Map
    rfi_cpi_detection_map = np.zeros(
        (num_cpi, num_rng_blks),
        dtype=bool,
    )

    # TB Skipped Map (tracks which TBs were skipped by RFI-presence check)
    tb_skipped_map = np.zeros((num_tb, num_rng_blks), dtype=np.bool_)

    # RFI Present Map (RFI-presence characterization result)
    rfi_present_map = np.zeros((num_tb, num_rng_blks), dtype=np.bool_)

    # Excluded pulse count map (per CPI)
    excluded_pulse_count_map = np.zeros((num_cpi, num_rng_blks), dtype=np.int16)

    # Modify raw_data in-place
    if raw_data_mitigated is None:
        raw_data_mitigated = raw_data
    else:
        if raw_data_mitigated.shape != raw_data.shape:
            raise ValueError(
                "Shape mismatch: output mitigated data array must have the same shape"
                " as the input data"
            )

    # Verify min_rank_frac
    if not (0.0 < min_rank_frac <= 1.0):
        raise ValueError(
            f"min_rank_frac must be in (0, 1], got {min_rank_frac}."
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

    # Verify mask_valid: check to see if it is None or populated.
    if (swaths is not None) and (swaths.shape[1] != raw_data.shape[0]):
        raise ValueError("Require same number of rows in swaths and raw_data")

    # Determine a valid Eigenvalue index to estimate minimum-Eigenvalue statistics,
    # ensuring robustness against zero Eigenvalues caused by insufficient valid samples in a CPI.
    min_ev_valid_idx = max(1, int(np.round(min_rank_frac * cpi_len)) - 1)

    # Maximum number of degrees of freedom must be less than min_ev_valid_idx
    if max_deg_freedom >= min_ev_valid_idx:
        warnings.warn(
            f"max_deg_freedom ({max_deg_freedom}) >= min_ev_valid_idx ({min_ev_valid_idx})."
            "This is not recommended since it may lead to insufficient noise subspace.",
            RuntimeWarning
        )

    # Run RFI Detection and Mitigation
    for idx_tb, tb_slow_time in enumerate(slice_gen(num_pulses_proc, num_pulses_tb)):
        # Get valid data mask for all rows in current block.
        if swaths is not None:
            swaths_tb = swaths[:, tb_slow_time, :]
            mask_valid = np.zeros((swaths_tb.shape[1], raw_data.shape[1]), dtype=bool)
            for i in range(mask_valid.shape[0]):
                for start, end in swaths_tb[:, i, :]:
                    mask_valid[i, start:end] = True

        for idx_rng, tb_fast_time in enumerate(rng_slices):
            raw_tb_blk = raw_data[tb_slow_time, tb_fast_time]
            mask_valid_tb = None if swaths is None else mask_valid[:, tb_fast_time]

            (
                rfi_cpi_flag_tb,
                evec_sort_tb,
                diag_valid_array_tb,
                signal_tb_skipped_tb,
                rfi_present_tb,
                num_rfi_candidate_cpi_tb,
            ) = rfi_detect(
                raw_tb_blk,
                cpi_len,
                max_deg_freedom,
                min_ev_valid_idx,
                num_max_trim=num_max_trim,
                num_min_trim=num_min_trim,
                max_num_rfi_ev=max_num_rfi_ev,
                num_rfi_buffer=num_rfi_buffer,
                off_diag_overlap_ratio=off_diag_overlap_ratio,
                diag_valid_ratio=diag_valid_ratio,
                rx_dynamic_range_db=rx_dynamic_range_db,
                mask_valid=mask_valid_tb,
                threshold_method=threshold_method,
                threshold_params=threshold_params,
                rfi_check=rfi_check,
                max_ev_spread_thresh_db=max_ev_spread_thresh_db,
                eff_rank_std_thresh=eff_rank_std_thresh,
                sig_ev_margin_db=sig_ev_margin_db,
                rfi_candidate_tolerance_db=rfi_candidate_tolerance_db,
            )

            # Global CPI indices for this threshold block
            cpi_start = idx_tb * num_cpi_per_threshold_block
            cpi_end = cpi_start + rfi_cpi_flag_tb.shape[0]

            # Count excluded pulses per CPI only for valid/processed TBs
            if not signal_tb_skipped_tb:  # Only count if TB was actually processed
                excluded_pulse_count_tb = count_excluded_pulses_per_cpi(diag_valid_array_tb)
            else:
                # Invalid or skipped TB: set excluded pulse count to zero to avoid confusion
                excluded_pulse_count_tb = np.zeros(diag_valid_array_tb.shape[0], dtype=np.int16)

            excluded_pulse_count_map[cpi_start:cpi_end, idx_rng] = excluded_pulse_count_tb

            # Check if any RFI was detected
            has_rfi = np.any(rfi_cpi_flag_tb)

            # Compute number of CPIs detected with RFI presence
            num_rfi_ev_cpi = np.sum(rfi_cpi_flag_tb, axis=1).astype(np.int16)
            rfi_cpi_count = np.sum(num_rfi_ev_cpi != 0)
            rfi_cpi_count_sum += rfi_cpi_count

            # Populate detection maps
            rfi_ev_count_map[cpi_start:cpi_end, idx_rng] = num_rfi_ev_cpi
            rfi_cpi_detection_map[cpi_start:cpi_end, idx_rng] = num_rfi_ev_cpi > 0

            # Populate diagnostic maps (TB-level)
            tb_skipped_map[idx_tb, idx_rng] = signal_tb_skipped_tb
            rfi_present_map[idx_tb, idx_rng] = rfi_present_tb

            # Run Mitigation:
            if mitigate_enable and has_rfi:
                rfi_mitigate_tb(
                    raw_tb_blk,
                    evec_sort_tb,
                    rfi_cpi_flag_tb,
                    raw_data_mitigated[tb_slow_time, tb_fast_time],
                )
            else:
                raw_data_mitigated[tb_slow_time, tb_fast_time] = raw_tb_blk

    # Percentage of RFI Eigenvalues based on slow-time min EV slope detection
    rfi_likelihood = rfi_cpi_count_sum / (num_cpi * num_rng_blks)

    # Fill the remaining few pulses with original raw data samples
    if num_pulses > num_pulses_proc:
        raw_data_mitigated[num_pulses_proc:] = raw_data[num_pulses_proc:]

    return rfi_likelihood
