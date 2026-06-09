"""
RFI Detection using DUAL-CHECK: Condition Number + Power Stationarity
Enhanced version - uses BOTH condition number variability AND power spread check
"""
import numpy as np
from isce3.signal.compute_evd_cpi import compute_evd_tb
from dataclasses import dataclass, field
from typing import List
import warnings

@dataclass
class ThresholdParams:
    """This dataclass computes the interpolated value of the number
    of sigmas (standard deviation) of the first difference of minimum
    Eigenvalues across threshold block.

    Parameters
    ----------
    x: list of floats
        This is the computed sigma ratio of maximum and minimum Eigenvalue
        first differences. It is a dimensionless figure of merit. Larger value
        of x indicates higher likelihood of RFI presence.
        Defaults are [2.0, 20.0]
    y: list of floats
        Estimated range of number of sigmas of the first difference of
        minimum Eigenvalues across threshold block as a function of input x,
        e.g., smaller x results in larger value of y, therefore relaxing the
        final threshold. The values of x outside of the defined range of y are
        extrapolated.
        Defaults are [5.0, 2.0]
    """
    x: List[float] = field(default_factory=lambda: [2.0, 20.0])
    y: List[float] = field(default_factory=lambda: [5.0, 2.0])

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError("length mismatch: x and y must have the same size")
        if len(self.x) < 2:
            raise ValueError("At least two points are required")

def rfi_detect(
    raw_data,
    cpi_len,
    max_deg_freedom,
    min_ev_valid_idx,
    *,
    num_rfi_buffer=3,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    off_diag_overlap_ratio=0.25,
    diag_valid_ratio=0.20,
    rx_dynamic_range_db=50.0,
    mask_valid=None,
    threshold_method='max_ev',
    threshold_params: ThresholdParams = ThresholdParams(),
    rfi_fig_merit_thresh=1.0,
    bright_target_check=True,
    bt_condition_num_thresh_db=2.0,
    bt_pwr_spread_thresh_db=6.0,
    pwr_ref_percentile=50,
    pwr_upper_percentile=99.5,
    sig_ev_margin_upper_db=3.0,
    sig_ev_margin_lower_db=0.0,
    pcr_range=[0.1, 0.4],
):

    """This wrapper performs Eigenvalue Decomposition of input raw data as well as
    RFI Eigenvalue slope threshold estimation and detection.

    DUAL-CHECK VERSION: Uses condition number check AND power spread check

    Workflow:
    1. Perform EVD on all CPIs in the threshold block
    2. Validate TB (check for sufficient usable eigenvalues)
    3. DUAL-CHECK to filter out bright targets:
       a. Condition number variability check
       b. Power spread check (upper tail spread of diagonal power)
       SKIP if BOTH checks pass (stable eigenvalue structure + low power spread)
    4. If TB fails either check, apply selected threshold method (ev_slope/max_ev)
    5. Detect RFI eigenvalues per CPI based on computed threshold

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    cpi_len: int
        Number of slow-time pulses within a Coherent Processing Interval or CPI
    max_deg_freedom: int
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len to avoid unintended removal of signal data.
    min_ev_valid_idx: int
        Eigenvalue index used by threshold estimation to estimate the slow-time minimum
        Eigenvalue slope. This parameter is also used to validate that the threshold block
        has enough usable eigenvalues for robust sample covaraince estimation of a CPI.
    num_rfi_buffer: int, default=3
        Number of buffer eigenvalue indices to skip after last possible RFI EV before
        starting clean segment interpolation for 'max_ev' method. The clean segment
        starts at index (max_deg_freedom + num_rfi_buffer - 1).
    num_max_trim: int, default=0
        Number of large-value outliers to be trimmed in slow-time minimum Eigenvalues.
    num_min_trim: int, default=0
        Number of small-value outliers to be trimmed in slow-time minimum Eigenvalues
    max_num_rfi_ev: int, default=2
        A detection error (miss) happens when a maximum power RFI emitter contaminates
        multiple consecutive CPIs, resulting in a flat maximum Eigenvalue slope in slow
        time. Hence the standard (STD) deviation of multiple dominant EVs across slow time
        defined by this parameter are compared. The one with the maximum STD is used for RFI
        Eigenvalue first difference computation.
    off_diag_overlap_ratio : float, default=0.25
        Minimum overlap ratio used by gap exclusion covariance estimation
    diag_valid_ratio : float, default=0.20
        Minimum fraction of valid samples required to compute a diagonal term in the
        sample covariance matrix entry R_ii.
    rx_dynamic_range_db: float, default=50.0
        Radar platform receiver dynamic range. This is applied as a threshold
        to determine if the Eigenvalue under test is meaningfully signficant. If the
        Eigenvalue under test is less than this threshold, it will be viewed as unusable.
    mask_valid : np.ndarray bool or None, default=None
        Valid-sample mask with same shape as raw_data If provided, it has the shape of
        [num_pulses x num_rng_samples]. CPI sample covariance matrix will be normalized
        differently by excluding the invalid data gaps.
    threshold_method : str, default='max_ev'
        Detection method: 'ev_slope' or 'max_ev'
    threshold_params: ThresholdParams dataclass object, default=ThresholdParams()
        RFI detection threshold interpolation parameters. The x field defines STD
        ratio between maximum and minimum Eigenvalue slopes (MMES) of the
        slow-time threshold interval. The y field defines the number of sigma (STD)
        from the mean of MMES.
    rfi_fig_merit_thresh : float or None, default=1.0
        Figure of merit threshold for 'max_ev' method. Default of 1.0 ensures
        aggressive RFI detection - checks as many TBs as possible.
    bright_target_check : bool, default=True
        Enable dual bright target rejection check using BOTH condition number variability
        AND power spread. When True, TB is skipped only if BOTH checks pass:
        - Condition number std <= bt_condition_num_thresh_db (stable eigenvalue structure)
        - Upper tail spread <= bt_pwr_spread_thresh_db (low power spread)
        When False, bright target checks are disabled (all TBs are processed).
    bt_condition_num_thresh_db : float, default=2.0
        Condition number std threshold in dB. TBs with std(cond#) <= this value
        pass the condition# check (stable eigenvalue structure).
        Only used when bright_target_check=True.
    bt_pwr_spread_thresh_db : float, default=6.0
        Power spread threshold in dB. TBs with upper tail spread <= this value
        pass the power check (low power spread across slow time).
        Upper tail spread = pwr_upper_percentile - pwr_ref_percentile of diagonal power.
        Only used when bright_target_check=True.
    pwr_ref_percentile : float, default=50
        Reference (baseline) percentile for power spread computation.
        Anchors the lower end of the spread measurement. Only used when bright_target_check=True.
    pwr_upper_percentile : float, default=99.5
        Upper tail percentile for power spread computation (recommended: 98.0 or 99.5).
        99.5 = top 0.5% of power samples, 98.0 = top 2% of power samples.
        Only used when bright_target_check=True.
    sig_ev_margin_upper_db : float, default=3.0
        Conservative safety margin in dB applied when PCR is low (weak RFI).
        Upper bound of the adaptive margin range. Only used when threshold_method='max_ev'.
    sig_ev_margin_lower_db : float, default=0.0
        Aggressive safety margin in dB applied when PCR is high (strong RFI).
        Lower bound of the adaptive margin range. Only used when threshold_method='max_ev'.
    pcr_range : list of 2 floats, default=[0.1, 0.4]
        (lower, upper) bounds of the PCR interpolation range.
        CPIs with PCR <= lower use sig_ev_margin_upper_db (conservative).
        CPIs with PCR >= upper use sig_ev_margin_lower_db (aggressive).
        Only used when threshold_method='max_ev'.

    DUAL-CHECK LOGIC (when bright_target_check=True):
    --------------------------------------------------
    TB is skipped only if BOTH checks pass (AND logic):
        IF (cond# std <= bt_condition_num_thresh_db) AND (upper tail spread <= bt_pwr_spread_thresh_db):
            SKIP TB (Clean or Bright Target)
        ELSE:
            PROCEED with RFI detection

    When bright_target_check=False:
        All TBs proceed to RFI detection (no bright target filtering)

    Returns
    --------
    rfi_cpi_flag_array: 2D array of bool, [num_cpi x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as either RFI or signal.
        1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
    eig_vec_sort: 3D array of complex, [num_cpi x cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on indices of sorted Eigenvalues
    """
    num_pulses = raw_data.shape[0]

    # Verify total number of pulses is greater than number of pulses per CPI
    if num_pulses < cpi_len:
        raise ValueError(
            "Total number of pulses must be greater or equal to number of pulses per single CPI."
        )

    (
        eig_val_sort_array,
        eig_vec_sort_array,
        diag_power_array,
        diag_valid_array,
        tb_is_valid,
    ) = compute_evd_tb(
        raw_data,
        cpi_len=cpi_len,
        mask_valid=mask_valid,
        off_diag_overlap_ratio=off_diag_overlap_ratio,
        diag_valid_ratio=diag_valid_ratio,
        min_ev_valid_idx=min_ev_valid_idx,
        rx_dynamic_range_db=rx_dynamic_range_db,
    )

    # If any CPI within a threshold block is determined to be invalid
    # Then skip threshold computation for this block by setting rfi_cpi_flag_array
    # to all zeros
    if not tb_is_valid:
        num_cpi = eig_val_sort_array.shape[0]
        rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
        fig_merit_detect_tb = 0

        return (
            rfi_cpi_flag_array,
            eig_vec_sort_array,
            diag_power_array,
            diag_valid_array,
            fig_merit_detect_tb,
        )

    # DUAL-CHECK: Condition number variability + Power spread
    # Both checks must pass to skip TB (conservative approach)
    if bright_target_check:
        # Check 1: Condition number variability
        condition_number_std_db = compute_tb_condition_number_std(
            eig_val_sort_array,
            min_ev_valid_idx,
        )
        cond_num_passes = (np.isfinite(condition_number_std_db) and
                          condition_number_std_db <= bt_condition_num_thresh_db)

        # Check 2: Power spread (upper tail spread)
        upper_tail_spread_db = compute_tb_upper_tail(
            diag_power_array,
            diag_valid_array,
            pwr_ref_percentile=pwr_ref_percentile,
            pwr_upper_percentile=pwr_upper_percentile,
        )
        power_stat_passes = (np.isfinite(upper_tail_spread_db) and
                            upper_tail_spread_db <= bt_pwr_spread_thresh_db)

        # DECISION: SKIP only if BOTH checks pass
        if cond_num_passes and power_stat_passes:
            # Clean or Bright Target: stable structure + stationary power
            num_cpi = eig_val_sort_array.shape[0]
            rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
            fig_merit_detect_tb = 0

            return (
                rfi_cpi_flag_array,
                eig_vec_sort_array,
                diag_power_array,
                diag_valid_array,
                fig_merit_detect_tb,
            )

    # TB has potential RFI (at least one check failed): proceed with detection
    if threshold_method == 'ev_slope':
        # Estimate a single threshold for all CPIs
        detect_threshold, fig_merit_detect_tb = threshold_estimate_evd(
            eig_val_sort_array,
            num_max_trim,
            num_min_trim,
            max_num_rfi_ev,
            min_ev_valid_idx,
            threshold_params,
        )
    elif threshold_method == 'max_ev':
        detect_threshold, fig_merit_detect_tb = threshold_estimate_max_ev(
            eig_val_sort_array,
            max_deg_freedom=max_deg_freedom,
            num_rfi_buffer=num_rfi_buffer,
            num_max_trim=num_max_trim,
            num_min_trim=num_min_trim,
            max_num_rfi_ev=max_num_rfi_ev,
            min_ev_valid_idx=min_ev_valid_idx,
            rfi_fig_merit_thresh=rfi_fig_merit_thresh,
            sig_ev_margin_upper_db=sig_ev_margin_upper_db,
            sig_ev_margin_lower_db=sig_ev_margin_lower_db,
            pcr_range=pcr_range,
        )

        # FoM below threshold: eigenvalue structure does not indicate RFI presence.
        # Return early with no detections — distinct from invalid TB and bright target skip.
        if not np.isfinite(detect_threshold).all():
            num_cpi = eig_val_sort_array.shape[0]
            rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
            return (
                rfi_cpi_flag_array,
                eig_vec_sort_array,
                diag_power_array,
                diag_valid_array,
                fig_merit_detect_tb,
            )
    else:
        raise ValueError(f"Unsupported threshold method: {threshold_method}")

    # Detect RFI Eigenvalues of each CPI based on input detection threshold
    rfi_cpi_flag_array = rfi_detect_evd_tb(
        eig_val_sort_array, detect_threshold, max_deg_freedom, threshold_method,
    )

    return rfi_cpi_flag_array, eig_vec_sort_array, diag_power_array, diag_valid_array, fig_merit_detect_tb

def threshold_estimate_max_ev(
    eig_val_sort_array,
    max_deg_freedom=4,
    num_rfi_buffer=3,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    min_ev_valid_idx=10,
    rfi_fig_merit_thresh=1.0,
    sig_ev_margin_upper_db=3.0,
    sig_ev_margin_lower_db=0.0,
    pcr_range=[0.1, 0.4],
):
    """Estimate signal max eigenvalue threshold using max_ev detection algorithm.

    Algorithm:
    1. Compute FoM using EST method
    2. Check if FoM indicates potential RFI
    3. If RFI confirmed (condition number check already performed in rfi_detect),
       compute signal_ev_max_estimate by:
       - Computing adaptive margin per CPI based on Principal Component Ratio (PCR)
       - PCR = ev[0] / sum(ev[1:]) measures dominance of first eigenvalue
       - High PCR (strong RFI) -> reduced margin for aggressive detection
       - Low PCR (weak RFI) -> full margin for conservative detection
       - Using clean eigenvalue segment from (max_deg_freedom+1) to min_ev_valid_idx
       - Performing linear regression across this segment for each CPI independently
       - Extrapolating each CPI's fitted line to index 0 to estimate maximum signal eigenvalue
       - Adding adaptive margin (varies per CPI based on PCR)
       - Returning per-CPI thresholds (no averaging across CPIs)

    Parameters
    ----------
    eig_val_sort_array : 2D array of float, [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    max_deg_freedom : int, default=4
        Maximum number of independent RFI emitters designed to be detected.
        Eigenvalue indices [0, ..., max_deg_freedom-1] can potentially be RFI.
    num_rfi_buffer : int, default=3
        Number of buffer eigenvalue indices to skip after last possible RFI EV
        before starting clean segment interpolation.
        start_idx = max_deg_freedom + num_rfi_buffer - 1.
    num_max_trim : int, default=0
        Number of large-value outliers to trim for FoM computation
    num_min_trim : int, default=0
        Number of small-value outliers to trim for FoM computation
    max_num_rfi_ev : int, default=2
        Maximum number of dominant EVs to check for FoM computation
    min_ev_valid_idx : int, default=10
        Eigenvalue index for noise floor estimation
    rfi_fig_merit_thresh : float, default=1.0
        Figure of merit threshold. TB proceeds to threshold estimation only if
        fig_merit >= rfi_fig_merit_thresh. Lower values allow more TBs to be checked.
    sig_ev_margin_upper_db : float, default=3.0
        Conservative safety margin in dB applied when PCR is low (weak RFI).
        Upper bound of the adaptive margin interpolation range.
    sig_ev_margin_lower_db : float, default=0.0
        Aggressive safety margin in dB applied when PCR is high (strong RFI).
        Lower bound of the adaptive margin interpolation range.
        Higher PCR indicates stronger RFI dominance, allowing more aggressive detection.
    pcr_range : list of 2 floats, default=[0.1, 0.4]
        (lower, upper) bounds of the PCR interpolation range.
        CPIs with PCR <= lower use sig_ev_margin_upper_db (conservative).
        CPIs with PCR >= upper use sig_ev_margin_lower_db (aggressive).

    Returns
    -------
    detect_threshold : float or ndarray
        Per-CPI signal max eigenvalue estimates in dB if RFI detected, else np.inf.
        Returns a 1D array of length num_cpi, where each element is the threshold
        for the corresponding CPI.
    fig_merit : float
        Figure of merit value computed
    """

    # Step 1: Compute FoM using EST method
    _, fig_merit = threshold_estimate_evd(
        eig_val_sort_array,
        num_max_trim=num_max_trim,
        num_min_trim=num_min_trim,
        max_num_rfi_ev=max_num_rfi_ev,
        min_ev_valid_idx=min_ev_valid_idx,
        threshold_params=ThresholdParams(x=[2.0, 20.0], y=[5.0, 2.0]),
    )

    # Step 2: Check if FoM indicates potential RFI
    if fig_merit < rfi_fig_merit_thresh:
        # Low FoM, no RFI detected
        return np.inf, fig_merit

    # Step 3: RFI confirmed (high FoM, and power spread already checked in rfi_detect)
    # Compute signal_ev_max_estimate using clean eigenvalue segment
    eps = np.finfo(np.float64).tiny
    eig_val = np.maximum(np.real(eig_val_sort_array), eps)
    eig_val_sort_db_array = 10.0 * np.log10(eig_val)

    # Define clean eigenvalue segment: from (max_deg_freedom + num_rfi_buffer - 1) to min_ev_valid_idx
    # max_deg_freedom indices [0, ..., max_deg_freedom-1] can be RFI
    # Add num_rfi_buffer indices as buffer, then start at max_deg_freedom + num_rfi_buffer - 1
    # This segment should be free from RFI contamination and usable for interpolation
    start_idx = max_deg_freedom + num_rfi_buffer - 1
    end_idx = min_ev_valid_idx

    # Validate that clean segment has sufficient points
    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid clean eigenvalue segment: start_idx ({start_idx}) >= end_idx ({end_idx}). "
            f"Require min_ev_valid_idx ({min_ev_valid_idx}) > max_deg_freedom + num_rfi_buffer - 1 "
            f"({max_deg_freedom} + {num_rfi_buffer} - 1 = {start_idx})."
        )

    # Ensure we have at least 2 points for linear fit
    if end_idx <= start_idx + 1:
        # Fallback: segment too short for regression (< 2 points)
        # Use simple slope estimate from last 2 eigenvalues before min_ev_valid_idx
        # Returns a scalar threshold (averaged across all CPIs)
        slopes = (eig_val_sort_db_array[:, min_ev_valid_idx - 1] -
                  eig_val_sort_db_array[:, min_ev_valid_idx])
        slope_avg = np.mean(slopes)
        ev_min_avg = np.mean(eig_val_sort_db_array[:, min_ev_valid_idx])
        signal_ev_max_estimate = slope_avg * min_ev_valid_idx + ev_min_avg + sig_ev_margin_upper_db
    else:
        # Use linear regression across clean eigenvalue segment for each CPI
        # Generate one threshold per CPI (no averaging)
        num_cpi = eig_val_sort_db_array.shape[0]
        signal_ev_max_estimate = np.zeros(num_cpi)

        for i in range(num_cpi):
            # Compute adaptive margin based on Principal Component Ratio (PCR)
            # PCR = ev[0] / sum(ev[1:]) measures dominance of first eigenvalue
            # High PCR (strong RFI) -> reduce margin for more aggressive detection
            # Low PCR (weak RFI or clutter) -> keep full margin for conservative detection
            ev_dominant = eig_val[i, 0]
            ev_sum_rest = np.sum(eig_val[i, 1:])

            if ev_sum_rest > eps:
                pcr = ev_dominant / ev_sum_rest
            else:
                pcr = 0.0  # Fallback if sum is zero

            # Adaptive margin interpolation:
            # PCR < pcr_range[0]: use upper margin (conservative)
            # PCR in pcr_range: interpolate between upper and lower margin
            # PCR > pcr_range[1]: use lower margin (aggressive)
            pcr_clamped = np.clip(pcr, pcr_range[0], pcr_range[1])
            adaptive_margin_db = np.interp(
                pcr_clamped,
                pcr_range,                                     # PCR range
                [sig_ev_margin_upper_db, sig_ev_margin_lower_db]  # Margin range
            )

            # Extract clean eigenvalue segment for this CPI
            ev_segment = eig_val_sort_db_array[i, start_idx:end_idx]
            indices = np.arange(start_idx, end_idx)

            # Linear regression: ev_db = slope * index + intercept
            # Using polyfit (degree=1 for linear)
            coeffs = np.polyfit(indices, ev_segment, deg=1)
            intercept = coeffs[1]  # dB at index 0

            # Extrapolate to index 0 for this CPI with adaptive margin
            signal_ev_max_estimate[i] = intercept + adaptive_margin_db

    return signal_ev_max_estimate, fig_merit

def threshold_estimate_evd(
    eig_val_sort_array,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    min_ev_valid_idx=10,
    threshold_params: ThresholdParams = ThresholdParams(),
):
    """Perform data-centric thresholding algorithm: "Slow-Time Eigenvalue Slope
    Thresholding "(ST-EST)"[1]_ based on the assumption that first difference of
    minimum Eigenvalue across slow time is an estimate of signal power variation
    as if there is no RFI"

    Algorithm Overview for applying ST-EST on one raw data block:
    1. Remove a specified number of outliers in maximum and minimum Eigenvalues
    2. Compute slow-time standard deviation (STD) of maximum Eigenvalue slope.
    3. Compute slow-time standard deviation (STD) of minimum Eigenvalue slope.
    4. Compute STD ratio of maximum and minimum Eigenvalue slopes (SRMMES).
       SRMMES will be applied as input to a linear interpolator to derive
       the detection threshold in dB / Eigenvalue Index.
    5. Apply SRMMES in step #4 as input to a linear interpolator
       defined by threshold_params. The final detection threshold tau is a
       function of sigma and mu (mean) computed in step #4 such that:
       tau = alpha * sigma(min_EV_slope) + mu(min_EV_slope)
       where alpha is the curve-fitted value.

    Parameters
    ----------
    eig_val_sort_array: 2D array of float, [num_cpi x cpi_len]
        Sorted Eigenvalues in descending order in linear units of all CPIs in raw data
        Eigenvalues will subsequently converted into dB for threshold estimation.
    num_max_trim: int, default = 0
        Number of large value outliers to be trimmed in slow-time minimum Eigenvalues.
    num_min_trim: int, default = 0
        Number of small value outlliers to be trimmed in slow-time minimum Eigenvalues
    max_num_rfi_ev: int, default = 2
        A detection error (miss) happens when a maximum power RFI emitter contaminates
        multiple consecutive CPIs, resulting in a flat maximum Eigenvalue slope in slow
        time. Hence the standard (STD) deviation of multiple dominant EVs across slow time
        defined by this parameter are compared. The one with the maximum STD is used for RFI
        Eigenvalue first difference computation.
    min_ev_valid_idx: int
        Eigenvalue index used by threshold estimation to estimate the slow-time minimum
        Eigenvalue slope. This parameter is also used to validate that the threshold block
        has enough usable eigenvalues for robust sample covaraince estimation of a CPI.
    threshold_params: ThresholdParams dataclass object, default=ThresholdParams()
        RFI detection threshold interpolation parameters

    Returns
    -------
    detect_threshold: float
        RFI detection threshold used by EVD detection algorithm in dB/Eigenvalue index
        All CPIs shares a common threshold.

    References
    ----------
    ..[1] Bo Huang, Heresh Fattahi, Hirad Ghaemi, Brian Hawkins, Geoffrey Gunter,
    "Radio Frequency Interference Detection and Mitigation of NISAR DATA using
    Slow Time Eigenvalue Decomposition", IGARSS 2023.'
    """

    if max_num_rfi_ev < 1:
        raise ValueError('max_num_rfi_ev" shall be larger than zero.')

    # Max power Eigenvalue (RFI) can appear in multiple consecutive CPIs which results
    # in zero (flat) slope across slow time, compute slow-time standard deviation (STD)
    # of top N max power Eigenvalues, default=2, and use the one with highest STD.
    eval_sort_max_db = 10 * np.log10(np.abs(eig_val_sort_array[:, 0:max_num_rfi_ev]))
    eval_sort_max_std = np.std(eval_sort_max_db, axis=0)

    # Find slow-time Principal Component Eigenvalue array with largest STD
    ev_max_std_idx = np.argmax(eval_sort_max_std)
    ev_max_db = eval_sort_max_db[:, ev_max_std_idx]

    # For Dithered PRF mode, min_ev_valid_idx is selected to avoid zeros in the tail
    # of the Eigenvalue spectrum due to invalid data gaps for each pulse.
    ev_min_db = 10 * np.log10(np.abs(eig_val_sort_array[:, min_ev_valid_idx]))

    # Remove possible outliers in max and min Eigenvalues without reordering.
    if num_min_trim > 0:
        ev_min_trim_idx = np.argsort(ev_min_db)[:num_min_trim]
        ev_min_db = np.delete(ev_min_db, ev_min_trim_idx)

    if num_max_trim > 0:
        ev_max_trim_idx = np.argsort(ev_max_db)[-num_max_trim:]
        ev_max_db = np.delete(ev_max_db, ev_max_trim_idx)

    # Compute STD of the slope of max and min Eigenvalues
    ev_slope_max = np.diff(ev_max_db)
    ev_slope_min = np.diff(ev_min_db)

    ev_slope_max_std = ev_slope_max.std()
    ev_slope_min_std = ev_slope_min.std()
    ev_slope_min_mean = ev_slope_min.mean()

    # Max(dB)/min(dB) Eigenvalue slope STD ratio: indicator of RFI severity and
    # input to RFI linear interpolator for final detection threshold
    std_ratio_ev_slope = ev_slope_max_std / ev_slope_min_std

    # Threshold interpolation parameters
    std_ratio = threshold_params.x
    threshold_sigma = threshold_params.y

    num_sigma = np.interp(std_ratio_ev_slope, std_ratio, threshold_sigma)

    detect_threshold = ev_slope_min_mean + num_sigma * ev_slope_min_std

    return detect_threshold, std_ratio_ev_slope


def rfi_detect_evd(
    eig_val_db_slope,
    detect_threshold,
    max_deg_freedom=8,
):
    """Perform RFI detection of Eigenvalues within a CPI based on input detection
    threshold in dB/Eigenvalue index. The threshold is set to be a negative value.
    If the magnitude of an Eigenvalue slope exceeds this threshold, then it is identified
    as RFI.

    Parameters
    ----------
    eig_val_db_slope: 1D array of float
        Eigenvalue slope (first difference of Eigenvalues)
    detect_threshold: float
        A positive RFI detection threshold used by EVD detection algorithm to identify
        RFI Eigenvalue slope valules.
    max_deg_freedom: int, default=8
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len.

    Returns
    -------
    sig_ev_idx_start: int
        Start index of signal Eigenvalues
    """

    sig_ev_idx_start = 0
    rfi_ev_idx = np.where(eig_val_db_slope[:max_deg_freedom] < -detect_threshold)[0]

    if rfi_ev_idx.size:
        sig_ev_idx_start = rfi_ev_idx[-1] + 1

    return sig_ev_idx_start


def rfi_detect_evd_tb(
    eig_val_sort_array,
    detect_threshold,
    max_deg_freedom=4,
    threshold_method='max_ev',
):
    """Wrapper function which performs RFI detection of data within a Threshold Block (TB)
    one CPI at a time based on input detection threshold.

    Supports two detection methods:
    - 'ev_slope': Eigenvalue Slope Thresholding (slope-based detection)
    - 'max_ev': EV Maximum Estimation (absolute eigenvalue threshold from noise extrapolation)

    Parameters
    ----------
    eig_val_sort_array: 2D array of float, [num_cpi x cpi_len]
        Sorted Eigenvalues in descending order of all CPIs in raw data
    detect_threshold: float or ndarray
        RFI detection threshold. For 'ev_slope': positive slope threshold in dB/index.
        For 'max_ev': absolute eigenvalue threshold in dB.
        Can be a scalar (same threshold for all CPIs) or a 1D array of length num_cpi
        (one threshold per CPI).
    max_deg_freedom: int, default = 4
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len.
    threshold_method: str, default='max_ev'
        Detection method: 'ev_slope' or 'max_ev'

    Returns
    -------
    rfi_cpi_flag_array: 2D array of bool, [num_cpi x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as either RFI or signal.
        True = RFI Eigenvalue index; False = Signal Eigenvalue index
    """

    # Number of pulses, CPI length, and number of range blocks in a CPI
    num_cpi, cpi_len = eig_val_sort_array.shape

    # Convert threshold to array for uniform handling
    detect_threshold_arr = np.atleast_1d(detect_threshold)

    # Validate threshold array shape
    if detect_threshold_arr.size == 1:
        # Scalar threshold - broadcast to all CPIs
        detect_threshold_arr = np.full(num_cpi, detect_threshold_arr[0])
    elif detect_threshold_arr.size != num_cpi:
        raise ValueError(f"Threshold array size ({detect_threshold_arr.size}) must match num_cpi ({num_cpi})")

    # Validate threshold values
    if threshold_method == 'ev_slope':
        # Ensure detection threshold is a positive value
        if np.any(~np.isfinite(detect_threshold_arr)) or np.any(detect_threshold_arr <= 0):
            warnings.warn("Warning: Non-positive detection threshold. Skipping TB detection.")
            return np.zeros((num_cpi, cpi_len), dtype=np.bool_)
    elif threshold_method == 'max_ev':
        if np.any(~np.isfinite(detect_threshold_arr)):
            warnings.warn(f"Warning: Non-finite {threshold_method.upper()} detection threshold. Skipping TB detection.")
            return np.zeros((num_cpi, cpi_len), dtype=np.bool_)
    else:
        raise ValueError(f"Unsupported threshold_method: {threshold_method}")

    # Maximum number of degrees of freedom must be less than cpi_len
    if max_deg_freedom >= cpi_len:
        raise ValueError(
            "Max number of deg. of freedom must be less than number of pulses in a CPI."
        )

    # RFI flag for each eigenvalue index in all CPIs: RFI=1, signal=0
    rfi_cpi_flag_array = np.ones((num_cpi, cpi_len), dtype=np.bool_)

    # Compute Eigenvalue in dB for all CPIs
    eig_val_sort_db_array = 10 * np.log10(np.abs(eig_val_sort_array))

    if threshold_method == 'ev_slope':
        eig_val_db_slope_array = np.diff(eig_val_sort_db_array, axis=1)
        for idx_cpi in range(num_cpi):
            eig_val_db_slope = eig_val_db_slope_array[idx_cpi]

            # Use per-CPI threshold
            sig_ev_idx_start = rfi_detect_evd(eig_val_db_slope, detect_threshold_arr[idx_cpi], max_deg_freedom)

            # Sets signal eigenvalue indices to False
            rfi_cpi_flag_array[idx_cpi, sig_ev_idx_start:] = False
    else:
        # 'max_ev' uses absolute eigenvalue comparison
        for idx_cpi in range(num_cpi):
            eig_val_db_valid = eig_val_sort_db_array[idx_cpi, :max_deg_freedom]

            # Use per-CPI threshold
            rfi_ev_idx = np.where(eig_val_db_valid > detect_threshold_arr[idx_cpi])[0]

            if rfi_ev_idx.size:
                sig_ev_idx_start = rfi_ev_idx[-1] + 1

                # Sets signal eigenvalue indices to zero
                rfi_cpi_flag_array[idx_cpi, sig_ev_idx_start:] = False
            else:
                rfi_cpi_flag_array[idx_cpi, :] = False

    return rfi_cpi_flag_array

def compute_tb_upper_tail(
    diag_power_array,
    diag_valid_array,
    pwr_ref_percentile=50,
    pwr_upper_percentile=99.5,
    eps=1e-12,
):
    pwr = np.asarray(diag_power_array)[diag_valid_array]

    pwr = pwr[np.isfinite(pwr)]

    if pwr.size == 0:
        return np.nan

    # Convert to dB
    pwr_db = 10 * np.log10(np.maximum(pwr, eps))

    # Remove non-finite values
    pwr_db = pwr_db[np.isfinite(pwr_db)]

    # Compute upper-tail spread
    baseline = np.percentile(pwr_db, pwr_ref_percentile)
    upper = np.percentile(pwr_db, pwr_upper_percentile)

    upper_tail_db = upper - baseline

    return upper_tail_db

def compute_tb_condition_number_std(
    eig_val_sort_array,
    min_ev_valid_idx,
    eps=1e-12,
):
    """Compute standard deviation of condition numbers across CPIs in a TB.

    Parameters
    ----------
    eig_val_sort_array : 2D array [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    min_ev_valid_idx : int
        Index of minimum valid eigenvalue
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    condition_number_std_db : float
        Standard deviation of condition numbers in dB
    """
    num_cpi = eig_val_sort_array.shape[0]
    condition_numbers_db = np.zeros(num_cpi)

    for i in range(num_cpi):
        ev_max = np.maximum(eig_val_sort_array[i, 0], eps)
        ev_min = np.maximum(eig_val_sort_array[i, min_ev_valid_idx], eps)
        condition_numbers_db[i] = 10 * np.log10(ev_max) - 10 * np.log10(ev_min)

    # Remove non-finite values
    condition_numbers_db = condition_numbers_db[np.isfinite(condition_numbers_db)]

    if condition_numbers_db.size == 0:
        return np.nan

    condition_number_std_db = np.std(condition_numbers_db)

    return condition_number_std_db
