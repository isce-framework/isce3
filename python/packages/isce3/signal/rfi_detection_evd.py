"""
Performs RFI detection of input data using Slow-Time Eigenvalue Slope
Thresholding algorithm (ST-EST).
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
    num_rfi_buffer=2,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    off_diag_overlap_ratio=0.25,
    diag_valid_ratio=0.20,
    rx_dynamic_range_db=50.0,
    mask_valid=None,
    threshold_method='max_ev',
    threshold_params: ThresholdParams = None,
    rfi_check=True,
    max_ev_spread_thresh_db=2.0,
    eff_rank_std_thresh=1.0,
    sig_ev_margin_db=1.0,
    rfi_candidate_tolerance_db=3.0,
):

    """This wrapper performs Eigenvalue Decomposition of input raw data as well as
    RFI Eigenvalue slope threshold estimation and detection.

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
        Detection method: 'ev_slope' or 'max_ev'.
        'max_ev' uses per-CPI adaptive thresholds with fixed margin.
    threshold_params: ThresholdParams dataclass object or None, default=None
        RFI detection threshold interpolation parameters. If None, default is
        ThresholdParams() (x=[2.0, 20.0], y=[5.0, 2.0]) for 'ev_slope'.
        The x field defines STD ratio between maximum and minimum Eigenvalue
        slopes (MMES) of the slow-time threshold interval. The y field defines
        the number of sigma (STD) from the mean of MMES.
    num_rfi_buffer: int, default=2
        Number of buffer eigenvalue indices to skip after last possible RFI EV before
        starting clean segment interpolation. The clean segment starts at index
        (max_deg_freedom + num_rfi_buffer - 1). Used by both 'ev_slope' and 'max_ev' methods.
    sig_ev_margin_db : float, default=1.0
        Safety margin in dB added to the extrapolated estimate for 'max_ev' method.
        Only used when threshold_method='max_ev'.
    rfi_candidate_tolerance_db : float, default=3.0
        Tolerance in dB for RFI candidate selection. CPIs where
        (actual_EV0 - predicted_clean_EV0) > this value are flagged as RFI candidates.
        Used by both 'ev_slope' and 'max_ev' methods.
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

    RFI-PRESENCE CHARACTERIZATION LOGIC:
    -------------------------------------
    rfi_check=False:
        All TBs proceed to RFI detection (no RFI-trait screening).
    rfi_check=True:
        rfi_present = True  if max_ev_spread_db > max_ev_spread_thresh_db
        rfi_present = True  elif eff_rank_std > eff_rank_std_thresh
        rfi_present = False otherwise
        SKIP TB if rfi_present is False

    Returns
    --------
    rfi_cpi_flag_array: 2D array of bool, [num_cpi x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as either RFI or signal.
        1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
    eig_vec_sort: 3D array of complex, [num_cpi x cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on indices of sorted Eigenvalues
    diag_valid_array: 2D array of bool, [num_cpi x cpi_len]
        Per-pulse validity array. True indicates the pulse had sufficient valid samples
        to compute its diagonal covariance entry (R_ii). False indicates the pulse was
        excluded due to insufficient valid samples (below diag_valid_ratio threshold).
    signal_tb_skipped: bool
        True if the TB was skipped because RFI-presence characterization found no RFI traits.
        False in all other cases (invalid TB or normal detection path).
    rfi_present: bool
        True if the RFI-presence characterization determined the TB exhibits
        RFI-like Eigenvalue traits (or if rfi_check is False). False if the check
        determined the TB does not exhibit RFI traits (TB skipped), or if the TB is invalid.
    num_rfi_candidate_cpi: int or nan
        Number of CPIs flagged as RFI candidates by rfi_cpi_candidate_sel().
        These CPIs receive detailed eigenvalue detection via the chosen threshold method.
        NaN only when the TB is skipped/invalid before candidate selection is reached.
    """
    num_pulses = raw_data.shape[0]

    # Verify total number of pulses is greater than number of pulses per CPI
    if num_pulses < cpi_len:
        raise ValueError(
            "Total number of pulses must be greater or equal to number of pulses per single CPI."
        )

    # Select method-specific threshold_params default if not explicitly provided
    if threshold_params is None:
        threshold_params = ThresholdParams()

    # Need to validate sample covariance rank
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

    # Default rfi_present: assume RFI may be present unless the check determines otherwise
    rfi_present = True
    # Default RFI candidate CPI count: only populated for 'max_ev' threshold estimation
    num_rfi_candidate_cpi = np.nan

    # If any CPI within a threshold block is determined to be invalid
    # Then skip threshold computation for this block by setting rfi_cpi_flag_array
    # to all zeros
    if not tb_is_valid:
        num_cpi = eig_val_sort_array.shape[0]
        rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
        signal_tb_skipped = False  # invalid TB, not an RFI-presence skip
        rfi_present = False  # invalid TB: no RFI determination possible

        return (
            rfi_cpi_flag_array,
            eig_vec_sort_array,
            diag_valid_array,
            signal_tb_skipped,
            rfi_present,
            num_rfi_candidate_cpi,
        )

    # RFI-presence characterization: controlled by rfi_check
    if rfi_check:
        rfi_present, _, _ = rfi_check_tb(
            eig_val_sort_array,
            max_deg_freedom,
            min_ev_valid_idx,
            max_ev_spread_thresh_db=max_ev_spread_thresh_db,
            eff_rank_std_thresh=eff_rank_std_thresh,
        )

        if not rfi_present:
            # No RFI-like Eigenvalue traits detected: stable structure
            num_cpi = eig_val_sort_array.shape[0]
            rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
            signal_tb_skipped = True  # TB skipped: RFI-presence characterization found no RFI traits

            return (
                rfi_cpi_flag_array,
                eig_vec_sort_array,
                diag_valid_array,
                signal_tb_skipped,
                rfi_present,
                num_rfi_candidate_cpi,
            )

    # TB has potential RFI: proceed with detection
    # First, identify which CPIs are RFI candidates
    # This mask will be used by both threshold methods
    cpi_rfi_mask, num_rfi_candidate_cpi = rfi_cpi_candidate_sel(
        eig_val_sort_array,
        max_deg_freedom,
        num_rfi_buffer,
        min_ev_valid_idx,
        rfi_candidate_tolerance_db=rfi_candidate_tolerance_db,
    )

    if threshold_method == 'ev_slope':
        # Estimate a single threshold for all CPIs (TB-wise)
        # Will be applied only to RFI candidate CPIs via cpi_rfi_mask
        detect_threshold = threshold_estimate_evd(
            eig_val_sort_array,
            num_max_trim,
            num_min_trim,
            max_num_rfi_ev,
            min_ev_valid_idx,
            threshold_params,
        )
    elif threshold_method == 'max_ev':
        # Estimate per-CPI thresholds
        # Will be applied only to RFI candidate CPIs via cpi_rfi_mask
        detect_threshold = threshold_estimate_max_ev(
            eig_val_sort_array,
            max_deg_freedom=max_deg_freedom,
            num_rfi_buffer=num_rfi_buffer,
            min_ev_valid_idx=min_ev_valid_idx,
            sig_ev_margin_db=sig_ev_margin_db,
        )
    else:
        raise ValueError(f"Unsupported threshold method: {threshold_method}")

    # Detect RFI Eigenvalues of each CPI based on input detection threshold
    rfi_cpi_flag_array = rfi_detect_evd_tb(
        eig_val_sort_array,
        detect_threshold,
        max_deg_freedom,
        threshold_method,
        cpi_rfi_mask=cpi_rfi_mask,
    )

    signal_tb_skipped = False  # TB processed normally (not skipped)
    return (
        rfi_cpi_flag_array,
        eig_vec_sort_array,
        diag_valid_array,
        signal_tb_skipped,
        rfi_present,
        num_rfi_candidate_cpi,
    )


def compute_tb_max_ev_spread(
    eig_val_sort_array,
    max_deg_freedom,
    eps=1e-12,
):
    """Compute the maximum spread (std across CPIs) among dominant Eigenvalues in a TB.

    Parameters
    ----------
    eig_val_sort_array : 2D array [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    max_deg_freedom : int
        Number of dominant Eigenvalue indices considered (indices [0, max_deg_freedom))
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    max_ev_spread : float
        Maximum standard deviation (in dB) among the dominant Eigenvalues across CPIs
    """
    eig_val_db_array = 10.0 * np.log10(
        np.maximum(np.real(eig_val_sort_array), eps)
    )

    # Use dominant EVs only
    dominant_ev_db = eig_val_db_array[:, :max_deg_freedom]

    # Standard deviation of each EV index across CPIs
    ev_std = np.std(dominant_ev_db, axis=0)

    # Maximum spread among dominant EVs
    max_ev_spread = np.max(ev_std)

    return max_ev_spread


def compute_tb_eff_rank_std(
    eig_val_sort_array,
    min_ev_valid_idx,
    eps=1e-12,
):
    """Compute standard deviation of per-CPI effective rank across a TB.

    Effective rank is computed via Shannon entropy of the normalized eigenvalue
    distribution over the valid eigenvalue indices [0, min_ev_valid_idx].

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
    eff_rank_std : float
        Standard deviation of effective rank across CPIs in the TB
    """
    num_cpi = eig_val_sort_array.shape[0]
    eff_rank_array = np.zeros(num_cpi)

    for i in range(num_cpi):
        ev = np.maximum(np.real(eig_val_sort_array[i, :min_ev_valid_idx + 1]), eps)
        p = ev / np.sum(ev)
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        eff_rank_array[i] = np.exp(entropy)

    eff_rank_array = eff_rank_array[np.isfinite(eff_rank_array)]

    if eff_rank_array.size == 0:
        return np.nan

    eff_rank_std = np.std(eff_rank_array)

    return eff_rank_std


def rfi_check_tb(
    eig_val_sort_array,
    max_deg_freedom,
    min_ev_valid_idx,
    max_ev_spread_thresh_db=2.0,
    eff_rank_std_thresh=1.0,
):
    """Check if a threshold block (TB) exhibits RFI-like traits.

    This function performs RFI-presence characterization by examining:
    1. Spread of dominant eigenvalues across CPIs
    2. Effective rank variability

    Parameters
    ----------
    eig_val_sort_array : 2D array [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    max_deg_freedom : int
        Number of dominant Eigenvalue indices to consider
    min_ev_valid_idx : int
        Index of minimum valid eigenvalue
    max_ev_spread_thresh_db : float, default=2.0
        Threshold in dB for the spread (std across CPIs) of dominant Eigenvalues.
        If the maximum spread exceeds this value, RFI is determined to be present.
    eff_rank_std_thresh : float, default=1.0
        Threshold for the standard deviation of per-CPI effective rank.
        If exceeded, indicates RFI presence.

    Returns
    -------
    rfi_present : bool
        True if RFI-like traits are detected, False otherwise
    max_ev_spread_db : float
        Maximum spread (std) of dominant eigenvalues in dB
    eff_rank_std : float
        Standard deviation of effective rank across CPIs
    """
    # Compute max eigenvalue spread
    max_ev_spread_db = compute_tb_max_ev_spread(
        eig_val_sort_array,
        max_deg_freedom,
    )

    # Compute effective rank variability
    eff_rank_std = compute_tb_eff_rank_std(
        eig_val_sort_array,
        min_ev_valid_idx,
    )

    # Cascaded decision logic
    if max_ev_spread_db > max_ev_spread_thresh_db:
        rfi_present = True
    elif eff_rank_std > eff_rank_std_thresh:
        rfi_present = True
    else:
        rfi_present = False

    return rfi_present, max_ev_spread_db, eff_rank_std


def estimate_ev_from_clean_segment(
    eig_val_sort_array,
    start_idx,
    end_idx,
    eval_index,
):
    """Estimate eigenvalue level at a specified index using a linear fit.

    A separate line is fitted to the clean eigenvalue segment of each CPI.

    Parameters
    ----------
    eig_val_sort_array : ndarray, shape (num_cpi, cpi_len)
        Sorted eigenvalues in descending order, in linear units.

    start_idx : int
        Inclusive start index of the clean fitting segment.

    end_idx : int
        Exclusive end index of the clean fitting segment.

    eval_index : int
        Eigenvalue index at which to evaluate the fitted line.

    Returns
    -------
    ev_est_db : ndarray, shape (num_cpi,)
        Estimated eigenvalue levels in dB at eval_index for each CPI.
    """
    eig_val_sort_array = np.asarray(eig_val_sort_array)

    if eig_val_sort_array.ndim != 2:
        raise ValueError(
            "eig_val_sort_array must be a 2D array with shape "
            "(num_cpi, cpi_len)."
        )

    num_cpi, cpi_len = eig_val_sort_array.shape

    if end_idx - start_idx < 2:
        raise ValueError(
            f"Invalid clean eigenvalue segment: "
            f"need at least 2 points for linear regression, "
            f"but got {end_idx - start_idx}. "
            f"start_idx={start_idx}, end_idx={end_idx}."
        )

    eig_val_db_array = 10.0 * np.log10(eig_val_sort_array)

    fit_indices = np.arange(
        start_idx,
        end_idx,
        dtype=np.float64,
    )

    ev_est_db_array = np.empty(num_cpi, dtype=np.float64)

    for idx_cpi in range(num_cpi):
        ev_segment_db = eig_val_db_array[
            idx_cpi,
            start_idx:end_idx,
        ]

        slope, intercept = np.polyfit(
            fit_indices,
            ev_segment_db,
            deg=1,
        )

        ev_est_db_array[idx_cpi] = slope * eval_index + intercept

    return ev_est_db_array


def rfi_cpi_candidate_sel(
    eig_val_sort_array,
    max_deg_freedom,
    num_rfi_buffer,
    min_ev_valid_idx,
    rfi_candidate_tolerance_db=3.0,
):
    """Select RFI candidate CPIs within a threshold block.

    This function identifies which CPIs likely contain RFI by:
    1. Performing linear regression on the "clean" eigenvalue segment
    2. Extrapolating to predict what EV[0] should be if the CPI were clean
    3. Comparing actual EV[0] with predicted clean EV[0]
    4. Flagging CPIs where the excess exceeds the tolerance

    This is a pre-processing step that can be used by any threshold algorithm
    to focus detection only on problematic CPIs.

    Parameters
    ----------
    eig_val_sort_array : 2D array of float, [num_cpi x cpi_len]
        Sorted eigenvalues in descending order (linear units, not dB)
    max_deg_freedom : int
        Maximum number of independent RFI emitters designed to be detected.
        Eigenvalue indices [0, ..., max_deg_freedom-1] can potentially be RFI.
    num_rfi_buffer : int
        Number of buffer eigenvalue indices to skip after last possible RFI EV
        before starting clean segment interpolation.
        Clean segment starts at index (max_deg_freedom + num_rfi_buffer - 1).
    min_ev_valid_idx : int
        Eigenvalue index for noise floor estimation.
        Clean segment ends at this index.
    rfi_candidate_tolerance_db : float, default=3.0
        Tolerance in dB for RFI candidate selection. If (actual_EV0 - predicted_clean_EV0)
        exceeds this value, the CPI is flagged as an RFI candidate.

    Returns
    -------
    rfi_cpi_candidate : ndarray of bool, shape (num_cpi,)
        Per-CPI mask. True = RFI candidate (likely contains RFI);
        False = good CPI (likely clean signal).
    num_rfi_candidates : int
        Number of CPIs flagged as RFI candidates.

    Raises
    ------
    ValueError
        If the clean eigenvalue segment has fewer than 2 points for regression.
    """
    # Define clean eigenvalue segment
    start_idx = max_deg_freedom + num_rfi_buffer - 1
    end_idx = min_ev_valid_idx

    # Estimate clean EV[0] for all CPIs using helper function
    ev0_est_db_array = estimate_ev_from_clean_segment(
        eig_val_sort_array,
        start_idx,
        end_idx,
        eval_index=0,
    )

    # Convert actual EV[0] to dB for comparison
    eps = np.finfo(np.float64).tiny
    eig_val_0 = np.maximum(np.real(eig_val_sort_array[:, 0]), eps)
    eig_val_0_db = 10.0 * np.log10(eig_val_0)

    # Select RFI CPI candidates by comparing actual vs predicted EV[0]
    ev0_excess_db = eig_val_0_db - ev0_est_db_array
    rfi_cpi_candidate = ev0_excess_db > rfi_candidate_tolerance_db

    num_rfi_candidates = np.sum(rfi_cpi_candidate)

    return rfi_cpi_candidate, num_rfi_candidates


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

    return detect_threshold


def threshold_estimate_max_ev(
    eig_val_sort_array,
    max_deg_freedom=4,
    num_rfi_buffer=2,
    min_ev_valid_idx=10,
    sig_ev_margin_db=1.0,
):
    """Estimate per-CPI signal max eigenvalue thresholds using max_ev algorithm.

    This function computes per-CPI adaptive thresholds by:
    1. Performing linear regression on the "clean" eigenvalue segment
    2. Extrapolating to max_deg_freedom - 1 (the last possible RFI index)
    3. Adding a safety margin

    Note: RFI candidate selection is performed separately in rfi_detect().
    This function only computes the thresholds.

    Parameters
    ----------
    eig_val_sort_array : 2D array of float, [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    max_deg_freedom : int, default=4
        Maximum number of independent RFI emitters designed to be detected.
        Eigenvalue indices [0, ..., max_deg_freedom-1] can potentially be RFI.
    num_rfi_buffer : int, default=2
        Number of buffer eigenvalue indices to skip after last possible RFI EV
        before starting clean segment interpolation.
        start_idx = max_deg_freedom + num_rfi_buffer - 1.
    min_ev_valid_idx : int, default=10
        Eigenvalue index for noise floor estimation
    sig_ev_margin_db : float, default=1.0
        Aggressive safety margin in dB added to the extrapolated estimate.

    Returns
    -------
    detect_threshold : ndarray of float, shape (num_cpi,)
        Per-CPI signal max eigenvalue thresholds in dB.
        Computed as ev_extrap_db_array + sig_ev_margin_db.
    """
    # Define clean eigenvalue segment
    start_idx = max_deg_freedom + num_rfi_buffer - 1
    end_idx = min_ev_valid_idx

    # Reference index for aggressive threshold (RFI candidates)
    sig_ev_ref_idx = max_deg_freedom - 1

    # Extrapolate to sig_ev_ref_idx using helper function
    ev_extrap_db_array = estimate_ev_from_clean_segment(
        eig_val_sort_array,
        start_idx,
        end_idx,
        eval_index=sig_ev_ref_idx,
    )

    # Generate per-CPI thresholds
    signal_ev_max_estimate_array = ev_extrap_db_array + sig_ev_margin_db

    return signal_ev_max_estimate_array


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
    cpi_rfi_mask=None,
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
    cpi_rfi_mask : ndarray of bool or None, default=None
        Optional per-CPI RFI candidate mask used by both
        'ev_slope' and 'max_ev' methods.

        When provided, detailed eigenvalue detection is applied
        only to CPIs where the mask is True. CPIs where the mask
        is False have been exonerated by candidate selection and
        are returned with all-False RFI flags.

        If None, all CPIs are processed.

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

    # Validate optional CPI detection mask
    if cpi_rfi_mask is not None:
        cpi_rfi_mask = np.asarray(cpi_rfi_mask, dtype=bool)
        if cpi_rfi_mask.shape != (num_cpi,):
            raise ValueError(
                f"cpi_rfi_mask shape {cpi_rfi_mask.shape} must be ({num_cpi},)"
            )

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

    # Compute Eigenvalue in dB for all CPIs
    eig_val_sort_db_array = 10 * np.log10(np.abs(eig_val_sort_array))

    if threshold_method == 'ev_slope':
        # RFI flag for each eigenvalue index in all CPIs: RFI=1, signal=0
        # Start with all CPIs marked clean. Only RFI candidate CPIs are compared.
        rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
        if cpi_rfi_mask is None:
            cpi_rfi_mask = np.ones(num_cpi, dtype=bool)

        eig_val_db_slope_array = np.diff(eig_val_sort_db_array, axis=1)
        for idx_cpi in range(num_cpi):
            if not cpi_rfi_mask[idx_cpi]:
                continue

            eig_val_db_slope = eig_val_db_slope_array[idx_cpi]

            # Use per-CPI threshold
            sig_ev_idx_start = rfi_detect_evd(eig_val_db_slope, detect_threshold_arr[idx_cpi], max_deg_freedom)

            # Sets RFI eigenvalue indices to True
            rfi_cpi_flag_array[idx_cpi, :sig_ev_idx_start] = True
    else:
        # Absolute eigenvalue comparison ('max_ev').
        # Start with all CPIs marked clean. If a max_ev mask is provided, only
        # masked/bad CPIs are compared against the aggressive threshold.
        rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
        if cpi_rfi_mask is None:
            cpi_rfi_mask = np.ones(num_cpi, dtype=bool)

        for idx_cpi in range(num_cpi):
            if not cpi_rfi_mask[idx_cpi]:
                continue

            eig_val_db_valid = eig_val_sort_db_array[idx_cpi, :max_deg_freedom]

            # Use per-CPI threshold
            rfi_ev_idx = np.where(eig_val_db_valid > detect_threshold_arr[idx_cpi])[0]

            if rfi_ev_idx.size:
                sig_ev_idx_start = rfi_ev_idx[-1] + 1
                rfi_cpi_flag_array[idx_cpi, :sig_ev_idx_start] = True

    return rfi_cpi_flag_array
