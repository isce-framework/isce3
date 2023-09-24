"""
Performs RFI detection of input data using Slow-Time Eigenvalue Slope
Thresholding algorithm (ST-EST).
"""
import numpy as np
from isce3.signal.compute_evd_cpi import compute_evd_tb
from dataclasses import dataclass, field
from typing import List


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
    num_max_trim,
    num_min_trim,
    max_num_rfi_ev,
    threshold_params,
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
    num_max_trim: int, default = 0
        Number of large-value outliers to be trimmed in slow-time minimum Eigenvalues.
    num_min_trim: int, default = 0
        Number of small-value outliers to be trimmed in slow-time minimum Eigenvalues
    max_num_rfi_ev: int
        A detection error (miss) happens when a maximum power RFI emitter contaminates 
        multiple consecutive CPIs, resulting in a flat maximum Eigenvalue slope in slow 
        time. Hence the standard (STD) deviation of multiple dominant EVs across slow time 
        defined by this parameter are compared. The one with the maximum STD is used for RFI
        Eigenvalue first difference computation.
    threshold_params: ThresholdParams dataclass object
        RFI detection threshold interpolation parameters. The x field defines STD
        ratio between maximum and minimum Eigenvalue slopes (MMES) of the
        slow-time threshold interval. The y field defines the number of sigma (STD)
        from the mean of MMES.

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

    # Compute EVD of input data
    eig_val_sort_array, eig_vec_sort_array = compute_evd_tb(raw_data, cpi_len)

    # Estimate a single threshold for all CPIs
    detect_threshold = threshold_estimate_evd(
        eig_val_sort_array,
        num_max_trim,
        num_min_trim,
        max_num_rfi_ev,
        threshold_params,
    )

    # Detect RFI Eigenvalues of each CPI based on input detection threshold
    rfi_cpi_flag_array = rfi_detect_evd_tb(
        eig_val_sort_array, detect_threshold, max_deg_freedom
    )

    return rfi_cpi_flag_array, eig_vec_sort_array


def threshold_estimate_evd(
    eig_val_sort_array,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
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

    ev_min_db = 10 * np.log10(np.abs(eig_val_sort_array[:, -1]))

    # Remove possible outliers in max and min Eigenvalues without reordering.
    if num_min_trim > 0:
        ev_min_trim_idx = np.argsort(ev_min_db)[:num_min_trim]
        ev_min_db = np.delete(ev_min_db, ev_min_trim_idx)

    if num_max_trim > 0:
        ev_max_trim_idx = np.argsort(ev_min_db)[-num_max_trim:]
        ev_min_db = np.delete(ev_min_db, ev_max_trim_idx)

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


def rfi_detect_evd(
    eig_val_db_slope,
    detect_threshold,
    max_deg_freedom=12,
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
    max_deg_freedom: int, default = 12
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
    max_deg_freedom=12,
):
    """Wrapper function which performs RFI detection of data within a Threshold Block (TB) 
    one CPI at a time base don input detection threshold in dB/Eigenvalue index. 

    Parameters
    ----------
    eig_val_sort_array: 2D array of float, [num_cpi x cpi_len]
        Sorted Eigenvalues in descending order of all CPIs in raw data
    detect_threshold: float
        A positive RFI detection threshold used by EVD detection algorithm to identify
        RFI Eigenvalues in a CPI in dB/Eigenvalue index. All CPIs of the input data
        share a common detection threshold.
    max_deg_freedom: int, default = 12
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len.

    Returns
    -------
    rfi_cpi_flag_array: 2D array of bool, [num_cpi x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as either RFI or signal.
        1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
    """

    # Number of pulses, CPI length, and number of range blocks in a CPI
    num_cpi, cpi_len = eig_val_sort_array.shape

    # Ensure detection threshold is a positive value
    if detect_threshold <= 0:
        raise ValueError("Detection threshold must be a positive value!")

    # Maximum number of degrees of freedom must be less than cpi_len
    if max_deg_freedom >= cpi_len:
        raise ValueError(
            "Max number of deg. of freedom must be less than number of pulses in a CPI."
        )

    # RFI flag for each eigenvalue index in all CPIs: RFI=1, signal=0:w
    rfi_cpi_flag_array = np.ones((num_cpi, cpi_len), dtype=np.bool_)

    # Compute Eigenvalue Slope or first difference of all CPIs
    eig_val_sort_db_array = 10 * np.log10(np.abs(eig_val_sort_array))
    eig_val_db_slope_array = np.diff(eig_val_sort_db_array, axis=1)

    for idx_cpi in range(num_cpi):
        eig_val_db_slope = eig_val_db_slope_array[idx_cpi]

        # Determine starting index of signal EV
        sig_ev_idx_start = rfi_detect_evd(eig_val_db_slope, detect_threshold, max_deg_freedom)

        # Sets signal eigenvalue indices to zero
        rfi_cpi_flag_array[idx_cpi, sig_ev_idx_start:] = 0

    return rfi_cpi_flag_array
