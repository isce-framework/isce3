"""
Perform RFI Detection and mitigation by evaluating raw data frequency domain
averaged Power Spectra. Detected RFI-contaminated frequency bins are nulled.
"""

import numpy as np
from scipy.fft import fft, ifft
from numpy.polynomial import Polynomial
from scipy import stats
from collections.abc import Iterator

def overlap_slice_gen(total_size: int, batch_size: int, overlap: int=0) -> Iterator[slice]:
    """Generate slices with size defined by batch_size and number of 
    overlapping samples defined by overlap.

    Parameters
    ----------
    total_size: int
        size of data to be manipulated by the slice generator
    batch_size: int
        designated data chunk size in which data is sliced into.
    overlap: int
        Number of overlapping samples in each of the slices
        Default = 0

    Yields
    ------
    slice: slice obj
        Iterable slices of data with specified input batch size, bounded by start_idx
        and stop_idx.
    """

    step_size = batch_size - overlap
    overlap_batch_size = batch_size + step_size

    if batch_size < overlap:
        raise ValueError(
            f"The value of 'overlap' ({overlap}) must be less than that of 'batch_size' ({batch_size})."
        )

    # If total_size is less than overlap_batch_size, generate the slice that
    # corresponds to the entire dataset.
    if total_size < overlap_batch_size:
        yield slice(0, total_size)
    else:
        # The intention is to extend the last full batch to include the remainder
        # samples. Hence the last batch is always larger than all previous batches.
        for start_idx in range(0, total_size - (overlap_batch_size-1), step_size):
            stop_idx = start_idx + batch_size
            yield slice(start_idx, stop_idx)

        # Include remainder samples to include the final remainder samples, if any
        last_blk_start = stop_idx - overlap
        last_blk_stop = total_size

        yield slice(last_blk_start, last_blk_stop)


def run_freq_notch(
    raw_data: np.ndarray,
    num_pulses_az,
    *,
    num_rng_blks: int = 1,
    az_winsize: int = 256,
    rng_winsize: int = 100,
    trim_frac: float = 0.01,
    pvalue_threshold: float = 0.005,
    cdf_threshold: float = 0.1,
    nb_detect: bool = True,
    wb_detect: bool = True,
    mitigate_enable=False,
    raw_data_mitigated: np.ndarray = None,
):
    """Top-level function which performs RFI mitigation on input raw data
    using Frequency Domain Notch Filtering (FDNF) approach by calling 
    lower-level mitigation algorithm.

    Parameters
    ------------
    raw_data: numpy.ndarray complex [num_pulses x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    num_pulses_az: int
        Number of azimuth pulses within the raw data to be processed at once
        It is desirable to divide the raw data into processing blocks
        consisted of dimensions [num_pulses_az x num_samples_rng_blk]
    num_rng_blks: int, default=1
        Number of blocks in the range dimension, default=1
        When num_rng_blks=1, all the range samples of the pulses within a 
        processing block are used to estimate detection mask.
    az_winsize: int, default=256
        The size (in number of pulses) of moving average Azimuth window 
        in which the averaged range spectrum is computed for narrowband detector.
    rng_winsize: int, default=100
        The size (in number of range bins) of moving average Range window 
        in which the total power in range spectrum is computed for wideband
        detector.
    trim_frac: float, optional
        Total proportion of data to trim. `trim_frac/2` is cut off of both tails
        of the distribution. Defaults to 0.01.
    pvalue_threshold: float, default=0.005
        Confidence Level = 1 - pvalue_threshold
        Null hypothesis: No RFI
        Alternative hypothesis: RFI is present
        If p-value of the range-frequency power spectra is less than p-value threshold, 
        alternative hypothesis is accepted. Otherwise, null hypothesis is accepted.
    cdf_threshold: float, default=0.1
        It is a threshold for the cumulative probability density function (CDF) 
        of the input Time Stationary Narrowband (TSNB) and Time Varying Wideband 
        (TVWB) masks. It represents an estimate of the probability of RFI likelihood
        in the input raw_data. A small cdf_threshold value results in a high threshold 
        for RFI detection.
    nb_detect: bool, default=True
        Controls whether narrowband RFI detection mask should be generated.
        If false, narrowband RFI detection mask are populated by zeros.
    wb_detect: bool, default=True
        Controls whether wideband RFI detection mask should be generated.
        If false, wideband RFI detection mask are populated by zeros.
    mitigate_enable: bool, default=False
        Enable mitigation
    raw_data_mitigated: numpy.ndarray complex [num_pulses x num_rng_samples] or None, optional
         output array in which the mitigated data values is placed. It
         must be an array-like object supporting `multidimensional array access
         <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
         The array should have the same shape and dtype as the input raw data array.
         If None (the default), the input 'raw' data array will be modified in-place.

    Returns
    -------
    rfi_likelihood: float
        The fraction of pulses that are contaminated by RFI.

    References
    __________
    ..[1]  F. Meyer, J. Nicoll, and A. Doulgeris, “Correction and Characterization 
    of Radio Frequency Interference Signatures in L-Band Synthetic Aperture Radar 
    Data”, IEEE Trans. Geosci. Remote Sens., vol. 51, no. 10, pp. 4961–4972, Oct. 2013.
    """

    # Modify raw_data in-place
    if raw_data_mitigated is None:
        raw_data_mitigated = raw_data
    else:
        if raw_data_mitigated.shape != raw_data.shape:
            raise ValueError(
                "Shape mismatch: output mitigated data array must have the same shape"
                " as the input data"
            )

    num_pulses, num_rng_samples = raw_data.shape
    num_samples_rng_blk = num_rng_samples // num_rng_blks

    # Count the total number of detected RFI frequency bins
    rfi_pulse_count_sum = 0

    # Run RFI Frequency Domain Notch Filtering Mitigation
    # Generate blocks with increased number of rows and columns
    for idx_az, blk_slow_time in enumerate(
        overlap_slice_gen(
            num_pulses, num_pulses_az + az_winsize-1, az_winsize-1)):
        for idx_rng, blk_fast_time in enumerate(
            overlap_slice_gen(num_rng_samples, num_samples_rng_blk + rng_winsize-1, rng_winsize-1)
        ):
            raw_blk = raw_data[blk_slow_time, blk_fast_time]
            rfi_detect_mask = rfi_freq_notch(
                raw_blk,
                az_winsize=az_winsize,
                rng_winsize=rng_winsize,
                trim_frac=trim_frac,
                pvalue_threshold=pvalue_threshold,
                cdf_threshold=cdf_threshold,
                nb_detect=nb_detect,
                wb_detect=nb_detect,
                mitigate_enable=mitigate_enable,
                raw_data_mitigated=raw_data_mitigated[blk_slow_time, blk_fast_time],
            )

            num_rfi_fft_bins_az = np.sum(rfi_detect_mask, axis=1)
            rfi_pulse_count = np.sum(num_rfi_fft_bins_az != 0)
            rfi_pulse_count_sum += rfi_pulse_count

    rfi_likelihood = rfi_pulse_count_sum / (num_pulses * num_rng_blks)

    return rfi_likelihood


def rfi_freq_notch(
    raw_data,
    *,
    az_winsize=256,
    rng_winsize=100,
    trim_frac=0.01,
    pvalue_threshold=0.005,
    cdf_threshold=0.1,
    nb_detect=True,
    wb_detect=True,
    mitigate_enable=False,
    raw_data_mitigated=None,
):
    """Wrapper function which takes raw data input and performs
    RFI mitigation using Frequency Domain Notch Filtering (FDNF) approach
    by generating a frequency domain detection mask based on individual 
    narrowband detection and wideband detection mask, and subsequently 
    utilize the final detection mask for RFI mitigation.

    Parameters
    ------------
    raw_data: numpy.ndarray complex [num_pulses x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    az_winsize: int, default=256
        The size (in number of pulses) of moving average Azimuth window 
        in which the averaged range spectrum is computed for narrowband detector.
    rng_winsize: int, default=100
        The size (in number of range bins) of moving average Range window 
        in which the total power in range spectrum is computed for wideband
        detector.
    trim_frac: float, optional
        Total proportion of data to trim. `trim_frac/2` is cut off of both tails
        of the distribution. Defaults to 0.01.
    pvalue_threshold: float, default=0.005
        Confidence Level = 1 - pvalue_threshold
        Null hypothesis: No RFI
        Alternative hypothesis: RFI is present
        If p-value of the range-frequency power spectra is less than p-value threshold, 
        alternative hypothesis is accepted. Otherwise, null hypothesis is accepted.
    cdf_threshold: float, default=0.1
        It is a threshold for the cumulative probability density function (CDF) 
        of the input Time Stationary Narrowband (TSNB) and Time Varying Wideband 
        (TVWB) masks. It represents an estimate of the probability of RFI likelihood
        in the input raw_data. A small cdf_threshold value results in a high threshold 
        for RFI detection.
    nb_detect: bool, default=True
        Controls whether narrowband RFI detection mask should be generated.
        If false, narrowband RFI detection mask are populated by zeros.
    wb_detect: bool, default=True
        Controls whether wideband RFI detection mask should be generated.
        If false, wideband RFI detection mask are populated by zeros.
    mitigate_enable: bool, default=False
        Enable mitigation
    raw_data_mitigated: numpy.ndarray complex [num_pulses x num_rng_samples] or None, optional
         output array in which the mitigated data values is placed. It
         must be an array-like object supporting `multidimensional array access
         <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
         The array should have the same shape and dtype as the input raw data array.
         If None (the default), the input 'raw' data array will be modified in-place.

    Returns
    -------
    rfi_detect_mask: 2D array, bool
        Frequency domain binary RFI mask of combined TSNB and TVWB types of RFI.
        Same shape as input raw_data
    """

    # Contributed by Josh Cohen

    if raw_data_mitigated is None:
        raw_data_mitigated = raw_data
    else:
        if raw_data_mitigated.shape != raw_data.shape:
            raise ValueError(
                "Shape mismatch: output mitigated data array must have the same shape"
                " as the input data"
            )

    if not np.iscomplexobj(raw_data_mitigated):
        raise TypeError(
            "Data type mismatch: raw_data_mitigated should be complex."
        )

    num_rng_samples = raw_data.shape[1]
    raw_data_fft = fft(raw_data, axis=1)
    raw_data_freq_psd = np.abs(raw_data_fft)**2

    # Enable narrowband RFI detection
    if nb_detect:
        mask_tsnb = detect_rfi_tsnb(
            raw_data_freq_psd, 
            az_winsize, 
            pvalue_threshold
        )
    else:
        mask_tsnb = np.zeros(raw_data_fft.shape, dtype=np.int32)

    # Enable wideband RFI detection
    if wb_detect:
        mask_tvwb = detect_rfi_tvwb(raw_data_freq_psd, rng_winsize, pvalue_threshold)
    else:
        mask_tvwb = np.zeros(raw_data_fft.shape, dtype=np.int32) # image mask

    # Generate combined frequency-domain binary detection mask.
    rfi_detect_mask = gen_rfi_detect_mask(
        mask_tsnb, 
        mask_tvwb, 
        cdf_threshold,
    )

    # Remove RFI using combined detection mask
    if mitigate_enable:
        rfi_freq_removal(
            raw_data,
            raw_data_fft, 
            rfi_detect_mask, 
            az_winsize,
            rng_winsize,
            raw_data_mitigated,
        )

    return rfi_detect_mask


def trim_mean_and_std(a, trim_frac=0.01, axis=0):
    """
    Return the mean and standard deviation of an array after trimming both tails.

    Parameters
    ----------
    a : numpy.ndarray
        Input array.
    trim_frac: float, optional
        Total proportion of data to trim. `trim_frac/2` is cut off of both tails
        of the distribution. Defaults to 0.01.
    axis : int or None, optional
        Axis along which the trimmed means are computed. Default is 0.
        If None, compute over the whole array `a`.

    Returns
    -------
    trim_mean : numpy.ndarray
        Mean of trimmed array.
    trim_std : numpy.ndarray
        Standard deviation of trimmed array.
    """

    # Contributed by Geoffrey Gunter

    # Adapted from `scipy.stats.trim_mean`
    nobs = a.shape[axis]
    lowercut = int(round(0.5 * trim_frac * nobs))
    uppercut = nobs - lowercut
    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    p = np.partition(a, (lowercut, uppercut - 1), axis)

    s = [slice(None)] * a.ndim
    if axis is not None:
        s[axis] = slice(lowercut, uppercut)

    trim_mean = np.mean(p[tuple(s)], axis=axis)
    trim_std = np.std(p[tuple(s)], axis=axis)

    return trim_mean, trim_std


def detect_rfi_tsnb(
    raw_data_fft_psd, 
    az_winsize=256, 
    pvalue_threshold=0.005,
    trim_frac=0.01
):
    """Estimate Time-Stationary Narrowband (TSNB) frequency domain RFI mask.
    The TSNB type of RFI is assumed to be consistently present in the 
    same range bin(s) across numerous slow time pulses. Therefore, TSNB RFI 
    can be identified by its signatures in the averaged range power spectrum 
    in azimuth-time direction. The averaging azimuth-time window size is 
    defined by 'az_winsize'.

    Parameters
    ------------
    raw_data_fft_psd: array-like real [num_pulses x num_rng_samples]
        raw data range frequency power spectra
    az_winsize: int, default=256
        The size (in number of pulses) of moving average Azimuth window 
        in which the averaged range spectrum is computed for narrowband detector.
    pvalue_threshold: float, default=0.005
        Time Stationary Narrowband (TSNB) p-value threshold. 
        Confidence Level = 1 - pvalue_threshold
        If p-value of the range-frequency power spectra is less than TSNB
        p-value threshold, alternative hypothesis is accepted.  
        Otherwise, null hypothesis is accepted.
        Null hypothesis: No TSNB RFI
        Alternative hypothesis: TSNB RFI is present
    trim_frac: float, optional
        Total proportion of data to trim. `trim_frac/2` is cut off of both tails
        of the distribution. Defaults to 0.01.

    Returns
    -------
    mask_tsnb: 2D array of same size as input, int32
        Frequency domain TSNB hit mask based on p-value of the PDF of
        range frequency power spectra
    """

    # Contributed by Josh Cohen

    num_pulses, num_rng_samples = raw_data_fft_psd.shape

    if (az_winsize >  num_pulses):
        raise ValueError(f'Azimuth window size shall be equal to or less than {num_pulses}.')

    # TSNB hit mask
    tsnb_hits_buffer = np.zeros((az_winsize, num_rng_samples), dtype=np.int32)
    mask_tsnb = np.zeros((num_pulses, num_rng_samples), dtype=np.int32)

    for i in range(num_pulses - az_winsize + 1):
        az_block = raw_data_fft_psd[i : i + az_winsize]
        avg_az_time = np.mean(az_block, axis=0)  # Average in Azimuth time

        #compute Z-Scores of data which indicates the number of
        #standard deviations a data value lies from the mean
        mu, sigma = trim_mean_and_std(avg_az_time, trim_frac, axis=0)
        zscores = (avg_az_time - mu) / sigma

        # Convert Z-Scores to one-tailed P-Values
        pvalues = stats.norm.sf(zscores)
        
        # Isolate significant values (p < threshold) and convert to ints
        # The results are significant (alternative hypothesis is True) if
        # p < threshold and vice versa.
        rfi_hit_tsnb = (pvalues < pvalue_threshold).astype(int)
        tsnb_hits_buffer += rfi_hit_tsnb  # Add RFI hits to overall mask

        count = i % az_winsize
        mask_tsnb[i] = tsnb_hits_buffer[count].astype(np.int32)

        tsnb_hits_buffer[count] = 0  # Zero out mask line

    return mask_tsnb


def detect_rfi_tvwb(
    raw_data_fft_psd, 
    rng_winsize=100, 
    pvalue_threshold=0.005, 
    trim_frac=0.01
):
    """Estimate Time-Varying Wideband (TVWB) frequency-domain RFI mask.
    TVWB RFI signatures can be best identified by averaging frequency power
    spectrum in range-frequency direction. The averaging window size is defined
    as 'rng_winsize'.

    Parameters
    ----------
    raw_data_fft_psd: array-like real [num_pulses x num_rng_samples]
        raw data range frequency power spectra
    rng_winsize: int, default=100
        The size (in number of range bins) of moving average Range window 
        in which the total power in range spectrum is computed for wideband
        detector.
    pvalue_threshold: float, default=0.005
        Time Varying Wideband (TVWB) p-value threshold. 
        Confidence Level = 1 - pvalue_threshold
        If p-value of the range-frequency power spectra is less than TVWB 
        p-value threshold, alternative hypothesis is accepted.  
        Otherwise, null hypothesis is accepted.
        Null hypothesis: No TVWB  RFI
        Alternative hypothesis: TVWB RFI is present
    trim_frac: float, optional
        Total proportion of data to trim. `trim_frac/2` is cut off of both tails
        of the distribution. Defaults to 0.01.

    Returns
    -------
    mask_tvwb: 2D array, int32, same dimension as input
        Frequency domain TVWB hit mask based on p-value of the probability
        density function (PDF) of range frequency power spectra. Must be
        of the same shape as 'mask_tsnb'.
    """

    # Contributed by Josh Cohen
    
    num_pulses, num_rng_samples = raw_data_fft_psd.shape

    if (rng_winsize >  num_rng_samples):
        raise ValueError(f'Range window size shall be equal to or less than {num_rng_samples}.')

    # Transposing to match TSNB Block code
    raw_data_fft_psd = np.transpose(raw_data_fft_psd)

    # TVWB hit mask
    tvwb_hits_buffer = np.zeros((rng_winsize, num_pulses), dtype=np.int32)
    mask_tvwb = np.zeros((num_pulses, num_rng_samples), dtype=np.int32)
    x = np.arange(num_pulses)

    for i in range(num_rng_samples - rng_winsize + 1):
        rng_block = raw_data_fft_psd[i : i + rng_winsize]
        avg_rng_freq = np.mean(rng_block, axis=0) # Average in range frequency

        # In Meyer's paper, detrending is applied to linear-scale averaged power 
        # spectra. However, in the code below, linear-scale averaged power spectra 
        # is converted to log scale before detrending. The reason is that detrending 
        # with linear curve fitting approach implemented in this code is more accurate 
        # with respect to log-scale averaged power spectra due to extremely large difference
        # in power between RFI and signal in linear-scale averaged power spectra.

        avg_rng_freq_db = np.log10(avg_rng_freq)  # Convert to log-space

        # Generate numpy function with best-fit linear coeffs
        pfit = Polynomial.fit(x, avg_rng_freq_db, 1)

        # Detrend the average with the best-fit (normalize the row's mean)
        avg_rng_freq_db -= pfit(x)

        mu, sigma = trim_mean_and_std(avg_rng_freq_db, trim_frac, axis=0)
        zscores = (avg_rng_freq_db - mu) / sigma

        # Isolate significant values (p < threshold) and convert to ints
        # The results are significant (alternative hypothesis is True) if
        # p < alpha and vice versa.
        pvalues = stats.norm.sf(zscores)

        rfi_hit_tvwb = (pvalues < pvalue_threshold).astype(int)
        tvwb_hits_buffer += rfi_hit_tvwb

        count = i % rng_winsize
        mask_tvwb[:, i] = tvwb_hits_buffer[count].astype(np.int32)
        tvwb_hits_buffer[count] = 0

    return mask_tvwb


def gen_rfi_detect_mask(
    mask_tsnb,
    mask_tvwb,
    cdf_threshold=0.1,
):
    """Estimate combined frequency-domain RFI mask using input
    TSNB and TVWB masks.

    Parameters
    ------------
    mask_tsnb: 2D array, int32
        Frequency domain TSNB hit mask based on p-value of the PDF of
        range frequency power spectra
    mask_tvwb: 2D array, int32
        Frequency domain TVWB hit mask based on p-value of the probability
        density function (PDF) of range frequency power spectra. Must be
        of the same shape as 'mask_tsnb'.
    cdf_threshold: float, default=0.1
        This is the  threshold for the cumulative probability density function (CDF) 
        of the input Time Stationary Narrowband (TSNB) and Time Varying Wideband 
        (TVWB) masks. It represents an estimate of the probability of RFI likelihood
        in the input raw_data. A small cdf_threshold value results in a high threshold 
        for RFI detection.

    Returns
    -------
    rfi_detect_mask: 2D array, bool, same dimension as input TSNB and TVWB masks
        Frequency domain binary RFI mask of combined TSNB and TVWB types of RFI.
    """

    # Contributed by Josh Cohen and Geoff Gunter

    if mask_tsnb.shape != mask_tvwb.shape:
        raise ValueError("shape mismatch: RFI masks must have the same shape")

    tsnb_thresh = np.quantile(mask_tsnb, 1.0 - cdf_threshold)
    tvwb_thresh = np.quantile(mask_tvwb, 1.0 - cdf_threshold)

    # Generate Binary RFI mask for TSNB and TVWB
    tsnb_detect = (mask_tsnb > tsnb_thresh)
    tvwb_detect = (mask_tvwb > tvwb_thresh)

    # Generate final combined RFI mask
    rfi_detect_mask = tsnb_detect | tvwb_detect

    return rfi_detect_mask


def rfi_freq_removal(
    raw_data,
    raw_data_fft, 
    rfi_detect_mask, 
    az_winsize=256,
    rng_winsize=100,
    raw_data_mitigated=None
):
    """Null the frequency domain FFT bins identified as RFI by the
    binary rfi_detect_mask, and then take inverse FFT of RFI-mitigated 
    frequency response to generate its time-domain counterpart.

    Parameters
    ------------
    raw_data: numpy.ndarray complex [num_pulses x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    raw_data_fft: 2D array of complex, [num_pulses x num_rng_samples]
        Range frequency spectrum response of raw data
    raw_detect_mask: 2D array, bool, [num_pulses x num_rng_samples]
        Frequency domain binary RFI mask: 1 = RFI, 0 = No RFI
    az_winsize: int, default=256
        The size (in number of pulses) of moving average Azimuth window 
        in which the averaged range spectrum is computed for narrowband detector.
    rng_winsize: int, default=100
        The size (in number of range bins) of moving average Range window 
        in which the total power in range spectrum is computed for wideband
        detector.
    raw_data_mitigated: numpy.ndarray complex [num_pulses x num_rng_samples] or None, optional
         output array in which the mitigated data values is placed. It
         must be an array-like object supporting `multidimensional array access
         <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
         The array should have the same shape and dtype as the input raw data array.
         If None (the default), the input 'raw' data array will be modified in-place.
    """

    num_pulses, num_rng_samples = raw_data.shape

    # Compute the dimensions of non-overlapped data blocks
    num_pulses_no_overlap = num_pulses - az_winsize + 1
    num_samples_no_overlap = num_rng_samples - rng_winsize + 1

    # Trim the overlapped inputs to the dimensions of non-overlapped data blocks
    # Only process the pulses and range samples with respect to the non-overlapped data block
    rfi_detect_mask = rfi_detect_mask[:num_pulses_no_overlap, :num_samples_no_overlap]
    raw_data = raw_data[:num_pulses_no_overlap, :num_samples_no_overlap]
    raw_data_fft = raw_data_fft[:num_pulses_no_overlap, :num_samples_no_overlap]

    # Create an alias for raw_data if there is no 'raw_data_mitigated' as input
    if raw_data_mitigated is None:
        raw_data_mitigated = raw_data
    else:
        raw_data_mitigated = raw_data_mitigated[:num_pulses_no_overlap, :num_samples_no_overlap]

    for i in range(num_pulses_no_overlap):
        if np.all(rfi_detect_mask[i] == 0):
            raw_data_mitigated[i] = raw_data[i]
        else:
            notch_line = (rfi_detect_mask[i] == 0)
            # Null frequency bins identified as RFI by RFI mask of the matching line of image
            raw_line_fft_nulled = raw_data_fft[i] * notch_line
            # Inverse FFT to tranlate into time domain
            raw_line_nulled = ifft(raw_line_fft_nulled)
            raw_data_mitigated[i] = raw_line_nulled
