import h5py
import logging
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from . import cola_windows
from isce3.noise.noise_power_est_func import cpi_slice_gen

log = logging.getLogger("isce3.signal.rfi")


def exp_from_quantile(p, vp, bw=1.0):
    """
    Determine the rate parameter of the lifted exponential distribution from a
    quantile and its corresponding value.

    Parameters
    ----------
    p : float | np.ndarray
        Quantile in [1-bw, 1).  For example, 0.5 for the median.
    vp : float | np.ndarray
        Value corresponding to the given quantile.  For example, the median
        value.
    bw : float, optional
        The portion of the distribution governed by an exponential distribution.
        In the context of the tone-rank algorithm, it's the chirp bandwidth
        normalized by the sample rate.  Values in interval (0, 1].
        See notes below.

    Returns
    -------
    λ : float | np.ndarray
        The rate parameter of the lifted exponential distribution.

    Notes
    -----
    The definition of the "lifted" exponential distribution is as follows:
        cdf(x) = (1 - bw) * u(x - x0) + bw * (1 - exp(-λ * x))
    where u(x) is the unit step function, x0 is some small value (e.g., the
    noise floor), and other parameters are as described above.

    The idea here is that spectral power in the chirp band should follow an
    exponential distribution, while the values outside the chirp band will be
    smaller and follow some other distribution whose shape we don't care about
    and just model with a step function.

    The formula is the same to solve for a value given the rate parameter.
    That is, you can also use this function to solve
        vp = exp_from_quantile(p, λ, bw)
    """
    p = np.asarray(p)
    if not np.all(p >= (1.0 - bw)):
        raise ValueError("invalid quantile for lifted exponential")
    vp = np.asarray(vp)
    return -np.log((1 - p) / bw) / vp


def abs2(z):
    return z.real**2 + z.imag**2


def get_spectral_mask(
    spectra,
    reference_quantile=0.5,
    nominal_false_positive_rate=0.0005,
    bandwidth=5 / 6,
):
    """
    Detect RFI in spectral domain data using lifted exponential model.

    Uses a statistical model to identify spectral samples that are anomalously
    loud compared to the expected distribution of the signal. The method assumes
    that the power spectrum follows a lifted exponential distribution and uses
    a quantile of the data to estimate the distribution parameters.

    Parameters
    ----------
    spectra : np.ndarray
        Complex spectral data, arbitrary shape.
    reference_quantile : float, optional
        Quantile (in [1-bandwidth, 1)) used to estimate the rate parameter
        of the lifted exponential distribution. Default is 0.5 (median).
    nominal_false_positive_rate : float, optional
        Target false positive rate for RFI detection, in (0, 1).
        Lower values result in more conservative detection (fewer false alarms
        but potentially missed RFI).
    bandwidth : float, optional
        Fraction of samples in (0, 1] assumed to follow the exponential
        distribution (as opposed to being noise, filter roll-off, etc).

    Returns
    -------
    mask : np.ndarray[bool]
        Boolean mask marking detected RFI samples, same shape as spectra.
        True indicates RFI detection (sample exceeds threshold).
    isr : float
        Interference-to-signal ratio: ratio of total power in detected RFI
        samples to total power in clean samples.
    λ : float
        Estimated rate parameter of the lifted exponential distribution.

    Notes
    -----
    The detection threshold is determined by inverting the cumulative
    distribution function of the lifted exponential model at the desired
    false positive rate. See exp_from_quantile for details on the
    statistical model.
    """
    n = np.prod(spectra.shape)
    power_spectra = abs2(spectra)
    power_spectra.shape = (n,)
    # partition seems to be a little faster than quantile, I guess because the
    # latter does a full sort and interpolates?  For large n the difference is
    # small.
    k = round(reference_quantile * n)
    kth_power = np.partition(power_spectra, k)[k]
    if kth_power == 0.0:
        return np.zeros(spectra.shape, dtype=bool), 0.0, 0.0
    # Estimate the parameter of a lifted exponential distribution.
    λ = exp_from_quantile(reference_quantile, kth_power, bandwidth)
    # Determine threshold.  Since we've estimated a statistical model, we can
    # achieve a specified nominal false positive rate by thresholding the
    # nominal CDF at one minus that value.
    q = 1.0 - nominal_false_positive_rate
    too_loud = exp_from_quantile(q, λ, bandwidth)
    # Use threshold to generate mask.
    mask = power_spectra >= too_loud
    # Compute interference-to-signal ratio now that we've labeled the data.
    # Since masked values are presumably the sum of signal and interference,
    # we'll include an arithmetic correction using the stats model.
    mean_power = 1 / λ
    masked_sig_power = np.sum(mask) * mean_power
    rfi_power = np.sum(power_spectra[mask]) - masked_sig_power
    sig_power = np.sum(power_spectra[~mask]) + masked_sig_power
    isr = rfi_power / sig_power
    # Unflatten data.
    mask.shape = spectra.shape
    return mask, isr, λ


def fill_missing(z, fd, t, mask_replace, valid_rows, noise, interpolate=True,
                 fill_value="noise"):
    """
    Fill missing/RFI-contaminated samples in spectral domain data.

    Parameters
    ----------
    z : np.ndarray
        Complex spectral data to fill, shape (m, n)
    fd : float
        Doppler centroid frequency in Hz.  It is not scaled for each frequency
        bin in `z`.  If wideband processing is needed, just baseband `z` before
        calling this function and set `fd=0`.
    t : np.ndarray
        Pulse times in seconds since epoch, length m
    mask_replace : np.ndarray
        Boolean mask marking samples to replace, shape (m, n)
    valid_rows : np.ndarray
        Boolean mask marking valid rows, shape (m,)
    noise : np.ndarray
        Random noise samples in 1D array for fallback replacement.  Values may
        be used multiple times if length is less numpy.prod((m, n)).
    interpolate : bool, optional
        If True, attempt linear/nearest-neighbor interpolation. If False,
        replace directly with fill_value. Default is True.
    fill_value : str, optional
        Fallback strategy when interpolation fails or is disabled.
        "noise": use provided noise vector (default)
        "zero": use zero

    Returns
    -------
    zout : np.ndarray
        Filled data, same shape as z
    """
    m, n = z.shape
    if len(t) != m:
        raise ValueError(f"expected len(t)=={m} got {len(t)}")
    if mask_replace.shape != (m, n):
        raise ValueError(f"{mask_replace.shape=} does not match {z.shape=}")
    if valid_rows.shape != (m,):
        raise ValueError(f"{valid_rows.shape=} does not match {z.shape[0]=}")
    if interpolate and m < 2:
        log.warning(f"Disabling interpolation for block of {m} rows "
            "because there are no neighbors to use for interpolation.")
        interpolate = False

    # Deramp Doppler.
    deramp = np.exp(-1j * 2 * np.pi * fd * t).astype(z.dtype)
    zout = deramp[:, None] * z

    # Exclude RFI samples from being used as interpolation sources.
    mask_valid_clean = valid_rows[:, None] & ~mask_replace

    for i in range(m):
        # copy for modification
        cols_need_replacement = mask_replace[i, :].copy()

        if interpolate:
            # Previous and next pulse, with reflection boundary condition.
            iprev, inext = i - 1, i + 1
            if i == 0:
                iprev = i + 1
            if i == m - 1:
                inext = i - 1

            # Non-uniform time sampling, so let's weight closer samples more.
            # NOTE abs() since we might've reflected.
            dt_prev = abs(t[i] - t[iprev])
            dt_next = abs(t[inext] - t[i])
            w_prev = dt_next / (dt_prev + dt_next)
            w_next = dt_prev / (dt_prev + dt_next)

            # Four cases for replacement:
            # 1. prev and next both valid -> lerp between them
            j = np.where(mask_valid_clean[iprev, :] & mask_valid_clean[inext, :]
                & cols_need_replacement)[0]
            zout[i, j] = w_prev * zout[iprev, j] + w_next * zout[inext, j]
            cols_need_replacement[j] = False

            # 2. only prev valid. use it
            j = np.where(mask_valid_clean[iprev, :] & cols_need_replacement)[0]
            zout[i, j] = zout[iprev, j]
            cols_need_replacement[j] = False

            # 3. only next valid. use it
            j = np.where(mask_valid_clean[inext, :] & cols_need_replacement)[0]
            zout[i, j] = zout[inext, j]
            cols_need_replacement[j] = False

        # 4. Fallback for remaining samples (or all if not interpolating)
        j = np.where(cols_need_replacement)[0]
        if len(j) > 0:
            if fill_value == "zero":
                zout[i, j] = 0.0
            elif fill_value == "noise":
                noise_idx = ((i * n) + j) % len(noise)
                zout[i, j] = noise[noise_idx]
            else:
                raise ValueError(f"Invalid fill_value: {fill_value}")

    # Put Doppler back on.
    zout *= deramp[:, None].conj()
    return zout


def circular_gaussian_noise(n, σ=1, dtype=np.complex64):
    return σ * (np.random.normal(size=n) + 1j * np.random.normal(size=n))


def remove_loud_tones(
    z,
    t,
    r,
    swaths,
    doppler,
    block_dims=(512, 1024),
    reference_quantile=0.5,
    nominal_false_positive_rate=0.0005,
    bandwidth=5 / 6,
    detect_only=False,
    zout=None,
    interpolate=True,
    fill_value="noise",
    max_gap_fraction=0.5,
):
    """
    Detect and optionally mitigate narrowband RFI using spectral rank method.

    Processes raw data in overlapping blocks using a Short-Time Fourier Transform
    (STFT) approach. RFI is detected in the spectral domain using a lifted
    exponential statistical model. Detected RFI samples are replaced using
    temporal interpolation or fill values.

    Parameters
    ----------
    z : np.ndarray[complex64]
        Raw data, shape (num_pulses, num_range_bins).
    t : np.ndarray[float64]
        Pulse times in seconds since orbit/grid epoch, length num_pulses.
    r : isce3.core.Linspace
        Slant range to each sample in meters.
    swaths : np.ndarray[int]
        Valid subswath samples, shape (num_subswaths, num_pulses, 2).
        Last dimension contains [start, stop) indices of each subswath.
    doppler : isce3.core.LUT2d
        Raw data Doppler centroid look-up table in Hz. Must be valid over
        entire grid.
    block_dims : tuple[int, int], optional
        Processing block size (azimuth, range). Default is (512, 1024).
    reference_quantile : float, optional
        Quantile used to estimate the rate parameter of the lifted exponential
        distribution, in [1-bandwidth, 1). Default is 0.5 (median).
    nominal_false_positive_rate : float, optional
        Target false positive rate for RFI detection, in (0, 1].
    bandwidth : float, optional
        Ratio of chirp bandwidth to sample rate, in (0, 1].
    detect_only : bool, optional
        If True, only detect RFI without mitigation. If False, replace
        detected RFI samples.
    zout : np.ndarray[complex64], optional
        Output buffer for mitigated data. Must have same shape as z.
        If None, z is modified in-place.  In that case some rows will be cleaned
        twice if block_dims[0] doesn't divide evenly into z.shape[0].
    interpolate : bool, optional
        If True, attempt linear/nearest-neighbor interpolation for RFI samples.
        If False, replace directly with fill_value.
    fill_value : str, optional
        Fallback value when interpolation fails or is disabled.
        "noise": use random Gaussian noise (default)
        "zero": use zero
    max_gap_fraction : float, optional
        Max portion of a row that can be masked by a TX gap before its
        spectrum is considered invalid (0.0: any gap invalidates row,
        1.0: ignore gap mask).

    Returns
    -------
    block_times : np.ndarray[float]
        Azimuth time at center of each azimuth block in seconds since epoch,
        length num_az_blocks.
    block_ranges : np.ndarray[float]
        Slant range at center of each range block in meters,
        length num_range_blocks.
    f : np.ndarray[float]
        Normalized frequency axis for hits array, length block_dims[1].
        Units are cycles per sample.
    means : np.ndarray[float]
        Estimated mean signal power per block from lifted exponential model,
        shape (num_az_blocks, num_range_blocks). Equal to 1/λ where λ is the
        rate parameter.
    isr : np.ndarray[float]
        Interference-to-signal ratio per block,
        shape (num_az_blocks, num_range_blocks).
    hits : np.ndarray[float]
        Fraction of pulses with detected RFI at each frequency bin,
        shape (num_az_blocks, num_range_blocks, block_dims[1]).
        Values range from 0 (no RFI detected) to 1 (RFI detected in all pulses).

    Notes
    -----
    The algorithm processes data in overlapping blocks using COLA (Constant
    Overlap-Add) windowing in range. Each block is transformed to the spectral
    domain where RFI tones appear as anomalously loud samples. Detection uses
    the lifted exponential model (see get_spectral_mask). When mitigation is
    enabled, detected samples are replaced using temporal interpolation from
    adjacent clean pulses, with fallback to random noise or zeros.
    """
    # Check inputs
    if not (z.ndim == len(block_dims) == 2):
        raise ValueError("Data and block_dims must be 2-dimensional")
    block_dims = (block_dims[0], block_dims[1])  # copy
    for idim in (0, 1):
        if block_dims[idim] > z.shape[idim]:
            log.warning(f"Truncating block_dims[{idim}] to z.shape[{idim}]")
            block_dims[idim] = z.shape[idim]
    if not (0 < nominal_false_positive_rate <= 1):
        raise ValueError(
            "nominal_false_positive rate must be normalized to interval (0, 1]"
        )
    if not (0 < bandwidth <= 1):
        raise ValueError("bandwidth must be normalized to interval (0, 1]")
    if not ((1 - bandwidth) < reference_quantile < 1):
        raise ValueError(
            "reference_quantile must fall in the normalized signal "
            f"distribution ({1 - bandwidth}, 1)"
        )
    if zout is None:
        zout = z
    if (not detect_only) and (zout.shape != z.shape):
        raise ValueError("output shape must match input shape")

    mask_valid = np.zeros((block_dims[0], z.shape[1]), bool)

    slices_windows = list(cola_windows(z.shape[1], block_dims[1]))
    num_range_blocks = len(slices_windows)
    az_blocks = list(cpi_slice_gen(z.shape[0], block_dims[0]))
    num_az_blocks = len(az_blocks)
    zbshape = (block_dims[0], num_range_blocks, block_dims[1])
    z_block = np.zeros(zbshape, z.dtype)

    f = fftshift(fftfreq(block_dims[1]))
    meta_shape = (num_az_blocks, num_range_blocks)
    isr = np.zeros(meta_shape, dtype=np.float32)
    means = np.zeros(meta_shape, dtype=np.float32)
    hits = np.zeros(meta_shape + (block_dims[1],), dtype=np.float32)

    block_times = np.zeros(num_az_blocks)
    block_ranges = np.zeros(num_range_blocks)
    for j, (cols, _) in enumerate(slices_windows):
        # TODO could weight by window
        block_ranges[j] = r[(cols.start + cols.stop) // 2]

    # Generate a pool of noise with σ=1 once, and we'll scale it for each
    # block as needed.  We'll generate a random offset for each block to
    # increase entropy without having to regenerate noise for each block.
    max_offset = num_az_blocks * num_range_blocks
    num_noise = max_offset + np.prod(block_dims)
    std_noise = circular_gaussian_noise(num_noise)

    for iblock, rows in enumerate(az_blocks):
        block_results = []
        # calculate azimuth time of block
        block_times[iblock] = t[rows].mean()
        pulse_times = t[rows]
        # populate valid data mask
        mask_valid[...] = False
        for i, i_pulse in enumerate(range(rows.start, rows.stop)):
            for start, end in swaths[:, i_pulse, :]:
                mask_valid[i, start:end] = True
        # apply window to each range block
        for j, (cols, window) in enumerate(slices_windows):
            nw = len(window)
            z_block[:, j, :nw] = window[None, :] * z[rows, cols]
        # Range STFT.  Use consistent FFT size even for edges where window
        # may be shorter so that frequency metadata are consistent.
        spectra = fft(z_block, n=block_dims[1], axis=2)
        # filter RFI
        for j in range(z_block.shape[1]):
            mask_replace, isr[iblock,j], λ = get_spectral_mask(
                spectra[:, j, :],
                reference_quantile,
                nominal_false_positive_rate,
                bandwidth,
            )
            if not detect_only:
                σ = np.sqrt(0.5 / λ) if λ > 0.0 else 0.0
                noise = σ * std_noise[np.random.randint(0, max_offset):]
                fd = doppler.eval(block_times[iblock], block_ranges[j])
                cols, window = slices_windows[j]
                valid_rows = ((~mask_valid[:, cols]).mean(axis=1)
                    <= max_gap_fraction)
                spectra[:, j, :] = fill_missing(spectra[:,j,:], fd, pulse_times,
                    mask_replace, valid_rows, noise, interpolate, fill_value)
            hits[iblock, j, :] = fftshift(np.mean(mask_replace, axis=0))
            means[iblock, j] = 1 / λ if λ > 0.0 else 0.0
        # skip inverse FFTs and assignment if not required.
        if not detect_only:
            # range inverse STFT
            z_block[...] = ifft(spectra, axis=2)
            # sum COLA range blocks into output buffer
            zout[rows, ...] = 0
            for j, (cols, window) in enumerate(slices_windows):
                nw = len(window)
                zout[rows, cols] += z_block[:, j, :nw]

    return block_times, block_ranges, f, means, isr, hits


def write_tone_rank_results(group: h5py.Group, t, r, freq, means, isr, hits):
    """
    Write tone-rank RFI detection results to HDF5 group.

    Creates datasets for time/range/frequency axes and detection results
    (signal mean, interference-to-signal ratio, and hit counts) with
    appropriate metadata attributes.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to write datasets into.
    t : np.ndarray[float]
        Azimuth times at center of each azimuth block in seconds since epoch,
        length num_az_blocks. Written as "nativeDopplerTime" dataset.
    r : np.ndarray[float]
        Slant ranges at center of each range block in meters,
        length num_range_blocks. Written as "slantRange" dataset.
    freq : np.ndarray[float]
        Frequency axis in Hz, length num_freq_bins.
        Written as "frequency" dataset.
    means : np.ndarray[float]
        Estimated mean signal power per block from lifted exponential model,
        shape (num_az_blocks, num_range_blocks). Written as "signalMean" dataset.
    isr : np.ndarray[float]
        Interference-to-signal ratio per block,
        shape (num_az_blocks, num_range_blocks).
        Written as "interferenceSignalRatio" dataset.
    hits : np.ndarray[float]
        Fraction of pulses with detected RFI at each frequency bin,
        shape (num_az_blocks, num_range_blocks, num_freq_bins).
        Written as "hitCount" dataset.

    Notes
    -----
    All datasets are created with appropriate units and description attributes.
    Time/range/frequency axes are stored as float64, while data arrays (means,
    isr, hits) are stored as float32.
    """
    m, n = means.shape
    if isr.shape != (m, n):
        raise ValueError(f"Expected isr.shape={(m,n)} but got {isr.shape}")
    nf = len(freq)
    if hits.shape != (m, n, nf):
        raise ValueError(f"Expected hits.shape={(m,n,nf)} but got {hits.shape}")

    ds_time = group.create_dataset("nativeDopplerTime", data=t.astype("f8"))
    ds_range = group.create_dataset("slantRange", data=r.astype("f8"))
    ds_freq = group.create_dataset("frequency", data=freq.astype("f8"))

    ds_mean = group.create_dataset("signalMean", data=means.astype("f4"))
    ds_hits = group.create_dataset("hitCount", data=hits.astype("f4"))
    ds_isr = group.create_dataset("interferenceSignalRatio",
        data=isr.astype("f4"))

    ds_time.attrs["description"] = np.bytes_("Raw data slow-time midpoint of "
        "each analysis block")
    ds_time.attrs["units"] = np.bytes_("s")
    ds_range.attrs["description"] = np.bytes_("Raw data slant range midpoint "
        "of each analysis block")
    ds_range.attrs["units"] = np.bytes_("m")
    ds_freq.attrs["description"] = np.bytes_("Frequency axis for hitCount")
    ds_freq.attrs["units"] = np.bytes_("Hz")

    ds_mean.attrs["description"] = np.bytes_("Estimated mean signal power per "
        "block from lifted exponential model.  Expressed in linear power units,"
        " e.g., DN^2")
    ds_hits.attrs["description"] = np.bytes_("Fraction of pulses with detected "
        "RFI at each frequency bin.  Values range from 0 (no RFI) to 1 (RFI in "
        "all pulses)")
    ds_isr.attrs["description"] = np.bytes_("Interference-to-signal ratio.  "
        "Expressed in linear power units, e.g., DN^2 / DN^2")