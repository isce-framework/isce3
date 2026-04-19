import logging
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from . import cola_windows

log = logging.getLogger("isce3.signal.rfi")


def exp_from_quantile(p, vp, bw=1.0):
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
    isr = np.sum(power_spectra[mask]) / np.sum(power_spectra[~mask])
    mask.shape = spectra.shape
    return mask, isr, λ


def fill_missing(z, fd, t, mask_replace, mask_valid, noise):
    m, n = z.shape
    if len(t) != m:
        raise ValueError(f"expected len(t)=={m} got {len(t)}")
    if mask_replace.shape != (m, n):
        raise ValueError(f"{mask_replace.shape=} does not match {z.shape=}")
    if mask_valid.shape != (m, n):
        raise ValueError(f"{mask_valid.shape=} does not match {z.shape=}")

    # Deramp Doppler.
    deramp = np.exp(-1j * 2 * np.pi * fd * t).astype(z.dtype)
    zout = deramp[:, None] * z

    for i in range(m):
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

        # copy for modification
        cols_need_replacement = mask_replace[i, :].copy()

        # Four cases for replacement:
        # 1. prev and next both valid -> lerp between them
        j = np.where(mask_valid[iprev, :] & mask_valid[inext, :]
            & cols_need_replacement)[0]
        zout[i, j] = w_prev * z[iprev, j] + w_next * z[inext, j]
        cols_need_replacement[j] = False

        # 2. only prev valid. use it
        j = np.where(mask_valid[iprev, :] & cols_need_replacement)[0]
        zout[i, j] = z[iprev, j]
        cols_need_replacement[j] = False

        # 3. only next valid. use it
        j = np.where(mask_valid[inext, :] & cols_need_replacement)[0]
        zout[i, j] = z[inext, j]
        cols_need_replacement[j] = False

        # 4. prev and next both invalid -> fill noise
        j = np.where(cols_need_replacement)[0]
        noise_idx = ((i * n) + j) % len(noise)
        zout[i, j] = noise[noise_idx]

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
):
    """
    t : np.ndarray [float64]
        Pulse times (seconds since orbit/grid epoch).
    r : isce3.core.Linspace
        Range to each sample (meters).
    swaths : np.ndarray [int]
        Valid subswath samples, dims = (ns, nt, 2) where ns is the number of
        sub-swaths, nt is the number of pulses, and the trailing dimension is
        the [start, stop) indices of the sub-swath.
    doppler : isce3.core.LUT2d [double]
        Raw data Doppler look up table.  Must be valid over entire grid.
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
    num_az_blocks = 1 + (z.shape[0] - 1) // block_dims[0]
    zbshape = (block_dims[0], num_range_blocks, block_dims[1])
    z_block = np.zeros(zbshape, z.dtype)

    f = fftshift(fftfreq(block_dims[1]))
    meta_shape = (num_az_blocks, num_range_blocks)
    isr = np.zeros(meta_shape)
    hits = np.zeros(meta_shape + (block_dims[1],), dtype=np.uint32)

    block_ranges = np.zeros(num_range_blocks)
    for j, (cols, _) in enumerate(slices_windows):
        block_ranges[j] = r[(cols.start + cols.stop) // 2]

    for iblock, block_start in enumerate(range(0, z.shape[0], block_dims[0])):
        block_results = []
        # last block is smaller
        nb = min(block_dims[0], z.shape[0] - block_start)
        block_end = block_start + nb
        rows = slice(block_start, block_end)
        # calculate azimuth time of block
        block_time_mid = t[block_start + nb // 2]
        block_times = t[rows]
        # populate valid data mask
        mask_valid[...] = False
        for i, i_pulse in enumerate(range(block_start, block_end)):
            for start, end in swaths[:, i_pulse, :]:
                mask_valid[i, start:end] = True
        # apply window to each range block
        for j, (cols, window) in enumerate(slices_windows):
            nw = len(window)
            z_block[:nb, j, :nw] = window[None, :] * z[rows, cols]
        # crop for last azimuth block
        if nb < block_dims[0]:
            z_block = z_block[:nb, ...]
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
                num_noise = min(np.sum(mask_replace),
                    z_block.shape[1] * 991 // 97)
                noise = circular_gaussian_noise(num_noise, σ)
                fd = doppler.eval(block_time_mid, block_ranges[j])
                cols, window = slices_windows[j]
                nw = len(window)
                mask_valid_blk = np.zeros((nb, block_dims[1]), bool)
                mask_valid_blk[:, :nw] = mask_valid[:nb, cols]
                spectra[:, j, :] = fill_missing(spectra[:,j,:], fd, block_times,
                    mask_replace, mask_valid_blk, noise)
            hits[iblock, j, :] = fftshift(np.sum(mask_replace, axis=0))
        # skip inverse FFTs and assignment if not required.
        if not detect_only:
            # range inverse STFT
            z_block[...] = ifft(spectra, axis=2)
            # sum COLA range blocks into output buffer
            zout[rows, ...] = 0
            for j, (cols, window) in enumerate(slices_windows):
                nw = len(window)
                zout[rows, cols] += z_block[:nb, j, :nw]

    return isr, f, hits
