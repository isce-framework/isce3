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
    return mask, isr


def remove_loud_tones(
    z,
    block_dims,
    reference_quantile=0.5,
    nominal_false_positive_rate=0.0005,
    bandwidth=5 / 6,
    detect_only=False,
    zout=None,
):
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

    slices_windows = list(cola_windows(z.shape[1], block_dims[1]))
    num_range_blocks = len(slices_windows)
    num_az_blocks = 1 + (z.shape[0] - 1) // block_dims[0]
    zbshape = (block_dims[0], num_range_blocks, block_dims[1])
    z_block = np.zeros(zbshape, z.dtype)

    f = fftshift(fftfreq(block_dims[1]))
    meta_shape = (num_az_blocks, num_range_blocks)
    isr = np.zeros(meta_shape)
    hits = np.zeros(meta_shape + (block_dims[1],), dtype=np.uint32)

    for iblock, block_start in enumerate(range(0, z.shape[0], block_dims[0])):
        block_results = []
        # last block is smaller
        nb = min(block_dims[0], z.shape[0] - block_start)
        block_end = block_start + nb
        rows = slice(block_start, block_end)
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
            mask, isr[iblock,j] = get_spectral_mask(
                spectra[:, j, :],
                reference_quantile,
                nominal_false_positive_rate,
                bandwidth,
            )
            if not detect_only:
                spectra[:, j, :][mask] = 0.0
            hits[iblock, j, :] = fftshift(np.sum(mask, axis=0))
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
