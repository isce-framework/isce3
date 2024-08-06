"""
Functions for noise power estimation based on various algorithms.
"""
from warnings import warn

import numpy as np


class TooFewNoiseLinesWarning(Warning):
    pass


def noise_pow_min_var_est(dset, *, scalar=1, remove_mean=False, diff=True,
                          threshold=True, diff_method='single'):
    """
    Noise power estimation based on Min Variance Estimator (MVE) or
    Maximum Likelihood Estimator (MLE).

    Parameters
    ----------
    dset : np.ndarray
        2-D complex array dataset with shape (range lines, range bins).
        All the range lines are assumed to be almost free from invalid
        values such as zero or NaN.
    scalar : float, default=1.0
        A scalar within (0, 1] applied to the estimator to get the
        noise power for a receive polarization channel such as H or V.
        In case the dataset is obtained from more than one set of
        observations, the scalar shall be set to a value less than 1
        determined by number of datasets and their respective range
        sampling rates!
    remove_mean : bool, default=False
        If True, the mean is assumed to be large enough to be removed.
        Default is assumed that the mean of data block is close to zero.
    diff : bool, default=True
        If True and there is enough range lines, the variance of differential
        dataset w.r.t to the one of the range line (e.g, last one) is
        calculated.
        In this case, the value of `scalar` is halved internally in order
        to estimate the noise power under the assumption that noise-only
        range lines are I.I.D.
    threshold : bool, default=True
        If true, it ignores samples with amplitude greater than two standard
        deviations above the square root of initially estimated noise power.
        This is to reduce large biases due to undesired contributors like
        RFI or caltone(or its residual after diff) to the additive thermal
        noise.
    diff_method : {'single', 'mean', 'diff'}
        It sets the method for differentiating the range lines (axis=0) only
        if `diff` is True. This will be ignoed in single range line case.
        For `single` (default), the difference of all range lines
        wrt a single range line will be used in the noise estimator.
        For `mean`, the difference of all range lines wrt the
        mean over all range lines will be used in the noise estimator.
        For `diff`, the consecutive difference of a pair of adjacent
        range lines will be used in the noise estimator.
        In all cases, it is assumed that the noise is identically and
        independently distributed (IID) over entire range lines.

    Returns
    -------
    float
        Noise power in digital numbers squared (DN ** 2) in linear scale.

    Warnings
    --------
    TooFewNoiseLinesWarning
        Issued when there is only one noise-only valid range line.

    Notes
    -----
    While the variance of the estimator is not generally minimized and
    unbiased, with more than one independent observations of the same
    noise population, the variance of the estimator can be reduced towards
    Cramer-Rao lower bound with less bias due to undesired signals like tone,
    RFI, etc on top of insufficient samples of the noise population.

    Key assumptions: additive thermal noise is relatively white
    (within entire TX bandwidth `A + B`) and is relatively stationary
    (fixed first- and second-order moment over entire datatake/slow-time).

    The simple idea here is similar to the MLE in [1]_ but is applied in a
    different way for a different purpose. See notes in [2]_ for details.

    Amplitude thresholding is applied to reduce possible large biases
    due to RFI and residual caltone even after differentiating range lines.
    The amplitude threshold is based on two-sigma of the square root of
    initial noise power estimate (so-called noise variance).

    References
    ----------
    .. [1] M. Villano, "SNR and Noise Variance Estimation in Polarimetric SAR
        Data," IEEE Geosci. Remote Sens., Lett., vol. 11, pp. 278-282,
        January 2014.
    .. [2] H. Ghaemi, "NISAR Noise Power and NESZ Estimation Strategies and
        Anlyses,", JPL Report, Rev A, April 2024.

    """
    if dset.ndim != 2:
        raise ValueError(f'Expected 2-D array but got {dset.ndim}-D array!')
    nrgls, _ = dset.shape
    if nrgls == 1 or diff is False:
        warn('Noise MVE could be largely biased!',
             category=TooFewNoiseLinesWarning
             )
        dset1 = dset
    else:
        # there exists more than one range line. Let's reduce the
        # bias as well as the variance of estimate to some extent.
        if diff_method == 'single':
            dset1 = dset[:-1] - dset[-1]
            scalar = 0.5 * scalar
        elif diff_method == 'mean':
            dset1 = dset - np.nanmean(dset, axis=0)
        elif diff_method == 'diff':
            dset1 = np.diff(dset, axis=0)
            scalar = 0.5 * scalar
        else:
            raise ValueError(f'Wrong diff method {diff_method}! '
                             'Must be either of "single", "mean", or "diff".')

    sig_var = np.nanmean(abs(dset1) ** 2)
    # remove the mean squared if requested.
    if remove_mean:
        sig_var -= abs(np.nanmean(dset1)) ** 2
    if threshold:
        # filter out outliers based on overestimated two sigma (~96%)
        thrsh = 2.0 * np.sqrt(sig_var)
        abs_dset = abs(dset1)
        sig_var = np.nanmean(abs_dset[abs_dset < thrsh] ** 2)
    return scalar * sig_var


def noise_pow_min_eigval_est(dset, cpi=2, *, scalar=1, remove_mean=False,
                             median_ev=False):
    """
    Noise power estimation based on Min Eigenvalue Estimator (MEE).

    Parameters
    ----------
    dset : np.ndarray
        2-D complex array dataset with shape (range lines, range bins).
        All the range lines are assumed to be valid ones almost free from
        invalid values!
    cpi : int, default=2
        Number of range lines representing coherent processing interval.
        This shall be equal or greater than 2!
    scalar : float, default=1.0
        A scalar within (0, 1] applied to the estimator to get the
        noise power for a receive polarization channel such as H or V.
        In case the dataset is obtained from more than one set of
        observations, the scalar shall be set to a value less than 1
        determined by number of datasets and their respective range
        sampling rates!
    remove_mean : bool, default=False
        If True, the mean is assumed to be large enough to be removed.
        Default is assumed that the mean of data block is close to zero.
    median_ev : bool, default=False
        If True, noise power is the median of the first smallest
        `cpi - 1` eigenvalues. If False, simply the min eigenvalue will
        be reported as noise power.

    Returns
    -------
    float
        Noise power (DN ** 2) in linear scale.

    Notes
    -----
    Key assumptions: additive thermal noise is relatively white
    (within entire TX bandwidth `A + B`) and is relatively staionary
    (fixed second-order moment over entire datatake/slow-time).

    The simple idea here is similar to the MEE in [1]_ but applied in a
    different way for a different purpose. See notes in [2]_ for details.

    References
    ----------
    .. [1] I. Hajnsek, E. Pottier, and S.R. Cloude, "Inversion of Surface
        Parameters from Polarimetric SAR, " IEEE Trans. Geosci. Remote Sens.
        , vol 41, pp. 727-744, April 2003.
    .. [2] H. Ghaemi, "NISAR Noise Power and NESZ Estimation Strategies and
        Anlyses,", JPL Report, Rev A, April 2024.

    """
    if cpi < 2:
        raise ValueError('CPI is less than min 2!')
    if dset.ndim != 2:
        raise ValueError(f'Expected 2-D array but got {dset.ndim}-D array!')
    nrgls, _ = dset.shape
    if nrgls < cpi:
        raise ValueError(
            f'Valid noise lines # {nrgls} v.s. CPI # {cpi}!'
        )
    nrgls, nrgbs = dset.shape
    # loop over CPI slices to compute cov matrix.
    # One can accumulate/average Cov Mat prior to eigenvalue decomposition.
    # This can overestimate additive noise and lead to estimation of
    # thermal noise plus residual-tone plus weak rfi (if any).
    # The outcome is closer to MVE representing total background noise.
    # On the other hand, one can estimate noise power locally within CPI and
    # eventually take a median or mean of all noise power estimations over
    # all CPIs/range lines. This will lead to noise power est closer to the
    # thermal noise. The assumption is that noise stat is stationary over
    # all range lines.
    cpi_slices = _cpi_slice_gen(nrgls, cpi)
    pw_ns_all = []
    for cpi_slice in cpi_slices:
        d = dset[cpi_slice]
        if remove_mean:
            d -= np.nanmean(d)
        cov_mat = d @ d.conj().T
        # noise observation is assumed to be independent per cpi block.
        # Whereas the systematic terms are added coherently and thus well
        # separated from small random contribution of noise!
        cov_mat *= 1 / nrgbs
        # get min eigen value from averaged cov matrix and scale the outcome.
        # note to ignore relatively very small imag part in diagonal term
        # of an approximately Hermitian matrix (lead to positive definite)!
        # sort eigen values in ascending order.
        eig_vals = np.linalg.eigvalsh(cov_mat)
        # mean or median the first CPI - 1 eigen values and then
        # scale the final power to get true noise power.
        # In case of CPI=2, this is the min eigen value!
        if median_ev:
            pow_noise = np.nanmedian(eig_vals[:cpi - 1])
        else:
            pow_noise = eig_vals[0]
        pw_ns_all.append(pow_noise)
    # use median to exclude outliers due to RFI, etc.
    pow_noise = scalar * np.nanmedian(pw_ns_all)
    return pow_noise


# helper function
def _cpi_slice_gen(nrgl, cpi):
    """Helper function for CPI slice generator

    Parameters
    ----------
    nrgl : int
        Total number of range lines
    cpi : int
        The number of lines within a cpi.
        It is assumed that cpi is not larger than nrgl!

    Yields
    ------
    slice
        Slice of (start, stop) range lines.

    Notes
    -----
    The output guarantees fixed-length CPI per CPI block.

    """
    for i_start in range(0, nrgl, cpi):
        i_stop = min(i_start + cpi, nrgl)
        i_start = i_stop - cpi
        yield slice(i_start, i_stop)
