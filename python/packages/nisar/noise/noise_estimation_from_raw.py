"""
Functions and classes for noise power estimation from Raw data
"""
from warnings import warn
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from isce3.noise import noise_pow_min_var_est, noise_pow_min_eigval_est
from isce3.focus import fill_gaps
from nisar.antenna import get_calib_range_line_idx
from nisar.log import set_logger
from isce3.core import DateTime


# Global Noise-related Constants
# Min number of range bins recommended per noise range block
RGB_MIN_NOISE = 50
# A default threshold that defines percentage of total number of range bins
# per range block above which the noise range block is assumed to be invalid.
PERC_INVALID_NOISE = 10
# Invalid float value to fill in TX gaps if TX gap is not already mitigated
INVALID_VALUE = 0
# Suggested max number of range lines per CPI
MAX_CPI_LEN = 100


class TooShortNoiseRangeBlockWarning(Warning):
    pass


class InvalidNoiseRangeBlockWarning(Warning):
    pass


class ZeroBandwidthNoiseWarning(Warning):
    pass


@dataclass
class NeszProduct:
    """Noise Equivalent Sigma Zero (NESZ) Product.

    Attributes
    ----------
    power_linear : 2-D array of float
        Noise power in linear scale in (DN ** 2) as a function of range
        provided at several AZ time stamps with shape=(az times, slant ranges)
    slant_range : 1-D array of float
        Slant range vector in (m). Must be the same size as the number of
        columns in `power_linear`.
    az_time : 1-D array of float
        Azimuth times in seconds w.r.t. reference epoch.
        Must be the same size as the number of rows in `power_linear`.
    ref_epoch : isce3.core.DateTime
        Reference epoch
    freq_band : str
        Frequency band char such as A or B.
    txrx_pol : str
        TxRx polarization, such as HH, HV, etc.
        What actually matters is the RX pol!

    """
    power_linear: np.ndarray
    slant_range: np.ndarray
    az_time: np.ndarray
    ref_epoch: DateTime
    freq_band: str
    txrx_pol: str

    def __post_init__(self):
        if self.txrx_pol[1] not in ('H', 'V'):
            raise ValueError('RX Pol must be either "H" or "V"!')
        if self.power_linear.shape != (self.az_time.size,
                                       self.slant_range.size):
            raise ValueError(
                'Shape of noise power array does not match '
                '(size of AZ time, size of slant-range)!'
            )


@dataclass
class NoiseEstProduct:
    """Noise estimator product.

    Attributes
    ----------
    power_linear : 1-D array of float
        Noise power in linear scale in (DN ** 2) as a function of range
    slant_range : 1-D array of float
        Slant range vecTor in (m). Must be the same size as `power_linear`
    enbw : float
        Equivalent noise bandwidth (ENBW) in (Hz)
    txrx_pol : str
        TxRx polarization, such as HH, HV, etc.
        What actually matters is the RX pol!
    freq_band : str
        Frequency band char such as A or B.
    method : str
        Method name used for noise estimation.

    """
    power_linear: np.ndarray
    slant_range: np.ndarray
    enbw: float
    txrx_pol: str
    freq_band: str
    method: str

    def __post_init__(self):
        if not (self.enbw > 0):
            raise ValueError('ENBW must be positive value!')
        if self.txrx_pol[1] not in ('H', 'V'):
            raise ValueError('RX Pol must be either "H" or "V"!')
        if len(self.slant_range) != len(self.power_linear):
            raise ValueError(
                'Size mismatch between Noise Power and Slant-range arrays!'
            )


def extract_noise_only_lines(raw, freq_band, txrx_pol):
    """Extract noise-only range lines from a L0B raw dataset.

    Parameters:
    -----------
    raw: nisar.products.readers.Raw
        Raw L0B product reader
    freq_band: str
        frequency band such as 'A' or 'B'
    txrx_pol: str
        Tx and Rx polarization such as
        'HH', 'HV', 'VH', 'VV', 'LH', 'LV', 'RH', 'RV'

    Returns:
    --------
    2D array of complex
        Noise-only echo data
    1-D array of int
        Noise-only true range line indexes

    """
    cal_path_mask = raw.getCalType(freq_band, tx=txrx_pol[0])
    _, _, _, noise_index = get_calib_range_line_idx(cal_path_mask)
    dset = raw.getRawDataset(freq_band, txrx_pol)
    return dset[noise_index], noise_index


def enbw_from_raw(raw, freq_band, tx_pol):
    """
    Get approximate ENBW (equivalent noise bandwidth) in (Hz) from raw per
    desired frequency band and TxRx Pol by assuming the spectrum is shaped
    roughly like a trapezoid.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
        Input NISAR L0B product reader
    freq_band : str
        frequency band char "A", "B"
    tx_pol : str
        TX polarization such as "H", "V"

    Returns
    -------
    float
        ENBW in Hz.

    Warnings
    --------
    ZeroBandwidthNoiseWarning
        Issued if TX bandwidth is zero!
        In this case, the bandwidth is set to a value equals to
        (sampling rate / 1.2)!

    Notes
    -----
    A nearly perfect trapezoidal envelope of range spectrum is assumed, that
    is a fixed value, flat, over TX bandwidth while zero value at the Nyquist.
    Thus, the ENBW is the average between sampling rate and bandwidth.

    """
    _, fs, cr, pw = raw.getChirpParameters(freq_band, tx_pol)
    if np.isclose(fs, 0):
        raise ValueError(
            f'Sampling rate for band "{freq_band}" and '
            f'TX Pol "{tx_pol}" is zero!')
    bw = abs(cr * pw)
    # assumed perfect trapezoidal envelope of range spectrum, that is
    # a fixed value, flat, over TX bandwidth while zero value at the Nyquist.
    # Thus, the ENBW is the average between sampling rate and bandwidth.
    # Users can improve ENBW by a fudge factor to account for
    # non-trapezoidal shape.
    if np.isclose(bw, 0):
        bw = fs / 1.2
        warn(f'BW for band "{freq_band}" and TX Pol "{tx_pol}" is assumed'
             ' to be (1 / 1.2) of sampling rate!',
             category=ZeroBandwidthNoiseWarning)
    enbw = 0.5 * (fs + bw)
    return enbw


def est_noise_power_from_raw(
        raw, *, num_rng_block=None, algorithm='MEE', cpi=None, diff=True,
        diff_method='single', median_ev=True, dif_quad=False,
        remove_mean=False, perc_invalid_rngblk=PERC_INVALID_NOISE,
        exclude_first_last=False, logger=None):
    """Estimate noise power from a L0B raw product.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
    num_rng_block : int, optional.
        Number of range blocks.
        Default is set by min required `2 * C  - 1` where `C`
        is the number of active RX channels.
    algorithm : {'MEE', 'MVE'}
        MVE: min var estimator based on maximum likelihood (ML) [1]_.
        MEE: min eigenvalue estimator based on eigenvalue decomposition (EVD)
        [2]_.
    cpi : int, optional
        Number of range lines of coherent processing interval (CPI)
        used simply in MEE. Must be greater than 1!
        If None, it will be set based on number of valid noise-only range
        lines. The min desired CPI is set to 3 while the max number of CPI
        blocks is set to 8 to obtain the default cpi if there exists enough
        valid noise-only range lines.
    diff : bool, default=True
        If True and there is enough range lines`, the
        differential dataset will be used in the MVE noise estimator.
    diff_method : {'single', 'mean', 'diff'}
        It sets the method for differentiating the range lines (axis=0) only
        if `diff` is True and algorithm=`MVE`.
        This will be ignoed in single range line case.
        For `single` (default), the difference of all range lines
        wrt a single range line will be used in the noise estimator.
        For `mean`, the difference of all range lines wrt the
        mean over all range lines will be used in the noise estimator.
        For `diff`, the consecutive difference of a pair of adjacent
        range lines will be used in the noise estimator.
        In all cases, it is assumed that the noise is identically and
        independently distributed (IID) over entire range lines.
    median_ev : bool, default=True
        If True, noise power is the median of the first smallest
        `cpi - 1` eigenvalues. If False, simply the min eigenvalue will
        be reported as noise power. This is only used in MEE.
    dif_quad : bool, default=False
        If True, it will differentiate Co/Cx-pol datasets
        with the same RX pol in a joint noise est for quad-pol cases.
        This assumes that different TX setups has no impact on a RX channel!
    remove_mean : bool, default=False
        If True, the mean is assumed to be large enough to be removed.
        Default is assumed that the mean of data block is close to zero.
    perc_invalid_rngblk : float, default=PERC_INVALID_NOISE
        If the percentage of a range block with invalid values is above
        this threshold, the noise power estimation is skipped and set to NaN.
        This value shall be within [0, 100]
    exclude_first_last : bool, default=False
        If true, the first and the last noise-only range lines will be
        excluded in noise estimation.
        In NISAR case, the first one may not be scaled by onboard
        RX cal while the last one can be affected by a new config at the
        transition mode. Thus, to avoid any bias, it is safer to exclude them.
    logger : logging.Logger, optional

    Returns
    -------
    List of NoiseEstProduct
        List of noise products over all frequency bands and polarizations.

    Warnings
    --------
    TooFewNoiseLinesWarning
        Issued when there is only one noise-only valid range line.

    See Also
    --------
    noise_pow_min_var_est
    noise_pow_min_eigval_est

    Notes
    -----
    Key assumptions: additive thermal noise is relatively white
    (within entire TX bandwidth `A + B`) and is relatively staionary
    (fixed second-order moment over entire datatake/slow-time).

    The simple idea here is similar to the [1-2]_ but applied in a
    different way for a different purpose. See notes in [3]_ for details.

    Currently, the science-mode L0B is simply supported but not the
    diagnostic ones!

    References
    ----------
    .. [1] M. Villano, "SNR and Noise Variance Estimation in Polarimetric SAR
        Data," IEEE Geosci. Remote Sens., Lett., vol. 11, pp. 278-282,
        January 2014.
    .. [2] I. Hajnsek, E. Pottier, and S.R. Cloude, "Inversion of Surface
        Parameters from Polarimetric SAR, " IEEE Trans. Geosci. Remote Sens.
        , vol 41, pp. 727-744, April 2003.
    .. [3] H. Ghaemi, "NISAR Noise Power and NESZ Estimation Strategies and
        Anlyses,", JPL Report, Rev A, April 2024.

    """
    # const
    methods = ('MEE', 'MVE')
    # check if the L0B is in science/DBF mode
    dm_flag = raw.identification.diagnosticModeFlag
    if dm_flag != 0:
        raise NotImplementedError(
            f'Only science mode is supported not mode # {dm_flag}!')
    # check some values
    if perc_invalid_rngblk < 0 or perc_invalid_rngblk > 100:
        raise ValueError('"perc_invalid_rngblk" shall be within [0, 100]!')
    # set logger if None
    if logger is None:
        logger = set_logger('est_noise_power_from_raw')

    logger.info(f'The noise estimator algorithm -> {algorithm}')
    if algorithm not in methods:
        raise ValueError(f'only algorithms {methods} are supported!')

    # parse all polarizations
    frq_pol = raw.polarizations
    # the very first frequency band
    freq = sorted(raw.frequencies)[0]
    is_dither = raw.isDithered(freq)
    logger.info(f'PRF dithering -> {is_dither}')

    # get list of RX channels from the first frequency band given they shall
    # be the same over all bands and also over all polarizations of NISAR
    # nominal science modes.
    list_rx = raw.getListOfRxTRMs(freq, frq_pol[freq][0])
    num_rx = len(list_rx)
    logger.info(f'Number of RX channels per pol -> {num_rx}')

    # determine the min and final number of range blocks
    n_rg_blk_min = 2 * num_rx - 1
    logger.info(f'Min recommended number of range blocks -> {n_rg_blk_min}')

    if num_rng_block is None:
        num_rng_block = n_rg_blk_min
    else:
        if num_rng_block < 1:
            raise ValueError('Number of range blocks must a positive integer!')
        if num_rng_block < n_rg_blk_min:
            logger.warning(
                f'Number of range blocks is smaller than min {n_rg_blk_min}'
            )
    logger.info(f'Number of range blocks -> {num_rng_block}')

    # container for all noise products
    noise_prods = []
    # if quad pol, then do MVE or MEE
    # where co-pol and cx-pol with the same receiver pol will
    # be combined provided that `dif_quad=True`.
    # The same could be done for quasi-quad if both bands have equal
    # sampling rate and ENBW! However, that's not true for all QQs of
    # L-band NISAR.
    # loop over freq bands
    for freq_band in frq_pol:
        # check if it is QP and product differentiation is set to True
        if dif_quad and _is_quad_pol(frq_pol[freq_band]):
            logger.info('The difference of co-pol and cx-pol with'
                        ' the same RX pol will be used in Noise est!')
            # let's combine datasets with the same RX Pol
            # to reduce variance of noise estimation while
            # removing undesired deterministic signals
            # repeated almost equally in both products.
            # Thus, simply loop over RX pols per band!

            for rx_pol in ('H', 'V'):
                txrx_pols = [tx_pol + rx_pol for tx_pol in ('H', 'V')]
                logger.info(
                    'Processing TX co-pol and cx-pol jointly for frequency '
                    f'band {freq_band} and Rx Pol {rx_pol} ...'
                )
                # parse two noise datasets
                dset_noise1, idx_rgl_ns = extract_noise_only_lines(
                    raw, freq_band, txrx_pols[0])
                dset_noise2, _ = extract_noise_only_lines(
                    raw, freq_band, txrx_pols[1])
                assert dset_noise1.shape == dset_noise2.shape, (
                    f"Shape mismatch between {txrx_pols[0]} and {txrx_pols[1]}"
                )
                # subtract the two products with the same RX pol
                dset_noise = dset_noise1 - dset_noise2
                if exclude_first_last:
                    logger.info(
                        'Exclude the first and last noise range lines.')
                    dset_noise = dset_noise[1:-1]
                    idx_rgl_ns = idx_rgl_ns[1:-1]
                # get noise product
                ns_prod = _noise_product_rng_blocks(
                    raw, dset_noise, idx_rgl_ns, freq_band, 2 * rx_pol,
                    is_dither, algorithm, cpi, num_rng_block, 0.5, diff,
                    diff_method, median_ev, remove_mean,
                    perc_invalid_rngblk, logger
                )
                noise_prods.append(ns_prod)

        else:  # other pol types than QP
            for txrx_pol in frq_pol[freq_band]:
                logger.info(
                    'Processing individually frequency band '
                    f'{freq_band} and Pol {txrx_pol} ...'
                )
                # parse one noise dataset
                dset_noise, idx_rgl_ns = extract_noise_only_lines(
                    raw, freq_band, txrx_pol)
                if exclude_first_last:
                    logger.info(
                        'Exclude the first and last noise range lines.')
                    dset_noise = dset_noise[1:-1]
                    idx_rgl_ns = idx_rgl_ns[1:-1]
                # get noise product
                ns_prod = _noise_product_rng_blocks(
                    raw, dset_noise, idx_rgl_ns, freq_band, txrx_pol,
                    is_dither, algorithm, cpi, num_rng_block, 1.0, diff,
                    diff_method, median_ev, remove_mean,
                    perc_invalid_rngblk, logger
                )
                noise_prods.append(ns_prod)

    return noise_prods


# Helper functions
def _pow2db(p: float) -> float:
    """Linear power to dB"""
    return 10 * np.log10(p)


def _is_quad_pol(txrx_pols):
    """
    Whether the list of two-char TxRx Pols represents linear quad
    polarization or not.

    Parameters
    ----------
    txrx_pols : List of str
        List of TxRx polarizations

    Returns
    -------
    bool :
        True only if the pol list represents linear quad pol.
    """
    return set(txrx_pols) == {'HH', 'HV', 'VH', 'VV'}


def _range_slice_gen(n_rgb, n_blk, min_rgbs=RGB_MIN_NOISE):
    """Helper function to generate range block slice.

    Parameters
    ----------
    n_rgb : int
        Total number of range bins
    n_blk : int
        Number of blocks in range
    min_rgbs : int, default=RGB_MIN_NOISE
        Min required number of range bins.

    Yields
    ------
    slice
        Range bin slice

    Warnings
    --------
    TooShortNoiseRangeBlockWarning
        If number of samples per range block is less than "min_rgbs".

    """
    if n_blk > n_rgb:
        raise ValueError(
            f'Number of range blocks {n_blk} is larger than total '
            f'range bins {n_rgb}!'
        )
    n_bins = n_rgb // n_blk
    if n_bins < min_rgbs:
        warn(f'Number of range bins -> {n_bins} less than {RGB_MIN_NOISE}!',
             category=TooShortNoiseRangeBlockWarning)

    for n in range(n_blk):
        i_start = n * n_bins
        i_stop = i_start + n_bins
        if n == (n_blk - 1):
            i_stop = n_rgb
        yield slice(i_start, i_stop)


def _check_noise_validity(
        noise, is_dither, *, perc_invalid=PERC_INVALID_NOISE,
        invalid_val=INVALID_VALUE):
    """
    Helper function to check whether some noise-only range lines within a
    range block are valid and returns list of relative valid range line
    indices.

    The validity is defined based on portion of range lines filled with
    invalid value (zero for L0B!) due to TX gaps and/or varying RD/WD/WL.
    If the portion is more than `perc_invalid` %, the entire range line is
    considered invalid.

    Parameters
    ----------
    noise : np.ndarray
        2-D array of noise-only range lines for a single range block.
        The shape is (range lines, range bins)
    is_dither : bool
        Whether it is dithering PRF or a fixed one.
    perc_invalid : float, default=PERC_INVALID_NOISE
        Percentage of range bins being masked by invalid values that make the
        corresponding range line invalid if above this threshold.
        A value within [0, 100].
    invalid_val : float, default=INVALID_VALUE
        Invalid float value such as zero for TX gap regions or other no-echo
        regions in the decoded raw data.

    Returns
    -------
    bool
        Whether the range block is valid over at least one range line.
    list
        List of valid rangeline relative indices

    Notes
    -----
    If PRF is fixed, then simply the first range line is checked for validity.
    It is assumed the number of non-tx-gap trailing or leading zeros which can
    vary over range lines is way smaller than those of tx gaps!
    This assumption can be ignored by setting `is_dither` to True!

    """
    nrgls, nrgbs = noise.shape
    ratio_invalid = perc_invalid / 100
    valid_lines = []
    if is_dither:  # check every range line
        for line in range(nrgls):
            n_bad = np.isclose(noise[line], invalid_val, equal_nan=True).sum()
            ratio_bad = n_bad / nrgbs
            if ratio_bad <= ratio_invalid:
                valid_lines.append(line)
    else:  # just check out the first range line
        n_bad = np.isclose(noise[0], invalid_val, equal_nan=True).sum()
        ratio_bad = n_bad / nrgbs
        if ratio_bad <= ratio_invalid:
            valid_lines = list(range(nrgls))
    return len(valid_lines) != 0, valid_lines


def _noise_product_rng_blocks(raw, dset_noise, idx_rgl_ns, freq_band,
                              txrx_pol, is_dither, algorithm, cpi,
                              num_rng_block, scalar, diff, diff_method,
                              median_ev, remove_mean, perc_invalid_rngblk,
                              logger):
    """Helper function ot get noise product per frequency band and RX Pol.

    Parameters
    ----------
    raw : nisar.products.reader.Raw
    dset_noise : np.ndarray
        2-D array of noisy dataset with shape (range lines, range bins)
    idx_rgl_ns : np.ndarray
        1-D array of indexes for noise-only range lines.
    freq_band : str, {'A', 'B'}
        frequency band char
    txrx_pol : str
        TxRx polarization such as {'HH', 'VV', 'HV', ...}.
    is_dither : bool
        Whether the PRF is dithered or not.
    algorithm : str, {'MEE', 'MVE'}
        Noise estimator algorithm.
    cpi : int
        Number of range lines of CPI (coherent processing interval)
        used simply in MEE.
    num_rng_block : int
        Number of range blocks
    scalar : float
        A scalar within (0, 1] applied to the estimator to get the
        noise power for a receive polarization channel such as H or V.
        In case the dataset is obtained from more than one set of
        observations, the scalar shall be set to a value less than 1
        determined by number of datasets.
    diff : bool
        If True and there is enough range lines`, the
        differential dataset will be used in the MVE noise estimator.
    diff_method : {'single', 'mean', 'diff'}
        It sets the method for differentiating the range lines (axis=0) only
        if `diff` is True and algorithm=`MVE`.
        This will be ignoed in single range line case.
        For `single` (default), the difference of all range lines
        wrt a single range line will be used in the noise estimator.
        For `mean`, the difference of all range lines wrt the
        mean over all range lines will be used in the noise estimator.
        For `diff`, the consecutive difference of a pair of adjacent
        range lines will be used in the noise estimator.
        In all cases, it is assumed that the noise is identically and
        independently distributed (IID) over entire range lines.
    median_ev : bool
        Whether or not use median of `cpi-1` eigenvalues in MEE.
        If True, noise power is the median of the first smallest `cpi - 1`.
        If False, simply the min eigenvalue will be reported as noise power.
    remove_mean : bool
        If True, the mean is assumed to be large enough to be removed.
    perc_invalid_rngblk : float
        If the percentage of a range block with invalid values is above
        this threshold, the noise power estimation is skipped and set to NaN.
    logger : logging.Logger

    Returns
    -------
    NoiseEstProduct
        Noise estimator product dataclass

    """
    # Constants
    # The number of CPI blocks is limited to 8 given around 40-sec
    # L0B duration with radar parameters updates of 10-sec interval.
    max_num_cpi_blocks = 8

    # blocking in range
    nrgls, nrgbs = dset_noise.shape
    logger.info('Number of noise-only range (lines, bins) '
                f'-> ({nrgls}, {nrgbs})')

    # parse valid sub-swath for noise-only range lines
    sbsw_ns = raw.getSubSwaths(freq_band, txrx_pol[0])[:, idx_rgl_ns]
    # fill-in TX gap regions with invalid value for noise-only range lines
    # This is to guarantee TX gap regions are mitigated and filled with
    # a common invalid value!
    fill_gaps(dset_noise, sbsw_ns, INVALID_VALUE)
    # get slant range vector
    sr_lsp = raw.getRanges(freq_band, txrx_pol[0])
    # get range block slices
    rg_slices = _range_slice_gen(nrgbs, num_rng_block)
    # initialize the outputs
    pow_noise = np.full(num_rng_block, np.nan, dtype='f8')
    sr_noise = np.zeros(num_rng_block, dtype='f8')
    # loop over range blocks
    for nn, rg_slice in enumerate(rg_slices):
        # assign a slant range to each noise range block.
        # Either calculate the mid range in each range block
        # or simply assign the starting slant range as a
        # slant range representative of that block.
        # Herein, we use the mid slantrange within a range block.
        sr_noise[nn] = 0.5 * (sr_lsp[rg_slice.stop - 1] +
                              sr_lsp[rg_slice.start])
        # check if the range bins mostly affected by TX gap
        # or invalid values!
        noise_rng_blk = dset_noise[:, rg_slice]
        # check the validity of the noise data given TX gaps and other
        # invalid range bins.
        flag_valid, idx_valid = _check_noise_validity(
            noise_rng_blk, is_dither, perc_invalid=perc_invalid_rngblk)
        nrgl_valid = len(idx_valid)
        logger.info('Number of valid noise-only '
                    f'range lines -> {nrgl_valid}')
        if not flag_valid:
            warn(f'Skip noise est for range block # {nn + 1}'
                 f' -> {rg_slice}!',
                 category=InvalidNoiseRangeBlockWarning)
            continue
        # run noise estimator per range block
        if algorithm == 'MEE':
            if cpi is None:
                # if not set, set CPI to max possible value equal or
                # greater than 3 with at least two CPI blocks if possible.
                cpi = min(max(
                    min(len(idx_valid), 3),
                    np.ceil(len(idx_valid) / max_num_cpi_blocks).astype(int)
                ), MAX_CPI_LEN)
            elif cpi > MAX_CPI_LEN:
                logger.warning(
                    f'Too large CPI value! It exceeds max {MAX_CPI_LEN}!'
                )
            logger.info(f'MEE CPI size -> {cpi}')
            pow_noise[nn] = noise_pow_min_eigval_est(
                noise_rng_blk[idx_valid], cpi, scalar=scalar,
                remove_mean=remove_mean, median_ev=median_ev)
        elif algorithm == 'MVE':
            pow_noise[nn] = noise_pow_min_var_est(
                noise_rng_blk[idx_valid], scalar=scalar,
                remove_mean=remove_mean, diff=diff,
                diff_method=diff_method)
        else:
            # should be unreachable.
            assert False, f"Unexpected algorithm: {algorithm}"
        # report final est noise power and its slant range
        logger.info(
            'Noise (Power, Slantrange) in (dB, km) -> ('
            f'{_pow2db(pow_noise[nn]):.2f}, '
            f'{sr_noise[nn] * 1e-3:.3f})'
        )
    # store noise product for all blocks
    # calculate approximate ENBW for relatively white noise!
    enbw = enbw_from_raw(raw, freq_band, txrx_pol[0])
    logger.info(f'Approximate ENBW in (MHz) -> {enbw * 1e-6}')
    return NoiseEstProduct(
        pow_noise, sr_noise, enbw, txrx_pol, freq_band, algorithm)


def est_noise_power_in_focus(
        dset_noise, slant_range, subswath_noise, is_dither,
        algorithm='MEE', *, cpi=10, num_range_block=80,
        diff=True, diff_method='single', median_ev=True,
        remove_mean=False,
        threshold_invalid_range_block=PERC_INVALID_NOISE,
        fill_nan_ends=True, logger=None
):
    """
    Helper function to estimate noise power as a function of slant range
    that can be used in "focus.py" workflow.

    Parameters
    ----------
    dset_noise : np.ndarray
        2-D array of noise-only data with shape (range lines, range bins)
    slant_range : isce3.core.Linspace
        Slant range in meters with the same size as the number of columns of
        noise data.
    subswath_noise : np.ndarray
        3-D array representing valid [start,stop) range bins of subswath
        for noise-only range lines with shape
        (number of subswaths, number of noise-only range lines, 2)
    is_dither : bool
        Whether the PRF is dithered or not.
    algorithm : {'MEE', 'MVE'}
        MVE: min var estimator based on maximum likelihood (ML) [1]_.
        MEE: min eigenvalue estimator based on eigenvalue decomposition (EVD)
        [2]_.
    cpi : int, default=10
        Number of range lines of coherent processing interval (CPI)
        used simply in MEE. Must be greater than 1!
        If the number of noise-only range line is less than this value,
        it will be limited to the number of noise-only range lines and
        a warning message will be issued in the log.
        On the other hand, if this is larger than `MAX_CPI_LEN`,
        A warning message will be logged.
    num_range_block : int, default=80
        Number of range blocks. Default is based on nominal
        NISAR 12 active RX channel and min range sampling rate.
    diff : bool, default=True
        If True and there are enough range lines, the
        differential dataset will be used in the MVE noise estimator.
    diff_method : {'single', 'mean', 'diff'}
        It sets the method for differentiating the range lines (axis=0) only
        if `diff` is True and algorithm=`MVE`.
        This will be ignored in single range line case.
        For `single` (default), the difference of all range lines
        wrt a single range line will be used in the noise estimator.
        For `mean`, the difference of all range lines wrt the
        mean over all range lines will be used in the noise estimator.
        For `diff`, the consecutive difference of a pair of adjacent
        range lines will be used in the noise estimator.
        In all cases, it is assumed that the noise is identically and
        independently distributed (IID) over entire range lines.
    median_ev : bool, default=True
        If True, noise power is the median of the first smallest
        `cpi - 1` eigenvalues. If False, simply the min eigenvalue will
        be reported as noise power. This is only used in MEE.
    remove_mean : bool, default=False
        If True, the mean is assumed to be large enough to be removed.
        Default is assumed that the mean of data block is close to zero.
    threshold_invalid_range_block : float, default=PERC_INVALID_NOISE
        If the percentage of a range block with invalid values is above
        this threshold, the noise power estimation is skipped and set to NaN.
        This value shall be within [0, 100]
    fill_nan_ends : bool, default=True
        Fill NaN values introduced by TX gaps as well as missing end points,
        (first, last) range bins, with interpolated/extrapolated noise power
        via Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
        interpolator.
    logger : logging.Logger, optional

    Returns
    -------
    noise_power : np.ndarray[float]
        1-D array of float representing noise power (DN^2) in linear scale
    slant_range : np.ndarray[float]
        1-D array of float representing slant ranges in meters.
        This has the same size as noise power.

    References
    ----------
    .. [1] M. Villano, "SNR and Noise Variance Estimation in Polarimetric SAR
        Data," IEEE Geosci. Remote Sens., Lett., vol. 11, pp. 278-282,
        January 2014.
    .. [2] I. Hajnsek, E. Pottier, and S.R. Cloude, "Inversion of Surface
        Parameters from Polarimetric SAR, " IEEE Trans. Geosci. Remote Sens.
        , vol 41, pp. 727-744, April 2003.
    .. [3] H. Ghaemi, "NISAR Noise Power and NESZ Estimation Strategies and
        Analyses,", JPL Report, Rev A, April 2024.

    """
    # Constants
    # scalar factor set to unity
    scalar = 1.0
    # set logger if None
    if logger is None:
        logger = set_logger('est_noise_power_in_focus')
    # blocking in range
    nrgls, nrgbs = dset_noise.shape
    logger.info('Number of noise-only range (lines, bins) '
                f'-> ({nrgls}, {nrgbs})')
    logger.info(f'Noise estimation algorithm -> {algorithm}')
    # fill-in TX gap regions with invalid value for noise-only range lines
    # This is to guarantee TX gap regions are mitigated and filled with
    # a common invalid value!
    fill_gaps(dset_noise, subswath_noise, INVALID_VALUE)
    if num_range_block < 1:
        raise ValueError(
            'Number of nosie range blocks must a positive integer!')
    logger.info(f'Number of noise range blocks -> {num_range_block}')
    # get range block slices
    rg_slices = _range_slice_gen(nrgbs, num_range_block)
    # initialize the outputs
    pow_noise = np.full(num_range_block, np.nan, dtype='f4')
    sr_noise = np.zeros(num_range_block, dtype='f8')
    # loop over range blocks
    for nn, rg_slice in enumerate(rg_slices):
        # assign a slant range to each noise range block.
        # Either calculate the mid range in each range block
        # or simply assign the starting slant range as a
        # slant range representative of that block.
        # Herein, we use the mid slantrange within a range block.
        sr_noise[nn] = 0.5 * (slant_range[rg_slice.stop - 1] +
                              slant_range[rg_slice.start])
        # check if the range bins mostly affected by TX gap
        # or invalid values!
        noise_rng_blk = dset_noise[:, rg_slice]
        # check the validity of the noise data given TX gaps and other
        # invalid range bins.
        flag_valid, idx_valid = _check_noise_validity(
            noise_rng_blk,
            is_dither,
            perc_invalid=threshold_invalid_range_block
        )
        nrgl_valid = len(idx_valid)
        logger.info('Number of valid noise-only '
                    f'range lines -> {nrgl_valid}')
        if not flag_valid:
            warn(f'Skip noise est for range block # {nn + 1}'
                 f' -> {rg_slice}!',
                 category=InvalidNoiseRangeBlockWarning)
            continue
        # run noise estimator per range block
        if algorithm == 'MEE':
            if nrgl_valid < 2:
                raise ValueError(
                    f'Number of noise-only range lines {nrgl_valid} '
                    'is less than 2 in "MEE"! Use algorithm="MVE" instead!'
                )
            if cpi > nrgl_valid:
                logger.warning(
                    f'CPI={cpi} is larger than valid noise-only range lines '
                    f'{nrgl_valid}. CPI is set to {nrgl_valid}!')
                cpi = nrgl_valid
            if cpi > MAX_CPI_LEN:
                logger.warning(
                    f'Too large CPI value! It exceeds max {MAX_CPI_LEN}!'
                )
            logger.info(f'MEE CPI size -> {cpi}')
            pow_noise[nn] = noise_pow_min_eigval_est(
                noise_rng_blk[idx_valid], cpi, scalar=scalar,
                remove_mean=remove_mean, median_ev=median_ev)
        elif algorithm == 'MVE':
            pow_noise[nn] = noise_pow_min_var_est(
                noise_rng_blk[idx_valid], scalar=scalar,
                remove_mean=remove_mean, diff=diff,
                diff_method=diff_method)
        else:
            raise ValueError('Not a valid algorithm. Either "MEE" or "MVE".')
        # report final est noise power and its slant range
        logger.info(
            'Noise (Power, Slantrange) in (dB, km) -> ('
            f'{_pow2db(pow_noise[nn]):.2f}, '
            f'{sr_noise[nn] * 1e-3:.3f})'
        )
    # fill NaNs if any and add extra end points, near/far, while making sure
    # the samples are uniform in range direction.
    if fill_nan_ends:
        logger.info('Fill NaN values and add extra near/far range bins '
                    'with interpolated noise power via PCHIP!')
        pow_noise, sr_noise = _fill_nan_ends(pow_noise, sr_noise)
        logger.info(f'The final size of noise power array -> {sr_noise.size}')
    else:
        logger.warning('The noise product can be non-uniform in range and '
                       'also carry NaN values! Not suitable for ISCE3 LUT')
    return pow_noise, sr_noise


def _fill_nan_ends(pow_noise, sr_noise):
    """
    Helper function to fill NaN values with valid ones and
    add two end points for full slant range coverage in noise power.
    The NaN filling is done via scipy PCHIP interpolator while
    the end points are extrapolated via first-order polynominal.

    Parameters
    ----------
    pow_noise : np.ndarray(float)
        1-D array of Noise power in linear scale.
    sr_noise : np.ndarray(float)
        1-D array of slant ranges in meters with the same
        size as `pow_noise`.

    Returns
    -------
    noise_power : np.ndarray[float]
        1-D array of updated noise powers with two extra values
    slant_range : np.ndarray[float]
        1-D array of updated slant ranges with two extra values

    Raises
    ------
    RuntimeError
        If number of valid noise samples is less than 4!

    Notes
    -----
    The last range bin for partially overlapped range block will be
    replaced to preserve uniform spacing in slant range prior to adding
    end points.
    Given expected linear trend at the end points of NESZ, a first-order
    polynominal based on the first and last four points are used to
    extrapolate the respective noise powers for the first ad last slant
    ranges.
    The end points of slant range is slightly beyond that of echo to
    avoid any need for extrapolation. This with the combination of uniformity
    will make NESZ product suitable for representation and application via
    ISCE3 LUT.

    """
    mask_nan = np.isnan(pow_noise)
    mask_valid = ~ mask_nan
    if pow_noise[mask_valid].size < 4:
        raise RuntimeError(
            'Not enough noise range blocks to fill nan and end points!'
        )
    # form PCHIP interpolant per valid data
    p_intrp = PchipInterpolator(
        sr_noise[mask_valid], pow_noise[mask_valid], extrapolate=True
    )
    idx_nan = np.nonzero(mask_nan)[0]
    if idx_nan.size > 0:
        pow_noise = pow_noise.copy()
        pow_noise[idx_nan] = p_intrp(sr_noise[idx_nan])
    # add end points and expand output arrays by two points
    # use the first/last four points for the extrapolation via linear polyfit
    # start with the near range
    pf_near = np.polyfit(sr_noise[:4], pow_noise[:4], deg=1)
    sr_spacing = sr_noise[1] - sr_noise[0]
    sr_near = sr_noise[0] - sr_spacing
    pow_near = np.polyval(pf_near, sr_near)
    # extrapolate end point for the far range
    pf_far = np.polyfit(sr_noise[-4:], pow_noise[-4:], deg=1)
    # first check if the last possibly overlapped range block has the same
    # spacing as the rest
    is_uniform = np.isclose(
        sr_noise[-1] - sr_noise[-2], sr_spacing, rtol=0, atol=1e-7
    )
    if not is_uniform:
        # replace the last range bin with an extrapolated one
        #  to form uniform spacing
        sr_last = sr_noise[-2] + sr_spacing
        pow_last = np.polyval(pf_far, sr_last)
        # make a copy to avoid modifying the input
        sr_noise = np.copy(sr_noise)
        pow_noise = np.copy(pow_noise)
        sr_noise[-1] = sr_last
        pow_noise[-1] = pow_last
    # now extrapolate the far-range point
    sr_far = sr_noise[-1] + sr_spacing
    pow_far = np.polyval(pf_far, sr_far)

    # update final uniformly-spaced noise power vs. slant range
    # with newly added near/far points which are slightly beyond
    # echo slant range to avoid any need for extrapolation.
    num_ranges = pow_noise.size
    pow_noise = np.insert(pow_noise, [0, num_ranges], [pow_near, pow_far])
    sr_noise = np.insert(sr_noise, [0, num_ranges], [sr_near, sr_far])

    return pow_noise, sr_noise
