from __future__ import annotations
from warnings import warn
from typing import Tuple, Dict
from dataclasses import dataclass
import logging

import numpy as np

from nisar.products.readers.Raw import (
    Raw, chirpcorrelator_caltype_from_raw, caltone_frequency_from_raw
)
from nisar.antenna import get_calib_range_line_idx


log = logging.getLogger("nisar.antenna.rx_channel_imbalance_helpers")


@dataclass(frozen=True)
class RxChannelImbalanceProduct:
    """
    RX channel imbalance product extracted from LNA/CALTONE ratio
    for a certain frequency band and polarization.

    Attributes
    ----------
    lna_caltone_ratio: np.ndarray(complex)
        Peak-normalized complex LNA/CALTONE ratio over all RXs (typically 12)
    ntap_dominant: np.ndarray(int)
        Dominant tap number, a value within [1,3] over all RXs.
    time_delays_sec: np.ndarray(float)
        Time delays from the phase of outlier qFSP in seconds for all RXs.
    max_amp_ratio: float
        Max amplitude ratio used in peak normalizing `lna_caltone_ratio`.

    """
    lna_caltone_ratio: np.ndarray
    ntap_dominant: np.ndarray
    time_delays_sec: np.ndarray
    max_amp_ratio: float

    def __post_init__(self):
        # XXX Size of all arrays must be 12 for L-band NISAR but
        # not enforced due to failure of special cases such as unit test
        if (self.lna_caltone_ratio.size != self.ntap_dominant.size
                != self.time_delays_sec.size):
            raise ValueError('The size of all arrays must be equal!')
        if self.lna_caltone_ratio.size != 12:
            warn('The size of LNA-CALTONE ratio is '
                 f'{self.lna_caltone_ratio.size} instead of 12!')


def compute_all_rx_channel_imbalances_from_l0b(
        l0b_file: str | Raw,
        *,
        caltone_freq: float | None = None,
        freq_band: str | None = None,
        txrx_pol: str | None = None
) -> Dict[Tuple[str, str], RxChannelImbalanceProduct]:
    """
    Compute 12 complex RX channel imbalance based on LNA/CALTONE ratio
    for over all bands and polarizations. The bands and polarizations are
    used as dictionary keys in the form of [freq_band, txrx_pol].

    Also report the dominant tap number our of 3 for LNA three-tap
    correlator as well as detected relative time delays for all RX channels
    for debugging purposes.

    Parameters
    ----------
    l0b_file : str or nisar.products.readers.Raw
        L0B filename or Raw object
    caltone_freq : float or None. Optional
        Caltone frequency in Hz.
        If None (default), it will be extracted from DRT in L0B.
    freq_band : str. Optional
        "A" or "B". Default is all.
    txrx_pol : str. Optional
        TR pol in `freq_band` such as "HH", "HV", etc.
        Default is all.

    Returns
    -------
    dict:
        A dict with keys (freq_band, txrx_pol) and values of type
        `RxChannelImbalanceProduct`

    """
    if isinstance(l0b_file, str):
        raw = Raw(hdf5file=l0b_file)
    else:
        raw = l0b_file
    frq_pols = raw.polarizations
    # get freq_bands and txrx_pols
    if freq_band is not None:
        frq_pols = {freq_band: frq_pols[freq_band]}
    if txrx_pol is not None:
        frq_pols = {f: [txrx_pol] for f in frq_pols if txrx_pol in frq_pols[f]}

    out = dict()
    for freq_band in frq_pols:
        for txrx_pol in frq_pols[freq_band]:
            (lna_caltone_ratio, n_tap_dominant, time_delays, max_ratio
             ) = compute_rx_channel_imbalance(
                raw=raw,
                freq_band=freq_band,
                txrx_pol=txrx_pol,
                caltone_freq=caltone_freq
            )
            out[freq_band, txrx_pol] = RxChannelImbalanceProduct(
                lna_caltone_ratio=lna_caltone_ratio,
                ntap_dominant=n_tap_dominant,
                time_delays_sec=time_delays,
                max_amp_ratio=max_ratio
            )
    return out


def compute_rx_channel_imbalance(
        raw: Raw,
        freq_band: str,
        txrx_pol: str,
        caltone_freq: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute 12 complex RX channel imbalance based on LNA/CALTONE ratio
    for a desired frequency band and TR polarization, averaged over all
    records in the raw data file.

    Also report the dominant tap number our of 3 for LNA three-tap
    correlator as well as detected relative time delays for all RX channels
    for debugging purposes.

    Parameters
    ----------
    raw : Raw
        ISCE3 NISAR L0B product parser
    freq_band: str,
        A or B
    txrx_pol : str
        HH, HV, etc
    caltone_freq : float or None, optional.
        Caltone frequency in Hz. If None, it will be parsed from DRT
        field of L0B product.

    Returns
    -------
    lna_caltone_ratio: np.ndarray(complex)
        Peak-normalized complex LNA/CALTONE ratio over all 12 RXs
    n_tap_dominant: np.ndarray(int)
        Dominant tap number, a value within [1,3] over all 12 RXs.
    time_delays: np.ndarray(float)
        Time delays from the phase of qFSP outlier in seconds
    max_ratio : float
        Report peak power among all channels used for amplitude
        normalization of RX channel imbalances.

    """
    lna_mean, n_tap_dominant = get_lna_cal_mean(raw, txrx_pol)
    # get caltone mean over all RX channels
    caltone_mean = get_caltone_mean(raw, freq_band, txrx_pol)
    # Get complex ratio LNA/Caltone over all channels
    lna_caltone_ratio = lna_mean / caltone_mean
    # correct the ratio for the second band if necessary
    lna_caltone_ratio, time_delays = correct_lna_caltone_ratio_for_second_band(
        lna_caltone_ratio,
        raw,
        freq_band,
        txrx_pol,
        caltone_freq=caltone_freq
    )
    # peak normalized
    max_ratio = np.nanmax(abs(lna_caltone_ratio))
    if not np.isclose(max_ratio, 0):
        lna_caltone_ratio /= max_ratio
    return lna_caltone_ratio, n_tap_dominant, time_delays, max_ratio


def get_lna_cal_mean(
    raw: Raw,
    txrx_pol: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns mean complex LNA values and dominant tap
    numbers within [1, 2, 3] for all channels of a
    desired polarization.

    Parameters
    ----------
    raw : Raw
        ISCE3 NISAR L0B product parser
    txrx_pol : str
        HH, HV, etc

    Returns
    -------
    np.ndarray(complex)
        1-D array of slow-time averaged LNA CAL for all RX channels.
        The size is the number of RX channels.
    np.ndarray(int)
        1-D array of dominant tap number of LNA three-tap chirp-correlator,
        a value within [1, 3] for all RX channels.
        The size is the number of RX channels.

    """
    chp_cor, cal_type = chirpcorrelator_caltype_from_raw(
        raw=raw,
        txrx_pol=txrx_pol
    )
    n_rxs = chp_cor.shape[1]
    _, idx_byp, idx_lna, _ = get_calib_range_line_idx(cal_type)
    if len(idx_lna) == 0:
        warn('No LNA CAL to represent RX! Use BYPASS Cal instead!')
        if len(idx_byp) == 0:
            # XXX to avoid failure in unit test or very short L0B
            # lacking LNA/BYP CAL datasets, a warning will be issued
            # and the values will all be set to unity!
            warn('No LNA or BYPASS CAL! LNA mean will be all unity. '
                 'The results will be invalid!')
            lna_mean = np.ones(n_rxs, dtype='c8')
            n_tap_dominant = np.full(n_rxs, fill_value=2)
            return lna_mean, n_tap_dominant
        idx_lna = idx_byp
    # get  LNA for all three taps (or BYPASS)
    lna_mean_tap3 = np.zeros((3, n_rxs), dtype='c16')
    for nn in range(3):
        lna_cal = chp_cor[idx_lna, :, nn]
        # get complex mean for all RX channels
        lna_mean_tap3[nn] = _mean_2d(lna_cal)
    # get dominat taps
    abs_lna_mean_tap3 = abs(lna_mean_tap3)
    idx_lna_taps = np.nanargmax(abs_lna_mean_tap3, axis=0)
    amp_lna_mean = np.zeros(n_rxs)
    for nn in range(n_rxs):
        amp_lna_mean[nn] = abs_lna_mean_tap3[idx_lna_taps[nn], nn]
    _check_if_zero(amp_lna_mean, msg=f'{txrx_pol[0]}-pol LNA Cal')
    # get the phase part at a fixed common tap rather than dominant one
    phs_lna_mean = np.angle(lna_mean_tap3[1])
    # form complex lna
    lna_mean = amp_lna_mean * np.exp(1j * phs_lna_mean)
    n_tap_dominant = idx_lna_taps + 1
    return lna_mean, n_tap_dominant


def correct_lna_caltone_ratio_for_second_band(
        lna_caltone_ratio: np.ndarray,
        raw: Raw,
        freq_band: str,
        txrx_pol: str,
        caltone_freq: float | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct complex LNA/CALTONE ratio already obtained from the
    first frequency band for the second frequency band if exists
    in raw L0B product per desired polarization
    and band.
    It also return the respective time delays for all three qFSP
    (12 RX channels).

    Parameters
    ----------
    lna_caltone_ratio: np.ndarray(complex)
        Complex LNA/CALTONE ratio over all 12 RXs
    raw : Raw
        ISCE3 NISAR L0B product parser
    freq_band: str,
        A or B
    txrx_pol : str
        HH, HV, etc
    caltone_freq : float or None, optional.
        Caltone frequency in Hz. If None, it will be parsed from DRT
        field of L0B product.

    Returns
    -------
    np.ndarray(complex)
        1-D array of slow-time averaged Caltone for all RX channels
    np.ndarray(float)
        1-D array of relative time delays of three qFSP (12 RX channels)
        in seconds.

    """
    # Get caltone frequency from DRT if not provided
    if caltone_freq is None:
        caltone_freq = caltone_frequency_from_raw(raw, txrx_pol)
        log.info(f'Caltone frequency is extracted from {txrx_pol[1]}-pol DRT '
                 f'-> {caltone_freq * 1e-6:.3f} (MHz)')
    # Loopback cal is only ever measured on main sub-band.
    # Thus, check if product from the second band so we can
    # modify the results from the main band only if there is a
    # relative delay offset in one of qFSP vs others, that is
    # one of the qFSP is an outlier due to  ADC clock/delay issue
    # check if there is delay anomaly among three qFSP.
    if _is_product_from_second_band(raw, freq_band, txrx_pol):
        log.info(
            f'correcting LNA/CALTONE for band={freq_band} and pol={txrx_pol}')
        fc_a, _, _, _ = raw.getChirpParameters('A', txrx_pol[0])
        # get diff of chirp (band=A) and caltone freq for delay detection
        dif_chirp_caltone_freq = fc_a - caltone_freq
        time_delay = _get_qfsp_delay_anomaly(
            lna_caltone_ratio, dif_chirp_caltone_freq)
        # if there is then get diff of frequency bands A and B
        # to be used to correct phase from A for B
        fc_b, _, _, _ = raw.getChirpParameters('B', txrx_pol[0])
        phs_adj = 2 * np.pi * (fc_b - fc_a) * time_delay
        # correct the LNA/CALTONE by delay amount via phase if any.
        lna_caltone_ratio *= np.exp(1j * phs_adj)
    else:  # simply first band either A or B!
        fc, _, _, _ = raw.getChirpParameters(freq_band, txrx_pol[0])
        # get diff of chirp (band=A) and caltone freq for delay detection
        dif_chirp_caltone_freq = fc - caltone_freq
        time_delay = _get_qfsp_delay_anomaly(
            lna_caltone_ratio, dif_chirp_caltone_freq)
    return lna_caltone_ratio, time_delay


def get_caltone_mean(
        raw: Raw,
        freq_band: str,
        txrx_pol: str
) -> np.ndarray:
    """
    Get average Caltone across all range lines for a desired
    band and polarization from L0B.

    Parameters
    ----------
    raw : Raw
        ISCE3 NISAR L0B product parser
    freq_band: str,
        A or B
    txrx_pol : str
        HH, HV, etc

    Returns
    -------
    np.ndarray(complex)
        1-D array of slow-time averaged Caltone for all RX channels

    """
    # now get caltone always from swath
    caltone = raw.getCaltone(freq_band, txrx_pol)
    caltone_mean = _mean_2d(caltone)
    _check_if_zero(caltone_mean, msg=f'{txrx_pol}-pol Caltone')
    return caltone_mean


def _is_product_from_second_band(
        raw: Raw,
        freq_band: str,
        txrx_pol: str) -> bool:
    """
    Determine whether the raw prodcut with dersied polarization
    is available on both frequency bands and it is representing
    the second frequency band "B".

    Parameters
    ----------
    raw : Raw
        ISCE3 NISAR L0B product parser
    freq_band: str,
        A or B
    txrx_pol : str
        HH, HV, etc

    Returns
    -------
    bool
        True if the product band/pol represent the second frequency band
        in split-spectrum case, otherwise false.
    """
    return (freq_band == "B" and len(raw.frequencies) == 2 and
            txrx_pol in raw.polarizations['A'])


def _get_qfsp_delay_anomaly(
        lna_caltone_ratio: np.ndarray,
        dif_chirp_caltone_freq: float,
        adc_clock: float = 240e6) -> np.ndarray:
    """
    If the product is a 12-channel NISAR L-band product,
    return the time delays for a qFSP with phase anomaly.
    Else, return zeros.

    Parameters
    ----------
    lna_caltone_ratio : np.ndarray(complex)
        1-D array of complex LNA/CALTONE ratio with size equals
        to the number of RX channels
    dif_chirp_caltone_freq: float
        Frequency difference between chirp and caltone (chirp - caltone)
        in Hz.
    adc_clock : float, default=240e6
        Analogue-to-digital (ADC) clock rate of NISAR in Hz.

    Returns
    -------
    time_delays : np.ndarray(float)
        Rleative time delays of all three qFSP (12 RX channels) in seconds.
        Same size as `lna_caltone_ratio`.

    Notes
    -----
    In case of non-NISAR case with less than 12 channels, all
    delays are zero to zeros.

    """
    if lna_caltone_ratio.size == 12:
        # group them into three 4-channels, one per qFSP
        lna2cal_ratio = lna_caltone_ratio.reshape(3, 4)
        # get unwrap phase across 4 channels per qFSP (radians)
        lna2cal_phs = np.unwrap(np.angle(lna2cal_ratio), axis=1)
        # get median phase per qfsp, total 3 phase values (radians)
        # and then unwrap three values
        qfps_phs = np.unwrap(np.nanmedian(lna2cal_phs, axis=1))
        # use median among all three to be used as a reference to
        # catch a single outlier
        phs_ref = np.median(qfps_phs)
        # phase due to ADC delay
        phs_adc_delay = 2 * np.pi * dif_chirp_caltone_freq / adc_clock
        if np.isclose(phs_adc_delay, 0):
            warn('Caltone and chirp center frequency is the same. This '
                 'can lead to no qFSP delay anomaly detection if any!')
            n_delay_qfsp = np.zeros_like(qfps_phs)
        else:
            n_delay_qfsp = np.round((qfps_phs - phs_ref) / phs_adc_delay)
        # now repeat sample delay 4x per qFSP
        n_delays = np.repeat(
            n_delay_qfsp[:, np.newaxis], repeats=4, axis=1).ravel()
        time_delays = n_delays / adc_clock
    else:
        time_delays = np.zeros(lna_caltone_ratio.size)
    return time_delays


def _mean_2d(data: np.ndarray, perc: float = 5.0) -> np.asarray:
    """
    Compute mean within percentile [perc, 100-perc] along range lines,
    of a 2-D complex array with shape (rangelines, channels)
    due to bad telemetry.

    Parameters
    ----------
    data : np.ndarray(complex)
        2-D complex float data with shape (rangelines, channels)
    perc : float, default=5
        Percentile, a value within [0, 100].
        The values within [perc, 100 - perc] are used in the mean
        calculation. If no value in `data` that fullfills this,
        the median of `data` will be used instead.
        Note that, `perc` and `100-perc` will lead to the same
        exact outcome.
        Also, `perc=50` is equivalent to `np.nanmedian`.

    Returns
    -------
    np.ndarray(complex)
        1-D array of size equal to number of channels representing
        the mean across range lines.

    """
    # or simply np.nanmean(data, axis=0)
    d = np.sort(np.abs(data), axis=0)
    q1_all, q3_all = np.percentile(d, q=[perc, 100 - perc], axis=0)
    mean_all = []
    for cc, (q1, q3) in enumerate(zip(q1_all, q3_all)):
        q1, q3 = np.sort([q1, q3])
        data_q1_q3 = data[(d[:, cc] >= q1) & (d[:, cc] <= q3), cc]
        if data_q1_q3.size == 0:
            # use median (perc=50) if no values within desired percentiles.
            mean_all.append(np.nanmedian(data[:, cc]))
        else:  # there is at least one value within desired percentiles
            mean_all.append(np.nanmean(data_q1_q3))
    return np.asarray(mean_all)


def _check_if_zero(arr: np.ndarray, msg: str) -> None:
    """
    Check a telemetry array to see if all or some of its values are zero
    and then issue an appropriate warning with message `msg`
    If all values are zero, then input wiill be filled with unity to avoid
    failure for older L0B products.

    Parameters
    ----------
    arr : np.ndarray
        Input array to be modified in place only if all its elements are zero!.
    msg : str
        Message to be used as part of a warning message.

    """
    is_zero = np.isclose(arr, 0, atol=1e-9)
    if is_zero.all():
        # XXX to avoid unit test failure and old sim L0B
        # a warning will be issued and all values will be set
        # to unity!
        warn(f'All values are zero for {msg}! They are set to untiy. '
             'Result may be invalid!')
        arr[...] = 1.0
    if is_zero.any():
        warn(f'Some values are zero for {msg}!')
