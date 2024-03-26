from enum import IntEnum, unique
from isce3.core import Orbit, Attitude, Linspace
from isce3.geometry import DEMInterpolator
import logging
from nisar.products.readers.antenna import AntennaParser
from nisar.products.readers.instrument import InstrumentParser
from nisar.products.readers.Raw import Raw
from nisar.antenna import TxTrmInfo, RxTrmInfo, TxBMF, RxDBF
from nisar.antenna.beamformer import get_pulse_index
import numpy as np

log = logging.getLogger("nisar.antenna.pattern")


def find_changes(x):
    """
    Find indices where the value of x changes.

    Parameters
    ----------
    x : np.ndarray
        One- or two-dimensional array of values.

    Returns
    -------
    changes : numpy.ndarray
        Sorted indices i where x[i] is different from x[i-1].
    """
    if x.ndim == 1:
        i = np.diff(x).nonzero()[0]
        return i + 1
    if x.ndim > 2:
        raise NotImplementedError("unsupported ndim for find_changes")
    # The difference between each adjacent pair of rows.
    rowdiff = np.diff(x, axis=0)
    # True for each pair of rows if any corresponding element was different
    # between them.
    anydiff = np.any(rowdiff, axis=1)
    # Get indices of True values in the above.
    i = anydiff.nonzero()[0]
    return i + 1


class TimingFinder:
    """Helper class for finding when timing telemetry changes.

    Parameters
    ----------
    pulse_times : sequence of float
        Time tag of each pulse in seconds since an epoch.
    rd : sequence of sequence of int
        Round trip time at starting elevation angle in Time-To-Angle LUT
        calculated with respect to transmit pulse.  This value along with `wd`
        determines data window position (DWP) for each channel.  Time units are
        'fs_win' clock sample counts (typically 240 MHz for NISAR).  The object
        should have shape (m, n) where m == len(pulse_times) and n is the
        number of RX channels.
    wd : sequence of sequence of int
        Receive window delay to first valid data sample relative to RD.
        Value is provided as 'fs_win' clock sample counts.
        Same units and shape as `rd`.
    wl : sequence of sequence of int
        Length of receive data window provided as 'fs_win' clock sample counts.
        Same units and shape as `rd`.

    Attributes
    ------------
    changes : array of int
        Pulse indices where changes of input parameters occur
    time_changes : array of float
        Pulse times where changes of input parameter occur
    """

    def __init__(self, pulse_times, rd, wd, wl):
        self.pulse_times = pulse_times
        self.rd = rd
        self.wd = wd
        self.wl = wl
        rows = np.hstack((rd, wd, wl))
        # append 0 to simplify logic
        self.changes = np.hstack(([0], find_changes(rows)))
        self.time_changes = pulse_times[self.changes]

    def get_dbf_timing(self, t):
        """
        Get range timing parameters corresponding to a given azimuth time.

        Parameters
        ----------
        t : float
            Time in seconds (since same epoch as pulse_times)

        Returns
        -------
        tuple[Sequence[int], Sequence[int], Sequence[int]]
            RD, WD, WL at time `t`.
            Each has shape (n,) where n is the number of RX channels.
            See constructor for the meaning of these timing values.
        """
        # Searching through just the change times is faster than searching
        # the whole `pulse_times` array (which may be nonuniform).  However
        # we have to give up `nearest` semantics.  NISAR timing can't update
        # faster than once every 10 seconds, and we assume it's okay if we
        # possibly mess up that often.
        k = get_pulse_index(self.time_changes, t, nearest=False)
        i = self.changes[k]
        return self.rd[i], self.wd[i], self.wl[i]


@unique
class PolType(IntEnum):
    """Enumeration for polarization types"""
    single = 1
    dual = 2
    quad = 3
    quasi_dual = 4
    quasi_quad = 5
    compact_left = 6
    compact_right = 7


def pols_type_from_raw(raw: Raw):
    """
    Get all Tx/Rx pols and polarization type from raw.
    The info will be used in 2-way antenna pattern formation.

    Parameters
    ----------
    raw : Raw
        NISAR raw data object

    Returns
    -------
    pol_type : PolType
        Polarization type enumeration
    is_ssp : bool
        Whether the product is split-spectrum (SSP) or not.
    tx_pols : list
        List of all TX polarizations from all frequency bands.
    rx_pols : list
        List of all RX polarizations from all frequency bands.
    freq : str
        The first frequency band character (for the highest sampling rate).
        The Tx/Rx polarizations are mainly reported for this band except
        for quasi cases where all unique polarizations obtained from both
        frequency bands for RX and TX are stored together.

    """
    frq_list = np.sort(raw.frequencies)
    if frq_list.size > 2:
        raise NotImplementedError(
            'More than two frequency bands are not supported!'
        )
    is_ssp = False
    # use the first frequency to get individual RX and TX pols
    txrx_pols = raw.polarizations
    freq = frq_list[0]
    txrx_pol_list = txrx_pols[freq]
    rx_pols = {p[1] for p in txrx_pol_list}
    # NOTE using fromkeys to preserve ordering
    tx_pols = list(dict.fromkeys([p[0] for p in txrx_pol_list]))
    # check if it is compact pol (circular on TX)
    if tx_pols[0] in ['L', 'R']:
        if tx_pols[0] == 'L':
            pol_type = PolType.compact_left
        else:
            pol_type = PolType.compact_right
    else:
        if len(tx_pols) == 2:
            pol_type = PolType.quad
        else:
            assert len(tx_pols) == 1
            if len(rx_pols) == 2:
                pol_type = PolType.dual
            else:
                assert len(rx_pols) == 1
                pol_type = PolType.single

    # check second band to see if it is Quasi-Quad or Quasi-Dual
    if frq_list.size == 2:
        is_ssp = True
        freq_b = frq_list[1]
        txrx_pol_list_b = txrx_pols[freq_b]
        if set(txrx_pol_list) != set(txrx_pol_list_b):
            tx_pols.extend(
                dict.fromkeys([p[0] for p in txrx_pol_list_b])
            )
            rx_pols_b = set([p[1] for p in txrx_pol_list_b])
            rx_pols = rx_pols.union(rx_pols_b)
            if len(rx_pols_b) == 2:
                pol_type = PolType.quasi_quad
            else:
                assert len(rx_pols_b) == 1
                pol_type = PolType.quasi_dual

    return pol_type, is_ssp, tx_pols, list(rx_pols), freq


def build_tx_trm(raw: Raw, pulse_times: np.ndarray, freq_band: str,
                 tx_pol: str):
    """Build TxTrmInfo object """
    # Parse Tx-related Cal stuff used for Tx BMF
    tx_chanl = raw.getListOfTxTRMs(freq_band, tx_pol)
    corr_tap2 = raw.getChirpCorrelator(freq_band, tx_pol)[..., 1]
    cal_type = raw.getCalType(freq_band, tx_pol)
    # build TxTRM  from Tx Cal stuff w/o optional "tx_phase"
    return TxTrmInfo(pulse_times, tx_chanl, corr_tap2,
                     cal_type)


class AntennaPattern:
    """
    Convenience class for generating combined TX*RX elevation pattern for
    use in RSLC processing.

    Parameters
    ----------
    raw: Raw
        NISAR raw data (L0B) reader
    dem: DEMInterpolator
        Digital elevation model
    ant: AntennaParser
        NISAR antenna pattern object
    ins: InstrumentParser
        NISAR instrument tables (TA and AC tables)
    orbit: Orbit
        Antenna orbit ephemeris
    attitude: Attitude
        Antenna orientation
    el_spacing_min: float, default=8.72665e-5
        Min EL angle spacing in (radians) used to determine min slant range
        spacing over entire swath.
        The default is around 5 mdeg where antenna pattern magnitude and phase
        vairations expected to be less than 0.05 dB and 0.25 deg, respectively.
        This can speed up the antenna pattern computation. If None, it will be
        ignored.

    """

    def __init__(self, raw: Raw, dem: DEMInterpolator,
                 ant: AntennaParser, ins: InstrumentParser,
                 orbit: Orbit, attitude: Attitude, norm_weight=True,
                 el_spacing_min=8.72665e-5):

        self.orbit = orbit.copy()
        self.attitude = attitude.copy()
        self.dem = dem
        self.norm_weight = norm_weight
        self.el_spacing_min = el_spacing_min

        # get linear pols and pol type
        (self.pol_type, self.is_ssp, self.tx_pols, self.rx_pols,
         self.freq_band) = pols_type_from_raw(raw)

        # Parse ref epoch, pulse time, slant range, orbit and attitude from Raw
        # Except for quad-pol, pulse time is the same for all TX pols.
        # In case of quad-pol the time offset is half single-pol PRF and thus
        # shouldn't affect the time index finding process below, otherwise,
        # two sets of pulse time is required in case of quad-pol!
        self.reference_epoch, self.pulse_times = raw.getPulseTimes(
            self.freq_band, self.tx_pols[0])

        # Harmonize epochs.  Already made copies above, so not modifying input.
        self.orbit.update_reference_epoch(self.reference_epoch)
        self.attitude.update_reference_epoch(self.reference_epoch)

        # Sampling rate in (Hz) for range window parameters RD/WD/WL in
        # NISAR case, assumed same on all channels.  This is only different
        # from 240 MHz for simulated data.
        self.fs_win = raw.getSampleRateDBF(self.freq_band)

        # parse active RX channels and fs_ta which are polarization
        # independent!
        txrx_pol = raw.polarizations[self.freq_band][0]
        self.rx_chanl = raw.getListOfRxTRMs(self.freq_band, txrx_pol)
        self.fs_ta = ins.sampling_rate_ta(txrx_pol[1])

        # Parse DBF-related RD/WD/WL, time-to-angle(TA) and angle-to-coeffs(AC)
        # tables to be used for Rx DBF pattern.
        rd_all, wd_all, wl_all = dict(), dict(), dict()
        self.finder = dict()

        # get RD/WD/WL for all unique RX polarizations of first freq band
        for pol in self.rx_pols:
            if self.pol_type == PolType.quasi_dual:
                txrx_pol = 2 * pol
            else:
                txrx_pol = self.tx_pols[0] + pol
            rd_all[pol], wd_all[pol], wl_all[pol] = raw.getRdWdWl(
                self.freq_band, txrx_pol)
            self.finder[pol] = TimingFinder(self.pulse_times, rd_all[pol],
                                            wd_all[pol], wl_all[pol])

        # build RxTRMs  and the first RxDBF for all possible RX
        # linear polarizations
        self.rx_trm = dict()
        self.rx_dbf = dict()
        self.el_pat_rx = dict()
        self.ta_switch = dict()
        self.dbf_coef = dict()
        self.ela_dbf = dict()
        self.channel_adj_fact_rx = dict()
        for nn, rx_p in enumerate(self.rx_pols):

            # fetch RX channel adjustment complex factors from
            # instrument file per RX pol.
            self.channel_adj_fact_rx[rx_p] = ins.channel_adjustment_factors_rx(
                rx_p)

            # get rx el-cut patterns
            self.el_pat_rx[rx_p] = ant.el_cut_all(rx_p)

            # get instrument DBF tables
            self.ta_switch[rx_p] = ins.get_time2angle(rx_p)
            self.dbf_coef[rx_p] = ins.get_angle2coef(rx_p)
            self.ela_dbf[rx_p] = ins.el_angles_ac(rx_p)

            # build RxTRM object for the very first pulse
            self.rx_trm[rx_p] = RxTrmInfo(
                self.pulse_times, self.rx_chanl, rd_all[rx_p][0],
                wd_all[rx_p][0], wl_all[rx_p][0], self.dbf_coef[rx_p],
                self.ta_switch[rx_p], self.ela_dbf[rx_p], self.fs_win,
                self.fs_ta)

            # construct RX DBF object
            if nn == 0:
                # for the first pol use min EL spacing to determine
                # min slant range spacing
                self.rx_dbf[rx_p] = RxDBF(
                    self.orbit, self.attitude, self.dem, self.el_pat_rx[rx_p],
                    self.rx_trm[rx_p], self.reference_epoch,
                    norm_weight=self.norm_weight,
                    el_spacing_min=self.el_spacing_min,
                )
                self.rg_spacing_min = self.rx_dbf[rx_p].rg_spacing_min
            else:
                self.rx_dbf[rx_p] = RxDBF(
                    self.orbit, self.attitude, self.dem, self.el_pat_rx[rx_p],
                    self.rx_trm[rx_p], self.reference_epoch,
                    norm_weight=self.norm_weight,
                    rg_spacing_min=self.rg_spacing_min,
                )

        # build all TxBMFs for all possible TX linear polarizations
        # in case of compact pol, both H and V are taken into account.
        self.tx_bmf = dict()
        self.channel_adj_fact_tx = dict()
        if (self.pol_type == PolType.compact_left or
                self.pol_type == PolType.compact_right):
            tx_pols_lin = ['H', 'V']
        else:
            tx_pols_lin = self.tx_pols

        for tx_p, tx_lp in zip(self.tx_pols, tx_pols_lin):

            # fetch TX channel adjustment complex factors from
            # instrument file per TX linear pol.
            self.channel_adj_fact_tx[tx_lp] = (
                ins.channel_adjustment_factors_tx(tx_lp)
                )

            # get tx el-cut patterns
            el_pat_tx = ant.el_cut_all(tx_lp)

            # build Tx TRM
            # Note that in QD and QQ modes the subbands may have different TX
            # frequencies.  Need to make sure raw data queries have consistent
            # pairings of freq band and TX pol.
            tx_band = raw.frequencies[0]
            if self.is_ssp:
                tx0 = [pol[0] for pol in raw.polarizations[raw.frequencies[0]]]
                if tx_p not in tx0:
                    tx_band = raw.frequencies[1]
            tx_trm = build_tx_trm(raw, self.pulse_times, tx_band, tx_p)

            # construct TX BMF object
            self.tx_bmf[tx_lp] = TxBMF(
                self.orbit, self.attitude, self.dem, el_pat_tx, tx_trm,
                self.reference_epoch, norm_weight=self.norm_weight,
                rg_spacing_min=self.rg_spacing_min)

    def form_pattern(self, t: float, slant_range: Linspace,
                     nearest: bool = False):
        """
        Get the two-way antenna pattern at a given time and set of ranges for
        all polarization combinations.

        Parameters
        ----------
        t: float
            Azimuth time in seconds (since same epoch as pulse_times)
        slant_range: isce3.core.Linspace
            Range vector (in meters)
        nearest : bool
            For `nearest=False` and `pulse_times[i] <= t < pulse_times[i+1]`
            then `i` will be returned (e.g., a floor operation).  If
            `nearest=True` then return the closer of the two (e.g., a round
            operation).

        Returns
        -------
        dict
            Two-way complex antenna patterns as a function of range bin
            over all TxRx polarization products. The format of dict is
            {pol: np.ndarray[complex]}.
        """
        # form one-way RX patterns for all linear pols
        rx_dbf_pat = dict()
        for p in self.rx_pols:
            rd, wd, wl = self.finder[p].get_dbf_timing(t)
            # Only update RxDBF if range timing changes, otherwise use cached.
            # Presumption is that timing changes infrequently and user is
            # likely to call form_pattern serially in time-sorted order.
            if not ((all(rd == self.rx_trm[p].rd) and
                     all(wd == self.rx_trm[p].wd) and
                     all(wl == self.rx_trm[p].wl))):
                log.info(f'Updating {p}-pol RX antenna pattern because'
                         ' change in RD/WD/WL')

                self.rx_trm[p] = RxTrmInfo(
                    self.pulse_times, self.rx_chanl, rd, wd, wl,
                    self.dbf_coef[p], self.ta_switch[p], self.ela_dbf[p],
                    self.fs_win, self.fs_ta)

                self.rx_dbf[p] = RxDBF(
                    self.orbit, self.attitude, self.dem, self.el_pat_rx[p],
                    self.rx_trm[p], self.reference_epoch,
                    norm_weight=self.rx_dbf[p].norm_weight)

            rx_dbf_pat[p] = self.rx_dbf[p].form_pattern(
                t, slant_range, channel_adj_factors=self.channel_adj_fact_rx[p]
            )

        # form one-way TX patterns for all TX pols
        tx_bmf_pat = dict()
        if self.pol_type == PolType.compact_left:
            tx_bmf_pat['L'] = (
                self.tx_bmf['H'].form_pattern(
                    t, slant_range, nearest=nearest,
                    channel_adj_factors=self.channel_adj_fact_tx['H']) +
                1j * self.tx_bmf['V'].form_pattern(
                    t, slant_range, nearest=nearest,
                    channel_adj_factors=self.channel_adj_fact_tx['V'])
            )

        elif self.pol_type == PolType.compact_right:
            tx_bmf_pat['R'] = (
                self.tx_bmf['H'].form_pattern(
                    t, slant_range, nearest=nearest,
                    channel_adj_factors=self.channel_adj_fact_tx['H']) -
                1j * self.tx_bmf['V'].form_pattern(
                    t, slant_range, nearest=nearest,
                    channel_adj_factors=self.channel_adj_fact_tx['V'])
            )
        else:  # other non-compact pol types
            for p in self.tx_pols:
                tx_bmf_pat[p] = self.tx_bmf[p].form_pattern(
                    t, slant_range, nearest=nearest,
                    channel_adj_factors=self.channel_adj_fact_tx[p])

        # build two-way pattern for all unique TxRx products obtained from all
        # freq bands
        pat2w = dict()
        for tx_p in self.tx_pols:
            if self.pol_type == PolType.quasi_dual:
                txrx_p = 2 * tx_p
                pat2w[txrx_p] = np.squeeze(
                        tx_bmf_pat[tx_p] * rx_dbf_pat[tx_p]
                        )
            else:  # non quasi-dual mode
                for rx_p in self.rx_pols:
                    txrx_p = tx_p + rx_p
                    pat2w[txrx_p] = np.squeeze(
                        tx_bmf_pat[tx_p] * rx_dbf_pat[rx_p]
                        )

        return pat2w
