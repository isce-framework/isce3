from collections import defaultdict
from enum import IntEnum, unique
from isce3.core import Orbit, Attitude, Linspace
from isce3.geometry import DEMInterpolator
import logging
from nisar.mixed_mode.logic import PolChannelSet
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
    el_lut : LUT2d, optional
        LUT2d to be used for range/azimuth to EL lookups.
        If not provided, EL will be computed on-the-fly using elaz2slantrange.
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
                 orbit: Orbit, attitude: Attitude,
                 *, el_lut=None,
                 norm_weight=True,
                 el_spacing_min=8.72665e-5):

        self.orbit = orbit.copy()
        self.attitude = attitude.copy()
        self.dem = dem
        self.norm_weight = norm_weight
        self.el_spacing_min = el_spacing_min
        self.el_lut = el_lut

        # get pols
        channels = PolChannelSet.from_raw(raw)
        freqs = tuple({chan.freq_id for chan in channels})
        self.freq_band = "A" if "A" in freqs else freqs[0]
        self.txrx_pols = tuple({chan.pol for chan in channels})

        # Parse ref epoch, pulse time, slant range, orbit and attitude from Raw
        # Except for quad-pol, pulse time is the same for all TX pols.
        # In case of quad-pol the time offset is half single-pol PRF and thus
        # shouldn't affect the time index finding process below, otherwise,
        # two sets of pulse time is required in case of quad-pol!
        self.reference_epoch, self.pulse_times = raw.getPulseTimes(
            self.freq_band)

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

        # get RD/WD/WL for all unique RX polarizations
        # Loop over all freqs & pols since some RX pols may be found only on
        # freq B (e.q. the QQP case).  Assume RD/WD/WL are the same for all
        # freqs/pols that have the same RX polarization.
        for chan in channels:
            rxpol = chan.pol[1]
            if rxpol in rd_all:
                continue
            rd_all[rxpol], wd_all[rxpol], wl_all[rxpol] = raw.getRdWdWl(
                chan.freq_id, chan.pol)
            self.finder[rxpol] = TimingFinder(self.pulse_times, rd_all[rxpol],
                                            wd_all[rxpol], wl_all[rxpol])

        # build RxTRMs  and the first RxDBF for all possible RX
        # linear polarizations
        self.rx_trm = dict()
        self.rx_dbf = dict()
        self.el_pat_rx = dict()
        self.ta_switch = dict()
        self.dbf_coef = dict()
        self.ela_dbf = dict()
        self.channel_adj_fact_rx = dict()
        rx_pols = {pol[1] for pol in self.txrx_pols}
        for nn, rx_p in enumerate(rx_pols):

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
                    el_lut=self.el_lut,
                    norm_weight=self.norm_weight,
                    el_spacing_min=self.el_spacing_min,
                )
                self.rg_spacing_min = self.rx_dbf[rx_p].rg_spacing_min
            else:
                self.rx_dbf[rx_p] = RxDBF(
                    self.orbit, self.attitude, self.dem, self.el_pat_rx[rx_p],
                    self.rx_trm[rx_p], self.reference_epoch,
                    el_lut=self.el_lut,
                    norm_weight=self.norm_weight,
                    rg_spacing_min=self.rg_spacing_min,
                )

        # build all TxBMFs for all possible TX linear polarizations
        # in case of compact pol, both H and V are taken into account.
        self.tx_bmf = dict()
        self.channel_adj_fact_tx = dict()
        tx_pols = set(pol[0] for pol in self.txrx_pols)

        for tx_p in tx_pols:
            # build Tx TRM
            # Note that in QD and QQ modes the subbands may have different TX
            # frequencies.  Need to make sure raw data queries have consistent
            # pairings of freq band and TX pol.
            for tx_band, pols in raw.polarizations.items():
                if tx_p in {pol[0] for pol in pols}:
                    tx_trm = build_tx_trm(raw, self.pulse_times, tx_band, tx_p)
                    break
            else:
                assert False, f"couldn't find freq_id for tx_pol={tx_p}"

            if tx_p in {"L", "R"}:
                tx_pols_lin = ["H", "V"]
            else:
                tx_pols_lin = [tx_p]

            for tx_lp in tx_pols_lin:
                # fetch TX channel adjustment complex factors from
                # instrument file per TX linear pol.
                self.channel_adj_fact_tx[tx_lp] = (
                    ins.channel_adjustment_factors_tx(tx_lp)
                    )

                # get tx el-cut patterns
                el_pat_tx = ant.el_cut_all(tx_lp)

                # construct TX BMF object
                self.tx_bmf[tx_lp] = TxBMF(
                    self.orbit, self.attitude, self.dem, el_pat_tx, tx_trm,
                    self.reference_epoch,
                    el_lut=self.el_lut, norm_weight=self.norm_weight,
                    rg_spacing_min=self.rg_spacing_min)


    def form_pattern(self, tseq, slant_range: Linspace,
                     nearest: bool = False, txrx_pols = None):
        """
        Get the two-way antenna pattern at a given time and set of ranges for
        either all or specified polarization combinations if Tx/Rx pols are
        provided.

        Parameters
        ----------
        tseq: float or np.ndarray
            Azimuth times in seconds (since same epoch as pulse_times)
        slant_range: isce3.core.Linspace
            Range vector (in meters)
        nearest : bool
            For `nearest=False` and `pulse_times[i] <= t < pulse_times[i+1]`
            then `i` will be returned (e.g., a floor operation).  If
            `nearest=True` then return the closer of the two (e.g., a round
            operation).
        txrx_pols : Optional[Iterable[str]]
            List of TxRx pols to use. Default is all available pols.

        Returns
        -------
        dict
            Two-way complex antenna patterns as a function of range bin
            over either all or specified TxRx polarization products. The format of dict is
            {pol: np.ndarray[complex]}.
        """
        if txrx_pols is None:
            txrx_pols = self.txrx_pols
        elif not set(txrx_pols).issubset(self.txrx_pols):
            raise ValueError(f"Specified txrx_pols {txrx_pols} is out of "
                f"available pols {self.txrx_pols}!")

        tseq = np.atleast_1d(tseq)
        rx_pols = {pol[1] for pol in txrx_pols}

        # form one-way RX patterns for all linear pols
        rx_dbf_pat = dict()
        for p in rx_pols:

            # Split up provided timespan into groups with the same range timing
            # (Adding one because get_pulse_index uses floor but we want ceil)
            change_indices = [
                get_pulse_index(tseq, t) + 1 for t in self.finder[p].time_changes
                if t > tseq[0] and t < tseq[-1]
            ]
            tgroups = np.split(tseq, change_indices)

            # Running start-index of tgroup within entire tspan
            i0 = 0
            for tgroup in tgroups:
                t = tgroup[0]
                rd, wd, wl = self.finder[p].get_dbf_timing(t)

                log.info(f'Updating {p}-pol RX antenna pattern because'
                         ' change in RD/WD/WL')

                self.rx_trm[p] = RxTrmInfo(
                    self.pulse_times, self.rx_chanl, rd, wd, wl,
                    self.dbf_coef[p], self.ta_switch[p], self.ela_dbf[p],
                    self.fs_win, self.fs_ta)

                self.rx_dbf[p] = RxDBF(
                    self.orbit, self.attitude, self.dem, self.el_pat_rx[p],
                    self.rx_trm[p], self.reference_epoch,
                    el_lut=self.el_lut,
                    norm_weight=self.rx_dbf[p].norm_weight)

                pat = self.rx_dbf[p].form_pattern(
                    tgroup, slant_range,
                    channel_adj_factors=self.channel_adj_fact_rx[p]
                )
                # Initialize the pattern array so we can slice this range timing
                # group into it - TODO move this outside the loop for clarity?
                if p not in rx_dbf_pat:
                    rx_dbf_pat[p] = np.empty((len(tseq), slant_range.size),
                        dtype=np.complex64)

                # Slice it into the full array, and
                # bump up the index for the next slice
                iend = i0 + len(tgroup)
                rx_dbf_pat[p][i0:iend] = pat
                i0 = iend

        # form one-way TX patterns for all TX pols
        tx_bmf_pat = defaultdict(lambda: np.empty((len(tseq), slant_range.size),
            dtype=np.complex64))
        for tx_pol in {pol[0] for pol in txrx_pols}:
            if tx_pol == "L":
                tx_bmf_pat[tx_pol] = (
                    self.tx_bmf['H'].form_pattern(
                        t, slant_range, nearest=nearest,
                        channel_adj_factors=self.channel_adj_fact_tx['H']) +
                    1j * self.tx_bmf['V'].form_pattern(
                        t, slant_range, nearest=nearest,
                        channel_adj_factors=self.channel_adj_fact_tx['V'])
                ).astype(np.complex64)

            elif tx_pol == "R":
                tx_bmf_pat[tx_pol] = (
                    self.tx_bmf['H'].form_pattern(
                        t, slant_range, nearest=nearest,
                        channel_adj_factors=self.channel_adj_fact_tx['H']) -
                    1j * self.tx_bmf['V'].form_pattern(
                        t, slant_range, nearest=nearest,
                        channel_adj_factors=self.channel_adj_fact_tx['V'])
                ).astype(np.complex64)

            else:  # other non-compact pol types
                adj = self.channel_adj_fact_tx[tx_pol]
                tx_bmf_pat[tx_pol] = self.tx_bmf[tx_pol].form_pattern(
                        t, slant_range, nearest=nearest, channel_adj_factors=adj
                    ).astype(np.complex64)

        # build two-way pattern for all unique TxRx products obtained from all
        # freq bands
        pat2w = dict()
        for pol in txrx_pols:
            tx_p, rx_p = pol[0], pol[1]
            pat2w[pol] = np.squeeze(tx_bmf_pat[tx_p] * rx_dbf_pat[rx_p])

        return pat2w
