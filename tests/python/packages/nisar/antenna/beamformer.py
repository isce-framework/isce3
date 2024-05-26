import iscetest
from isce3.geometry import DEMInterpolator
from nisar.products.readers.antenna import AntennaParser
from nisar.products.readers.instrument import InstrumentParser
from nisar.products.readers.Raw import Raw
from nisar.antenna import TxTrmInfo, RxTrmInfo, TxBMF, RxDBF
from nisar.antenna.beamformer import get_pulse_index
from nisar.workflows.focus import make_doppler_lut
from isce3.focus import make_el_lut
from isce3.antenna import ant2rgdop

import isce3
import math
import numpy as np
import numpy.testing as npt
import os
from copy import deepcopy
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


#  Some helper functions
def amp2db(amp):
    """Complex amplitude to power in dB"""
    return 20 * np.log10(abs(amp))


def amp2deg(amp):
    """Complex amplitude to phase in deg"""
    return np.angle(amp, deg=True)


def ref_rxdbf_txbmf_from_ant(ant, orbit, attitude, dem_interp, slant_range,
                             pulse_time_mid, txrx_pol):
    """Get RX DBF and TX BMF EL-cut paterns from input antenna object to be
    used as references for validation of final averaged beamformed patterns.

    These complex beamformed patterns are resampled at mid
    azimuth/pulse time as a function of slant range.

    Parameters
    ----------
    ant : nisar.products.readers.antenna.AntennaParser
    orbit : isce3.core.orbit
    attitude : isce3.core.attitude
    dem_interp : isce3.geometry.DEMInterpolator
    slant_range : isce3.core.Linspace
    pulse_time_mid : float
            Mid pulse time in seconds wrt to a common ref epoch
    txrx_pol : str
            Two-character transmit-receive polarization

    Returns
    -------
    np.ndarray(complex)
        1-D complex EL-cut RX DBF pattern
    np.ndarray(complex)
        1-D complex EL-cut TX BMF pattern

    """
    # get RX DBF pattern for a certain RX pol
    ant_rx_dbf_pat = ant.fid[
        f'RX_DBF_{txrx_pol[1]}/elevation/copol_pattern'][()]
    # get EL angles
    ant_el_ang = ant.fid[f'RX_DBF_{txrx_pol[1]}/elevation/angle'][()]
    # get azimuth angle for el cuts
    ant_cut_angle = ant.fid[
        f'RX_DBF_{txrx_pol[1]}/elevation/angle'].attrs['cut_angle']
    # get TX BMF pattern for a certain TX pol
    ant_tx_bmf_pat = ant.fid[
        f'TX_BMF_{txrx_pol[1]}/elevation/copol_pattern'][()]
    # calculate slant range from EL angle at mid pulse time
    pos_mid, vel_mid = orbit.interpolate(pulse_time_mid)
    quat_mid = attitude.interpolate(pulse_time_mid)
    ant_sr, _, _ = ant2rgdop(ant_el_ang, ant_cut_angle, pos_mid, vel_mid,
                             quat_mid, 1.0, dem_interp, abs_tol=10)
    # interpolate antenna BMF pattern over output slant range
    sr_out = np.asarray(slant_range)
    ant_tx_bmf_out = np.interp(sr_out, ant_sr, ant_tx_bmf_pat)
    ant_rx_dbf_out = np.interp(sr_out, ant_sr, ant_rx_dbf_pat)

    return ant_rx_dbf_out, ant_tx_bmf_out


def plot_el_cut_data_vs_ref(data, ref, sr, tag):
    """
    Plot EL pattern for the first and last pulse of data and
    compare it with its reference pattern.
    """
    sr_km = np.asarray(sr) * 1e-3
    plt.figure(figsize=(7, 7))
    plt.subplot(211)
    plt.plot(sr_km, amp2db(data).T,
             sr_km, amp2db(ref), 'k--', linewidth=2)
    plt.legend(['First Pulse', 'Last Pulse', 'Reference'], loc='best')
    plt.xlabel('Slant Range (km)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.title(
        f'{tag} EL patterns for the first and last pulse v.s. ref')
    plt.subplot(212)
    plt.plot(sr_km, amp2deg(data).T,
             sr_km, amp2deg(ref), 'k--', linewidth=2)
    plt.legend(['First Pulse', 'Last Pulse', 'Reference'], loc='best')
    plt.xlabel('Slant Range (km)')
    plt.ylabel('Phase (deg)')
    plt.grid(True)
    plt.show()
    plt.close()

# Main test fxiture


class TestElevationBeamformer:

    # sub directory for all test files under "isce3/tests/data"
    sub_dir = 'bf'

    # List of input file under "sub_dir"
    # Note that L0B contains BYPASS cal data!
    l0b_file = 'REE_L0B_ECHO_DATA.h5'
    ant_file = 'REE_ANTPAT_CUTS_DATA.h5'
    instrument_file = 'REE_INSTRUMENT_TABLE.h5'

    # common input parameters
    txrx_pol = 'HH'
    freq_band = 'A'
    ref_height = 0.0  # (m)

    # EL spacing 5mdeg corresponds to around 50 m in slant range
    # Note that RX DBF can take an angle and determine min slant range
    # spacing. Then this value can be used for RxDBF constructor for TxBMF.
    # Alternatively, both constructor can take min spacing in range w/o any
    # internal computation. Both options are exercised for testing RxDBF.
    el_spacing_min = np.deg2rad(5e-3)  # (rad)
    rg_spacing_min = 52.0  # (m)

    # optional plotting of mag/phs EL patterns for debuging
    plot = False

    # absolute tolerances used for phase (deg) and magnitude (dB) errors,
    # both STD and MEAN errors.
    atol_mag_db = 0.05
    atol_phs_deg = 0.25

    # build common objects from input file
    _raw = Raw(hdf5file=os.path.join(iscetest.data, sub_dir, l0b_file))
    _ant = AntennaParser(os.path.join(iscetest.data, sub_dir, ant_file))
    _ins = InstrumentParser(os.path.join(iscetest.data, sub_dir,
                                         instrument_file))

    # Parse ref epoch, pulse time, slant range, orbit and attitude from Raw
    ref_epoch, pulse_time = _raw.getPulseTimes(freq_band, txrx_pol[0])
    orbit = _raw.getOrbit()
    attitude = _raw.getAttitude()
    slant_range = _raw.getRanges(freq_band, txrx_pol[0])

    # pulse time tag used for forming active RX DBF and TX BMF EL paterns
    # to be generated as either any subset of "pulse_time" or any values
    # within [pule_time[0], pulse_time[-1]].
    pulse_time_out = pulse_time[::3]

    # form DEM interpolator object per ref height
    dem_interp = DEMInterpolator(ref_height)

    fc, dop = make_doppler_lut([os.path.join(iscetest.data, sub_dir, l0b_file)],
                               0, orbit, attitude, dem_interp)

    wavelength = isce3.core.speed_of_light / fc
    rdr2geo_params = dict(
        tol_height = 1e-5,
        look_min = 0,
        look_max = math.pi / 2,
    )
    el_lut = make_el_lut(orbit, attitude,
                         _raw.identification.lookDirection,
                         dop, wavelength, dem_interp,
                         rdr2geo_params)

    # Parse Tx-related Cal stuff used only for Tx BMF test cases
    _tx_chanl = _raw.getListOfTxTRMs(freq_band, txrx_pol[0])
    _corr_tap2 = _raw.getChirpCorrelator(freq_band, txrx_pol[0])[..., 1]
    _cal_type = _raw.getCalType(freq_band, txrx_pol[0])
    # build TxTRM  from Tx Cal stuff w/o optional "tx_phase"
    tx_trm = TxTrmInfo(pulse_time, _tx_chanl, _corr_tap2, _cal_type)
    # parse EL patterns for all beams on TX side used only for TX BMF
    # test cases
    el_pat_tx = _ant.el_cut_all(txrx_pol[0])

    # Parse DBF-related RD/WD/WL, time-to-angle(TA) and angle-to-coeffs(AC)
    # tables to be used only  Rx DBF pattern test cases
    # Get DBF-related RD/WD/WL arrays for all RX channels simply for
    # the first range line.
    _rd = _raw.getRD(freq_band, txrx_pol)[0]
    _wd = _raw.getWD(freq_band, txrx_pol)[0]
    _wl = _raw.getWL(freq_band, txrx_pol)[0]
    # Sampling rate in (Hz) for range window parameters RD/WD/WL in NISAR case
    _fs_win = 240e6

    _ta_switch = _ins.get_time2angle(txrx_pol[1])
    _fs_ta = _ins.sampling_rate_ta(txrx_pol[1])
    _dbf_coef = _ins.get_angle2coef(txrx_pol[1])
    _ela_dbf = _ins.el_angles_ac(txrx_pol[1])
    # parse active RX channels
    _rx_chanl = _raw.getListOfRxTRMs(freq_band, txrx_pol)
    # build RxTRM object
    rx_trm = RxTrmInfo(pulse_time, _rx_chanl, _rd, _wd, _wl, _dbf_coef,
                       _ta_switch, _ela_dbf, _fs_win, _fs_ta)
    # parse EL patterns for all beams on RX side used only for RX DBF
    # test cases
    if txrx_pol[0] == txrx_pol[1]:
        el_pat_rx = deepcopy(el_pat_tx)
    else:
        el_pat_rx = _ant.el_cut_all(txrx_pol[1])

    # compute reference RX DBF & TX BMF EL-cut patterns as a function of
    # slant_range at mid azimuth time for a desired TxRx polarization to be
    # used for validation process.
    _mid_time = pulse_time.mean()
    ref_ant_rx_dbf, ref_ant_tx_bmf = ref_rxdbf_txbmf_from_ant(
        _ant, orbit, attitude, dem_interp, slant_range, _mid_time, txrx_pol)

    # check the plot flag
    if plot and plt is None:
        plot = False

    def _validate_el_pattern(self, el_pat_ref, el_pat, err_msg,
                             ignore_mean=False):
        """Validate EL beamformed pattern against its expected values (ref)
        extracted from AntennaParser object

        Parameters
        ----------
        el_pat_ref : np.ndarray(complex)
                Reference 1-D complex EL pattern obtained from antenna file
                as a function of slant range
        el_pat : np.ndarray(complex)
                Computed averaged 1-D complex EL pattern as a function of
                slant range
        err_msg : str, default=''
                Extra error messages appended to the default one.
        ignore_mean : bool, default=False
                Ignore the mean computation and comparison for phase only.
                In reality, both absolute end-to-end phase magnitude of TRMs
                won't be available! Simulated data is an exception!
                See Notes below.

        Notes
        -----
        In general, the STD of the error (difference between Ref & computed)
        shall be solely investigated here because the MEAN can be different
        depending on whether the absolute phase/mag are captured in forming
        computed patterns. Besides, we don't care about overall offset (mean)
        in relative calibration (our main goal is relative calibration here).
        However, per simulation data, MEAN values are computed and verified
        except for one of TX BMF test case where the true TX-path phase is not
        available. In that case, it is well expected to have a phase offset!

        """
        # complex amp ratio
        pat_ratio = el_pat / el_pat_ref

        # compare std [and MEAN] error defined by difference between
        # Ref & Computed magnitudes of EL patterns in (dB)
        pow_dif = amp2db(pat_ratio)

        npt.assert_allclose(pow_dif.std(), 0.0, atol=self.atol_mag_db,
                            err_msg=f'Large STD Mag error (dB) {err_msg}')

        npt.assert_allclose(pow_dif.mean(), 0.0, atol=self.atol_mag_db,
                            err_msg=f'Large MEAN Mag error (dB) {err_msg}')

        # compare std [and MEAN] error defined by difference between
        # Ref & Computed phases of EL patterns in (deg) if requested
        phs_dif = amp2deg(pat_ratio)

        npt.assert_allclose(phs_dif.std(), 0.0, atol=self.atol_phs_deg,
                            err_msg=f'Large STD Phase error (deg) {err_msg}')

        if not ignore_mean:
            npt.assert_allclose(
                phs_dif.mean(), 0.0, atol=self.atol_phs_deg,
                err_msg=f'Large MEAN Phase error (deg) {err_msg}'
            )

    def test_rx_dbf(self):
        for el_lut in (None, self.el_lut):
            # construct RX DBF object
            rx_dbf = RxDBF(self.orbit, self.attitude, self.dem_interp,
                           self.el_pat_rx, self.rx_trm, self.ref_epoch,
                           el_lut=el_lut,
                           norm_weight=True, el_spacing_min=self.el_spacing_min)
            # form RX DBF pattern w/o channel adjustment
            rx_dbf_pat = rx_dbf.form_pattern(self.pulse_time_out, self.slant_range)
            # validate slow-time averaged EL pattern as a function of range
            self._validate_el_pattern(self.ref_ant_rx_dbf, rx_dbf_pat.mean(axis=0),
                                      err_msg='for RX DBF')
            if self.plot:
                plot_el_cut_data_vs_ref(
                    rx_dbf_pat[::self.pulse_time_out.size - 1],
                    self.ref_ant_rx_dbf, self.slant_range, 'RX')

    def test_rx_dbf_with_elofs_chanladj(self):
      for el_lut in (None, self.el_lut):
        # Just to test the code and compare with the same ref pattern for now,
        # consider trivial EL offset relative to DBF Coeffs angular resolution
        # to avoid any noticeable changes in pattern per the abs tolerance.
        el_ofs_deg = 0.01
        # construct RX DBF object w/ el offset and channel adjustment.
        rx_dbf = RxDBF(self.orbit, self.attitude, self.dem_interp,
                       self.el_pat_rx, self.rx_trm, self.ref_epoch,
                       el_lut=el_lut,
                       norm_weight=True, el_ofs_dbf=np.deg2rad(el_ofs_deg),
                       rg_spacing_min=self.rg_spacing_min)
        # Total number of RX channels
        num_chanl_rx, _ = self.rx_trm.ac_dbf_coef.shape
        # fudge factor for RX or None
        rx_chan_adj = np.ones(num_chanl_rx, dtype=complex)
        # form RX DBF pattern w/o channel adjustment
        rx_dbf_pat = rx_dbf.form_pattern(self.pulse_time_out, self.slant_range,
                                         rx_chan_adj)
        # validate slow-time averaged EL pattern as a function of range
        self._validate_el_pattern(self.ref_ant_rx_dbf, rx_dbf_pat.mean(axis=0),
                                  err_msg='for RX DBF w/ trivial EL angle'
                                  'offset and uniform adjustment of channels')

    def test_tx_bmf(self):
      for el_lut in (None, self.el_lut):
        # construct TX BMF object
        tx_bmf = TxBMF(self.orbit, self.attitude, self.dem_interp,
                       self.el_pat_tx, self.tx_trm,
                       self.ref_epoch, el_lut=el_lut, norm_weight=True,
                       rg_spacing_min=self.rg_spacing_min)
        # for TX BMF pattern w/o TX channel adjustment weights
        tx_bmf_pat = tx_bmf.form_pattern(self.pulse_time_out, self.slant_range)
        # validate slow-time averaged EL pattern as a function of range
        # Notice that due to lack of absolute TX-path phase, there will be an
        # offset between Ref and Computed ones. Set "ignore_mean" to True.
        self._validate_el_pattern(self.ref_ant_tx_bmf, tx_bmf_pat.mean(axis=0),
                                  ignore_mean=True, err_msg='for TX BMF')
        if self.plot:
            plot_el_cut_data_vs_ref(
                tx_bmf_pat[::self.pulse_time_out.size - 1],
                self.ref_ant_tx_bmf, self.slant_range, 'TX')

    def test_tx_bmf_with_txphase_chanladj(self):
        for el_lut in (None, self.el_lut):
            # make a copy of the tx_trm object to be modified
            tx_trm = deepcopy(self.tx_trm)
            # set TX-path phases for TxTRMInfo from Raw
            tx_trm.tx_phase = np.deg2rad(
                self._raw.getTxPhase(self.freq_band, self.txrx_pol[0]))
            # construct TX BMF object
            tx_bmf = TxBMF(self.orbit, self.attitude, self.dem_interp,
                           self.el_pat_tx, tx_trm,
                           self.ref_epoch, el_lut=el_lut, norm_weight=True,
                           rg_spacing_min=self.rg_spacing_min)
            # form TX BMF pattern w/ TX channel adjustment weights
            # fudge factors for Tx or None
            _, num_chanl_tx = tx_trm.correlator_tap2.shape
            tx_chanl_adj = np.ones(num_chanl_tx, dtype=complex)
            tx_bmf_pat = tx_bmf.form_pattern(self.pulse_time_out, self.slant_range,
                                             tx_chanl_adj)
            # validate slow-time averaged EL pattern as a function of range
            self._validate_el_pattern(self.ref_ant_tx_bmf, tx_bmf_pat.mean(axis=0),
                                      err_msg='for TX BMF w/ Tx phase data and '
                                      'uniform adjustment of channels')


def test_get_pulse_index():
    times = np.arange(10, dtype=float)
    # floor
    npt.assert_equal(get_pulse_index(times, 1.0), 1)
    npt.assert_equal(get_pulse_index(times, 1.1), 1)
    npt.assert_equal(get_pulse_index(times, 1.9), 1)  # floor
    npt.assert_equal(get_pulse_index(times, 9.0), 9)
    # check snapping, so that 2-eps -> 2
    t = np.nextafter(2.0, 0.0)
    npt.assert_equal(get_pulse_index(times, t), 2)  # doesn't floor to 1.0
    # rounding
    npt.assert_equal(get_pulse_index(times, 1.0, nearest=True), 1)
    npt.assert_equal(get_pulse_index(times, 1.1, nearest=True), 1)
    npt.assert_equal(get_pulse_index(times, 1.9, nearest=True), 2)  # round
    npt.assert_equal(get_pulse_index(times, t, nearest=True), 2)
    # out of bounds
    npt.assert_equal(get_pulse_index(times, -1.0), 0)
    npt.assert_equal(get_pulse_index(times, -0.1), 0)
    npt.assert_equal(get_pulse_index(times, 9.1), 9)
    npt.assert_equal(get_pulse_index(times, 11.0), 9)
