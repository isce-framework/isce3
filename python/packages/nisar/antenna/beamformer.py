import bisect
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d

from isce3.antenna import ant2rgdop, ant2geo
from isce3.core import Linspace, speed_of_light
from nisar.antenna import CalPath
from isce3.geometry import DEMInterpolator


class ElevationBeamformer(ABC):
    """Abstract base class for general (Transmit or Receiver) beamformers in
    Elevation (EL) direction.

    Parameters
    ----------
    orbit : isce3.core.Orbit
        Orbit data for an interval spanning the datatake.
    attitude : isce3.core.Attitude
        Platform attitude data for an interval spanning the datatake.
    dem_interp : isce3.geometry.DEMInterpolator
        Digital elevation model (DEM) of the imaged surface.
    el_ant_info : collections.namedtuple
        Info about antenna multi-channel elevation-cut patterns
        copol_pattern : np.ndarray(complex)
            2-D array of complex multi-channel co-pol patterns of
            shape (beams, el angles)
        angle : np.ndarray(float)
            Elevation angles in (rad)
        cut_angle : float
            AZ angle around which EL-cut patterns are provided in (rad)
    trm_info: either nisar.antenna.TxTrmInfo or nisar.antenna.RxTrmInfo
        data class that contains all relevant attributes to
        compute weights needed to form active pattern in EL.
    ref_epoch : isce3.core.DateTime
        Reference epoch for all time tags in `trm_info`, pulse_time.
        In case, orbit and attitude have different reference epochs,
        the corresponding class attributes will be adjusted to this reference.
        (Input objects will not be modified.)
    norm_weight : bool, default=False
        Whether or not power normalize weights.
    interp_method : str, default='linear'
        Interpolation method for final resampling of patterns in slant range.
        It shall be one the values of `kind` in scipy.interpolate.interp1d.
    num_pulse_skip: int, default=12
        Number of pulses to skip in geomtery computation from antenna frame
        to slant range for the sake of speed-up. The default value is around
        the number of pulses in the air for mid swath of NISAR orbit.

    Attributes
    ----------
    weights : np.ndarray(complex)
        2-D array of complex weights
    active_channel_idx : np.ndarray(int)
        0-based channel indexes of only active channels

    Raises
    ------
    ValueError
        Reference epoch mismtach between orbit and attitude

    """

    def __init__(self, orbit, attitude, dem_interp, el_ant_info, trm_info,
                 ref_epoch, norm_weight=False, interp_method='linear',
                 num_pulse_skip=12):
        # check ref epoch of orbit/attitude
        if orbit.reference_epoch != ref_epoch:
            orbit = orbit.copy()
            orbit.update_reference_epoch(ref_epoch)
        if attitude.reference_epoch != ref_epoch:
            attitude = attitude.copy()
            attitude.update_reference_epoch(ref_epoch)

        self.orbit = orbit
        self.attitude = attitude
        self.dem_interp = dem_interp
        self.el_ant_info = el_ant_info
        self.trm_info = trm_info
        self.ref_epoch = ref_epoch
        self.norm_weight = norm_weight
        self.interp_method = interp_method
        self.num_pulse_skip = num_pulse_skip

        # pre-computed read-only attributes
        self._weights = self._compute_weights()
        self._active_channel_idx = np.asarray(trm_info.channels) - 1

        # get peak location of first and last active beam in EL direction
        self._el_peak_first, self._el_peak_last = self._peak_loc_beam_el()

    @property
    def weights(self):
        return self._weights

    @property
    def active_channel_idx(self):
        return self._active_channel_idx

    @abstractmethod
    def form_pattern(self, pulse_time, slant_range):
        pass

    @abstractmethod
    def _compute_weights(self):
        pass

    def _peak_loc_beam_el(self):
        """Peak location in EL direction for the first and last beam

        Returns
        -------
        tuple(float, float)
            EL angles for the first and the last active beam in radians

        """
        idx_first = abs(self.el_ant_info.copol_pattern[
            self._active_channel_idx[0]]).argmax()
        idx_last = abs(self.el_ant_info.copol_pattern[
            self._active_channel_idx[-1]]).argmax()
        return (self.el_ant_info.angle[idx_first],
                self.el_ant_info.angle[idx_last])

    def _elaz2slantrange(self, pulse_time):
        """
        Get slant ranges over entire antenna elevation angle coverage
        at a specific pulse time.

        Parameters
        ----------
        pulse_time : float
            Transmit pulse time of range line in seconds w.r.t `ref_epoch`.

        Returns
        -------
        np.ndarray(float)
            Slant range computed at antenna pattern elevation angles

        Notes
        -----
        The `pulse_time` is assumed to be wrt the same reference epoch as
        that of orbit and attitude data.

        """
        # No need for high accuracy in height tolerance when it comes to
        # antenna EL pattern due to no changes in its phs/mag within 1 mdeg EL
        # angle which is equivalent to around 10 m in height change
        # at around mid swath.
        abs_hgt_tol = 10.0  # (m)

        # get pos/vel in ECEF
        pos_ecef, vel_ecef = self.orbit.interpolate(pulse_time)

        # get quaternion from antenna to ECEF
        q_ant2ecef = self.attitude.interpolate(pulse_time)

        # convert "N" EL angles within peak location of [first, last] beam
        # into geodetic location LLH and then compute mean DEM height
        # across the entire swath at a desired pulse/azimuth time.
        # "N" is determined by setting EL angle spacing = 150 mdeg.
        ela_spacing = np.deg2rad(0.15)
        num_ela = round((self._el_peak_last - self._el_peak_first) /
                        ela_spacing) + 1
        ela = np.linspace(self._el_peak_first, self._el_peak_last, num=num_ela)
        llh, _ = ant2geo(ela, self.el_ant_info.cut_angle, pos_ecef, q_ant2ecef,
                         self.dem_interp, abs_tol=abs_hgt_tol)
        dem_avg = np.asarray(llh)[:, -1].mean()
        # build a new DEM interplator based on locallly averaged height.
        # This DEM will be used to avoid non-monotonic slant ranges issue
        # with large topography due to overlays (non-homomorphism)
        # As a result, there is an error between actual antenna patterns and
        # the computed ones, but that error is miminized on average across
        # the swath at an azimuth time. Besides, there is no way to get a
        # unique el angle and thus antenna weight for a desired slant range
        # in overlay case by starting from radar geometry.
        dem_interp_avg = DEMInterpolator(dem_avg)

        # set wavelength to 1.0 below given Doppler is not needed here!
        # Convert Elevation angle to slant range
        slant_range, _, _ = ant2rgdop(
            self.el_ant_info.angle,
            self.el_ant_info.cut_angle,
            pos_ecef,
            vel_ecef,
            q_ant2ecef,
            1.0,
            dem_interp_avg,
            abs_tol=abs_hgt_tol
        )
        return slant_range


class TxBMF(ElevationBeamformer):
    """Transmit beamformer (TxBMF) in Elevation (EL) direction.

    This class sets up Tx beamformer per transmit-receiver-module (TRM) info
    and allows formation of active transmit pattern as a function of
    slant range per polarization.

    Parameters
    ----------
    orbit : isce3.core.Orbit
    attitude : isce3.core.Attitude
    dem_interp : isce3.geometry.DEMInterpolator
    el_ant_info : collections.namedtuple
        Info about antenna multi-channel elevation-cut patterns
        copol_pattern : np.ndarray(complex)
            2-D array of complex multi-channel co-pol patterns of
            shape (beams, el angles)
        angle : np.ndarray(float)
            Elevation angles in (rad)
        cut_angle : float
            AZ angle around which EL-cut patterns are provided in (rad)
    trm_info : nisar.antenna.TxTrmInfo
        data class that contains all relevant attributes needed to
        compute Tx weights needed to form active TX pattern
    ref_epoch : isce3.core.DateTime
        Reference epoch for all time tags in `trm_info`, pulse_time.
        In case, or bit and attitude have difference reference epoch,
        they will be adjusted to this reference.
    norm_weight : bool, default=False
        Whether or not power normalize TX weights.
    interp_method : str, default='linear'
        Interpolation method for final resampling of patterns in slant range.
        It shall be one of the values of `kind` in scipy.interpolate.interp1d.
    num_pulse_skip: int, default=12
        Number of pulses to skip in geomtery computation from antenna frame
        to slant range for the sake of speed-up. The default value is around
        the number of pulses in the air for mid swath of NISAR orbit.

    Attributes
    ----------
    weights : np.ndarray(complex)
        2-D array with the shape (rangelines, active TX channels)
    active_channel_idx : np.ndarray(int)
        0-based Tx active channel indexes

    Raises
    ------
    ValueError
        Reference epoch mismtach between orbit and attitude
        Mismatch between total number of beams and number of channels

    """

    def form_pattern(self, pulse_time, slant_range, channel_adj_factors=None,
                     is_nearest=False):
        """
        Form transmit beamformed (BMF) pattern, as a function
        of slant range @ each azimuth pulse time.

        Parameters
        ----------
        pulse_time : scalar or array of float
             transmit time of range line in seconds w.r.t `ref_epoch`,
             single or multiple range lines. In case of array, it is assumed
             to be sorted in ascending order.
        slant_range : isce3.core.Linspace
            Slant ranges in meters.
        channel_adj_factors : Sequence of complex, optional
            These are extra fudge factors to balance/adjust Tx channels.
            The TX weights will be multiplied by these factors per
            channel. The size is total number of TX channels. If None,
            no correction will be applied to TX weights.
            Note that these factors will NOT be peak/power normalized.
        is_nearest: bool, default=False
            Whether to get neartest TX-TRM pulse index for a desired pulse
            time. If False, the index search operation will be "floor".
            Otherwise, it will be "nearest".

        Returns
        -------
        np.ndarray(complex)
            2D array Tx BMFed pattern with
            shape (pulse_time.size, slant_range.size)

        Riases
        ------
        ValueError
            pulse_time is out of the range of time tags of `tx_trm_info`.
            `channel_adj_factors` are zeros for all active TX channels.
            Size of `channel_adj_factors` is not equal to total TX channels.

        """
        # absolute tolerance in time error (sec) used for getting pulse index
        atol_time_err = 5e-10

        if self.num_pulse_skip < 1:
            raise ValueError('Number of pulses to be skipped shall be a'
                             ' positive integer!')
        if np.isscalar(pulse_time):
            pulse_time = [pulse_time]

        # check the pulse_time to be within time tag of TxTRM
        if (pulse_time[0] < self.trm_info.time[0] or
                pulse_time[-1] > self.trm_info.time[-1]):
            raise ValueError(
                f'Pulse time is out of Tx time tag [{self.trm_info.time[0]}, '
                f'{self.trm_info.time[-1]}] (sec, sec)!'
            )
        # get total number of TX channels
        _, num_chanl = self.trm_info.correlator_tap2.shape

        # apply correction/fudge factor to Tx weights along active channels
        # if provided
        if channel_adj_factors is not None:
            # check the size of corection factor container
            if len(channel_adj_factors) != num_chanl:
                raise ValueError('Size of TX "channel adjustment factor" '
                                 f'must be {num_chanl}')
            # check if the correction factor is zero for all active channels
            cor_fact = np.asarray(channel_adj_factors)[self.active_channel_idx]
            if np.isclose(abs(cor_fact).max(), 0):
                raise ValueError('"channel_adj_factors" are zeros for all '
                                 'active TX channels!')
            tx_weights = self.weights * cor_fact
        else:
            tx_weights = self.weights

        # get the antenna EL patterns for active TX channels
        ant_pat_el = self.el_ant_info.copol_pattern[self.active_channel_idx]

        # initialize the tx pattern
        tx_pat = np.zeros((len(pulse_time), slant_range.size), dtype='complex')

        # get slant range vector from its Linspace to be used in the
        # interpolation
        sr = np.asarray(slant_range)

        # loop over pulses
        for pp, tm in enumerate(pulse_time):

            # Get the respective pulse index
            # Note that for floor operation, the closest TRM time which is less
            # or equal to desired pulse time will be picked.
            # This is due to possible step-like phase jumps between TX pulses
            # dominated by random-like phase toggeling. The phase of TX can not
            # noticeably vary faster than one PRI during which the TX-path
            # phase is pretty constant! So, for all times limited between "i"
            # (inclusive) and "i+1" (exclusive) pulses, the corresponding TX
            # weights for pulse "i" shall be used. Super slow-varying thermal
            # phase/amp gradient is ignored.
            # On the other hand, in nearest case, the closest TRM time will be
            # picked for a desired pulse time. That is either "i" or "i+1".

            # Floor operation
            idx_right = bisect.bisect_right(self.trm_info.time, tm)
            idx_pulse = idx_right - 1
            # Nearest opration
            if is_nearest:
                if (idx_right < self.trm_info.time.size and
                    ((self.trm_info.time[idx_right] - tm) <
                     (tm - self.trm_info.time[idx_pulse]))):
                    idx_pulse = idx_right
            else:
                # Check for nano-second resolution time offset at the right
                # side for "Floor" case to avoid precision issue assuming
                # slow-time tags are within one nano-second resolution!
                # That is, the acceptable max abs error is "0.5 nano second".
                if (self.trm_info.time[idx_right] - tm) <= atol_time_err:
                    idx_pulse = idx_right

            # form TX pattern in antenna EL angle domain
            tx_pat_el = tx_weights[idx_pulse] @ ant_pat_el

            # Compute the respective slant range for beamformed antenna pattern
            # Simply calculate slant range for every few pulses where
            # S/C pos/vel and DEM barely changes. This speeds up the process!
            if (pp % self.num_pulse_skip == 0):
                sr_ant = self._elaz2slantrange(tm)

            # form TX BMF pattern at desird slant ranges
            func_interp = interp1d(sr_ant, tx_pat_el, kind=self.interp_method,
                                   fill_value='extrapolate', copy=False,
                                   assume_sorted=True)
            tx_pat[pp] = func_interp(sr)

        return tx_pat

    def _compute_weights(self):
        """Compute TX weights for all range lines and active TX channels.

        For those noise-only range lines, the nearest neighnors with
        HCAL will be reported.

        In case of missing BCAL, use the HCAL rather than HCAL/BCAL ratio.

        Returns
        -------
        np.ndarray(complex)
            2-D array with the shape (rangelines, active TX channels).

        Raises
        ------
        ValueError:
            Shape mistmatch between 2-D array of correlator and tx phase
            if exists.
        RuntimeError
            For zero BCAL values.

        Notes
        -----
        The weights are formed by |HCAL/(BCAL/BCAL[0])|*exp(j*TxPhase).
        In case, TxPhase is not provided(None), use HCAL/(BCAL/BCAL[0])
        as complex weights.

        """
        return compute_transmit_pattern_weights(self.trm_info,
                                                norm=self.norm_weight)


class RxDBF(ElevationBeamformer):
    """Receive digital beamformer (RxDBF) in Elevation (EL) direction.

    This class sets up Rx beamformer per transmit-receiver-module (TRM) info
    and allows  formation of active receive pattern as a function of
    slant range per polarization.

    Parameters
    ----------
    orbit : isce3.core.Orbit
    attitude : isce3.core.Attitude
    dem_interp : isce3.geometry.DEMInterpolator
    el_ant_info : collections.namedtuple
        Info about antenna multi-channel elevation-cut patterns
        copol_pattern : np.ndarray(complex)
            2-D array of complex multi-channel co-pol patterns of
            shape (beams, el angles)
        angle : np.ndarray(float)
            Elevation angles in (rad)
        cut_angle : float
            AZ angle around which EL-cut patterns are provided in (rad)
    trm_info : nisar.antenna.RxTrmInfo
        data class that contains all relevant attributes needed to
        compute Rx weights needed to form active RX pattern
    ref_epoch : isce3.core.DateTime
        Reference epoch for all time tags in `trm_info`, pulse_time.
        In case, or bit and attitude have difference reference epoch,
        they will be adjusted to this reference.
    norm_weight : bool, default=False
        Whether or not power normalize RX weights.
    interp_method : str, default='linear'
        Interpolation method for final resampling of patterns in slant range.
        It shall be one of vlues of the `kind` in scipy.interpolate.interp1d.
    num_pulse_skip: int, default=12
        Number of pulses to skip in geomtery computation from antenna frame
        to slant range for the sake of speed-up. The default value is around
        the number of pulses in the air for mid swath of NISAR orbit.
    el_ofs_dbf : float, default=0.0
        Elevation angle offset (in radians) used in adjusting angle indexes
        prior to grabing DBF coeffs from AC table in DBF process.

    Attributes
    ----------
    weights : np.ndarray(complex)
        2-D array with the shape (active RX channels, rangebins) @ DBF
        sampling rate
    active_channel_idx : np.ndarray(int)
        0-based Rx active channel indexes
    slant_range_dbf : isce3.core.Linspace
        Slant ranges correspond to fast-time DBF weights @ TA sampling rate
        ,`rx_trm_info.fs_ta`, used for all range lines within pulse time.

    Raises
    ------
    ValueError
        Reference epoch mismtach between orbit and attitude
        Mismatch between total number of beams and number of channels

    """

    def __init__(self, orbit, attitude, dem_interp, el_ant_info, trm_info,
                 ref_epoch, norm_weight=False, interp_method='linear',
                 num_pulse_skip=12, el_ofs_dbf=0.0):
        self.el_ofs_dbf = el_ofs_dbf
        super().__init__(orbit, attitude, dem_interp, el_ant_info, trm_info,
                         ref_epoch, norm_weight, interp_method, num_pulse_skip)

    @property
    def slant_range_dbf(self):
        return self._slant_range_dbf

    def form_pattern(self, pulse_time, slant_range, channel_adj_factors=None):
        """
        Form receive digitally beamformed (DBF) pattern, as a function
        of slant range @ each azimuth pulse time.

        Parameters
        ----------
        pulse_time : scalar or array of float
             transmit time of range line in seconds w.r.t `ref_epoch`,
             single or multiple range lines. In case of array, it is assumed
             to be sorted in ascending order.
        slant_range : isce3.core.Linspace
            Slant ranges in meters.
        channel_adj_factors : Sequence of complex, optional
            A place holder for applying possbile secondary Rx channel
            corrections or any extra fudge factors to balance/adjust Rx
            channels. The RX weights will be multiplied by these factors per
            channel. The size is total number of RX channels. If None,
            no correction will be applied to RX weights. Note that these
            factors will NOT be peak/power normalized.

        Returns
        -------
        np.ndarray(complex)
            2D array Rx DBFed pattern with
            shape (pulse_time.size, slant_range.size)

        Raises
        ------
        ValueError
            pulse_time is out of the range of time tags of `rx_trm_info`.
            `channel_adj_factors` are zeros for all active RX channels.
            Size of `channel_adj_factors` is not equal to total RX channels.

        """
        if self.num_pulse_skip < 1:
            raise ValueError('Number of pulses to be skipped shall be a'
                             ' positive integer!')
        # get total number of RX channels
        num_chanl, _ = self.trm_info.ac_dbf_coef.shape

        if np.isscalar(pulse_time):
            pulse_time = [pulse_time]

        # check the pulse_time to be within time tag of RxTRM
        if (pulse_time[0] < self.trm_info.time[0] or
                pulse_time[-1] > self.trm_info.time[-1]):
            raise ValueError(
                f'Pulse time is out of Rx time tag [{self.trm_info.time[0]}, '
                f'{self.trm_info.time[-1]}] (sec, sec)!'
            )

        # EL-cut pattern with shape active beams by EL angles
        ant_pat_el = self.el_ant_info.copol_pattern[self.active_channel_idx]

        # get slant range vector from its Linspace to be used in the
        # interpolation
        sr = np.asarray(slant_range)

        # resample RX weightings to the output slant range
        # use simply nearest neighbor given RX weights are very finely sampled!
        func_interp_wgt = interp1d(self.slant_range_dbf, self.weights,
                                   kind='nearest', fill_value='extrapolate',
                                   copy=False, assume_sorted=True, axis=1)
        # RX weights with shape active beams by output slant ranges
        rx_wgt = func_interp_wgt(sr)

        # apply correction/fudge factor to Rx weights along active channels
        # if provided
        if channel_adj_factors is not None:
            # check the size of corection factor container
            if len(channel_adj_factors) != num_chanl:
                raise ValueError('Size of RX "channel adjustment factor" '
                                 f'must be {num_chanl}')
            # check if the correction factor is zero for all active channels
            cor_fact = np.asarray(channel_adj_factors)[self.active_channel_idx]
            if np.isclose(abs(cor_fact).max(), 0):
                raise ValueError('"channel_adj_factors" are zeros for all '
                                 'active RX channels!')
            rx_wgt *= cor_fact[:, None]

        # initialize the RX DBF pattern
        rx_pat = np.zeros((len(pulse_time), slant_range.size), dtype='complex')

        # loop over pulses
        for pp, tm in enumerate(pulse_time):
            # Compute the respective slant range for beamformed antenna pattern
            # Simply calculate slant range for every few pulses where
            # S/C pos/vel and DEM barely changes. This speeds up the process!
            if (pp % self.num_pulse_skip == 0):
                sr_ant = self._elaz2slantrange(tm)

            # resample EL-cut antenna patterns to the output slant range
            func_interp_ant = interp1d(sr_ant, ant_pat_el,
                                       kind=self.interp_method,
                                       fill_value='extrapolate', axis=1,
                                       copy=False, assume_sorted=True)

            # form the RX DBF pattern for output slant ranges per pulse
            rx_pat[pp] = (func_interp_ant(sr) * rx_wgt).sum(axis=0)

        return rx_pat

    def _compute_weights(self):
        """Compute RX weights for all range bins and active RX channels.

        Returns
        -------
        np.ndarray(complex)
            2-D array with the shape (active RX channels, rangebins).

        Raises
        ------
        ValueError:
            Shape mistmatch between DBF TA and AC arrays

        """
        rx_weights, slant_range_dbf = compute_receive_pattern_weights(
            self.trm_info, el_ofs=self.el_ofs_dbf, norm=self.norm_weight)
        # set the read-only property for slant range values @ DBF clock rate
        self._slant_range_dbf = slant_range_dbf
        return rx_weights


# List of public functions

def get_calib_range_line_idx(cal_path_mask):
    """
    Get range line index for each calbration path enumeration type
    "CalPath" from calibration mask array stored in L0B product.

    Each rangeline contains either of calibration measurments HCAL,
    BCAL or LCAL.

    It also reports noise-only (no transmit) range line indexes.

    Parameters
    ----------
    cal_path_mask : np.ndarray(CalPath)

    Returns
    -------
    np.ndarray(uint32)
        HPA CAL range line indices
    np.ndarray(uint32)
        BYPASS CAL range line indices
    np.ndarray(uint32)
        LNA CAL range line indices
    np.ndarray(uint32)
        Noise-only range line indices

    """
    rng_lines_idx = np.arange(cal_path_mask.size, dtype='uint32')
    hcal_lines_idx = rng_lines_idx[cal_path_mask == CalPath.HPA]
    lcal_lines_idx = rng_lines_idx[cal_path_mask == CalPath.LNA]
    bcal_lines_idx = rng_lines_idx[cal_path_mask == CalPath.BYPASS]
    noise_lines_idx = rng_lines_idx[cal_path_mask != CalPath.HPA]

    return hcal_lines_idx, bcal_lines_idx, lcal_lines_idx, noise_lines_idx


def compute_transmit_pattern_weights(tx_trm_info, norm=False):
    """Compute TX weights for all range lines and active TX channels.

    This is part of beam formed (BMF) Tx antenna pattern.
    See [1]_ for detailed descripion of the Tx calibration algorithm.

    Parameters
    ----------
    tx_trm_info : nisar.antenna.TxTrmInfo
    norm : bool, default=False
        Whether or not power-normalize the weights.

    Returns
    -------
    np.ndarray(complex)
        2-D array with the shape (rangelines, active TX channels).

    Raises
    ------
    ValueError:
        Shape mistmatch between 2-D array of correlator and tx phase if exists
        Wrong size of correction factor
    RuntimeError
        For zero BCAL values.

    Notes
    -----
    The weights are formed by |HCAL/(BCAL/BCAL[0])|*exp(j*TxPhase).
    In case, TxPhase is not provided(None), use HCAL/(BCAL/BCAL[0])
    as complex weights.

    For those noise-only range lines, the nearest neighnors with
    HCAL will be reported.
    In case of missing BCAL, use the HCAL rather than HCAL/BCAL ratio.

    References
    ----------
    .. [1] H. Ghaemi, "DSI SweepSAR On-Board DSP Algorithms Description ,"
        JPL D-95646, Rev 14, 2018.

    """
    num_rgl, num_chanl = tx_trm_info.correlator_tap2.shape

    # get the index of active TX channels
    active_tx_idx = np.asarray(tx_trm_info.channels) - 1

    # get range line index for each type of Cal Path
    hcal_lines_idx, bcal_lines_idx, _, noise_lines_idx = \
        get_calib_range_line_idx(tx_trm_info.cal_path_mask)

    # initialize tx weights with 2nd tap of correlator
    tx_weights = tx_trm_info.correlator_tap2[:, active_tx_idx]

    # If BCAL exists compute ratio HCAL/(BCAL/BCAL[0])
    if bcal_lines_idx.size:
        # check BCAL to make sure it's non-zero value for active channels only!
        if np.isclose(abs(tx_weights[bcal_lines_idx]).min(), 0):
            raise RuntimeError('Zero-value BCAL data is encountered!')

        for line_start, line_stop in zip(bcal_lines_idx[:-1],
                                         bcal_lines_idx[1:]):
            # normalize BCAL wrt to the first TX channel to get relative
            # channel-to-channel BCAL
            bcal_rel = tx_weights[line_start] / tx_weights[line_start, 0]
            # get the ratio except for the last one
            tx_weights[line_start:line_stop] /= bcal_rel

        # get the last ratio
        bcal_rel = (tx_weights[bcal_lines_idx[-1]] /
                    tx_weights[bcal_lines_idx[-1], 0])
        tx_weights[bcal_lines_idx[-1]:] /= bcal_rel

    # Now fill in noise-only range lines with nearest neighbor values
    # from HCAL ones
    func_nearest = interp1d(hcal_lines_idx, tx_weights[hcal_lines_idx],
                            kind='nearest', fill_value='extrapolate',
                            assume_sorted=True, axis=0)
    tx_weights[noise_lines_idx] = func_nearest(noise_lines_idx)

    # power normalize across channels if requested
    if norm:
        # replace zero-value norms with unity to avoid bad normalization for
        # bad/trivial values!
        norm_vals = np.linalg.norm(tx_weights, axis=1)
        norm_vals[np.isclose(norm_vals, 0)] = 1
        tx_weights /= norm_vals[:, None]

    # check if tx_phase exists and if so use it for the phase part of
    # final weights
    if tx_trm_info.tx_phase is None:
        return tx_weights

    # check the size of tx_phase
    if tx_trm_info.tx_phase.shape != (num_rgl, num_chanl):
        raise ValueError(
            'Shape mismtach between "tx_phase" and "correlator_tap2"')
    return abs(tx_weights) * np.exp(1j * tx_trm_info.tx_phase)


def compute_receive_pattern_weights(rx_trm_info, el_ofs=0.0, norm=False):
    """Compute RX weights for all range bins and active RX channels.

    This is part of digitally beam formed (DBF) Rx antenna pattern.
    See [1]_ for detailed descripion of the DBF algorithm.

    Parameters
    ----------
    rx_trm_info : nisar.antenna.RxTrmInfo
    el_ofs : float, default=0.0
        Elevation (EL) angle offset in radians.
        This will adjust angle index used to grab DBF coeffs from
        angle-to-coefficient (AC) table. This will account for any known
        considerable mis-pointing in EL on DBF side of active RX pattern.
    norm : bool, default=False
        Whether or not power-normalize the weights.

    Returns
    -------
    np.ndarray(complex)
        2-D array of weights with shape (active RX channels, rangebins).
    isce3.core.Linspace
        Slant ranges in (m) @ sampling rate equals to `rx_trm_info.fs_ta`

    Raises
    ------
    ValueError:
        Shape mistmatch between DBF TA and AC arrays
        Wrong size of correction factor

    References
    ----------
    .. [1] H. Ghaemi, "DSI SweepSAR On-Board DSP Algorithms Description ,"
        JPL D-95646, Rev 14, 2018.

    """
    # check the shape of two tables AC and TA to be consistent
    num_chanl, num_coefs = rx_trm_info.ac_dbf_coef.shape
    if rx_trm_info.ta_dbf_switch.shape != (num_chanl, num_coefs):
        raise ValueError('Shape mismtach between TA and AC table!')

    # get the index of active RX channels
    active_rx_idx = np.asarray(rx_trm_info.channels) - 1

    # Compute rd, wd, and wl based on data sampling rate of DBF process
    # which is sampling rate of entires of TA table.
    rd_dbf = np.round(rx_trm_info.rd * rx_trm_info.fs_ta /
                      rx_trm_info.fs_win).astype(int)
    wd_dbf = np.round(rx_trm_info.wd * rx_trm_info.fs_ta /
                      rx_trm_info.fs_win).astype(int)
    wl_dbf = np.round(rx_trm_info.wl * rx_trm_info.fs_ta /
                      rx_trm_info.fs_win).astype(int)

    max_idx_ta = np.size(rx_trm_info.ta_dbf_switch, axis=1) - 1

    # get index adjustment to AC table per EL angle offset for all
    # active channels
    if abs(el_ofs) > 0:
        # get (averaged) EL spacing for all active channels
        el_spacing = np.diff(
            rx_trm_info.el_ang_dbf[active_rx_idx]).mean(axis=1)
        idx_ofs = np.int_(np.round(el_ofs / el_spacing))
    else:  # no EL angle offset
        idx_ofs = np.zeros(len(active_rx_idx), dtype='int')

    # total number of range bins of a DBFed composite range line
    dwp_first = rd_dbf[active_rx_idx[0]] + wd_dbf[active_rx_idx[0]]
    dwp_last = rd_dbf[active_rx_idx[-1]] + wd_dbf[active_rx_idx[-1]]
    num_rgb_dbf = dwp_last - dwp_first + wl_dbf[active_rx_idx[-1]]

    # initialize the complex RX weights
    rx_weights = np.zeros((active_rx_idx.size, num_rgb_dbf), dtype='complex')

    # loop over active channels
    for cc, c_idx in enumerate(active_rx_idx):

        # get all indexes/addresses to angle-coefs table for each channel
        idx_ang = np.searchsorted(
            rx_trm_info.ta_dbf_switch[c_idx], np.arange(
                wd_dbf[c_idx], wd_dbf[c_idx] + wl_dbf[c_idx])
            )
        # adjust index to angle-coeffs table per EL angle offset if necessary
        if idx_ofs[cc] != 0:
            idx_ang += idx_ofs[cc]

        # check and limit (min, max) of final EL angle indexes
        idx_ang[idx_ang < 0] = 0
        idx_ang[idx_ang > max_idx_ta] = max_idx_ta

        # [start, stop) range bins of composite range line
        rgb_start = rd_dbf[c_idx] + wd_dbf[c_idx] - dwp_first
        rgb_stop = rgb_start + wl_dbf[c_idx]
        rx_weights[cc, rgb_start:rgb_stop] = \
            rx_trm_info.ac_dbf_coef[c_idx, idx_ang]

    # normalize the weights across channels per range bin if requested
    if norm:
        # replace zero-value norms with unity to avoid bad normalization for
        # bad/trivial values!
        norm_vals = np.linalg.norm(rx_weights, axis=0)
        norm_vals[np.isclose(norm_vals, 0)] = 1
        rx_weights /= norm_vals

    # form slant range vector for composite/DBF range line @ rx_trm_info.fs_ta
    sr_spacing = 0.5 * speed_of_light / rx_trm_info.fs_ta
    sr_first = dwp_first * sr_spacing
    sr_dbf = Linspace(sr_first, sr_spacing, num_rgb_dbf)

    return rx_weights, sr_dbf
